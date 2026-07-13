/// \file modeling_qwen3_5_omni_audio.cpp
/// \brief Qwen3_5_Omni audio preprocessing (log-mel spectrogram).
/// \author FastFlowLM Team
/// \note 128-bin log-mel pipeline ported from the gemma4e audio path; the omni
///       audio payload is field-for-field identical to gemma4e's.

#include "AutoModel/modeling_qwen3_5_omni.hpp"
#include "audio_process_utils/audioproc.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

audio_data_t Qwen3_5_Omni::load_audio(const std::string& filename, int resample_rate, MonoDownmixMode downmix) {
    audio_data_t result;
    if (!audio_reader_.load_audio(filename, result, resample_rate, downmix)) {
        header_print("ERROR", "Qwen3_5_Omni failed to load audio: " << filename);
        exit(-1);
    }
    return result;
}

audio_data_t Qwen3_5_Omni::load_audio_base64(const std::string& base64_str, int resample_rate, MonoDownmixMode downmix) {
    audio_data_t result;
    std::string audio_bytes = base64::from_base64(base64_str);
    if (!audio_reader_.load_audio_from_memory(
            reinterpret_cast<const uint8_t*>(audio_bytes.data()), audio_bytes.size(),
            result, resample_rate, downmix)) {
        header_print("ERROR", "Qwen3_5_Omni failed to load audio from base64 string");
        exit(-1);
    }
    return result;
}

std::vector<audio_data_t> Qwen3_5_Omni::clip_audio_length(audio_data_t& audio, double max_duration_second) {
    std::vector<audio_data_t> audio_chunks;
    size_t max_frames = static_cast<size_t>(max_duration_second * audio.sample_rate);

    size_t total_frames = audio.num_frames;
    size_t chunk_start_frame = 0;

    while (chunk_start_frame < total_frames) {
        size_t chunk_end_frame = std::min(chunk_start_frame + max_frames, total_frames);
        size_t chunk_start_sample = chunk_start_frame * audio.channels;
        size_t chunk_end_sample = chunk_end_frame * audio.channels;

        audio_data_t chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.channels = audio.channels;
        chunk.num_frames = chunk_end_frame - chunk_start_frame;
        chunk.num_samples = chunk.num_frames * audio.channels;
        chunk.duration_seconds = static_cast<double>(chunk.num_frames) / audio.sample_rate;
        chunk.samples.assign(audio.samples.begin() + chunk_start_sample,
                             audio.samples.begin() + chunk_end_sample);

        audio_chunks.push_back(std::move(chunk));
        chunk_start_frame = chunk_end_frame;
    }
    return audio_chunks;
}

/// \brief Number of audio soft tokens produced from a mel-frame count.
/// \note The engine's audio encoder is not implemented yet; use the gemma4e-style
///       two conv2d downsampling simulation (kernel=3, stride=2, padding=1) as a
///       reasonable placeholder so token expansion stays self-consistent.
unsigned int Qwen3_5_Omni::compute_audio_soft_tokens(int num_mel_frames) {
    if (num_mel_frames <= 0) return 0;
    constexpr int kernel = 3;
    constexpr int stride = 2;
    constexpr int padding = 1;
    int t = num_mel_frames;
    for (int layer = 0; layer < 2; layer++) {
        int t_padded = t + 2 * padding;
        t = (t_padded - kernel) / stride + 1;
    }
    return static_cast<unsigned int>(std::max(t, 0));
}

void Qwen3_5_Omni::extract_spectrogram(std::vector<audio_data_t>& audio_inputs,
                                       qwen3_5_omni_audio_payload_t& audio_payload) {
    audio_payload.num_audios = static_cast<unsigned int>(audio_inputs.size());
    audio_payload.mel_spectrograms.resize(audio_payload.num_audios);
    audio_payload.mel_spectrogram_frames_per_audio.resize(audio_payload.num_audios);
    audio_payload.mel_spectrogram_bins_per_audio.resize(audio_payload.num_audios);

    // Config (matches the gemma4e / HF feature extractor defaults)
    constexpr float frame_length_ms = 20.0f;
    constexpr float hop_length_ms   = 10.0f;
    constexpr float min_frequency   = 0.0f;
    constexpr float max_frequency   = 8000.0f;
    constexpr float mel_floor       = 1e-3f;
    constexpr int   feature_size    = 128; // num_mel_filters
    constexpr bool  fft_overdrive   = false;

    for (unsigned int audio_idx = 0; audio_idx < audio_payload.num_audios; audio_idx++) {
        audio_data_t& audio_input = audio_inputs[audio_idx];
        const int sampling_rate = audio_input.sample_rate;

        const int frame_length = static_cast<int>(std::round(sampling_rate * frame_length_ms / 1000.0f));
        const int hop_length   = static_cast<int>(std::round(sampling_rate * hop_length_ms / 1000.0f));

        int fft_length = 1;
        while (fft_length < frame_length) fft_length <<= 1;
        if (fft_overdrive) fft_length *= 2;

        const int num_frequency_bins = fft_length / 2 + 1;

        std::vector<float> window = audioproc::window_function_optimized(frame_length, "hann", /*periodic=*/true);
        std::vector<float> mel_filters = audioproc::mel_filter_bank_optimized(
            num_frequency_bins, feature_size,
            min_frequency, max_frequency,
            sampling_rate, /*apply_slaney_norm=*/false);

        const float* waveform_ptr = audio_input.samples.data();
        const int original_length = static_cast<int>(audio_input.num_frames);

        const int pad_left = frame_length / 2;
        const int padded_length = original_length + pad_left;
        std::vector<float> waveform(padded_length, 0.0f);
        std::memcpy(waveform.data() + pad_left, waveform_ptr, original_length * sizeof(float));

        const int frame_size_for_unfold = frame_length + 1;
        const int num_frames_out = (padded_length - frame_size_for_unfold) / hop_length + 1;
        if (num_frames_out <= 0) continue;

        std::vector<float> windowed_frames(static_cast<size_t>(num_frames_out) * frame_length);
        audioproc::apply_window_frames_optimized(
            waveform.data(), window.data(), windowed_frames.data(),
            num_frames_out, frame_length, hop_length);

        std::vector<float> magnitude_spec(static_cast<size_t>(num_frames_out) * num_frequency_bins);
        audioproc::rfft_magnitude_batch_optimized(
            windowed_frames.data(), magnitude_spec.data(),
            num_frames_out, frame_length, fft_length);

        std::vector<float> mel_spec(static_cast<size_t>(num_frames_out) * feature_size);
        audioproc::mel_spectrogram_optimized(
            magnitude_spec.data(), mel_filters.data(), mel_spec.data(),
            num_frames_out, num_frequency_bins, feature_size);

        std::vector<float> log_mel_spec(static_cast<size_t>(num_frames_out) * feature_size);
        audioproc::log_mel_floor_optimized(
            mel_spec.data(), log_mel_spec.data(),
            static_cast<size_t>(num_frames_out) * feature_size, mel_floor);

        const size_t total_bins = static_cast<size_t>(num_frames_out) * feature_size;
        std::vector<bf16> log_mel_spec_bf16(total_bins);
        for (size_t i = 0; i < total_bins; i++) {
            log_mel_spec_bf16[i] = static_cast<bf16>(log_mel_spec[i]);
        }
        audio_payload.mel_spectrograms[audio_idx] = std::move(log_mel_spec_bf16);
        audio_payload.mel_spectrogram_frames_per_audio[audio_idx] = num_frames_out;
        audio_payload.mel_spectrogram_bins_per_audio[audio_idx] = feature_size;
    }
}
