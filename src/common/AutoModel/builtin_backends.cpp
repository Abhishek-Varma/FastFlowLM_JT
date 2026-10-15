/// \file builtin_backends.cpp
/// \brief The backends this build ships, keyed by model family
/// \note One place knows both the family names from model_list.json and the
///       engine types behind them. Everything else goes through the registry.
/// \note Every family registers the flm backend. Only families with a corelib
///       engine also register a rai backend, and only when this build has one.
#include "AutoModel/automodel.hpp"
#include "AutoModel/flm_backend.hpp"
#include "AutoModel/model_backend.hpp"

#if defined(FLM_ENABLE_RAI)
#include "models/gemma4/rai/aie_next/gemma4_rai_backend.hpp"
#include "models/phi4/rai/aie_next/phi4_rai_backend.hpp"
#include "models/qwen3/rai/aie_next/qwen3_rai_backend.hpp"
#include "models/qwen35/rai/aie_next/qwen35_rai_backend.hpp"
#endif

// Pure/native-dispatch engine variants (Linux only; opt in with
// --backend purehrx / --backend purexrt). They bypass the flm_rt shim:
// purehrx dispatches through the raw libhrx C API, purexrt through XRT. A
// build is HRX or XRT (never both), so only the matching set is linked (see
// FLM_ENGINE_LINK_LIBS) and compiled in here. FLM_HAVE_PURE_ENGINES is set by
// CMake only where the prebuilt pure .so are shipped (src/lib/{hrx,xrt}).
#if defined(FLM_HAVE_PURE_ENGINES)
#if FLM_USE_HRX
#include "models/llama/llama_npu_pure_hrx.hpp"
#include "models/qwen2/qwen2_npu_pure_hrx.hpp"
#include "models/qwen2vl/qwen2vl_npu_pure_hrx.hpp"
#include "models/qwen3/qwen3_npu_pure_hrx.hpp"
#include "models/qwen3vl/qwen3vl_npu_pure_hrx.hpp"
#include "models/qwen3vl_flash/qwen3vl_flash_pure_hrx.hpp"
#include "models/qwen3_5vl/qwen3_5vl_npu_pure_hrx.hpp"
#include "models/qwen3_6_moe/qwen3_6_moe_npu_pure_hrx.hpp"
#include "models/gemma/gemma_npu_pure_hrx.hpp"
#include "models/gemma_text/gemma_text_npu_pure_hrx.hpp"
#include "models/gemma4e/gemma4e_npu_pure_hrx.hpp"
#include "models/gemma4e_flash/gemma4e_flash_pure_hrx.hpp"
#include "models/gemma4_12b/gemma4_12b_npu_pure_hrx.hpp"
#include "models/hunyuan/hunyuan_npu_pure_hrx.hpp"
#include "models/gpt_oss/gpt_oss_npu_pure_hrx.hpp"
#include "models/lfm2/lfm2_npu_pure_hrx.hpp"
#include "models/nanbeige/nanbeige_npu_pure_hrx.hpp"
#include "models/phi4/phi4_npu_pure_hrx.hpp"
#else
#include "models/llama/llama_npu_pure_xrt.hpp"
#include "models/qwen2/qwen2_npu_pure_xrt.hpp"
#include "models/qwen2vl/qwen2vl_npu_pure_xrt.hpp"
#include "models/qwen3/qwen3_npu_pure_xrt.hpp"
#include "models/qwen3vl/qwen3vl_npu_pure_xrt.hpp"
#include "models/qwen3vl_flash/qwen3vl_flash_pure_xrt.hpp"
#include "models/qwen3_5vl/qwen3_5vl_npu_pure_xrt.hpp"
#include "models/qwen3_6_moe/qwen3_6_moe_npu_pure_xrt.hpp"
#include "models/gemma/gemma_npu_pure_xrt.hpp"
#include "models/gemma_text/gemma_text_npu_pure_xrt.hpp"
#include "models/gemma4e/gemma4e_npu_pure_xrt.hpp"
#include "models/gemma4e_flash/gemma4e_flash_pure_xrt.hpp"
#include "models/gemma4_12b/gemma4_12b_npu_pure_xrt.hpp"
#include "models/hunyuan/hunyuan_npu_pure_xrt.hpp"
#include "models/gpt_oss/gpt_oss_npu_pure_xrt.hpp"
#include "models/lfm2/lfm2_npu_pure_xrt.hpp"
#include "models/nanbeige/nanbeige_npu_pure_xrt.hpp"
#include "models/phi4/phi4_npu_pure_xrt.hpp"
#endif
#endif  // FLM_HAVE_PURE_ENGINES

namespace flm::backend {
namespace {

/// \brief register the flm backend for one family
/// \tparam Engine the concrete engine type
/// \param registry the registry to populate
/// \param family the family name, as in details.family
template <class Engine>
void RegisterFlm(BackendRegistry& registry, const char* family) {
    registry.register_backend(family, kFlmBackendId,
                              flm_factory<Engine>());
}

#if defined(FLM_HAVE_PURE_ENGINES)
/// \brief register a pure/native-dispatch backend for one family
/// \tparam Engine the concrete pure engine type, e.g. gpt_oss_npu_pure_hrx
/// \param id the backend id, "purehrx" or "purexrt"
/// \note The pure engines are causal_lm subclasses with the same construct /
///       load_weights / clear_context shape as the stock engines, so the stock
///       flm_factory builds them unchanged.
template <class Engine>
void RegisterPure(BackendRegistry& registry, const char* family,
                  const char* id) {
    registry.register_backend(family, id, flm_factory<Engine>());
}
#endif

}  // namespace

void register_builtin_backends(BackendRegistry& registry) {
    RegisterFlm<llama_npu>(registry, "llama3");
    RegisterFlm<llama_npu>(registry, "deepseek-r1");
    RegisterFlm<qwen3_npu>(registry, "deepseek-r1-0528");
    RegisterFlm<qwen2_npu>(registry, "qwen2");
    RegisterFlm<qwen2vl_npu>(registry, "qwen2vl");
    RegisterFlm<qwen3_npu>(registry, "qwen3");
    RegisterFlm<qwen3_npu>(registry, "qwen3-it");
    RegisterFlm<qwen3_npu>(registry, "qwen3-tk");
    RegisterFlm<qwen3vl_npu>(registry, "qwen3vl");
    RegisterFlm<qwen3vl_flash>(registry, "qwen3vl-flash");
    RegisterFlm<qwen3_5vl_npu>(registry, "qwen3.5");
    RegisterFlm<minicpm_v_4_7_npu>(registry, "minicpm-v-4.7");
    RegisterFlm<qwen3_6_moe_npu>(registry, "qwen3.6-moe");
    RegisterFlm<qwen3_8mtp_npu>(registry, "qwen3.8-mtp");
    RegisterFlm<gemma_npu>(registry, "gemma3");
    RegisterFlm<gemma_text_npu>(registry, "gemma3-text");
    RegisterFlm<gemma4e_npu>(registry, "gemma4e");
    RegisterFlm<gemma4e_flash>(registry, "gemma4e-flash");
    RegisterFlm<gemma4_12b_npu>(registry, "gemma4-12b");
    RegisterFlm<hunyuan_npu>(registry, "hunyuan");
    RegisterFlm<gpt_oss_npu>(registry, "gpt-oss");
    RegisterFlm<lfm2_npu>(registry, "lfm2");
    RegisterFlm<lfm2_npu>(registry, "lfm2.5-tk");
    RegisterFlm<nanbeige_npu>(registry, "nanbeige");
    RegisterFlm<phi4_npu>(registry, "phi4");

#if defined(FLM_HAVE_PURE_ENGINES)
    // Pure/native-dispatch variants, selectable with --backend. A build is HRX
    // or XRT, so exactly one id ("purehrx" or "purexrt") is registered per
    // family. They reuse each family's existing NPU2/Q4NX weights -- only the
    // dispatch differs -- so no new catalog entry is needed.
#if FLM_USE_HRX
    RegisterPure<llama_npu_pure_hrx>(registry, "llama3", "purehrx");
    RegisterPure<qwen2_npu_pure_hrx>(registry, "qwen2", "purehrx");
    RegisterPure<qwen2vl_npu_pure_hrx>(registry, "qwen2vl", "purehrx");
    RegisterPure<qwen3_npu_pure_hrx>(registry, "qwen3", "purehrx");
    RegisterPure<qwen3vl_npu_pure_hrx>(registry, "qwen3vl", "purehrx");
    RegisterPure<qwen3vl_flash_pure_hrx>(registry, "qwen3vl-flash", "purehrx");
    RegisterPure<qwen3_5vl_npu_pure_hrx>(registry, "qwen3.5", "purehrx");
    RegisterPure<qwen3_6_moe_npu_pure_hrx>(registry, "qwen3.6-moe", "purehrx");
    RegisterPure<gemma_npu_pure_hrx>(registry, "gemma3", "purehrx");
    RegisterPure<gemma_text_npu_pure_hrx>(registry, "gemma3-text", "purehrx");
    RegisterPure<gemma4e_npu_pure_hrx>(registry, "gemma4e", "purehrx");
    RegisterPure<gemma4e_flash_pure_hrx>(registry, "gemma4e-flash", "purehrx");
    RegisterPure<gemma4_12b_npu_pure_hrx>(registry, "gemma4-12b", "purehrx");
    RegisterPure<hunyuan_npu_pure_hrx>(registry, "hunyuan", "purehrx");
    RegisterPure<gpt_oss_npu_pure_hrx>(registry, "gpt-oss", "purehrx");
    RegisterPure<lfm2_npu_pure_hrx>(registry, "lfm2", "purehrx");
    RegisterPure<nanbeige_npu_pure_hrx>(registry, "nanbeige", "purehrx");
    RegisterPure<phi4_npu_pure_hrx>(registry, "phi4", "purehrx");
#else
    RegisterPure<llama_npu_pure_xrt>(registry, "llama3", "purexrt");
    RegisterPure<qwen2_npu_pure_xrt>(registry, "qwen2", "purexrt");
    RegisterPure<qwen2vl_npu_pure_xrt>(registry, "qwen2vl", "purexrt");
    RegisterPure<qwen3_npu_pure_xrt>(registry, "qwen3", "purexrt");
    RegisterPure<qwen3vl_npu_pure_xrt>(registry, "qwen3vl", "purexrt");
    RegisterPure<qwen3vl_flash_pure_xrt>(registry, "qwen3vl-flash", "purexrt");
    RegisterPure<qwen3_5vl_npu_pure_xrt>(registry, "qwen3.5", "purexrt");
    RegisterPure<qwen3_6_moe_npu_pure_xrt>(registry, "qwen3.6-moe", "purexrt");
    RegisterPure<gemma_npu_pure_xrt>(registry, "gemma3", "purexrt");
    RegisterPure<gemma_text_npu_pure_xrt>(registry, "gemma3-text", "purexrt");
    RegisterPure<gemma4e_npu_pure_xrt>(registry, "gemma4e", "purexrt");
    RegisterPure<gemma4e_flash_pure_xrt>(registry, "gemma4e-flash", "purexrt");
    RegisterPure<gemma4_12b_npu_pure_xrt>(registry, "gemma4-12b", "purexrt");
    RegisterPure<hunyuan_npu_pure_xrt>(registry, "hunyuan", "purexrt");
    RegisterPure<gpt_oss_npu_pure_xrt>(registry, "gpt-oss", "purexrt");
    RegisterPure<lfm2_npu_pure_xrt>(registry, "lfm2", "purexrt");
    RegisterPure<nanbeige_npu_pure_xrt>(registry, "nanbeige", "purexrt");
    RegisterPure<phi4_npu_pure_xrt>(registry, "phi4", "purexrt");
#endif
#endif  // FLM_HAVE_PURE_ENGINES

#if defined(FLM_ENABLE_RAI)
    registry.register_backend("gemma4e", flm::backend::kRaiBackendId,
                              flm::gemma4::rai_factory(),
                              flm::gemma4::rai_traits());
    registry.register_backend("phi4", flm::backend::kRaiBackendId,
                              flm::phi4::rai_factory(),
                              flm::phi4::rai_traits());
    registry.register_backend("qwen3", flm::backend::kRaiBackendId,
                              flm::qwen3::rai_factory(),
                              flm::qwen3::rai_traits());
    registry.register_backend("qwen3.5", flm::backend::kRaiBackendId,
                              flm::qwen35::rai_factory(),
                              flm::qwen35::rai_traits());
#endif
}

}  // namespace flm::backend
