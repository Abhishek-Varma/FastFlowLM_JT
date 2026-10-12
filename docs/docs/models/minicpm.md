---
layout: docs
title: MiniCPM
nav_order: 15
parent: Models
---

## 🧩 Model Card: [MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes
- **Base Model:** [openbmb/MiniCPM-V-4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run minicpm-v-4.7:1b
```

📝 **Note**

- MiniCPM-V does not ship its own sampling defaults. FLM starts from temperature 0.7, top_p 0.8, top_k 20, and presence penalty 1.5. These are the Qwen3.5 non-thinking defaults, because MiniCPM-V-4.7-1B's language model has the same architecture as Qwen3.5-0.8B.
- Images are processed the same way as in the original MiniCPM-V: the model sees a downscaled view of the whole image, and large images are also split into up to 9 tiles. Because of this, the `--img-pre-resize` (`-r`) option has no effect on this model.
- Image TTFT stops growing at about 1080p: from there up, every image is split into the maximum 9 tiles, so a 4K image costs about the same as a 1080p one. Each tile is about 448 × 448 pixels, so very fine detail in large images is downscaled.
- Video and audio input are not supported.

---
