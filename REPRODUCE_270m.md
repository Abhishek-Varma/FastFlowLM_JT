# Reproducing gemma3:270m (JT / flm.exe side)

This repo builds `flm.exe`. The full, self-validating reproduction guide — including
the environment (Windows 11 build 26200, NPU driver 32.0.203.314, pinned HRX
2026.09.15, MSVC 19.44), the engine binaries, and the xclbin fix — lives in the
**FastFlowLM_IRON** repo at `repro/REPRODUCE.md` (branch
`windows-270m-sync-opt-repro`). Read that first.

## What is in this repo for the repro

- `src/include/AutoModel/all_models.hpp`, `src/CMakeLists.txt` — build an XRT-mode
  `flm.exe` that registers **only** the gemma_text pure-XRT engine
  (`FLM_XRT_GEMMA_TEXT_ONLY`), isolating risk to one engine. HRX builds unaffected.
- `src/lib/xrt/gemma_text_npu_pure_xrt.{dll,lib}` — the pure-XRT engine
  (sync-removed) consumed by the XRT build. HRX engine lives in `src/lib/hrx/`.
- `_build_xrt_handoff.ps1` — the XRT `flm.exe` build driver.

## Build (XRT-mode flm.exe)

```powershell
. .\_env_msvc.ps1                                   # MSVC+SDK env (reg.exe-policy workaround)
git submodule update --init --recursive third_party/tokenizers-cpp   # once
powershell -NoProfile -ExecutionPolicy Bypass -File .\_build_xrt_handoff.ps1
#   -> src\out\flm.exe   (FLM_USE_HRX=OFF)
```

Confirm it registers the family:

```powershell
Select-String -Path src\out\flm.exe -Pattern "gemma3-text-purexrt" -SimpleMatch -Quiet   # -> True
```

The HRX-mode build is the same with `-DFLM_USE_HRX=ON -DCMAKE_PREFIX_PATH=<pinned
HRX release>`.

## Run / validate

See `repro/REPRODUCE.md` in FastFlowLM_IRON for bundle assembly, the xclbin swap
(the real 270m fix), the exact test prompts, expected coherent outputs, the three
A/B validations, and the NPU-safety protocol.
