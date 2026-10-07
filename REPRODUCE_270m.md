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
(the real 270m fix), the exact test prompts, expected coherent outputs, the
A/B validations, and the NPU-safety protocol.

### Dispatch mode (individual vs chained)

With the swapped xclbins, 270m runs coherently in **both** dispatch modes on both
backends — see §9.1 of the IRON `REPRODUCE.md`. Mode is selected by the
`FLM_FORCE_INDIVIDUAL` env var, which is **presence-checked** (`getenv != nullptr`):

- Individual: `FLM_FORCE_INDIVIDUAL=1` (any value, even `0`).
- Chained: the var **fully unset** (`Remove-Item Env:\FLM_FORCE_INDIVIDUAL`).

Individual is the safe default; sync-removal's decode speedup is visible there,
while in chained the runlist already amortizes per-op cost (baseline ≈ sync_removed).
