# Build flm.exe with the XRT backend (FLM_USE_HRX=OFF) from the HANDOFF worktree.
# Our CMakeLists edits restrict XRT engines to gemma_text pure-XRT only
# (FLM_XRT_GEMMA_TEXT_ONLY). Engines come from src/lib/xrt (incl. freshly built
# gemma_text_npu_pure_xrt). Env from _env_msvc.ps1; XRT from C:/dev defaults.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT-handoff'
$log = 'C:\Users\abhvarma\FastFlowLM_JT-handoff\_build_xrt_handoff.log'
Remove-Item $log -ErrorAction SilentlyContinue

. 'C:\Users\abhvarma\FastFlowLM_JT-handoff\_env_msvc.ps1' *>> $log

$bdir = 'build_xrt'
Remove-Item $bdir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'src\build' -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'src\out' -Recurse -Force -ErrorAction SilentlyContinue

"==== CONFIGURE (XRT, FLM_USE_HRX=OFF, v1.0.5, Ninja) ====" | Tee-Object -FilePath $log -Append
cmake -S src -B $bdir -G Ninja `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DFLM_USE_HRX=OFF" `
  "-DFLM_VERSION=1.0.5" "-DNPU_VERSION=32.0.203.314" *>> $log
$cfg = $LASTEXITCODE
"CONFIGURE exit=$cfg" | Tee-Object -FilePath $log -Append
if ($cfg -ne 0) { "XRT_BUILD_DONE exit=1 stage=configure" | Tee-Object -FilePath $log -Append; exit 1 }

"==== BUILD ====" | Tee-Object -FilePath $log -Append
cmake --build $bdir --parallel *>> $log
$bld = $LASTEXITCODE
"BUILD exit=$bld" | Tee-Object -FilePath $log -Append

$exe = Get-ChildItem -Path 'C:\Users\abhvarma\FastFlowLM_JT-handoff' -Recurse -Filter 'flm.exe' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($exe) { "flm.exe present: $($exe.FullName) $($exe.Length) bytes $($exe.LastWriteTime)" | Tee-Object -FilePath $log -Append }
else { "flm.exe MISSING" | Tee-Object -FilePath $log -Append }

if ($bld -eq 0 -and $exe) { "XRT_BUILD_DONE exit=0" | Tee-Object -FilePath $log -Append }
else { "XRT_BUILD_DONE exit=1 stage=build" | Tee-Object -FilePath $log -Append }
