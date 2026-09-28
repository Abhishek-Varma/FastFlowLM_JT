# Baseline build: flm.exe from FLM_JT main against main's pinned HRX (2026.07.30).
# No rocm-npu: env from _env_msvc.ps1, HRX from JT hrx-integration/.hrx-release.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT'
$log = 'C:\Users\abhvarma\FastFlowLM_JT\_build_main.log'
Remove-Item $log -ErrorAction SilentlyContinue

. 'C:\Users\abhvarma\FastFlowLM_JT\_env_msvc.ps1' *>> $log

$hrxPrefix = (Resolve-Path 'hrx-integration\.hrx-release\hrx-amdxdna-2026.07.30-*-windows-x86_64').Path
"HRX prefix: $hrxPrefix" | Tee-Object -FilePath $log -Append

$bdir = 'build_main'
Remove-Item $bdir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'src\build' -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'src\out' -Recurse -Force -ErrorAction SilentlyContinue

"==== CONFIGURE (main, FLM_USE_HRX=ON, v1.0.5, Ninja) ====" | Tee-Object -FilePath $log -Append
cmake -S src -B $bdir -G Ninja `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DFLM_USE_HRX=ON" `
  "-DFLM_VERSION=1.0.5" "-DNPU_VERSION=32.0.203.304" `
  "-DCMAKE_PREFIX_PATH=$hrxPrefix" *>> $log
$cfg = $LASTEXITCODE
"CONFIGURE exit=$cfg" | Tee-Object -FilePath $log -Append
if ($cfg -ne 0) { "MAIN_BUILD_DONE exit=1 stage=configure" | Tee-Object -FilePath $log -Append; exit 1 }

"==== BUILD ====" | Tee-Object -FilePath $log -Append
cmake --build $bdir --parallel *>> $log
$bld = $LASTEXITCODE
"BUILD exit=$bld" | Tee-Object -FilePath $log -Append

$exe = 'C:\Users\abhvarma\FastFlowLM_JT\src\build\flm.exe'
if (Test-Path $exe) { "flm.exe present: $((Get-Item $exe).Length) bytes" | Tee-Object -FilePath $log -Append }
else { "flm.exe MISSING" | Tee-Object -FilePath $log -Append }

if ($bld -eq 0 -and (Test-Path $exe)) { "MAIN_BUILD_DONE exit=0" | Tee-Object -FilePath $log -Append }
else { "MAIN_BUILD_DONE exit=1 stage=build" | Tee-Object -FilePath $log -Append }
