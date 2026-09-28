# Configure + build flm.exe with FLM_USE_HRX=ON (bare Windows path: standalone
# Boost + vcpkg libs, no vcpkg toolchain), using Ninja. Logs to _build_flm_hrx.log.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT'
$log = 'C:\Users\abhvarma\FastFlowLM_JT\_build_flm_hrx.log'
Remove-Item $log -ErrorAction SilentlyContinue
$hrxPrefix = 'C:\Users\abhvarma\FastFlowLM_JT\hrx-integration\.hrx-release\hrx-amdxdna-2026.07.30-amdxdna-hal-native-rel-eb0b39f-windows-x86_64'

"==== CONFIGURE flm (FLM_USE_HRX=ON, Ninja) ====" | Tee-Object -FilePath $log -Append
cmake -S src -B build_hrx -G Ninja `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DFLM_USE_HRX=ON" `
  "-DFLM_VERSION=0.9.46" "-DNPU_VERSION=32.0.203.304" `
  "-DCMAKE_PREFIX_PATH=$hrxPrefix" *>> $log
$cfg = $LASTEXITCODE
"CONFIGURE exit=$cfg" | Tee-Object -FilePath $log -Append
if ($cfg -ne 0) { "FLM_BUILD_DONE exit=1 stage=configure" | Tee-Object -FilePath $log -Append; exit 1 }

"==== BUILD flm ====" | Tee-Object -FilePath $log -Append
cmake --build build_hrx --parallel *>> $log
$bld = $LASTEXITCODE
"BUILD exit=$bld" | Tee-Object -FilePath $log -Append

$exe = 'C:\Users\abhvarma\FastFlowLM_JT\build_hrx\flm.exe'
if (Test-Path $exe) { "flm.exe present: $((Get-Item $exe).Length) bytes" | Tee-Object -FilePath $log -Append }
else { "flm.exe MISSING" | Tee-Object -FilePath $log -Append }

if ($bld -eq 0 -and (Test-Path $exe)) { "FLM_BUILD_DONE exit=0" | Tee-Object -FilePath $log -Append }
else { "FLM_BUILD_DONE exit=1 stage=build" | Tee-Object -FilePath $log -Append }
