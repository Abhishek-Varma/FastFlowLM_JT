# Build flm.exe with FLM_USE_HRX=ON against the v2026.09.09 HRX runtime and the
# shipped PR engines in src/lib/hrx. Untracked helper; do not commit.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT'
$log = 'C:\Users\abhvarma\FastFlowLM_JT\_build_flm_hrx_0909.log'
Remove-Item $log -ErrorAction SilentlyContinue
$hrxPrefix = (Resolve-Path 'C:\Users\abhvarma\FastFlowLM_IRON\hrx-integration\.hrx-release\hrx-amdxdna-2026.09.09-*-windows-x86_64').Path
"HRX prefix: $hrxPrefix" | Tee-Object -FilePath $log -Append

"==== CONFIGURE flm (FLM_USE_HRX=ON, Ninja) ====" | Tee-Object -FilePath $log -Append
cmake -S src -B build_hrx0909 -G Ninja `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DFLM_USE_HRX=ON" `
  "-DFLM_VERSION=0.9.46" "-DNPU_VERSION=32.0.203.314" `
  "-DCMAKE_PREFIX_PATH=$hrxPrefix" *>> $log
$cfg = $LASTEXITCODE
"CONFIGURE exit=$cfg" | Tee-Object -FilePath $log -Append
if ($cfg -ne 0) { "FLM_BUILD_DONE exit=1 stage=configure" | Tee-Object -FilePath $log -Append; exit 1 }

"==== BUILD flm ====" | Tee-Object -FilePath $log -Append
cmake --build build_hrx0909 --parallel *>> $log
$bld = $LASTEXITCODE
"BUILD exit=$bld" | Tee-Object -FilePath $log -Append

$exe = 'C:\Users\abhvarma\FastFlowLM_JT\build_hrx0909\flm.exe'
if (Test-Path $exe) { "flm.exe present: $((Get-Item $exe).Length) bytes" | Tee-Object -FilePath $log -Append }
else { "flm.exe MISSING" | Tee-Object -FilePath $log -Append }

if ($bld -eq 0 -and (Test-Path $exe)) { "FLM_BUILD_DONE exit=0" | Tee-Object -FilePath $log -Append }
else { "FLM_BUILD_DONE exit=1 stage=build" | Tee-Object -FilePath $log -Append }
