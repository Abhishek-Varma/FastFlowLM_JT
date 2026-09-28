# Incremental flm.exe build helper (HRX, v1.0.5), persistent Ninja dir so
# successive engine ports relink fast. Untracked helper.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT'
$log = 'C:\Users\abhvarma\FastFlowLM_JT\_build_flm_inc.log'
Remove-Item $log -ErrorAction SilentlyContinue

. 'C:\Users\abhvarma\FastFlowLM_JT\_env_msvc.ps1' *>> $log

$hrxPrefix = (Resolve-Path 'C:\Users\abhvarma\FastFlowLM_IRON\hrx-integration\.hrx-release\hrx-amdxdna-2026.09.15-*-windows-x86_64').Path
"HRX prefix: $hrxPrefix" | Tee-Object -FilePath $log -Append

$bdir = 'build_flm_inc'

"==== CONFIGURE (re-glob sources) ====" | Tee-Object -FilePath $log -Append
cmake -S src -B $bdir -G Ninja `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DFLM_USE_HRX=ON" `
  "-DFLM_VERSION=1.0.5" "-DNPU_VERSION=32.0.203.314" `
  "-DCMAKE_PREFIX_PATH=$hrxPrefix" *>> $log
$cfg = $LASTEXITCODE
"CONFIGURE exit=$cfg" | Tee-Object -FilePath $log -Append
if ($cfg -ne 0) { "FLM_BUILD_DONE exit=1 stage=configure" | Tee-Object -FilePath $log -Append; exit 1 }

"==== BUILD ====" | Tee-Object -FilePath $log -Append
cmake --build $bdir --parallel *>> $log
$bld = $LASTEXITCODE
"BUILD exit=$bld" | Tee-Object -FilePath $log -Append

$exe = "C:\Users\abhvarma\FastFlowLM_JT\$bdir\flm.exe"
if (-not (Test-Path $exe)) { $exe = 'C:\Users\abhvarma\FastFlowLM_JT\src\build\flm.exe' }
if ($bld -eq 0 -and (Test-Path $exe)) { "FLM_BUILD_DONE exit=0 exe=$exe" | Tee-Object -FilePath $log -Append }
else { "FLM_BUILD_DONE exit=1 stage=build" | Tee-Object -FilePath $log -Append }
