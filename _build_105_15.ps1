# Rebuild flm.exe (v1.0.5, HRX) against the freshly regenerated 09.15 engines in
# src/lib/hrx (incl. flash) and the 09.15 HRX public package. Untracked helper.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT'
$log = 'C:\Users\abhvarma\FastFlowLM_JT\_build_105_15.log'
Remove-Item $log -ErrorAction SilentlyContinue

. 'C:\Users\abhvarma\FastFlowLM_JT\_env_msvc.ps1' *>> $log

$hrxPrefix = (Resolve-Path 'C:\Users\abhvarma\FastFlowLM_IRON\hrx-integration\.hrx-release\hrx-amdxdna-2026.09.15-*-windows-x86_64').Path
"HRX prefix: $hrxPrefix" | Tee-Object -FilePath $log -Append

$bdir = 'build_105_15'
Remove-Item $bdir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'src\build' -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'src\out' -Recurse -Force -ErrorAction SilentlyContinue

"==== CONFIGURE (FLM_USE_HRX=ON, v1.0.5, Ninja) ====" | Tee-Object -FilePath $log -Append
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

$exe = 'C:\Users\abhvarma\FastFlowLM_JT\src\build\flm.exe'
if (-not (Test-Path $exe)) { $exe = "C:\Users\abhvarma\FastFlowLM_JT\$bdir\flm.exe" }
if ($bld -ne 0 -or -not (Test-Path $exe)) { "FLM_BUILD_DONE exit=1 stage=build" | Tee-Object -FilePath $log -Append; exit 1 }

"==== INSTALL (self-contained bin) ====" | Tee-Object -FilePath $log -Append
$prefix = 'C:\Users\abhvarma\FastFlowLM_JT\run105_15_full'
Remove-Item $prefix -Recurse -Force -ErrorAction SilentlyContinue
cmake --install $bdir --prefix $prefix *>> $log
$inst = $LASTEXITCODE
"INSTALL exit=$inst" | Tee-Object -FilePath $log -Append

$instExe = Join-Path $prefix 'bin\flm.exe'
"BUILT_EXE=$exe" | Tee-Object -FilePath $log -Append
if ($inst -eq 0 -and (Test-Path $instExe)) { "FLM_BUILD_DONE exit=0 prefix=$prefix" | Tee-Object -FilePath $log -Append }
else { "FLM_BUILD_DONE exit=1 stage=install exe=$exe" | Tee-Object -FilePath $log -Append }
