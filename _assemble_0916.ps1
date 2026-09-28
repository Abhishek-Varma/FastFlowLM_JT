# Assemble a self-contained portable flm dir (run_hrx_0916) around the freshly built
# v1.0.6 flm.exe. Engines from src/lib/hrx (incl. flash), hrx.dll from IRON 09.15
# package, third-party from vcpkg, share/flm from src tree. Bypasses the fragile
# cmake --install fixup. Untracked helper.
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\abhvarma\FastFlowLM_JT'
Set-Location $root
$log = Join-Path $root '_assemble_0916.log'
Remove-Item $log -ErrorAction SilentlyContinue
function Log($m){ $m | Tee-Object -FilePath $log -Append }

$prefix = Join-Path $root 'run_hrx_0916'
Remove-Item $prefix -Recurse -Force -ErrorAction SilentlyContinue
$bin = Join-Path $prefix 'bin'
$share = Join-Path $prefix 'share\flm'
New-Item -ItemType Directory -Force -Path $bin,$share | Out-Null

$exe = Join-Path $root 'src\build\flm.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path $root 'build_hrx_0916\flm.exe' }
Copy-Item $exe $bin -Force
Log "flm.exe -> $((Get-Item (Join-Path $bin 'flm.exe')).Length) bytes"

Copy-Item (Join-Path $root 'src\lib\hrx\*.dll') $bin -Force
Log "engine dlls copied: $((Get-ChildItem (Join-Path $bin '*.dll')).Count)"
Log ("flash dlls present: " + ((Get-ChildItem (Join-Path $bin '*flash*.dll')).Name -join ', '))

$hrxDll = (Resolve-Path 'C:\Users\abhvarma\FastFlowLM_IRON\hrx-integration\.hrx-release\hrx-amdxdna-2026.09.15-*-windows-x86_64\bin\hrx.dll').Path
Copy-Item $hrxDll $bin -Force
Log "hrx.dll -> $((Get-Item (Join-Path $bin 'hrx.dll')).Length) bytes"

# Third-party runtime deps. The repo's src/lib bundles some (fftw libfftw3*-3,
# abseil, protobuf, zlib) that flm.exe links by those exact names; vcpkg provides
# the NEWER ffmpeg (avcodec-62 etc.) and curl that this build links. Copy repo
# bundle first, then let vcpkg win on name collisions.
Copy-Item (Join-Path $root 'src\lib\*.dll') $bin -Force
Log "repo src/lib dlls copied: $((Get-ChildItem (Join-Path $root 'src\lib\*.dll')).Count)"
$vcbin = 'C:\dev\vcpkg\installed\x64-windows\bin'
if (Test-Path $vcbin) { Copy-Item (Join-Path $vcbin '*.dll') $bin -Force; Log "vcpkg dlls copied: $((Get-ChildItem (Join-Path $vcbin '*.dll')).Count)" }

foreach($rt in 'vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll','vcomp140.dll'){
    $src = "C:\Users\abhvarma\flm-clean\fastflowlm-hrx-windows-x86_64\bin\$rt"
    if(Test-Path $src){ Copy-Item $src $bin -Force }
}

Copy-Item (Join-Path $root 'src\model_list.json') $share -Force
Copy-Item (Join-Path $root 'src\xclbins') $share -Recurse -Force
Log "share/flm: model_list.json + xclbins($((Get-ChildItem (Join-Path $share 'xclbins')).Count))"

Log "bin file count: $((Get-ChildItem $bin -File).Count)"
Log "ASSEMBLE_DONE prefix=$prefix"
