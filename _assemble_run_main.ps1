# Assemble self-contained portable dir (run_main) around the main flm.exe.
# Engines from main src/lib/hrx (07.30), hrx.dll from JT .hrx-release 07.30,
# third-party from src/lib + vcpkg, share/flm from main src tree.
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\abhvarma\FastFlowLM_JT'
Set-Location $root
$log = Join-Path $root '_assemble_run_main.log'
Remove-Item $log -ErrorAction SilentlyContinue
function Log($m){ $m | Tee-Object -FilePath $log -Append }

$prefix = Join-Path $root 'run_main'
Remove-Item $prefix -Recurse -Force -ErrorAction SilentlyContinue
$bin = Join-Path $prefix 'bin'
$share = Join-Path $prefix 'share\flm'
New-Item -ItemType Directory -Force -Path $bin,$share | Out-Null

Copy-Item (Join-Path $root 'src\build\flm.exe') $bin -Force
Log "flm.exe -> $((Get-Item (Join-Path $bin 'flm.exe')).Length) bytes"

Copy-Item (Join-Path $root 'src\lib\hrx\*.dll') $bin -Force
Log "engine dlls: $((Get-ChildItem (Join-Path $root 'src\lib\hrx\*.dll')).Count)"

$hrxDll = (Resolve-Path 'hrx-integration\.hrx-release\hrx-amdxdna-2026.07.30-*-windows-x86_64\bin\hrx.dll').Path
Copy-Item $hrxDll $bin -Force
Log "hrx.dll -> $((Get-Item (Join-Path $bin 'hrx.dll')).Length) bytes (07.30)"

$vcbin = 'C:\dev\vcpkg\installed\x64-windows\bin'
Copy-Item (Join-Path $vcbin '*.dll') $bin -Force
Copy-Item (Join-Path $root 'src\lib\*.dll') $bin -Force
foreach($rt in 'vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll'){
    $src = "C:\Users\abhvarma\flm-clean\fastflowlm-hrx-windows-x86_64\bin\$rt"
    if(Test-Path $src){ Copy-Item $src $bin -Force }
}

Copy-Item (Join-Path $root 'src\model_list.json') $share -Force
Copy-Item (Join-Path $root 'src\xclbins') $share -Recurse -Force
Log "share/flm: model_list.json + xclbins($((Get-ChildItem (Join-Path $share 'xclbins')).Count))"
Log "bin file count: $((Get-ChildItem $bin -File).Count)"
Log "ASSEMBLE_MAIN_DONE prefix=$prefix"
