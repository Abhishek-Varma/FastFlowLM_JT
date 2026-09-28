# Standalone MSVC + Windows SDK + vcpkg build environment (no rocm-npu).
# Mirrors the machine-global toolchain wiring needed to build flm on this box,
# working around the reg.exe policy that breaks vcvars SDK detection.
# cmake/ninja come from the VS BuildTools bundle; cargo from ~/.cargo.
$ErrorActionPreference = 'Stop'

$KitRoot = 'C:\Program Files (x86)\Windows Kits\10\'
$SdkVer  = '10.0.26100.0'
$VcVars  = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$VsCMake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake'

# 1) SDK presets so vcvars can find the SDK despite reg.exe being disabled.
$env:WindowsSdkDir      = $KitRoot
$env:UniversalCRTSdkDir = $KitRoot
$env:UCRTVersion        = $SdkVer

# 2) Import MSVC + SDK env from vcvars64 (only if cl not already loaded).
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $tmp = Join-Path $env:TEMP ("flmjt-vcvars-{0}.txt" -f $PID)
    cmd /c "call `"$VcVars`" >nul 2>&1 & set > `"$tmp`""
    Get-Content $tmp | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] }
    }
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

# 2b) Hardwire absolute SDK include/lib/bin paths (vcvars can leave these broken here).
$SdkInc = Join-Path $KitRoot ("Include\" + $SdkVer)
$SdkLib = Join-Path $KitRoot ("Lib\" + $SdkVer)
$SdkBin = Join-Path $KitRoot ("bin\" + $SdkVer + "\x64")
if (Test-Path $SdkInc) {
    $incDirs = @("ucrt","um","shared","winrt","cppwinrt") | ForEach-Object { Join-Path $SdkInc $_ } | Where-Object { Test-Path $_ }
    $env:INCLUDE = (($incDirs + ($env:INCLUDE -split ';' | Where-Object { $_ })) | Select-Object -Unique) -join ';'
    $libDirs = @("ucrt\x64","um\x64") | ForEach-Object { Join-Path $SdkLib $_ } | Where-Object { Test-Path $_ }
    $env:LIB = (($libDirs + ($env:LIB -split ';' | Where-Object { $_ })) | Select-Object -Unique) -join ';'
    if (Test-Path (Join-Path $SdkBin 'rc.exe')) { $env:Path = "$SdkBin;$env:Path" }
    $env:WindowsSDKVersion    = $SdkVer + '\'
    $env:WindowsSdkVerBinPath = $SdkBin.Substring(0, $SdkBin.Length - 4)
}

# 3) cmake + ninja from the VS bundle.
$env:Path = "$VsCMake\CMake\bin;$VsCMake\Ninja;$env:Path"

# 4) Cargo (Rust) for tokenizers-cpp.
$CargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path $CargoBin) { $env:Path = "$CargoBin;$env:Path" }

# 5) vcpkg (classic mode) + keep compiler env vars through vcpkg sanitization.
$env:VCPKG_ROOT = 'C:\dev\vcpkg'
$env:VCPKG_KEEP_ENV_VARS = 'PATH;INCLUDE;LIB;LIBPATH;WindowsSdkDir;UniversalCRTSdkDir;UCRTVersion'

$ErrorActionPreference = 'Continue'
Write-Host "msvc env ready:"
Write-Host ("  cl    : {0}" -f (Get-Command cl    -ErrorAction SilentlyContinue).Source)
Write-Host ("  rc    : {0}" -f (Get-Command rc    -ErrorAction SilentlyContinue).Source)
Write-Host ("  cmake : {0}" -f (Get-Command cmake -ErrorAction SilentlyContinue).Source)
Write-Host ("  ninja : {0}" -f (Get-Command ninja -ErrorAction SilentlyContinue).Source)
Write-Host ("  cargo : {0}" -f (Get-Command cargo -ErrorAction SilentlyContinue).Source)
