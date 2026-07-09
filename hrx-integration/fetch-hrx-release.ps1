<#
.SYNOPSIS
  Fetch and verify the pinned HRX amdxdna *Windows* release artifact.

.DESCRIPTION
  Windows counterpart of fetch-hrx-release.sh. Downloads the Windows .zip pinned
  in hrx-release.env (HRX_RELEASE_ASSET_WINDOWS / HRX_RELEASE_SHA256_WINDOWS),
  verifies its checksum, extracts it, and prints the extracted artifact root.

  The artifact is extracted into <script-dir>\.hrx-release\<artifact-root> — the
  same location the CMake HRX discovery logic globs (hrx-integration/.hrx-release/
  hrx-amdxdna-*), so no HRX_DIR / HRX_BUILD env vars are required after running it.

  When running in GitHub Actions, also writes `root=<artifact-root>` to
  $env:GITHUB_OUTPUT.

.PARAMETER OutDir
  Optional output directory (defaults to <script-dir>\.hrx-release).
#>
[CmdletBinding()]
param(
    [string]$OutDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here       = $PSScriptRoot
$releaseEnv = Join-Path $here 'hrx-release.env'
if (-not (Test-Path -LiteralPath $releaseEnv)) {
    throw "missing $releaseEnv"
}

# Parse the shell-style .env (KEY=VALUE or KEY="VALUE"; ignore comments/blanks).
$env_vars = @{}
foreach ($line in Get-Content -LiteralPath $releaseEnv) {
    $trimmed = $line.Trim()
    if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
    if ($trimmed -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $key = $Matches[1]
        $val = $Matches[2].Trim().Trim('"').Trim("'")
        $env_vars[$key] = $val
    }
}

function Require-Var([string]$name) {
    if (-not $env_vars.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($env_vars[$name])) {
        throw "missing $name in $releaseEnv"
    }
    return $env_vars[$name]
}

$repo   = Require-Var 'HRX_RELEASE_REPO'
$tag    = Require-Var 'HRX_RELEASE_TAG'
$asset  = Require-Var 'HRX_RELEASE_ASSET_WINDOWS'
$sha256 = (Require-Var 'HRX_RELEASE_SHA256_WINDOWS').ToLower()

if (-not $OutDir) { $OutDir = Join-Path $here '.hrx-release' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$baseUrl = "https://github.com/$repo/releases/download/$tag"
$zip     = Join-Path $OutDir $asset
$shaFile = "$zip.sha256"

function Download-File([string]$url, [string]$path) {
    Write-Host "download: $url"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
            return
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds 2
        }
    }
}

Download-File "$baseUrl/$asset" $zip
Download-File "$baseUrl/$asset.sha256" $shaFile

$remoteSha = ((Get-Content -LiteralPath $shaFile | Select-Object -First 1) -split '\s+')[0].ToLower()
if ([string]::IsNullOrWhiteSpace($remoteSha)) { throw "empty remote sha256 file: $shaFile" }
if ($remoteSha -ne $sha256) {
    throw "remote sha256 mismatch: expected $sha256, got $remoteSha"
}

$actualSha = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
if ($actualSha -ne $sha256) {
    throw "zip sha256 mismatch: expected $sha256, got $actualSha"
}

# The archive root directory mirrors the asset name without the .zip suffix.
$artifactRootName = [System.IO.Path]::GetFileNameWithoutExtension($asset)
if ([string]::IsNullOrWhiteSpace($artifactRootName)) { throw "could not determine zip root" }

$artifactRoot = Join-Path $OutDir $artifactRootName
if (Test-Path -LiteralPath $artifactRoot) {
    Remove-Item -LiteralPath $artifactRoot -Recurse -Force
}

Expand-Archive -LiteralPath $zip -DestinationPath $OutDir -Force

$envPs1    = Join-Path $artifactRoot 'env.ps1'
$hrxLib    = Join-Path $artifactRoot 'HRX_BUILD\libhrx\src\libhrx\hrx.lib'
$flatccLib = Join-Path $artifactRoot 'HRX_BUILD\flatcc_runtime.lib'
foreach ($f in @($envPs1, $hrxLib, $flatccLib)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "missing packaged file: $f" }
}

Write-Host "HRX_ARTIFACT_ROOT=$artifactRoot"
if ($env:GITHUB_OUTPUT) {
    "root=$artifactRoot" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
}
