# Serve one model tag, send fixed greedy (temp 0) prompts, capture outputs, stop.
# Usage: _validate_engine.ps1 -Tag llama3-purehrx:1b -Port 11541 -Out _val_llama_pure.txt
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [Parameter(Mandatory=$true)][int]$Port,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\abhvarma\FastFlowLM_JT\run105_15\bin'
$env:FLM_PUREHRX_PROFILE = '0'
$serveLog = "C:\Users\abhvarma\FastFlowLM_JT\_val_serve_$Port.log"
Remove-Item $serveLog -ErrorAction SilentlyContinue
Remove-Item $Out -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath '.\flm.exe' -ArgumentList @('serve', $Tag, '--port', "$Port") `
    -RedirectStandardOutput $serveLog -RedirectStandardError "$serveLog.err" -PassThru -WindowStyle Hidden

# wait up to 180s for the server to come up
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $serveLog) {
        if (Select-String -Path $serveLog -Pattern 'WebServer started' -Quiet -ErrorAction SilentlyContinue) { $ready = $true; break }
    }
    if ($proc.HasExited) { break }
}
if (-not $ready) {
    "SERVE_FAILED tag=$Tag port=$Port" | Out-File $Out
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    exit 1
}
Start-Sleep -Seconds 2

$prompts = @(
    'What is the capital of France? Answer in one short sentence.',
    'List the first 5 prime numbers.',
    'Write one sentence about the sun.'
)
"=== tag=$Tag ===" | Out-File $Out -Append
foreach ($p in $prompts) {
    $body = @{ model = $Tag; messages = @(@{ role = 'user'; content = $p }); temperature = 0; stream = $false; max_tokens = 80 } | ConvertTo-Json -Depth 6
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 120
        $content = $resp.choices[0].message.content
    } catch {
        $content = "REQUEST_ERROR: $($_.Exception.Message)"
    }
    "Q: $p" | Out-File $Out -Append
    "A: $content" | Out-File $Out -Append
    "---" | Out-File $Out -Append
}

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
"VALIDATE_DONE tag=$Tag" | Out-File $Out -Append
