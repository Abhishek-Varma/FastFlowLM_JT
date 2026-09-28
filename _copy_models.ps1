# Populate ~/.flm/models with the sweep-relevant model weights (data only).
# Skips the giant 20B/35B models and ASR-only whisper to keep sweep runtime sane.
$ErrorActionPreference = 'Continue'
$src = 'C:\Users\abhvarma\rocm-npu-staging\build\FastFlowLM-models\models'
$dst = 'C:\Users\abhvarma\.flm\models'
$log = 'C:\Users\abhvarma\FastFlowLM_JT\_copy_models.log'
Remove-Item $log -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$exclude = @(
  'GPT-OSS-20B-NPU2','GPT-OSS-Safeguard-20b-NPU2',
  'Qwen3.6-35B-A3B-NPU2','Whisper-V3-Turbo-NPU2','models'
)
$xd = @()
foreach($e in $exclude){ $xd += (Join-Path $src $e) }

"COPY_START $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
"excluding: $($exclude -join ', ')" | Tee-Object -FilePath $log -Append
# /E recurse incl empty, skip identical by default, multi-thread, quiet per-file
robocopy $src $dst /E /XD @xd /MT:8 /R:1 /W:1 /NFL /NDL /NP /NJH >> $log 2>&1
$rc = $LASTEXITCODE
"robocopy exit=$rc (0-7=success)" | Tee-Object -FilePath $log -Append
"models now in dst: $((Get-ChildItem $dst -Directory).Count)" | Tee-Object -FilePath $log -Append
(Get-ChildItem $dst -Directory).Name -join ', ' | Tee-Object -FilePath $log -Append
"COPY_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
