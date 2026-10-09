# v8: full data (train+val+test = 8189 -> 6961 train / 1228 holdout) + augmentation
#     + save-best-by-real-validation. Auto-resumes from the latest checkpoint.
# NOTE: keep this file ASCII-only. PowerShell 5.1 reads .ps1 as ANSI/GBK and
#       non-ASCII characters corrupt the parse.
param(
    [int]$Epochs = 330,
    [double]$BudgetMin = 570,
    [int]$MaxRetry = 8
)
$ErrorActionPreference = "Continue"
$py  = "C:\Users\chengyu\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$out = "anchor_v8.pth"
$log = "anchor_v8.log"
Set-Location "C:\Users\chengyu\order-tracking\snn_ab"

for ($try = 1; $try -le $MaxRetry; $try++) {
    $resume = @()
    if (Test-Path $out) {
        $ok = & $py -c "import torch;d=torch.load(r'$out',map_location='cpu',weights_only=False);print(int(all(torch.isfinite(p).all() for p in d['state_dict'].values())))" 2>$null
        if ("$ok".Trim() -eq "1") { $resume = @("--resume", $out) }
    }
    "[run_v8] attempt $try resume=$($resume -join ' ') at $(Get-Date -Format o)" |
        Out-File -FilePath $log -Append -Encoding utf8
    & $py -u anchor.py train `
        --image-size 256 --latent-channels 32 --num-steps 10 `
        --group-size 16 --anchor-frac 0.5 --lam 0.01 --lr 3e-4 --noise-floor 0.2 `
        --ref-wiring inp --ref-mode pairs --workers 8 `
        --epochs $Epochs --time-budget-min $BudgetMin `
        --save-every 5 --val-every 5 --val-batches 10 --eval-limit 1228 `
        --out $out @resume 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    if ($LASTEXITCODE -eq 0) { "[run_v8] finished cleanly"; break }
    "[run_v8] exit $LASTEXITCODE, retrying in 20s" | Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds 20
}

