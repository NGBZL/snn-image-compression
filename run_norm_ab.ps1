# Norm A/B: identical data, sampler and seed; only the decoder normalization differs.
# Each run is capped at 26 minutes (user directive: no long runs until issues are settled).
# ASCII only.
$ErrorActionPreference = "Continue"
$py = "C:\Users\chengyu\AppData\Local\Python\pythoncore-3.14-64\python.exe"
Set-Location "C:\Users\chengyu\order-tracking\snn_ab"

$common = @(
    "anchor.py", "train",
    "--image-size", "256", "--latent-channels", "32", "--num-steps", "10",
    "--group-size", "16", "--anchor-frac", "0.5", "--lam", "0.01", "--lr", "3e-4",
    "--noise-floor", "0.2", "--ref-wiring", "inp", "--ref-mode", "pairs",
    "--workers", "8", "--epochs", "18", "--time-budget-min", "26",
    "--save-every", "3", "--val-every", "2", "--val-batches", "8",
    "--eval-limit", "1228", "--seed", "42"
)

foreach ($n in @("gn", "bn", "none")) {
    "=== norm=$n start $(Get-Date -Format o) ===" | Out-File "normab_$n.log" -Encoding utf8
    & $py -u @common --norm $n --out "normab_$n.pth" 2>&1 |
        Out-File "normab_$n.log" -Append -Encoding utf8
    "=== norm=$n exit=$LASTEXITCODE $(Get-Date -Format o) ===" |
        Out-File "normab_$n.log" -Append -Encoding utf8
}
"ALL NORM A/B DONE $(Get-Date -Format o)" | Out-File normab_done.txt -Encoding utf8
