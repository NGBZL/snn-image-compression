# Download DIV2K (and Flickr2K) from hf-mirror at ~5 MB/s.
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI/GBK.
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\chengyu\order-tracking\snn_ab"
New-Item -ItemType Directory -Force -Path "data\div2k" | Out-Null

$jobs = @(
    @{ name = "DIV2K_train.zip"; url = "https://hf-mirror.com/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip" },
    @{ name = "Flickr2K.zip";   url = "https://hf-mirror.com/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip" }
)
foreach ($j in $jobs) {
    $dst = Join-Path "data\div2k" $j.name
    "start $($j.name) $(Get-Date -Format o)" | Out-File data\div2k\download.log -Append -Encoding utf8
    # -C - resumes a partial file, so a dropped connection does not restart from zero
    & curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors -o $dst $j.url 2>&1 |
        Out-File data\div2k\download.log -Append -Encoding utf8
    "done $($j.name) exit=$LASTEXITCODE size=$((Get-Item $dst -EA SilentlyContinue).Length)" |
        Out-File data\div2k\download.log -Append -Encoding utf8
}
"ALL DONE $(Get-Date -Format o)" | Out-File data\div2k\download.log -Append -Encoding utf8
