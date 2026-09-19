# Dong goi AutoSub Studio thanh thu muc chay duoc bang nhap doi.
# Ban dong goi tu chua FFmpeg, model nhan dang giong noi va model doc chu tren hinh,
# nen chep ca thu muc sang may Windows khac la dung duoc ngay.
#
# Cach dung:  powershell -ExecutionPolicy Bypass -File scripts\build.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# --- Kiem tra cac thanh phan se duoc nhung vao ------------------------------
$assets = Join-Path $root "assets"
$missing = @()
if (-not (Test-Path (Join-Path $assets "ffmpeg\ffmpeg.exe")))  { $missing += "assets\ffmpeg\ffmpeg.exe" }
if (-not (Test-Path (Join-Path $assets "ffmpeg\ffprobe.exe"))) { $missing += "assets\ffmpeg\ffprobe.exe" }
if (-not (Test-Path (Join-Path $assets "download\yt-dlp.exe"))) { $missing += "assets\download\yt-dlp.exe" }
$modelDirs = @()
if (Test-Path (Join-Path $assets "models")) {
    $modelDirs = @(Get-ChildItem (Join-Path $assets "models") -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "model.bin") })
}
if ($modelDirs.Count -eq 0) { $missing += "assets\models\<ten-model>\model.bin" }
$ocrModels = @(
    "ch_PP-OCRv4_det_infer.onnx",
    "ch_PP-OCRv4_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx"
)
foreach ($ocrModel in $ocrModels) {
    if (-not (Test-Path (Join-Path $assets "ocr\$ocrModel"))) {
        $missing += "assets\ocr\$ocrModel"
    }
}
$cudaCount = 0
if (Test-Path (Join-Path $assets "cuda")) {
    $cudaCount = @(Get-ChildItem (Join-Path $assets "cuda") -Filter *.dll).Count
}
if ($cudaCount -eq 0) {
    Write-Warning "Khong co assets\cuda: ban dong goi se chay bang CPU, khong tang toc GPU."
} else {
    Write-Host "Se nhung: $cudaCount tep thu vien CUDA (tang toc GPU)"
}

if ($missing.Count -gt 0) {
    Write-Warning "Thieu cac thanh phan sau, ban dong goi se KHONG chay doc lap duoc:"
    $missing | ForEach-Object { Write-Warning "  - $_" }
    Write-Warning "Xem muc 'Dong goi lai' trong README.md de biet cach chuan bi."
} else {
    Write-Host "Se nhung: FFmpeg + model $($modelDirs.Name -join ', ')"
}

# Ban chay portable luu du an ngay trong dist\AutoSubStudio\Data. Moi lan
# dong goi phai giu nguyen thu muc nay, neu khong thao tac don ban cu se xoa
# ca danh sach du an cua nguoi dung.
$liveData = Join-Path $root "dist\AutoSubStudio\Data"
$backupRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "AutoSubStudio_Data_" + [guid]::NewGuid().ToString("N")
)
$hadLiveData = Test-Path -LiteralPath $liveData
$dataRestored = $false
if ($hadLiveData) {
    New-Item -ItemType Directory -Path $backupRoot | Out-Null
    Get-ChildItem -LiteralPath $liveData -Force | Copy-Item -Destination $backupRoot -Recurse -Force
    Write-Host "Da sao luu tam du lieu du an truoc khi dong goi."
}

function Restore-PortableData {
    if (-not $hadLiveData -or $script:dataRestored) { return }
    New-Item -ItemType Directory -Force -Path $liveData | Out-Null
    Get-ChildItem -LiteralPath $backupRoot -Force | Copy-Item -Destination $liveData -Recurse -Force
    $script:dataRestored = $true
    Write-Host "Da khoi phuc nguyen ven du lieu du an vao ban dong goi."
}

try {

# --- Don ban cu -------------------------------------------------------------
Write-Host "Don thu muc build cu..."
if (Test-Path (Join-Path $root "build")) { Remove-Item -Recurse -Force (Join-Path $root "build") }
if (Test-Path (Join-Path $root "dist\AutoSubStudio")) {
    Remove-Item -Recurse -Force (Join-Path $root "dist\AutoSubStudio")
}

# --- Dong goi ---------------------------------------------------------------
Write-Host "Chay PyInstaller (buoc nay mat vai phut)..."
& $py -m PyInstaller --noconfirm --clean AutoSubStudio.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller that bai (ma loi $LASTEXITCODE)" }

$out = Join-Path $root "dist\AutoSubStudio"
$exe = Join-Path $out "AutoSubStudio.exe"
if (-not (Test-Path $exe)) { throw "Khong tim thay $exe sau khi dong goi" }

# ICU cua Poppler co the bi PyInstaller nhat nham tu PATH va lam PySide6/QtCore
# khong nap duoc. Qt tren Windows phai dung icuuc.dll cua he dieu hanh.
$foreignIcu = Join-Path $out "_internal\icuuc.dll"
if (Test-Path $foreignIcu) {
    throw "Phat hien DLL ICU ngoai lai trong ban dong goi: $foreignIcu"
}

# --- Tai lieu va tep tien ich di kem ---------------------------------------
foreach ($doc in @("HUONG-DAN.txt", "README.md", "NTS_FEATURE_MATRIX.md", "NTS_AUDIT.json")) {
    $p = Join-Path $root $doc
    if (Test-Path $p) { Copy-Item $p $out -Force }
}
$lic = Join-Path $root "LICENSE"
if (Test-Path $lic) { Copy-Item $lic (Join-Path $out "LICENSE-AutoSubStudio.txt") -Force }

@"
@echo off
rem Kiem tra nhanh xem may nay co chay duoc AutoSub Studio khong (~10 giay).
cd /d "%~dp0"
start "" "AutoSubStudio.exe" --selftest
"@ | Set-Content -Path (Join-Path $out "KIEM-TRA-MAY-NAY.bat") -Encoding ascii

@"
@echo off
rem Kiem tra sau: chay thu that OCR, nhan dang giong noi va render (1-2 phut).
cd /d "%~dp0"
start "" "AutoSubStudio.exe" --selftest-full
"@ | Set-Content -Path (Join-Path $out "KIEM-TRA-SAU.bat") -Encoding ascii

New-Item -ItemType Directory -Force (Join-Path $out "Data") | Out-Null
Restore-PortableData

# --- Kiem tra ket qua -------------------------------------------------------
$checks = @{
    "ffmpeg.exe"  = Join-Path $out "_internal\ffmpeg\ffmpeg.exe"
    "ffprobe.exe" = Join-Path $out "_internal\ffmpeg\ffprobe.exe"
    "model AI"    = Join-Path $out "_internal\models"
    "OCR Nhanh Nhu NTS" = Join-Path $out "_internal\ocr\ch_PP-OCRv4_rec_infer.onnx"
    "thu vien CUDA" = Join-Path $out "_internal\cuda\cudnn64_9.dll"
    "yt-dlp" = Join-Path $out "_internal\download\yt-dlp.exe"
    "thu vien LUT" = Join-Path $out "_internal\lut"
    "thu vien font" = Join-Path $out "_internal\fonts"
}
foreach ($name in $checks.Keys) {
    if (Test-Path $checks[$name]) { Write-Host "  [OK]   $name da duoc nhung" }
    else { Write-Warning "  [THIEU] $name khong co trong ban dong goi" }
}

$size = (Get-ChildItem $out -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Xong. Ban dong goi: {0} ({1:N0} MB)" -f $out, $size)
Write-Host "Chep ca thu muc nay sang may khac la dung duoc ngay."
}
finally {
    Restore-PortableData
    if (Test-Path -LiteralPath $backupRoot) {
        $resolvedBackup = [IO.Path]::GetFullPath($backupRoot)
        $resolvedTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if (-not $resolvedBackup.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Tu choi xoa thu muc sao luu ngoai TEMP: $resolvedBackup"
        }
        Remove-Item -LiteralPath $resolvedBackup -Recurse -Force
    }
}
