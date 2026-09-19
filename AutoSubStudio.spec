# -*- mode: python ; coding: utf-8 -*-
"""Cau hinh dong goi cho PyInstaller - ban chay doc lap.

Ban dong goi tu chua du moi thu de chay tren mot may Windows moi:
  - FFmpeg va FFprobe  (thu muc 'ffmpeg')
  - Model nhan dang giong noi (thu muc 'models')
  - Thu vien CUDA de tang toc bang card do hoa NVIDIA (thu muc 'cuda')
  - Model doc chu tren hinh (di kem thu vien rapidocr trong '_internal')
Du lieu nguoi dung (cau hinh, du an) nam trong thu muc 'Data' canh tep chay,
nen chep ca thu muc sang may khac la dung duoc ngay.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)
ASSETS = ROOT / "assets"

hidden = [
    "autosub_studio.ui.main_window",
    "autosub_studio.selftest",
    "autosub_studio.services.gpu",
    "sqlalchemy.dialects.sqlite",
]
datas = []

# Cac goi tuy chon: chi thu thap khi moi truong dong goi da cai san.
for optional in ("faster_whisper", "deep_translator", "anthropic",
                 "rapidocr"):
    try:
        __import__(optional)
    except ImportError:
        continue
    hidden += collect_submodules(optional)
    datas += collect_data_files(optional)

# FFmpeg va model di kem, dat canh tep chay chu khong nhet vao _internal.
binaries = []
ffmpeg_dir = ASSETS / "ffmpeg"
if ffmpeg_dir.is_dir():
    for exe in sorted(ffmpeg_dir.glob("*.exe")):
        binaries.append((str(exe), "ffmpeg"))

piper_dir = ASSETS / "piper"
if piper_dir.is_dir():
    espeak_dir = piper_dir / "espeak-ng-data"
    if espeak_dir.is_dir():
        datas.append((str(espeak_dir), "piper/espeak-ng-data"))
    for item in sorted(piper_dir.iterdir()):
        if item.is_file():
            binaries.append((str(item), "piper"))

cuda_dir = ASSETS / "cuda"
if cuda_dir.is_dir():
    for dll in sorted(cuda_dir.glob("*.dll")):
        binaries.append((str(dll), "cuda"))

models_dir = ASSETS / "models"
if models_dir.is_dir():
    for model in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for item in sorted(model.iterdir()):
            if item.is_file() and item.suffix.lower() not in (".md", ".gitattributes"):
                datas.append((str(item), f"models/{model.name}"))

# Bo PP-OCRv4 Mobile tieng Trung dung cho che do "Nhanh Nhu NTS".
ocr_dir = ASSETS / "ocr"
if ocr_dir.is_dir():
    for item in sorted(ocr_dir.glob("*.onnx")):
        datas.append((str(item), "ocr"))

download_dir = ASSETS / "download"
if download_dir.is_dir():
    for exe in sorted(download_dir.glob("*.exe")):
        binaries.append((str(exe), "download"))

for asset_name in ("lut", "fonts"):
    asset_dir = ASSETS / asset_name
    if asset_dir.is_dir():
        for item in sorted(asset_dir.iterdir()):
            if item.is_file():
                datas.append((str(item), asset_name))

a = Analysis(
    ["run_app.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "torch", "torchaudio", "torchvision"],
    noarchive=False,
)
# PyInstaller tu chep them mot so DLL CUDA ra goc _internal. Chung ta da dat
# san chung trong thu muc 'cuda' roi, nen bo ban trung de goi khong phinh them.
if cuda_dir.is_dir():
    bundled_names = {p.name.lower() for p in cuda_dir.glob("*.dll")}
    kept = []
    for entry in a.binaries:
        dest = str(entry[0])
        head, _, tail = dest.replace("/", "\\").rpartition("\\")
        if not head and tail.lower() in bundled_names:
            continue
        kept.append(entry)
    a.binaries = kept

# Qt 6 tren Windows dung ICU cua he dieu hanh (System32\icuuc.dll). Khi PATH
# cua may build co Poppler, PyInstaller co the nhat nham ICU 78 cua Poppler va
# dat no o goc _internal. DLL ngoai lai nay che ICU cua Windows, lam QtCore.pyd
# loi "The specified procedure could not be found" ngay khi mo giao dien.
a.binaries = [
    entry
    for entry in a.binaries
    if Path(str(entry[0])).name.lower() != "icuuc.dll"
    and not Path(str(entry[0])).name.lower().startswith("icudt")
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoSubStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AutoSubStudio",
)
