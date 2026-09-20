"""Quy uoc thu muc cua ung dung va cac ham tep an toan."""

from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

APP_DIR_NAME = "AutoSubStudio"
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def app_root() -> Path:
    """Thu muc chua ung dung (ho tro ca ban chay tu ma nguon va ban dong goi)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def is_portable() -> bool:
    """Ban dong goi dang chay o che do di dong hay khong.

    Che do di dong bat khi ung dung da dong goi va thu muc chua no ghi duoc.
    Khi do moi thu (cau hinh, du an, model) nam ngay canh tep chay, nen chep
    ca thu muc sang may khac la dung duoc ngay.
    """
    if not getattr(sys, "frozen", False):
        return False
    probe = app_root() / ".ghi-thu.tmp"
    try:
        probe.write_text("x", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def portable_data_dir() -> Path:
    """Thu muc du lieu nam canh tep chay khi o che do di dong."""
    return app_root() / "Data"


def bundled_dir(name: str) -> Path | None:
    """Thu muc kem theo ban dong goi (vi du 'ffmpeg', 'models'), None neu khong co."""
    for base in (app_root(), app_root() / "_internal"):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def fonts_dir() -> Path | None:
    """Thu muc font Unicode hop le cua ban dong goi hoac ma nguon."""
    if not getattr(sys, "frozen", False):
        dev = app_root() / "assets" / "fonts"
        if (dev / "NotoSans-Regular.ttf").is_file():
            return dev
    bundled = bundled_dir("fonts")
    if bundled is not None and (bundled / "NotoSans-Regular.ttf").is_file():
        return bundled
    return None


def config_dir() -> Path:
    """Thu muc luu cau hinh va khoa API cua nguoi dung."""
    if is_portable():
        d = portable_data_dir() / "config"
    else:
        base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
        d = root / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_workspace() -> Path:
    """Thu muc lam viec mac dinh chua du an, model va ket qua."""
    if is_portable():
        return portable_data_dir() / "workspace"
    docs = Path.home() / "Documents"
    root = docs if docs.is_dir() else Path.home()
    return root / APP_DIR_NAME


def ensure_workspace(path: str | Path) -> Path:
    """Tao day du cay thu muc lam viec."""
    root = Path(path).expanduser()
    for sub in ("projects", "models", "fonts", "luts", "presets", "logs", "db", "exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def safe_name(name: str, fallback: str = "du-an") -> str:
    """Chuyen ten nguoi dung nhap thanh ten thu muc hop le tren Windows."""
    text = unicodedata.normalize("NFC", str(name or "")).strip()
    text = _INVALID.sub("_", text).strip(" .")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return fallback
    if text.split(".")[0].upper() in _RESERVED:
        text = f"_{text}"
    return text[:80]


def unique_path(path: str | Path) -> Path:
    """Them hau to so neu tep/thu muc da ton tai."""
    p = Path(path)
    if not p.exists():
        return p
    stem, suffix, parent = p.stem, p.suffix, p.parent
    for i in range(1, 10000):
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
    raise FileExistsError(f"Khong tao duoc ten khac cho {p}")


def write_text_atomic(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Ghi tep qua tep tam roi doi ten, tranh mat du lieu khi ghi do."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(content, encoding=encoding, newline="\n")
    tmp.replace(p)
    return p


def human_size(num_bytes: float) -> str:
    """Doi so byte thanh chuoi de doc."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def free_space(path: str | Path) -> int:
    """So byte trong con lai tren o dia chua `path`."""
    try:
        return shutil.disk_usage(str(Path(path).anchor or Path(path))).free
    except OSError:
        return 0


def asr_models_dir() -> Path:
    """Thu muc luu tru model ASR (faster-whisper) doc lap voi workspace/du an/cap nhat."""
    env = os.environ.get("AUTOSUB_ASR_MODELS_DIR")
    if env:
        target = Path(env)
    elif is_portable():
        target = portable_data_dir() / "models" / "asr"
    else:
        target = app_root() / "Data" / "models" / "asr"
    target.mkdir(parents=True, exist_ok=True)
    return target


def migrate_v1_models(app_dir: Path | None = None) -> list[Path]:
    """Chuyen an toan cac model ASR tu ban cu (_internal/models) sang Data/models/asr.

    Dam bao:
      - Khong chep lai neu model dich da ton tai hop le (repeat-safe)
      - Chep qua thu muc tam va kiem tra toan ven truoc khi doi ten
      - Giu nguyen ban goc cho den khi kiem tra toan ven thanh cong
    """
    root = (app_dir or app_root()).resolve()
    old_models_dir = root / "_internal" / "models"
    if not old_models_dir.is_dir():
        old_models_dir = root / "models"
        if not old_models_dir.is_dir():
            return []

    if app_dir is not None:
        dest_base = root / "Data" / "models" / "asr"
    else:
        dest_base = asr_models_dir()
    dest_base.mkdir(parents=True, exist_ok=True)

    migrated: list[Path] = []
    for item in sorted(old_models_dir.iterdir()):
        if not item.is_dir():
            continue
        src_bin = item / "model.bin"
        if not src_bin.is_file():
            continue

        dst_dir = dest_base / item.name
        dst_bin = dst_dir / "model.bin"

        # Kiem tra xem da ton tai hop le chua (repeat-safe, khong chep lai)
        src_size = src_bin.stat().st_size
        if dst_bin.is_file() and dst_bin.stat().st_size == src_size and src_size > 0:
            continue

        # Chep an toan qua thu muc tam, bao toan ban goc
        tmp_dir = dest_base / f".tmp_migrate_{item.name}_{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for sub_item in item.iterdir():
                dst_sub = tmp_dir / sub_item.name
                if sub_item.is_file():
                    shutil.copy2(sub_item, dst_sub)
                elif sub_item.is_dir():
                    shutil.copytree(sub_item, dst_sub)

            # Kiem tra toan ven sau khi chep
            tmp_bin = tmp_dir / "model.bin"
            if not tmp_bin.is_file() or tmp_bin.stat().st_size != src_size:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            # Hoan tat: doi ten nguyen tu thu muc tam sang thu muc dich
            if dst_dir.exists():
                shutil.rmtree(dst_dir, ignore_errors=True)
            tmp_dir.replace(dst_dir)
            migrated.append(dst_dir)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return migrated


def piper_models_dir() -> Path:
    """Thu muc luu tru model Piper doc lap voi workspace/du an/cap nhat."""
    env = os.environ.get("AUTOSUB_PIPER_MODELS_DIR")
    if env:
        target = Path(env)
    elif is_portable():
        target = portable_data_dir() / "models" / "piper"
    else:
        target = app_root() / "Data" / "models" / "piper"
    target.mkdir(parents=True, exist_ok=True)
    return target


def piper_bin_path() -> Path | None:
    """Duong dan toi piper.exe neu co (uu tien bundled, sau do assets/piper hoac PATH)."""
    env = os.environ.get("AUTOSUB_PIPER_EXE")
    if env and Path(env).is_file():
        return Path(env)
    exe_name = "piper.exe" if os.name == "nt" else "piper"
    bundled = bundled_dir("piper")
    if bundled:
        candidate = bundled / exe_name
        if candidate.is_file():
            return candidate
    dev_candidate = app_root() / "assets" / "piper" / exe_name
    if dev_candidate.is_file():
        return dev_candidate
    which = shutil.which(exe_name) or shutil.which("piper")
    if which:
        return Path(which)
    return None


def piper_espeak_data_dir() -> Path | None:
    """Thu muc espeak-ng-data di kem piper neu co."""
    env = os.environ.get("AUTOSUB_PIPER_ESPEAK_DATA")
    if env and Path(env).is_dir():
        return Path(env)
    bundled = bundled_dir("piper")
    if bundled:
        candidate = bundled / "espeak-ng-data"
        if candidate.is_dir():
            return candidate
    dev_candidate = app_root() / "assets" / "piper" / "espeak-ng-data"
    if dev_candidate.is_dir():
        return dev_candidate
    piper_exe = piper_bin_path()
    if piper_exe:
        sibling = piper_exe.parent / "espeak-ng-data"
        if sibling.is_dir():
            return sibling
    return None
