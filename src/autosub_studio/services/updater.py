"""Dich vu kiem tra, tai ve va chuan bi cap nhat ung dung tu dong.

Thuc hien theo dung cac nguyen tac an toan:
  - Repository: longthao9820-alt/autosub-studio
  - Kenh on dinh (stable channel): bo qua ban draft va prerelease
  - Dung duy nhat phien ban trung tam APP_VERSION tu autosub_studio.version
  - Tai goi zip va xac thuc ma bam SHA256 tu .sha256 hoac manifest
  - Tai tep tam .part, ho tro bao tien do, huy tai giua chung va gioi han timeout
  - Giai nen an toan chong zip traversal vao Data/staging/update_{VERSION}
  - Khong de ung dung dang chay tu ghi de chinh no
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autosub_studio.version import APP_VERSION

from .paths import app_root, is_portable, portable_data_dir

logger = logging.getLogger(__name__)

DEFAULT_REPO = "longthao9820-alt/autosub-studio"
STABLE_CHANNEL = "stable"
DEFAULT_TIMEOUT = 20.0
DOWNLOAD_TIMEOUT = 60.0
USER_AGENT = f"AutoSubStudio-Updater/{APP_VERSION}"


class UpdateError(Exception):
    """Loi chung trong qua trinh cap nhat."""


class UpdateCancelledError(UpdateError):
    """Nguoi dung chu dong huy qua trinh tai cap nhat."""


class UpdateVerificationError(UpdateError):
    """Xac thuc ma bam SHA256 hoac cau truc ban cap nhat that bai."""


class SecurityError(UpdateError):
    """Phat hien hanh vi nguy hiem (nhu zip path traversal)."""


@dataclass
class ReleaseInfo:
    """Thong tin ve ban phat hanh moi nhat tren GitHub Releases."""

    version: str
    tag_name: str
    title: str
    changelog: str
    published_at: str
    asset_name: str
    asset_url: str
    asset_size: int
    sha256: str = ""
    sha256_url: str = ""
    is_newer: bool = False
    prerelease: bool = False


def parse_version_tuple(v: str) -> tuple[int, ...]:
    """Chuyen chuoi phien ban thanh tuple so de so sanh thu tu.

    Vi du: "v2.0.1" -> (2, 0, 1), "2.1.0-beta" -> (2, 1, 0).
    """
    clean = str(v or "").strip().lstrip("vV")
    # Tach phan so truoc cac hau to nhu -beta, -rc
    base = re.split(r"[-+]", clean)[0]
    parts: list[int] = []
    for part in base.split("."):
        digits = re.findall(r"\d+", part)
        parts.append(int(digits[0]) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def compare_versions(v1: str, v2: str) -> int:
    """So sanh hai chuoi phien ban: tra ve -1 neu v1 < v2, 0 neu bang, 1 neu v1 > v2."""
    t1 = parse_version_tuple(v1)
    t2 = parse_version_tuple(v2)
    if t1 < t2:
        return -1
    if t1 > t2:
        return 1
    return 0


def is_version_newer(candidate: str, current: str = APP_VERSION) -> bool:
    """Kiem tra xem phien ban ung vien co moi hon phien ban hien tai hay khong."""
    return compare_versions(candidate, current) > 0


def get_staging_dir(version: str) -> Path:
    """Thu muc staging co dinh trong Data/staging/update_{version}."""
    base = portable_data_dir() if is_portable() else app_root() / "Data"
    staging = base / "staging" / f"update_{version}"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def get_backup_dir(version: str = APP_VERSION) -> Path:
    """Thu muc backup chua ban cu truoc khi thay the."""
    base = portable_data_dir() if is_portable() else app_root() / "Data"
    backup = base / "backup" / f"backup_{version}"
    backup.mkdir(parents=True, exist_ok=True)
    return backup


def fetch_latest_release(
    repo: str = DEFAULT_REPO,
    api_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> ReleaseInfo | None:
    """Lay thong tin ban release moi nhat tren kenh on dinh GitHub Releases.

    Chi xet ban khong phai la draft va khong phai la prerelease.
    """
    url = api_url or f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github.v3+json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        logger.warning("Khong the lay thong tin release tu GitHub: %s", e)
        return None

    if not isinstance(data, dict):
        return None

    # Stable channel: bo qua neu release duoc danh dau la draft hoac prerelease
    if data.get("draft", False) or data.get("prerelease", False):
        return None

    tag_name = str(data.get("tag_name", "")).strip()
    version = tag_name.lstrip("vV")
    if not version:
        return None

    assets = data.get("assets", [])
    if not isinstance(assets, list):
        return None

    # Tim asset portable update ZIP (khong phai file ma nguon)
    zip_asset: dict[str, Any] | None = None
    sha256_asset: dict[str, Any] | None = None
    manifest_asset: dict[str, Any] | None = None

    for asset in assets:
        name = str(asset.get("name", "")).strip()
        name_lower = name.lower()
        if name_lower.endswith(".zip"):
            if not zip_asset or "portable" in name_lower or "windows" in name_lower:
                zip_asset = asset
        elif name_lower.endswith(".sha256") or name_lower.endswith(".sha256.txt"):
            sha256_asset = asset
        elif name_lower in ("manifest.json", "release-manifest.json"):
            manifest_asset = asset

    if not zip_asset:
        logger.info("Release %s khong co asset zip hop le", tag_name)
        return None

    asset_name = str(zip_asset.get("name", ""))
    asset_url = str(zip_asset.get("browser_download_url", ""))
    asset_size = int(zip_asset.get("size", 0))
    sha256_url = str(sha256_asset.get("browser_download_url", "")) if sha256_asset else ""
    known_sha256 = ""

    # Neu co manifest.json, co the doc sha256 tu do
    if manifest_asset and not sha256_url:
        sha256_url = str(manifest_asset.get("browser_download_url", ""))

    return ReleaseInfo(
        version=version,
        tag_name=tag_name,
        title=str(data.get("name", "") or tag_name),
        changelog=str(data.get("body", "")).strip(),
        published_at=str(data.get("published_at", "")),
        asset_name=asset_name,
        asset_url=asset_url,
        asset_size=asset_size,
        sha256=known_sha256,
        sha256_url=sha256_url,
        is_newer=is_version_newer(version, APP_VERSION),
        prerelease=bool(data.get("prerelease", False)),
    )


def fetch_checksum_from_url(
    url: str,
    target_filename: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Tai noi dung file .sha256 hoac manifest.json de lay ma hash 64 ky tu hex."""
    if not url:
        return ""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("Khong tai duoc file checksum: %s", e)
        return ""

    # Truong hop 1: file json manifest
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            if "sha256" in data:
                return str(data["sha256"]).strip().lower()
            if "assets" in data and isinstance(data["assets"], dict):
                for k, v in data["assets"].items():
                    if (
                        target_filename
                        and target_filename in k
                        and isinstance(v, dict)
                        and "sha256" in v
                    ):
                        return str(v["sha256"]).strip().lower()
    except json.JSONDecodeError:
        pass

    # Truong hop 2: file text chuan "<sha256>  <filename>" hoac chi co ma hash
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1 and len(parts[0]) == 64:
            return parts[0].lower()
        if len(parts) >= 2:
            h, f = parts[0], parts[1]
            if len(h) == 64 and (not target_filename or target_filename in f):
                return h.lower()

    # Match bat ky chuoi 64 hex characters
    matches = re.findall(r"\b[a-fA-F0-9]{64}\b", content)
    return matches[0].lower() if matches else ""


def download_release_asset(
    release: ReleaseInfo,
    dest_dir: Path,
    expected_sha256: str = "",
    on_progress: Callable[[int, int], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    timeout: float = DOWNLOAD_TIMEOUT,
    chunk_size: int = 65536,
) -> Path:
    """Tai tep cap nhat sang tep tam .part, kiem tra SHA256 roi moi doi ten.

    Neu nguoi dung huy hoac sha256 khong khop, tep .part se bi xoa de khong de rac.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / release.asset_name
    part_path = dest_dir / f"{release.asset_name}.part"

    if part_path.exists():
        part_path.unlink(missing_ok=True)
    if final_path.exists():
        final_path.unlink(missing_ok=True)

    # Lay sha256 neu chua co
    sha_target = (expected_sha256 or release.sha256).strip().lower()
    if not sha_target and release.sha256_url:
        sha_target = fetch_checksum_from_url(
            release.sha256_url, release.asset_name, timeout=DEFAULT_TIMEOUT
        ).strip().lower()

    # Bat buoc phai co ma checksum SHA-256 hop le (dung 64 ky tu hex)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha_target):
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        raise UpdateVerificationError(
            f"Khong co ma bam SHA256 hop le (dung 64 ky tu hex) cho asset {release.asset_name}: "
            f"'{sha_target}'"
        )

    req = urllib.request.Request(
        release.asset_url,
        headers={"User-Agent": USER_AGENT},
    )

    hasher = hashlib.sha256()
    downloaded_bytes = 0
    total_bytes = release.asset_size

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                total_bytes = int(content_length)

            with open(part_path, "wb") as f:
                while True:
                    if cancel_flag and cancel_flag():
                        raise UpdateCancelledError("Qua trinh tai da bi huy boi nguoi dung")

                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    hasher.update(chunk)
                    downloaded_bytes += len(chunk)

                    if on_progress:
                        on_progress(downloaded_bytes, total_bytes)

    except Exception:
        # Xoa tep tam neu bi loi hoac huy
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        raise

    calculated_sha256 = hasher.hexdigest().lower()

    # Kiem tra checksum SHA256
    if calculated_sha256 != sha_target:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        raise UpdateVerificationError(
            f"Kiem tra ma SHA256 that bai: mong doi {sha_target}, nhan duoc {calculated_sha256}"
        )

    # Hoan tat: doi ten nguyen tu .part -> .zip
    part_path.replace(final_path)
    return final_path


def extract_update_archive(zip_path: Path, staging_dir: Path) -> Path:
    """Giai nen an toan tap tin zip vao thu muc staging, ngan chan Zip Traversal (Zip Slip).

    Don dep noi dung cu da giai nen do truoc do nhung giu nguyen tep zip dang co.
    Tra ve duong dan toi goc payload cua ban cap nhat.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Khong tim thay tap tin cap nhat: {zip_path}")

    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    zip_resolved = zip_path.resolve()

    # Don dep noi dung cu trong staging_dir tu lan thu truoc ma khong xoa Data cha
    for item in list(staging_dir.iterdir()):
        try:
            if item.resolve() == zip_resolved or item.name == zip_path.name:
                continue
            if item.name.endswith(".part"):
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Khong the xoa muc cu trong staging (%s): %s", item, exc)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            filename = member.filename

            # Ngan chan path traversal: kiem tra .. hoac duong dan tuyet doi
            if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
                raise SecurityError(f"Phat hien duong dan nguy hiem trong tap tin zip: {filename}")

            if ":" in filename:  # Windows drive letter nhu C:
                raise SecurityError(f"Phat hien ky tu o dia Windows trong zip: {filename}")

            target_path = (staging_dir / filename).resolve()
            if not target_path.is_relative_to(staging_dir):
                raise SecurityError(f"Duong dan giai nen vuot ra ngoai thu muc dich: {filename}")

            # Giai nen an toan tung muc
            zf.extract(member, staging_dir)

    # Xac dinh thu muc payload thuc su:
    # Neu zip chua mot thu muc goc duy nhat (vi du AutoSubStudio/...), lay thu muc do
    payload_dir = resolve_payload_root(staging_dir)
    return payload_dir


def resolve_payload_root(staging_dir: Path) -> Path:
    """Xac dinh thu muc chua chuong trinh trong staging."""
    entries = [
        p
        for p in staging_dir.iterdir()
        if p.name not in ("__pycache__", ".part") and not p.name.endswith(".zip")
    ]
    # Neu staging chi chua mot thu muc duy nhat va thu muc do chua cac tep chuong trinh
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        # Neu thu muc trong co tep chuong trinh thi chon no
        if any(
            (inner / name).exists()
            for name in ("AutoSubStudio.exe", "run_app.py", "_internal", "src")
        ):
            return inner
    return staging_dir


def verify_package_layout(payload_dir: Path) -> bool:
    """Kiem tra tinh hop le cua cau truc goi cap nhat truoc khi thay the runtime."""
    if not payload_dir.is_dir():
        return False

    # Phai chua it nhat AutoSubStudio.exe hoac run_app.py hoac thu muc src hop le
    has_exe = (payload_dir / "AutoSubStudio.exe").is_file()
    has_run = (payload_dir / "run_app.py").is_file()
    has_src = (payload_dir / "src" / "autosub_studio" / "__init__.py").is_file()

    return has_exe or has_run or has_src


def find_updater_helper(app_dir: Path) -> list[str]:
    """Tim kiem chuong trinh helper de thuc hien cap nhat sau khi ung dung dong."""
    # 1. Ban dong goi co the co AutoSubUpdater.exe hoac updater.exe
    for exe_name in ("AutoSubUpdater.exe", "updater.exe"):
        for base in (app_dir, app_dir / "_internal"):
            cand = base / exe_name
            if cand.is_file():
                return [str(cand)]

    # 2. Ban chay tu ma nguon: dung python voi scripts/updater.py hoac module autosub_updater
    script_cand = app_dir / "scripts" / "updater.py"
    if script_cand.is_file():
        return [sys.executable, str(script_cand)]

    # 3. Chay duoi dang module
    return [sys.executable, "-m", "autosub_updater.main"]


def launch_updater_helper(
    target_root: Path,
    staging_payload: Path,
    backup_dir: Path | None = None,
    restart: bool = True,
    restart_cmd: list[str] | None = None,
    custom_helper_cmd: list[str] | None = None,
) -> subprocess.Popen[Any]:
    """Khoi chay helper tach biet de thay the file sau khi ung dung dong hoan toan."""
    helper_cmd = custom_helper_cmd or find_updater_helper(target_root)
    if not helper_cmd:
        raise UpdateError("Khong tim thay chuong trinh tro giup cap nhat (updater helper)")

    actual_backup = backup_dir or get_backup_dir(APP_VERSION)
    pid = os.getpid()

    # Neu helper la mot file .exe nam ben trong target_root, chep tam sang staging
    # de tranh bi khoa file Windows khi updater ghi de target_root
    first_arg = Path(helper_cmd[0])
    if first_arg.suffix.lower() == ".exe" and first_arg.is_file():
        try:
            if first_arg.resolve().is_relative_to(target_root.resolve()):
                runner_copy = staging_payload.parent / "AutoSubUpdater_runner.exe"
                shutil.copy2(first_arg, runner_copy)
                helper_cmd = [str(runner_copy)] + helper_cmd[1:]
        except (ValueError, OSError):
            pass

    args = list(helper_cmd) + [
        "--app-root",
        str(target_root.resolve()),
        "--staging-dir",
        str(staging_payload.resolve()),
        "--backup-dir",
        str(actual_backup.resolve()),
        "--pid",
        str(pid),
    ]
    if restart:
        args.append("--restart")
    if restart_cmd:
        args.extend(["--restart-cmd", subprocess.list2cmdline(restart_cmd)])

    flags = 0
    if os.name == "nt":
        # Tach hoan toan khoi tien trinh cha tren Windows
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    logger.info("Khoi chay updater helper: %s", args)
    return subprocess.Popen(
        args,
        creationflags=flags,
        close_fds=True,
    )


__all__ = [
    "DEFAULT_REPO",
    "DOWNLOAD_TIMEOUT",
    "STABLE_CHANNEL",
    "ReleaseInfo",
    "SecurityError",
    "UpdateCancelledError",
    "UpdateError",
    "UpdateVerificationError",
    "compare_versions",
    "download_release_asset",
    "extract_update_archive",
    "fetch_checksum_from_url",
    "fetch_latest_release",
    "find_updater_helper",
    "get_backup_dir",
    "get_staging_dir",
    "is_version_newer",
    "launch_updater_helper",
    "parse_version_tuple",
    "resolve_payload_root",
    "verify_package_layout",
]
