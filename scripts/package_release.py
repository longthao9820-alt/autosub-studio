"""Script tu dong hoa dong goi va phat hanh ban portable cho AutoSub Studio.

Nguyen tac:
  - Phien ban nhat quan tu APP_VERSION trung tam trong autosub_studio.version
  - Dong goi zip loai tru thu muc Data va du lieu nguoi dung
  - Tao tep checksum .sha256 va manifest.json
  - Ho tro --dry-run va --source-dir de kiem tra khong can build lai
  - Chi publish GitHub bang lenh gh khi co co --publish ro rang (khong tu dong publish am tham)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

# Nap src vao sys.path de lay APP_VERSION
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from autosub_studio.version import APP_NAME, APP_VERSION  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Release] %(levelname)s: %(message)s",
)
logger = logging.getLogger("package_release")

DEFAULT_REPO = "longthao9820-alt/autosub-studio"
EXCLUDED_DIR_NAMES = {"data", "__pycache__", ".git", ".pytest_cache", ".venv"}
EXCLUDED_EXTENSIONS = {".part", ".tmp", ".log"}
MAX_RELEASE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB = 2,147,483,648 bytes


def calculate_sha256(file_path: Path) -> str:
    """Tinh ma bam SHA256 cho tep tin."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def should_include_path(rel_path: Path) -> bool:
    """Kiem tra xem file/thu muc co duoc dua vao goi release hay khong."""
    parts_lower = [p.lower() for p in rel_path.parts]
    for excluded in EXCLUDED_DIR_NAMES:
        if excluded in parts_lower:
            return False
    # Loai tru tuyet doi bat ky thu muc models nao trong _internal hoac o goc
    if "_internal" in parts_lower and "models" in parts_lower:
        idx_int = parts_lower.index("_internal")
        if "models" in parts_lower[idx_int:]:
            return False
    if parts_lower and parts_lower[0] == "models":
        return False
    return rel_path.suffix.lower() not in EXCLUDED_EXTENSIONS


def create_release_zip(
    source_dir: Path,
    zip_dest: Path,
    compression: int = zipfile.ZIP_DEFLATED,
) -> int:
    """Dong goi source_dir vao zip_dest, loai bo thu muc Data va tep rac."""
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0

    with zipfile.ZipFile(zip_dest, "w", compression=compression) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(source_dir)
            if not should_include_path(rel):
                continue
            # Luu file vao zip voi duong dan tuong doi chuan
            arcname = str(rel).replace("\\", "/")
            zf.write(p, arcname)
            file_count += 1

    return file_count


def generate_manifest(
    version: str,
    asset_name: str,
    sha256_hash: str,
    file_size: int,
    channel: str = "stable",
) -> dict[str, str | int]:
    """Tao dict manifest cho ban release."""
    return {
        "version": version,
        "tag_name": f"v{version}",
        "channel": channel,
        "asset_name": asset_name,
        "sha256": sha256_hash,
        "size": file_size,
        "release_date": str(date.today()),
    }


def generate_release_notes(version: str, manifest: dict[str, str | int]) -> str:
    """Tao noi dung release notes mau cho ban phat hanh."""
    return f"""# {APP_NAME} v{version}

### Thông tin phát hành
- **Phiên bản:** v{version}
- **Kênh:** {manifest.get("channel", "stable")}
- **Tập tin:** `{manifest.get("asset_name")}`
- **Dung lượng:** {manifest.get("size")} bytes
- **Mã SHA256:** `{manifest.get("sha256")}`

### Hướng dẫn sử dụng
1. Tải về gói zip portable `{manifest.get("asset_name")}`.
2. Giải nén vào thư mục bất kỳ trên máy tính.
3. Chạy `AutoSubStudio.exe` để bắt đầu. Toàn bộ dữ liệu người dùng được lưu trong thư mục `Data`.
"""


def package_release(
    source_dir: Path,
    output_dir: Path,
    version: str = APP_VERSION,
    repo: str = DEFAULT_REPO,
    publish: bool = False,
    dry_run: bool = False,
) -> dict[str, Path | str | int]:
    """Thuc hien toan bo quy trinh dong goi ban release portable."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dong goi ban release: v%s", version)
    logger.info("Thu muc nguon: %s", source_dir)
    logger.info("Thu muc xuat: %s", output_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc nguon: {source_dir}")

    # Ten tep goi zip chuan
    asset_name = f"AutoSubStudio-v{version}-windows-x64.zip"
    zip_path = output_dir / asset_name
    sha256_path = output_dir / f"{asset_name}.sha256"
    manifest_path = output_dir / "manifest.json"
    notes_path = output_dir / f"release_notes_v{version}.md"

    if dry_run:
        logger.info("[DRY RUN] Kiem tra cau truc thu muc va mo phong dong goi...")

    # Kiem tra xem co updater helper khong
    has_updater = (source_dir / "AutoSubUpdater.exe").is_file() or (
        source_dir / "scripts" / "updater.py"
    ).is_file()
    if not has_updater:
        logger.warning(
            "Thu muc nguon chua co AutoSubUpdater.exe hoac scripts/updater.py "
            "(co the la ban build cu truoc khi tich hop updater)."
        )

    # Tao tep ZIP (dry_run dung ZIP_STORED de chay nhanh)
    comp = zipfile.ZIP_STORED if dry_run else zipfile.ZIP_DEFLATED
    file_count = create_release_zip(source_dir, zip_path, compression=comp)
    file_size = zip_path.stat().st_size
    sha256_hash = calculate_sha256(zip_path)

    if file_size >= MAX_RELEASE_BYTES:
        raise ValueError(
            f"Goi release vuot qua gioi han 2 GiB ({file_size} >= {MAX_RELEASE_BYTES} bytes)!"
        )

    logger.info("Da dong goi %d tep vao %s (Size: %d bytes)", file_count, zip_path.name, file_size)
    logger.info("SHA256: %s", sha256_hash)

    # Ghi tep .sha256
    sha256_content = f"{sha256_hash}  {asset_name}\n"
    sha256_path.write_text(sha256_content, encoding="utf-8")

    # Ghi manifest.json
    manifest_data = generate_manifest(version, asset_name, sha256_hash, file_size)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Ghi release notes
    notes_content = generate_release_notes(version, manifest_data)
    notes_path.write_text(notes_content, encoding="utf-8")

    gh_cmd = [
        "gh",
        "release",
        "create",
        f"v{version}",
        str(zip_path),
        str(sha256_path),
        str(manifest_path),
        "--repo",
        repo,
        "--title",
        f"{APP_NAME} v{version}",
        "--notes-file",
        str(notes_path),
    ]
    gh_cmd_str = " ".join(gh_cmd)

    if publish and not dry_run:
        logger.info("Dang thuc thi lenh publish: %s", gh_cmd_str)
        import subprocess

        res = subprocess.run(gh_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Lenh gh release that bai: {res.stderr}")
        logger.info("Publish GitHub Release thanh cong!")
    else:
        logger.info(
            "[Khong publish] Ban phat hanh da san sang trong %s. De publish, chay lenh:",
            output_dir,
        )
        logger.info("  %s", gh_cmd_str)

    return {
        "version": version,
        "zip_path": zip_path,
        "sha256_path": sha256_path,
        "manifest_path": manifest_path,
        "notes_path": notes_path,
        "sha256": sha256_hash,
        "size": file_size,
        "gh_command": gh_cmd_str,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI cho scripts/package_release.py."""
    parser = argparse.ArgumentParser(
        description="AutoSub Studio Release Packager & Artifact Generator"
    )
    parser.add_argument(
        "--source-dir",
        default=str(_ROOT / "dist" / "AutoSubStudio"),
        help="Thu muc ban build portable (mac dinh: dist/AutoSubStudio)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ROOT / "release"),
        help="Thu muc luu tru cac asset release",
    )
    parser.add_argument(
        "--version",
        default=APP_VERSION,
        help=f"Phien ban release (mac dinh: {APP_VERSION})",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub repository (mac dinh: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Thuc thi lenh gh release create de upload len GitHub (can dang nhap gh)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chay thu quy trinh dong goi va tao checksum, khong publish len GitHub",
    )

    args = parser.parse_args(argv)

    # Kiem tra tinh dong nhat cua phien ban
    if args.version != APP_VERSION:
        clean_arg = re.sub(r"^v", "", args.version)
        clean_app = re.sub(r"^v", "", APP_VERSION)
        if clean_arg != clean_app:
            logger.error("Phien ban %s khong khop voi APP_VERSION (%s)!", args.version, APP_VERSION)
            return 1

    try:
        package_release(
            source_dir=Path(args.source_dir),
            output_dir=Path(args.output_dir),
            version=APP_VERSION,
            repo=args.repo,
            publish=args.publish,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        logger.error("Dong goi release that bai: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
