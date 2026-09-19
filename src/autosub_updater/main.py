"""Tien trinh tro giup cap nhat tach biet (Updater Helper).

Dam bao cac bat bien:
  - Khong de file chay chinh tu ghi de chinh no
  - Cho tien trinh cha thoat sach se truoc khi thao tac
  - Giao dich thay the kem backup day du
  - Tuyet doi bao ve thu muc Data va du lieu nguoi dung (models, projects, config, db, presets)
  - Hoan tac (rollback) neu sao chep, kiem tra hoac khoi dong lai that bai
  - Giu lai ban backup cho toi khi cap nhat thanh cong
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Updater] %(levelname)s: %(message)s",
)
logger = logging.getLogger("autosub_updater")

PROTECTED_NAMES = {"Data", "data", "workspace"}


def wait_for_pid(pid: int, timeout: float = 30.0) -> bool:
    """Cho tien trinh co PID chi dinh ket thuc sach se."""
    if pid <= 0:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            # Tien trinh da thoat hoac khong ton tai
            return True
        time.sleep(0.2)
    return False


def resolve_payload_dir(staging_dir: Path) -> Path:
    """Xac dinh thu muc chua noi dung cap nhat thuc te ben trong staging."""
    entries = [
        p
        for p in staging_dir.iterdir()
        if p.name not in ("__pycache__", ".part") and not p.name.endswith(".zip")
    ]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        if any(
            (inner / name).exists()
            for name in ("AutoSubStudio.exe", "run_app.py", "_internal", "src")
        ):
            return inner
    return staging_dir


def verify_payload_layout(payload_dir: Path) -> bool:
    """Kiem tra xem thu muc cap nhat co chua cac thanh phan can thiet."""
    if not payload_dir.is_dir():
        return False
    has_exe = (payload_dir / "AutoSubStudio.exe").is_file()
    has_run = (payload_dir / "run_app.py").is_file()
    has_src = (payload_dir / "src" / "autosub_studio" / "__init__.py").is_file()
    return has_exe or has_run or has_src


def move_atomic_same_volume(src: Path, dst: Path) -> None:
    """Di chuyen tep hoac thu muc tren cung o dia mot cach nguyen tu."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_file():
            os.replace(src, dst)
            return
        if dst.is_dir():
            # Tren Windows, doi ten thu muc vao dich da ton tai se bi loi WinError 5
            # Can xoa thu muc dich cu (sau khi da duoc backup sang noi khac)
            shutil.rmtree(dst)
            os.replace(src, dst)
            return
    os.replace(src, dst)


def rollback(
    app_root: Path,
    moved_to_app: list[Path],
    moved_to_backup: list[tuple[Path, Path]],
) -> None:
    """Khoi phuc lai trang thai cu cua app_root neu gap loi."""
    logger.warning("Bat dau rollback phuc hoi phien ban cu...")
    # 1. Xoa cac tep/thu muc moi da chep vao app_root
    for item in moved_to_app:
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
        except Exception as e:
            logger.error("Loi khi xoa tep moi trong rollback (%s): %s", item, e)

    # 2. Dua cac tep/thu muc tu backup tro lai app_root
    for orig_path, backup_path in reversed(moved_to_backup):
        try:
            if backup_path.exists():
                move_atomic_same_volume(backup_path, orig_path)
        except Exception as e:
            logger.error(
                "Loi khi khoi phuc tep tu backup (%s -> %s): %s",
                backup_path,
                orig_path,
                e,
            )
    logger.info("Rollback hoan tat.")


def apply_update(
    app_root: Path,
    staging_dir: Path,
    backup_dir: Path,
    pid: int | None = None,
    restart: bool = False,
    restart_cmd: list[str] | None = None,
    timeout: float = 30.0,
) -> bool:
    """Thuc hien thay the ban cap nhat voi co che transactional va backup an toan."""
    app_root = app_root.resolve()
    staging_dir = staging_dir.resolve()
    backup_dir = backup_dir.resolve()

    logger.info("Bat dau quy trinh cap nhat cho: %s", app_root)
    logger.info("Staging: %s | Backup: %s", staging_dir, backup_dir)

    # 1. Cho tien trinh cha tat han neu co PID
    if pid is not None and pid > 0:
        logger.info("Dang cho tien trinh %d thoat...", pid)
        if not wait_for_pid(pid, timeout=timeout):
            logger.error("Tien trinh %d khong tat sau %.1f giay. Huy cap nhat.", pid, timeout)
            return False

    # 2. Kiem tra staging va layout
    if not staging_dir.is_dir():
        logger.error("Thu muc staging khong ton tai: %s", staging_dir)
        return False

    payload_dir = resolve_payload_dir(staging_dir)
    if not verify_payload_layout(payload_dir):
        logger.error("Cau truc goi cap nhat trong staging khong hop le.")
        return False

    # 3. Chuan bi backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)

    moved_to_backup: list[tuple[Path, Path]] = []
    moved_to_app: list[Path] = []

    try:
        # 4. Sao luu cac tep hien tai se bi ghi de
        payload_items = [
            p
            for p in payload_dir.iterdir()
            if p.name not in ("__pycache__", ".part")
            and not p.name.endswith(".zip")
            and p.name not in PROTECTED_NAMES
        ]

        for item in payload_items:
            dest = app_root / item.name
            # Tuyet doi khong ghi de hoac di chuyen Data
            if item.name in PROTECTED_NAMES:
                continue

            if dest.exists():
                backup_item = backup_dir / item.name
                if backup_item.exists():
                    if backup_item.is_dir():
                        shutil.rmtree(backup_item, ignore_errors=True)
                    else:
                        backup_item.unlink(missing_ok=True)
                move_atomic_same_volume(dest, backup_item)
                moved_to_backup.append((dest, backup_item))

        # 5. Di chuyen cac tep moi tu payload_dir vao app_root
        for item in payload_items:
            if item.name in PROTECTED_NAMES:
                continue
            dest = app_root / item.name
            move_atomic_same_volume(item, dest)
            moved_to_app.append(dest)

        # 6. Kiem tra tinh toan ven sau khi thay the
        if not verify_payload_layout(app_root):
            raise RuntimeError("Kiem tra app_root sau khi thay the that bai!")

        logger.info("Thay the tep phien ban moi thanh cong!")

        # 7. Khoi dong lai neu co yeu cau
        if restart:
            logger.info("Khoi dong lai ung dung...")
            cmd: list[str] = []
            if restart_cmd:
                cmd = restart_cmd
            elif (app_root / "AutoSubStudio.exe").is_file():
                cmd = [str(app_root / "AutoSubStudio.exe")]
            elif (app_root / "run_app.py").is_file():
                cmd = [sys.executable, str(app_root / "run_app.py")]
            else:
                raise RuntimeError("Khong xac dinh duoc lenh khoi dong lai.")

            flags = 0
            if os.name == "nt":
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

            try:
                subprocess.Popen(cmd, cwd=str(app_root), creationflags=flags, close_fds=True)
                logger.info("Da khoi chay ung dung thanh cong.")
            except Exception as e:
                logger.error("Khoi dong lai ung dung that bai: %s", e)
                raise

        # Giu lai backup an toan de phong truong hop nguoi dung can quay lai
        logger.info("Ban sao luu duoc luu tai: %s", backup_dir)
        return True

    except Exception as exc:
        logger.error("Xay ra loi trong qua trinh cap nhat: %s. Dang hoan tac...", exc)
        rollback(app_root, moved_to_app, moved_to_backup)
        return False


def main(argv: list[str] | None = None) -> int:
    """Diem khoi dau dong lenh cua updater helper."""
    parser = argparse.ArgumentParser(description="AutoSub Studio Standalone Updater Helper")
    parser.add_argument(
        "--app-root", required=True, help="Duong dan thu muc goc ung dung can cap nhat"
    )
    parser.add_argument(
        "--staging-dir", required=True, help="Duong dan thu muc chua ban cap nhat moi"
    )
    parser.add_argument(
        "--backup-dir", required=True, help="Duong dan thu muc sao luu ban cu"
    )
    parser.add_argument("--pid", type=int, default=None, help="PID cua ung dung can cho tat")
    parser.add_argument(
        "--restart", action="store_true", help="Tu dong khoi dong lai ung dung sau khi cap nhat"
    )
    parser.add_argument("--restart-cmd", default=None, help="Lenh tuy chinh de khoi dong lai")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Thoi gian cho PID thoat (giay)"
    )

    args = parser.parse_args(argv)

    restart_cmd_list = None
    if args.restart_cmd:
        import shlex

        restart_cmd_list = shlex.split(args.restart_cmd)

    success = apply_update(
        app_root=Path(args.app_root),
        staging_dir=Path(args.staging_dir),
        backup_dir=Path(args.backup_dir),
        pid=args.pid,
        restart=args.restart,
        restart_cmd=restart_cmd_list,
        timeout=args.timeout,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
