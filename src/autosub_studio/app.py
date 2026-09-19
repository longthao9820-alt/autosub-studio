"""Khoi dong ung dung va bat loi khong luong truoc de bao cho nguoi dung."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from .services.paths import app_root, config_dir


def _crash_log_path() -> Path:
    logs = config_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / "crash.log"


def _write_crash(text: str) -> Path:
    path = _crash_log_path()
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {stamp} =====\n{text}\n")
    except OSError:
        pass
    return path


def _install_excepthook() -> None:
    def hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        path = _write_crash(detail)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None,
                    "Ung dung gap loi",
                    "Da xay ra loi khong luong truoc. Cong viec dang lam co the "
                    "van con trong ban tu luu phuc hoi.\n\n"
                    f"Chi tiet ky thuat da ghi vao:\n{path}\n\n{exc_value}",
                )
        except Exception:  # khong duoc phep nem loi trong excepthook
            pass

    sys.excepthook = hook


def main() -> int:
    """Chay ung dung. Tra ve ma thoat."""
    _install_excepthook()
    root = app_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Phai dang ky truoc khi nap bat ky thu vien nao dung CUDA.
    from .services.gpu import register_cuda_dlls

    register_cuda_dlls()
    flags = {arg.lstrip("-/").lower() for arg in sys.argv[1:]}
    if flags & {"selftest", "kiemtra", "selftest-full", "kiemtrasau"}:
        from .selftest import main as selftest_main

        return selftest_main(deep=bool(flags & {"selftest-full", "kiemtrasau", "full"}))
    from .ui.main_window import run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
