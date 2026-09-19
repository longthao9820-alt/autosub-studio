"""Diem khoi dong dung cho ban dong goi va cho chay truc tiep tu ma nguon."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from autosub_studio.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
