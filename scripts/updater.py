"""Script chay updater helper doc lap."""

from __future__ import annotations

import sys
from pathlib import Path

# Them src vao sys.path neu chua co
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from autosub_updater.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
