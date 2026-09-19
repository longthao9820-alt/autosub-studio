"""Cau hinh chung cho bo kiem thu."""

from __future__ import annotations

import os

import pytest

# Chay Qt o che do khong man hinh de kiem thu duoc tren may khong co giao dien.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Mot QApplication dung chung cho ca phien kiem thu."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
