"""Cac hop thoai phu tro."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..core.editing import Issue


class IssueDialog(QDialog):
    """Danh sach loi thoi gian, bam doi de nhay toi cau tuong ung."""

    jumpRequested = Signal(int)

    def __init__(self, issues: list[Issue], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ket qua kiem tra lech thoi gian")
        self.resize(620, 420)
        self.list = QListWidget()
        errors = sum(1 for i in issues if i.severity == "error")
        summary = QLabel(
            f"Tim thay {len(issues)} van de ({errors} loi nang). "
            "Bam doi vao mot dong de nhay toi cau do."
            if issues
            else "Khong tim thay van de nao. Phu de dang on."
        )
        summary.setWordWrap(True)
        for issue in issues:
            item = QListWidgetItem(
                f"[{'LOI' if issue.severity == 'error' else 'Canh bao'}] "
                f"Cau {issue.index + 1}: {issue.message}"
            )
            item.setData(Qt.ItemDataRole.UserRole, issue.index)
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(self._jump)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(self.list, 1)
        layout.addWidget(buttons)

    def _jump(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(index, int):
            self.jumpRequested.emit(index)


class ShiftDialog(QDialog):
    """Nhap khoang thoi gian can doi."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Doi thoi gian phu de")
        self.edit = QLineEdit("0.5")
        self.edit.setPlaceholderText("Vi du: 0.5 hoac -1.25 (giay)")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Doi (giay):"))
        row.addWidget(self.edit, 1)
        layout.addLayout(row)
        layout.addWidget(QLabel("So duong = day phu de tre lai, so am = keo phu de som hon."))
        layout.addWidget(buttons)

    def value(self) -> float:
        try:
            return float(self.edit.text().replace(",", "."))
        except ValueError:
            return 0.0


class LogDialog(QDialog):
    """Cua so xem nhat ky chi tiet cua tac vu."""

    def __init__(self, title: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 480)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setPlainText(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(buttons)
