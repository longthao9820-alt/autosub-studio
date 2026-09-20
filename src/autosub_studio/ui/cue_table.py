"""Bang phu de: mo hinh du lieu, lich su chinh sua va khung hien thi."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from ..core.models import Cue, SubtitleDoc
from ..core.timecode import TimecodeError, format_display, parse_timecode
from .style import get_ui_font_family

COL_INDEX = 0
COL_SPEAKER = 1
COL_START = 2
COL_END = 3
COL_RATIO = 4
COL_TEXT = 5
COL_TRANS = 6
HEADERS = ("", "Vocal", "Time", "", "Ratio", "Original", "Translation")

ROW_NUMBER_TEXT = QColor("#e8e8e8")
VOCAL_TEXT = QColor("#dfff35")
TIME_TEXT = QColor("#ffeb00")
ORIGINAL_TEXT = QColor("#f5f0a0")
TRANSLATION_TEXT = QColor("#f5f0a0")
CURRENT_ROW_BG = QColor("#075742")

MAX_HISTORY = 80
_TIME_RANGE_RE = re.compile(r"\s*(?:-->|→|->)\s*")


@dataclass
class HistoryEntry:
    """Mot buoc trong lich su chinh sua."""

    label: str
    doc: SubtitleDoc


class CueTableModel(QAbstractTableModel):
    """Mo hinh bang cho danh sach cau phu de."""

    documentChanged = Signal()
    historyChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc = SubtitleDoc()
        self._undo: list[HistoryEntry] = []
        self._redo: list[HistoryEntry] = []
        self._current = -1

    # ------------------------------------------------------------------ du lieu

    @property
    def doc(self) -> SubtitleDoc:
        return self._doc

    def set_document(self, doc: SubtitleDoc, *, reset_history: bool = True) -> None:
        self.beginResetModel()
        self._doc = doc
        if reset_history:
            self._undo.clear()
            self._redo.clear()
        self.endResetModel()
        self.documentChanged.emit()
        self.historyChanged.emit()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._doc.cues)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return HEADERS[section]
        return None

    def flags(self, index) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() in (COL_SPEAKER, COL_START, COL_END, COL_TEXT, COL_TRANS):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if not 0 <= row < len(self._doc.cues):
            return None
        cue = self._doc.cues[row]

        if role == Qt.ItemDataRole.DisplayRole:
            if col == COL_INDEX:
                return f"No. {row + 1}"
            if col == COL_SPEAKER:
                return cue.speaker or "1"
            if col == COL_START:
                return f"{format_display(cue.start)} --> {format_display(cue.end)}"
            if col == COL_END:
                return format_display(cue.end)
            if col == COL_RATIO:
                return f"{cue.length_ratio:.2f}" if cue.translation.strip() else "-"
            if col == COL_TEXT:
                return cue.text.replace("\n", " ")
            if col == COL_TRANS:
                return cue.translation.replace("\n", " ")
        if role == Qt.ItemDataRole.EditRole:
            if col == COL_INDEX:
                return row + 1
            if col == COL_SPEAKER:
                return cue.speaker
            if col == COL_START:
                return f"{format_display(cue.start)} --> {format_display(cue.end)}"
            if col == COL_END:
                return format_display(cue.end)
            if col == COL_RATIO:
                return f"{cue.length_ratio:.2f}" if cue.translation.strip() else "-"
            if col == COL_TEXT:
                return cue.text.replace("\n", " ")
            if col == COL_TRANS:
                return cue.translation.replace("\n", " ")
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (
            COL_INDEX,
            COL_START,
            COL_END,
            COL_RATIO,
        ):
            return int(Qt.AlignmentFlag.AlignCenter)
        if role == Qt.ItemDataRole.BackgroundRole and row == self._current:
            return QBrush(CURRENT_ROW_BG)
        if role == Qt.ItemDataRole.ForegroundRole:
            if col == COL_INDEX:
                return QBrush(ROW_NUMBER_TEXT)
            if col == COL_SPEAKER:
                return QBrush(VOCAL_TEXT)
            if col == COL_START:
                return QBrush(TIME_TEXT)
            if col == COL_TEXT:
                return QBrush(ORIGINAL_TEXT)
            if col == COL_TRANS:
                return QBrush(TRANSLATION_TEXT)
            if col == COL_RATIO and cue.translation.strip() and cue.length_ratio > 1.6:
                return QBrush(QColor("#d99a2b"))
            if col in (COL_START, COL_END) and cue.end <= cue.start:
                return QBrush(QColor("#d05353"))
        if role == Qt.ItemDataRole.FontRole:
            font = QFont(get_ui_font_family())
            font.setBold(col in (COL_SPEAKER, COL_START, COL_TEXT, COL_TRANS))
            return font
        if role == Qt.ItemDataRole.ToolTipRole and col == COL_START:
            return f"{format_display(cue.start)} --> {format_display(cue.end)}"
        if role == Qt.ItemDataRole.ToolTipRole and col in (COL_TEXT, COL_TRANS):
            return cue.text if col == COL_TEXT else cue.translation
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        row, col = index.row(), index.column()
        if not 0 <= row < len(self._doc.cues):
            return False
        cue = self._doc.cues[row]
        text = str(value)
        if col == COL_START:
            parts = _TIME_RANGE_RE.split(text.strip(), maxsplit=1)
            try:
                start = parse_timecode(parts[0])
                end = parse_timecode(parts[1]) if len(parts) == 2 else cue.end
            except TimecodeError:
                return False
            if end <= start:
                return False
            self.push_history(f"Sua thoi gian cau {row + 1}")
            cue.start = start
            cue.end = end
        elif col == COL_END:
            try:
                seconds = parse_timecode(text)
            except TimecodeError:
                return False
            self.push_history(f"Sua thoi gian cau {row + 1}")
            cue.end = max(seconds, cue.start + 0.05)
        elif col == COL_SPEAKER:
            self.push_history(f"Sua nguoi noi cau {row + 1}")
            cue.speaker = text.strip()
        elif col == COL_TEXT:
            self.push_history(f"Sua noi dung cau {row + 1}")
            cue.text = text
        elif col == COL_TRANS:
            self.push_history(f"Sua ban dich cau {row + 1}")
            cue.translation = text
        else:
            return False
        self.dataChanged.emit(index.sibling(row, 0), index.sibling(row, COL_TRANS))
        self.documentChanged.emit()
        return True

    # ------------------------------------------------------------------ lich su

    def push_history(self, label: str) -> None:
        self._undo.append(HistoryEntry(label, self._doc.copy()))
        if len(self._undo) > MAX_HISTORY:
            del self._undo[0]
        self._redo.clear()
        self.historyChanged.emit()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str:
        if not self._undo:
            return ""
        entry = self._undo.pop()
        self._redo.append(HistoryEntry(entry.label, self._doc.copy()))
        self.beginResetModel()
        self._doc = entry.doc
        self.endResetModel()
        self.documentChanged.emit()
        self.historyChanged.emit()
        return entry.label

    def redo(self) -> str:
        if not self._redo:
            return ""
        entry = self._redo.pop()
        self._undo.append(HistoryEntry(entry.label, self._doc.copy()))
        self.beginResetModel()
        self._doc = entry.doc
        self.endResetModel()
        self.documentChanged.emit()
        self.historyChanged.emit()
        return entry.label

    def history_labels(self) -> list[str]:
        return [e.label for e in reversed(self._undo)]

    # ------------------------------------------------------------------ thao tac

    def refresh_row(self, row: int) -> None:
        if 0 <= row < len(self._doc.cues):
            self.dataChanged.emit(self.index(row, 0), self.index(row, COL_TRANS))

    def refresh_all(self) -> None:
        if self._doc.cues:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._doc.cues) - 1, COL_TRANS))

    def set_current(self, row: int) -> None:
        if row == self._current:
            return
        old = self._current
        self._current = row
        for r in (old, row):
            if 0 <= r < len(self._doc.cues):
                self.refresh_row(r)

    def insert_cue(self, at_seconds: float, duration: float = 2.0) -> int:
        self.push_history("Them cau moi")
        cue = Cue(start=at_seconds, end=at_seconds + duration, text="")
        self.beginResetModel()
        self._doc.cues.append(cue)
        self._doc.sort()
        self.endResetModel()
        self.documentChanged.emit()
        return self._doc.cues.index(cue)

    def remove_rows(self, rows: list[int]) -> int:
        rows = sorted({r for r in rows if 0 <= r < len(self._doc.cues)}, reverse=True)
        if not rows:
            return 0
        self.push_history(f"Xoa {len(rows)} cau")
        self.beginResetModel()
        for r in rows:
            del self._doc.cues[r]
        self.endResetModel()
        self.documentChanged.emit()
        return len(rows)

    def apply_change(self, label: str, func) -> None:
        """Chay mot thao tac tren tai lieu va ghi vao lich su."""
        self.push_history(label)
        self.beginResetModel()
        func(self._doc)
        self.endResetModel()
        self.documentChanged.emit()


class CueTableView(QTableView):
    """Bang hien thi cau phu de."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CueTable")
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.verticalHeader().setVisible(False)
        self.setWordWrap(False)
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    def apply_column_widths(self) -> None:
        widths = {
            COL_INDEX: 84,
            COL_SPEAKER: 100,
            COL_START: 204,
            COL_RATIO: 100,
            COL_TEXT: 310,
        }
        self.setColumnHidden(COL_INDEX, False)
        self.setColumnHidden(COL_END, True)
        for col, width in widths.items():
            self.setColumnWidth(col, width)
        self.verticalHeader().setDefaultSectionSize(39)

    def selected_rows(self) -> list[int]:
        return (
            sorted({i.row() for i in self.selectionModel().selectedRows()})
            if self.selectionModel()
            else []
        )
