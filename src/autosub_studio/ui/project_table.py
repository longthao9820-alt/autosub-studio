"""Bang danh sach du an."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QStyledItemDelegate, QTableView

from .style import get_ui_font_family

HEADERS = (
    "",
    "Chức năng",
    "ID",
    "Name Project",
    "Phụ Đề",
    "Dịch Sub",
    "Lồng Tiếng",
    "Render",
    "Ngôn Ngữ",
    "SL Ký Tự",
    "Nhiệm Vụ",
    "Xuất Dự Án",
    "Tiến trình",
    "Status",
)
COL_STT = 0
COL_ACTIONS = 1
COL_ID = 2
COL_NAME = 3
COL_PROGRESS = 12
COL_STATUS = 13
# Cac cot hien trang thai bang chu xanh / do.
_STATE_COLUMNS = {
    4: "has_subtitle",
    5: "has_translation",
    6: "has_dub",
    7: "has_render",
    11: "exported",
}

# Mau chu lay theo bang project trong giao dien NTS mau.
GREEN = QColor("#06ff6f")
RED = QColor("#f81919")
AMBER = QColor("#ffe400")
CYAN = QColor("#67ddfd")


@dataclass
class ProjectRow:
    """Mot dong trong bang du an (ban sao doc duoc, khong giu phien CSDL)."""

    id: int = 0
    name: str = ""
    folder: str = ""
    video_path: str = ""
    language: str = ""
    target_language: str = ""
    cue_count: int = 0
    char_count: int = 0
    has_subtitle: bool = False
    has_translation: bool = False
    has_dub: bool = False
    has_render: bool = False
    exported: bool = False
    current_task: str = ""
    progress: int = 0
    status: str = ""
    duration: float = 0.0
    extra: dict = field(default_factory=dict)


class ProjectTableModel(QAbstractTableModel):
    """Mo hinh bang du an."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[ProjectRow] = []

    def set_rows(self, rows: list[ProjectRow]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rows(self) -> list[ProjectRow]:
        return self._rows

    def row_at(self, index: int) -> ProjectRow | None:
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def update_row(self, project_id: int, **values) -> None:
        for i, row in enumerate(self._rows):
            if row.id == project_id:
                for key, value in values.items():
                    if hasattr(row, key):
                        setattr(row, key, value)
                self.dataChanged.emit(self.index(i, 0), self.index(i, len(HEADERS) - 1))
                return

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return [
                index.row() + 1,
                "",  # cot nut chuc nang, do delegate ve
                row.id,
                row.name,
                "Da Co" if row.has_subtitle else "Chua Co",
                "Da Dich" if row.has_translation else "Chua Dich",
                "Da Long Tieng" if row.has_dub else "Chua Long Tieng",
                "Da Render" if row.has_render else "Chua Render",
                row.language or row.target_language or "-",
                f"{row.char_count:,}".replace(",", "."),
                row.current_task or "",
                "Da Xuat" if row.exported else "Chua Xuat",
                row.progress,
                row.status or "-",
            ][col]
        if role == Qt.ItemDataRole.TextAlignmentRole and col not in (COL_NAME, 10, 13):
            return int(Qt.AlignmentFlag.AlignCenter)
        status_text = (row.status or "").lower()
        if role == Qt.ItemDataRole.ForegroundRole:
            field = _STATE_COLUMNS.get(col)
            if field is not None:
                return QBrush(GREEN if getattr(row, field, False) else RED)
            if col == COL_PROGRESS:
                return QBrush(AMBER)
            if col == COL_STATUS:
                text = status_text
                if "loi" in text:
                    return QBrush(RED)
                if "xong" in text or "hoan tat" in text:
                    return QBrush(GREEN)
                return QBrush(CYAN)
        if role == Qt.ItemDataRole.FontRole:
            font = QFont(get_ui_font_family())
            font.setBold(col not in (COL_STT,))
            return font
        return None


class ProgressDelegate(QStyledItemDelegate):
    """Ve thanh tien trinh trong cot 'Tien trinh'."""

    def paint(self, painter, option, index) -> None:
        if index.column() != COL_PROGRESS:
            super().paint(painter, option, index)
            return
        value = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            percent = max(0, min(100, int(value)))
        except (TypeError, ValueError):
            percent = 0
        rect = option.rect.adjusted(6, 6, -6, -6)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#232b34"))
        painter.drawRoundedRect(rect, 4, 4)
        if percent > 0:
            filled = rect.adjusted(0, 0, -int(rect.width() * (100 - percent) / 100), 0)
            painter.setBrush(GREEN if percent >= 100 else QColor("#2f9e6b"))
            painter.drawRoundedRect(filled, 4, 4)
        foreground = index.data(Qt.ItemDataRole.ForegroundRole)
        painter.setPen(foreground.color() if isinstance(foreground, QBrush) else AMBER)
        painter.drawText(option.rect, int(Qt.AlignmentFlag.AlignCenter), f"{percent}%")
        painter.restore()


class ProjectTableView(QTableView):
    """Bang du an voi loc va sap xep."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ProjectTable")
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Thu tu tren bang la thu tu tao project (moi o tren, cu o duoi).
        # Khong cho sap xep dong theo Status/Tien trinh vi khi OCR cap nhat,
        # project se nhay len xuong lam nguoi dung mat vi tri dang theo doi.
        self.setSortingEnabled(False)
        self.verticalHeader().setVisible(False)
        self.setItemDelegate(ProgressDelegate(self))
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setSortIndicatorShown(False)
        self.horizontalHeader().setStretchLastSection(True)

    def apply_column_widths(self) -> None:
        # Mau giao dien danh sach du an khong co cot Long Tieng rieng; trang
        # thai nay van duoc luu trong model va dung boi pipeline.
        self.setColumnHidden(6, True)
        for col, width in {
            0: 42,
            1: 116,
            2: 52,
            3: 230,
            4: 74,
            5: 84,
            6: 104,
            7: 92,
            8: 84,
            9: 78,
            10: 118,
            11: 180,
            12: 180,
        }.items():
            self.setColumnWidth(col, width)
        self.verticalHeader().setDefaultSectionSize(39)


class ProjectFilterProxy(QSortFilterProxyModel):
    """Loc du an theo tu khoa va trang thai."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.keyword = ""
        self.only_pending = False

    def set_keyword(self, text: str) -> None:
        self.keyword = (text or "").strip().lower()
        self.invalidateFilter()

    def set_only_pending(self, value: bool) -> None:
        self.only_pending = bool(value)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, ProjectTableModel):
            return True
        row = model.row_at(source_row)
        if row is None:
            return False
        if self.only_pending and row.has_render:
            return False
        if not self.keyword:
            return True
        haystack = f"{row.id} {row.name} {row.status} {row.current_task}".lower()
        return self.keyword in haystack
