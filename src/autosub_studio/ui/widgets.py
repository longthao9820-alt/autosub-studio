"""Cac thanh phan giao dien dung rieng cho AutoSub Studio."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from .style import BG_DEEP, BLUE, BORDER, GREEN, MUTED, TEXT, YELLOW

# Ma cac nut chuc nang tren tung dong cua bang du an.
ACTION_STOP = "stop"
ACTION_SRT = "srt"
ACTION_FOLDER = "folder"
ACTION_RUN = "run"
ROW_ACTIONS = (ACTION_STOP, ACTION_SRT, ACTION_FOLDER, ACTION_RUN)
ACTION_TIPS = {
    ACTION_STOP: "Dung tac vu cua du an nay",
    ACTION_SRT: "Chon tep SRT va nap vao du an nay",
    ACTION_FOLDER: "Mo thu muc du an",
    ACTION_RUN: "Chay kich ban cho du an nay",
}


class VerticalTabButton(QPushButton):
    """Nut dang vien thuoc, chu quay doc - dung lam tab ben canh."""

    def __init__(self, text: str, color: str = GREEN, parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._color = QColor(color)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setStyleSheet("background: transparent; border: none;")

    def sizeHint(self) -> QSize:  # noqa: N802
        metrics = self.fontMetrics()
        return QSize(26, metrics.horizontalAdvance(self._text) + 26)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)

        base = QColor(GREEN if self.isChecked() else BLUE)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(base))
        painter.drawRoundedRect(rect, rect.width() / 2, rect.width() / 2)

        if self.isChecked():
            painter.setPen(QPen(QColor(YELLOW), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, rect.width() / 2, rect.width() / 2)

        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(8.5)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(
            QRectF(-self.height() / 2, -self.width() / 2, self.height(), self.width()),
            int(Qt.AlignmentFlag.AlignCenter),
            self._text,
        )
        painter.restore()
        painter.end()


class VerticalTabStrip(QWidget):
    """Cot tab doc ben canh mot vung noi dung."""

    currentChanged = Signal(int)

    def __init__(self, labels: list[str], colors: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.buttons: list[VerticalTabButton] = []
        palette = colors or [GREEN, BLUE, GREEN, BLUE]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(6)
        for index, label in enumerate(labels):
            button = VerticalTabButton(label, palette[index % len(palette)])
            button.clicked.connect(lambda _c=False, i=index: self.set_current(i))
            layout.addWidget(button)
            self.buttons.append(button)
        layout.addStretch(1)
        self.setFixedWidth(30)
        if self.buttons:
            self.buttons[0].setChecked(True)

    def set_current(self, index: int) -> None:
        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)
        self.currentChanged.emit(index)

    def current(self) -> int:
        for i, button in enumerate(self.buttons):
            if button.isChecked():
                return i
        return 0


class PillTabBar(QWidget):
    """Hang tab dang vien thuoc, can giua - dung cho hang B1..B4."""

    currentChanged = Signal(int)

    def __init__(self, labels: list[str], parent=None) -> None:
        super().__init__(parent)
        self.buttons: list[QPushButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)
        layout.addStretch(1)
        for index, label in enumerate(labels):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setObjectName("Blue")
            button.clicked.connect(lambda _c=False, i=index: self.set_current(i))
            layout.addWidget(button)
            self.buttons.append(button)
        layout.addStretch(1)
        self._current = -1
        if self.buttons:
            self.set_current(0)

    def set_current(self, index: int) -> None:
        if not 0 <= index < len(self.buttons):
            return
        self._current = index
        for i, button in enumerate(self.buttons):
            active = i == index
            button.setChecked(active)
            button.setStyleSheet(
                f"background: {GREEN}; color: #ffffff; border-bottom: 3px solid "
                f"{YELLOW}; border-radius: 7px 7px 0 0; padding: 5px 10px; "
                "font-weight: 700;"
                if active
                else f"background: {BLUE}; color: #ffffff; border: none; "
                "border-radius: 7px 7px 0 0; padding: 5px 10px; font-weight: 700;"
            )
        self.currentChanged.emit(index)

    def current(self) -> int:
        return self._current


class RowActionDelegate(QStyledItemDelegate):
    """Ve cac nut chuc nang nho tren tung dong cua bang du an."""

    actionClicked = Signal(int, str)  # dong nguon, ma hanh dong

    ICON = 20
    GAP = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hover: tuple[int, int] = (-1, -1)

    # ------------------------------------------------------------------ ve

    def _icon_rects(self, rect: QRect) -> list[QRect]:
        total = len(ROW_ACTIONS) * self.ICON + (len(ROW_ACTIONS) - 1) * self.GAP
        x = rect.x() + max(2, (rect.width() - total) // 2)
        y = rect.y() + (rect.height() - self.ICON) // 2
        out = []
        for i in range(len(ROW_ACTIONS)):
            out.append(QRect(x + i * (self.ICON + self.GAP), y, self.ICON, self.ICON))
        return out

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rects = self._icon_rects(option.rect)
        for i, (name, box) in enumerate(zip(ROW_ACTIONS, rects, strict=False)):
            hovered = self._hover == (index.row(), i)
            self._draw_icon(painter, box, name, hovered)
        painter.restore()

    def _draw_icon(self, painter: QPainter, box: QRect, name: str, hovered: bool) -> None:
        bg = QColor("#3a3a3a") if not hovered else QColor("#4d4d4d")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(QRectF(box), 4, 4)

        inner = QRectF(box).adjusted(4.5, 4.5, -4.5, -4.5)
        if name == ACTION_STOP:
            painter.setBrush(QBrush(QColor("#e05252")))
            painter.drawRoundedRect(inner, 1.5, 1.5)
        elif name == ACTION_SRT:
            font = QFont(painter.font())
            font.setBold(True)
            font.setPointSizeF(6.0)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#7fc3ff")))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), "SRT")
        elif name == ACTION_FOLDER:
            painter.setBrush(QBrush(QColor("#e8b04b")))
            body = QRectF(inner.left(), inner.top() + 2.5, inner.width(), inner.height() - 2.5)
            painter.drawRoundedRect(body, 1.5, 1.5)
            tab = QRectF(inner.left(), inner.top(), inner.width() * 0.45, 3.0)
            painter.drawRoundedRect(tab, 1.0, 1.0)
        elif name == ACTION_RUN:
            painter.setBrush(QBrush(QColor("#3ecf8e")))
            triangle = QPolygonF(
                [
                    QPointF(inner.left() + 1, inner.top()),
                    QPointF(inner.right(), inner.center().y()),
                    QPointF(inner.left() + 1, inner.bottom()),
                ]
            )
            painter.drawPolygon(triangle)

    def sizeHint(self, option, index) -> QSize:  # noqa: N802, ARG002
        width = len(ROW_ACTIONS) * self.ICON + (len(ROW_ACTIONS) - 1) * self.GAP + 8
        return QSize(width, self.ICON + 6)

    # ------------------------------------------------------------------ chuot

    def editorEvent(self, event, model, option, index) -> bool:  # noqa: N802, ARG002
        kind = event.type()
        if kind == event.Type.MouseMove:
            hit = self._hit(event.position().toPoint(), option.rect)
            new = (index.row(), hit) if hit >= 0 else (-1, -1)
            if new != self._hover:
                self._hover = new
                view = self.parent()
                if isinstance(view, QAbstractItemView):
                    view.viewport().update()
            return False
        if kind == event.Type.MouseButtonRelease:
            hit = self._hit(event.position().toPoint(), option.rect)
            if hit >= 0:
                self.actionClicked.emit(index.row(), ROW_ACTIONS[hit])
                return True
        return False

    def _hit(self, point, rect: QRect) -> int:
        for i, box in enumerate(self._icon_rects(rect)):
            if box.contains(point):
                return i
        return -1


class StatusStrip(QWidget):
    """Hang chu trang thai nho, dung duoi cac bang dieu khien."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.label = QLabel("")
        self.label.setObjectName("Muted")
        self.label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label, 1)

    def set_text(self, text: str, tone: str = "muted") -> None:
        colors = {"muted": MUTED, "ok": "#3ecf8e", "bad": "#e86464", "warn": YELLOW}
        self.label.setStyleSheet(f"color: {colors.get(tone, MUTED)};")
        self.label.setText(text)


class PlayerButton(QPushButton):
    """Nut dieu khien trinh phat, ve bang hinh khoi don gian."""

    def __init__(self, kind: str, color: str = GREEN, parent=None) -> None:
        super().__init__(parent)
        self._kind = kind
        self._color = QColor(color)
        self.setFixedSize(30, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background: transparent; border: none;")

    def set_kind(self, kind: str) -> None:
        self._kind = kind
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(self._color)
        if self.underMouse():
            color = color.lighter(120)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        if self._kind == "back":
            for offset in (-5, 2):
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(cx + offset + 5, cy - 6),
                            QPointF(cx + offset + 5, cy + 6),
                            QPointF(cx + offset - 2, cy),
                        ]
                    )
                )
        elif self._kind == "forward":
            for offset in (-5, 2):
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(cx + offset - 2, cy - 6),
                            QPointF(cx + offset - 2, cy + 6),
                            QPointF(cx + offset + 5, cy),
                        ]
                    )
                )
        elif self._kind == "pause":
            painter.drawRoundedRect(QRectF(cx - 7, cy - 7, 5, 14), 1.5, 1.5)
            painter.setBrush(QBrush(QColor(YELLOW)))
            painter.drawRoundedRect(QRectF(cx + 2, cy - 7, 5, 14), 1.5, 1.5)
        elif self._kind == "play":
            painter.setBrush(QBrush(QColor("#e0433a")))
            painter.drawEllipse(QPointF(cx, cy), 10, 10)
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(cx - 3, cy - 5),
                        QPointF(cx + 6, cy),
                        QPointF(cx - 3, cy + 5),
                    ]
                )
            )
        elif self._kind == "expand":
            painter.setPen(QPen(QColor(TEXT), 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                x = cx + dx * 7
                y = cy + dy * 6
                painter.drawLine(QPointF(x, y), QPointF(x - dx * 4, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y - dy * 4))
        elif self._kind == "snapshot":
            painter.setPen(QPen(QColor(TEXT), 1.4))
            painter.setBrush(QBrush(QColor(BG_DEEP)))
            painter.drawRoundedRect(QRectF(cx - 9, cy - 6, 18, 13), 2, 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy + 0.5), 3.5, 3.5)
        painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().leaveEvent(event)


def section_label(text: str) -> QLabel:
    """Nhan tieu de nho mau xam cho tung khoi."""
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT}; font-weight: 700;")
    return label


def divider() -> QWidget:
    """Duong ke ngang mo."""
    line = QWidget()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {BORDER};")
    return line
