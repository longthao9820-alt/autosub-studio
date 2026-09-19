"""Trinh phat video kem lop phu de va cong cu khoanh vung."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.timecode import format_display
from ..services.settings import SubtitleStyle
from .style import BORDER, GREEN, MUTED, TEXT, YELLOW
from .widgets import PlayerButton

MODE_NONE = "none"
MODE_OCR = "ocr"
MODE_BLUR = "blur"

MIN_REGION = 24.0  # kich thuoc nho nhat cua khung, tinh theo diem anh cua video

# Vi tri 8 nut keo quanh khung.
H_TOP_LEFT, H_TOP, H_TOP_RIGHT, H_RIGHT = 0, 1, 2, 3
H_BOTTOM_RIGHT, H_BOTTOM, H_BOTTOM_LEFT, H_LEFT = 4, 5, 6, 7

_CURSORS = {
    H_TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
    H_TOP: Qt.CursorShape.SizeVerCursor,
    H_TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
    H_RIGHT: Qt.CursorShape.SizeHorCursor,
    H_BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    H_BOTTOM: Qt.CursorShape.SizeVerCursor,
    H_BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
    H_LEFT: Qt.CursorShape.SizeHorCursor,
}


def clamp_region(region: list[int], width: int, height: int) -> list[int]:
    """Ep mot vung da luu nam gon trong khung hinh hien tai."""
    if len(region) != 4 or width <= 0 or height <= 0:
        return []
    x, y, w, h = (int(v) for v in region)
    w = max(int(MIN_REGION), min(w, width))
    h = max(int(MIN_REGION), min(h, height))
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    return [x, y, w, h]


def default_region(width: float, height: float, mode: str = MODE_OCR) -> QRectF:
    """Khung goi y ban dau: dai ngang o phan duoi khung hinh, noi hay co phu de."""
    w = max(MIN_REGION, width * 0.80)
    h = max(MIN_REGION, height * 0.20)
    x = (width - w) / 2
    y = height * 0.74
    if y + h > height:
        y = max(0.0, height - h)
    return QRectF(x, y, w, h)


class RegionItem(QGraphicsObject):
    """Khung chu nhat keo duoc: keo giua de di chuyen, keo 8 nut de doi kich thuoc."""

    regionChanged = Signal(QRectF)  # phat khi tha chuot
    regionEdited = Signal(QRectF)  # phat lien tuc trong luc keo

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rect = QRectF(0, 0, 0, 0)
        self._bounds = QRectF(0, 0, 1920, 1080)
        self._grab = -1  # -1 = khong keo, -2 = di chuyen ca khung
        self._grab_offset = QPointF()
        self._start_rect = QRectF()
        self._handle = 9.0  # ban kinh nut keo, quy doi theo muc phong to
        self.setAcceptHoverEvents(True)
        self.setZValue(20)
        self.setVisible(False)

    # ------------------------------------------------------------------ du lieu

    def region(self) -> QRectF:
        return QRectF(self._rect)

    def set_region(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = self._clamp(QRectF(rect).normalized())
        self.update()

    def set_bounds(self, rect: QRectF) -> None:
        self._bounds = QRectF(rect)
        if not self._rect.isEmpty():
            self.set_region(self._rect)

    def set_view_scale(self, scale: float) -> None:
        """Giu cho nut keo luon to bang nhau tren man hinh du video to hay nho.

        Kich thuoc nut duoc chan tren de khung khong tu phinh ra lam sai
        muc phong to cua khung nhin.
        """
        if scale > 0:
            self.prepareGeometryChange()
            self._handle = max(3.0, min(40.0, 9.0 / scale))
            self.update()

    # ------------------------------------------------------------------ ve

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds.united(self._rect).adjusted(
            -self._handle * 3, -self._handle * 3, self._handle * 3, self._handle * 3
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        if self._rect.isEmpty():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        b, r = self._bounds, self._rect
        # Chi lam mo phan ngoai khung trong luc dang keo, de binh thuong van
        # xem video ro rang.
        if self._grab != -1:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 110)))
            painter.drawRect(QRectF(b.left(), b.top(), b.width(), r.top() - b.top()))
            painter.drawRect(QRectF(b.left(), r.bottom(), b.width(), b.bottom() - r.bottom()))
            painter.drawRect(QRectF(b.left(), r.top(), r.left() - b.left(), r.height()))
            painter.drawRect(QRectF(r.right(), r.top(), b.right() - r.right(), r.height()))

        # Vien khung.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#000000"), self._handle * 0.45))
        painter.drawRect(r)
        painter.setPen(QPen(QColor("#2f9e6b"), self._handle * 0.28))
        painter.drawRect(r)

        # Tam ngam giua khung cho de gong.
        painter.setPen(QPen(QColor(47, 158, 107, 150), self._handle * 0.16, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y()))
        painter.drawLine(QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()))

        # Tam nut keo.
        painter.setPen(QPen(QColor("#0d1116"), self._handle * 0.22))
        painter.setBrush(QBrush(QColor("#7ee2b8")))
        for point in self._handle_points():
            painter.drawRect(self._handle_rect(point))

        # Nhan kich thuoc.
        label = f"{int(r.width())} x {int(r.height())}"
        font = painter.font()
        font.setPointSizeF(max(5.0, self._handle * 1.25))
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(label) + self._handle * 1.2
        text_h = metrics.height() + self._handle * 0.5
        box = QRectF(
            r.left(), max(self._bounds.top(), r.top() - text_h - self._handle), text_w, text_h
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(13, 17, 22, 210)))
        painter.drawRoundedRect(box, self._handle * 0.4, self._handle * 0.4)
        painter.setPen(QPen(QColor("#e6edf3")))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), label)

    # ------------------------------------------------------------------ chuot

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._rect.isEmpty():
            event.ignore()
            return
        pos = event.pos()
        handle = self._handle_at(pos)
        if handle >= 0:
            self._grab = handle
        elif self._rect.contains(pos):
            self._grab = -2
            self._grab_offset = pos - self._rect.topLeft()
        else:
            event.ignore()
            return
        self._start_rect = QRectF(self._rect)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._grab == -1:
            event.ignore()
            return
        pos = event.pos()
        if self._grab == -2:
            new = QRectF(pos - self._grab_offset, self._rect.size())
        else:
            new = self._resized(self._grab, pos)
        self.prepareGeometryChange()
        self._rect = self._clamp(new)
        self.update()
        self.regionEdited.emit(self.region())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._grab == -1:
            event.ignore()
            return
        self._grab = -1
        self.update()
        self.regionChanged.emit(self.region())
        event.accept()

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        handle = self._handle_at(event.pos())
        if handle >= 0:
            self.setCursor(_CURSORS[handle])
        elif self._rect.contains(event.pos()):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    # ------------------------------------------------------------------ noi bo

    def _handle_points(self) -> list[QPointF]:
        r = self._rect
        return [
            r.topLeft(),
            QPointF(r.center().x(), r.top()),
            r.topRight(),
            QPointF(r.right(), r.center().y()),
            r.bottomRight(),
            QPointF(r.center().x(), r.bottom()),
            r.bottomLeft(),
            QPointF(r.left(), r.center().y()),
        ]

    def _handle_rect(self, point: QPointF) -> QRectF:
        h = self._handle
        return QRectF(point.x() - h, point.y() - h, h * 2, h * 2)

    def _handle_at(self, pos: QPointF) -> int:
        for index, point in enumerate(self._handle_points()):
            if self._handle_rect(point).adjusted(-2, -2, 2, 2).contains(pos):
                return index
        return -1

    def _resized(self, handle: int, pos: QPointF) -> QRectF:
        r = QRectF(self._start_rect)
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        if handle in (H_TOP_LEFT, H_LEFT, H_BOTTOM_LEFT):
            left = min(pos.x(), right - MIN_REGION)
        if handle in (H_TOP_RIGHT, H_RIGHT, H_BOTTOM_RIGHT):
            right = max(pos.x(), left + MIN_REGION)
        if handle in (H_TOP_LEFT, H_TOP, H_TOP_RIGHT):
            top = min(pos.y(), bottom - MIN_REGION)
        if handle in (H_BOTTOM_LEFT, H_BOTTOM, H_BOTTOM_RIGHT):
            bottom = max(pos.y(), top + MIN_REGION)
        return QRectF(QPointF(left, top), QPointF(right, bottom))

    def _clamp(self, rect: QRectF) -> QRectF:
        """Giu khung nam gon trong khung hinh va khong nho hon muc toi thieu."""
        b = self._bounds
        w = min(max(rect.width(), MIN_REGION), b.width())
        h = min(max(rect.height(), MIN_REGION), b.height())
        x = min(max(rect.left(), b.left()), b.right() - w)
        y = min(max(rect.top(), b.top()), b.bottom() - h)
        return QRectF(x, y, w, h)


class SubtitleItem(QGraphicsSimpleTextItem):
    """Dong sub xem truoc co the keo len/xuong trong Screen Render."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._movable = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def set_movable(self, enabled: bool) -> None:
        self._movable = bool(enabled)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, self._movable)
        self.setCursor(
            Qt.CursorShape.SizeVerCursor if self._movable else Qt.CursorShape.ArrowCursor
        )

    @property
    def movable(self) -> bool:
        return self._movable

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self._movable
            and self.scene() is not None
            and isinstance(value, QPointF)
        ):
            bounds = self.scene().sceneRect()
            box = self.boundingRect()
            # Render ASS hien tai can giua theo chieu ngang. Cho keo doc de vi
            # tri xem truoc va video render luon trung nhau.
            x = max(bounds.left(), (bounds.width() - box.width()) / 2)
            y = min(max(value.y(), bounds.top()), max(bounds.top(), bounds.bottom() - box.height()))
            return QPointF(x, y)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        if not self._movable or self.scene() is None:
            return
        bounds = self.scene().sceneRect()
        margin = max(0, round(bounds.bottom() - self.y() - self.boundingRect().height()))
        signal = getattr(self.scene(), "subtitleMoved", None)
        if signal is not None:
            signal.emit(margin)


class VideoScene(QGraphicsScene):
    """Canh chua video, phu de va khung khoanh vung."""

    subtitleMoved = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor("#000000")))
        self.video_item = QGraphicsVideoItem()
        self.video_item.setSize(QSizeF(1920, 1080))
        self.addItem(self.video_item)

        self.subtitle_item = SubtitleItem()
        self.subtitle_item.setBrush(QBrush(QColor("#FFFFFF")))
        self.subtitle_item.setPen(QPen(QColor("#000000"), 3))
        self.subtitle_item.setZValue(10)
        self.addItem(self.subtitle_item)

        self.region_item = RegionItem()
        self.addItem(self.region_item)

        self.mode = MODE_NONE
        self._native = QSizeF(1920, 1080)
        # Dat san vung hien thi de khung canh khong tu tinh theo kich thuoc
        # cua cac doi tuong ben trong, tranh phong to sai luc chua co video.
        self.setSceneRect(QRectF(QPointF(0, 0), self._native))
        self.region_item.set_bounds(self.sceneRect())

    def set_native_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        self._native = QSizeF(width, height)
        self.video_item.setSize(self._native)
        rect = QRectF(QPointF(0, 0), self._native)
        self.setSceneRect(rect)
        self.region_item.set_bounds(rect)

    @property
    def native_size(self) -> QSizeF:
        return self._native


class VideoPlayer(QWidget):
    """Khung xem video kem thanh dieu khien."""

    positionChanged = Signal(float)
    durationChanged = Signal(float)
    regionSelected = Signal(str, int, int, int, int)  # che do, x, y, w, h
    regionPreview = Signal(str, int, int, int, int)  # cap nhat lien tuc khi keo
    subtitlePositionChanged = Signal(int)  # le duoi moi sau khi keo dong sub

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._seeking = False
        self._resume_after_seek = False
        self._pending_seek_value: int | None = None
        self._style = SubtitleStyle()
        self._need_first_frame = False
        self._restore_volume = 0.8

        self.scene = VideoScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setStyleSheet(f"border: 1px solid {BORDER}; border-radius: 0px;")
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setMinimumHeight(240)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.scene.video_item)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.errorOccurred.connect(self._on_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.scene.video_item.nativeSizeChanged.connect(self._on_native_size)
        self.scene.region_item.regionChanged.connect(self._on_region_done)
        self.scene.region_item.regionEdited.connect(self._on_region_edited)
        self.scene.subtitleMoved.connect(self.subtitlePositionChanged)

        self.btn_back = PlayerButton("back", GREEN)
        self.btn_back.setToolTip("Lui 5 giay")
        self.btn_prev = PlayerButton("back", "#7fc3ff")
        self.btn_prev.setToolTip("Cau truoc")
        self.btn_fwd = PlayerButton("forward", YELLOW)
        self.btn_fwd.setToolTip("Tien 5 giay")
        self.btn_next = PlayerButton("forward", "#7fc3ff")
        self.btn_next.setToolTip("Cau sau")
        self.btn_pause = PlayerButton("pause", "#4aa3ff")
        self.btn_pause.setToolTip("Tam dung")
        self.btn_play = PlayerButton("play")
        self.btn_play.setToolTip("Phat / Tam dung")
        self.btn_expand = PlayerButton("expand", TEXT)
        self.btn_expand.setToolTip("Xem vua khung")
        self.btn_snapshot = PlayerButton("snapshot", TEXT)
        self.btn_snapshot.setToolTip("Chup khung hinh dang xem")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self._seek_timer = QTimer(self)
        self._seek_timer.setInterval(40)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.timeout.connect(self._flush_live_seek)
        self.slider.sliderPressed.connect(self._begin_seek)
        self.slider.sliderMoved.connect(self._queue_live_seek)
        self.slider.sliderReleased.connect(self._seek_finished)

        self.lbl_time = QLabel("00:00:00,000 / 00:00:00,000")
        self.lbl_time.setObjectName("Muted")

        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(80)
        self.volume.valueChanged.connect(lambda v: self.audio.setVolume(v / 100))

        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_back.clicked.connect(lambda: self.step(-5.0))
        self.btn_fwd.clicked.connect(lambda: self.step(5.0))
        self.btn_expand.clicked.connect(self.fit_view)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 0)
        controls.setSpacing(2)
        for widget in (
            self.btn_back,
            self.btn_prev,
            self.btn_fwd,
            self.btn_next,
            self.btn_pause,
            self.btn_play,
        ):
            controls.addWidget(widget)
        controls.addSpacing(6)
        controls.addWidget(self.slider, 1)
        controls.addSpacing(6)
        controls.addWidget(self.lbl_time)
        controls.addWidget(self.volume)
        controls.addWidget(self.btn_expand)
        controls.addWidget(self.btn_snapshot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.view, 1)
        layout.addLayout(controls)

        self._placeholder = QLabel(
            "Chua chon video. Dung tab B1 de them video vao du an.", self.view
        )
        self._placeholder.setObjectName("Empty")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {MUTED}; background: transparent;")
        self._placeholder.setGeometry(0, 0, 400, 40)
        self._update_placeholder()

    # ------------------------------------------------------------------ dieu khien

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        self._need_first_frame = True
        self.player.setSource(p.absolute().as_uri())
        self._placeholder.setVisible(False)
        return True

    def _on_media_status(self, status) -> None:
        """Hien ngay khung hinh dau tien thay vi de man hinh den.

        Qt chi ve khung hinh sau khi bat dau phat, nen ta phat rat ngan roi
        dung lai, va tam tat tieng de khong bi bup mot tieng.
        """
        ready = status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        )
        if not (ready and self._need_first_frame):
            return
        self._need_first_frame = False
        self._restore_volume = self.audio.volume()
        self.audio.setVolume(0.0)
        self.player.play()
        QTimer.singleShot(140, self._stop_first_frame)

    def _stop_first_frame(self) -> None:
        self.player.pause()
        self.player.setPosition(0)
        self.audio.setVolume(self._restore_volume)
        self.fit_view()

    def set_frame_size(self, width: int, height: int) -> None:
        """Dat kich thuoc khung hinh ngay khi biet, khong cho video phat moi biet.

        Nho vay khung khoanh vung luon tinh theo dung so diem anh cua video.
        """
        if width <= 0 or height <= 0:
            return
        self.scene.set_native_size(width, height)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._sync_handle_scale()
        self._layout_subtitle()

    @property
    def frame_size(self) -> tuple[int, int]:
        size = self.scene.native_size
        return int(size.width()), int(size.height())

    def clear(self) -> None:
        self.player.stop()
        self.player.setSource("")
        self.scene.subtitle_item.setText("")
        self._placeholder.setVisible(True)

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def pause(self) -> None:
        self.player.pause()

    def play(self) -> None:
        self.player.play()

    def fit_view(self) -> None:
        """Chinh lai cho video vua khung nhin."""
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._sync_handle_scale()

    def step(self, seconds: float) -> None:
        self.seek(max(0.0, self.position + seconds))

    def seek(self, seconds: float) -> None:
        self.player.setPosition(int(max(0.0, seconds) * 1000))

    def show_frame(self, seconds: float) -> None:
        """Tam dung va hien dung khung hinh tai moc thoi gian duoc chon."""
        target = max(0.0, min(float(seconds), self._duration or float(seconds)))
        self.pause()
        self.seek(target)

    @property
    def position(self) -> float:
        return self.player.position() / 1000.0

    @property
    def duration(self) -> float:
        return self._duration

    # ------------------------------------------------------------------ phu de

    def set_subtitle_style(self, style: SubtitleStyle) -> None:
        self._style = style
        font = QFont(style.font or "Arial", max(8, int(style.font_size)))
        font.setBold(bool(style.bold))
        self.scene.subtitle_item.setFont(font)
        self.scene.subtitle_item.setBrush(QBrush(QColor(style.primary_color)))
        self.scene.subtitle_item.setPen(
            QPen(QColor(style.outline_color), max(0.5, float(style.outline)))
        )
        self._layout_subtitle()

    def show_subtitle(self, text: str) -> None:
        value = text or ""
        if self.scene.subtitle_item.text() == value:
            return
        self.scene.subtitle_item.setText(value)
        self._layout_subtitle()

    def set_subtitle_movable(self, enabled: bool) -> None:
        """Chi cho keo dong sub trong Screen Render."""
        self.scene.subtitle_item.set_movable(enabled)

    def _layout_subtitle(self) -> None:
        item = self.scene.subtitle_item
        rect = item.boundingRect()
        size = self.scene.native_size
        x = (size.width() - rect.width()) / 2
        y = size.height() - rect.height() - max(10, self._style.margin_v)
        item.setPos(max(0.0, x), max(0.0, y))

    # ------------------------------------------------------------------ khoanh vung

    def begin_region(self, mode: str, region: list[int] | None = None) -> tuple[int, ...]:
        """Hien khung keo duoc de nguoi dung dat vao dung cho co phu de.

        Neu du an da luu vung truoc do thi dung lai, khong thi dat mot khung
        goi y o phan duoi khung hinh. Tra ve toa do khung dang hien.
        """
        self.scene.mode = mode
        size = self.scene.native_size
        item = self.scene.region_item
        item.set_bounds(QRectF(0, 0, size.width(), size.height()))
        if region and len(region) == 4 and region[2] > 0 and region[3] > 0:
            item.set_region(QRectF(region[0], region[1], region[2], region[3]))
        else:
            item.set_region(default_region(size.width(), size.height(), mode))
        item.setVisible(True)
        item.setEnabled(True)
        self._sync_handle_scale()
        rect = item.region()
        return (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

    def end_region(self) -> tuple[int, ...]:
        """Khoa khung lai va tra ve toa do cuoi cung."""
        item = self.scene.region_item
        rect = item.region()
        self.scene.mode = MODE_NONE
        item.setVisible(False)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)
        return (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

    def hide_region(self) -> None:
        """An han khung khoanh vung, dung cho che do chi xem lai video."""
        self.scene.mode = MODE_NONE
        self.scene.region_item.setVisible(False)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)

    @property
    def region_mode(self) -> str:
        return self.scene.mode

    def show_region(self, region: list[int] | None) -> None:
        """Hien khung o che do chi xem (khong keo duoc)."""
        item = self.scene.region_item
        if region and len(region) == 4 and region[2] > 0 and region[3] > 0:
            item.set_region(QRectF(region[0], region[1], region[2], region[3]))
            item.setVisible(True)
            item.setEnabled(False)
        else:
            item.setVisible(False)

    def _on_region_done(self, rect: QRectF) -> None:
        self.regionSelected.emit(
            self.scene.mode,
            int(rect.x()),
            int(rect.y()),
            int(rect.width()),
            int(rect.height()),
        )

    def _on_region_edited(self, rect: QRectF) -> None:
        self.regionPreview.emit(
            self.scene.mode,
            int(rect.x()),
            int(rect.y()),
            int(rect.width()),
            int(rect.height()),
        )

    def _sync_handle_scale(self) -> None:
        """Bao cho khung biet muc phong to hien tai de nut keo khong bi ti hon."""
        scale = self.view.transform().m11()
        self.scene.region_item.set_view_scale(scale if scale > 0 else 1.0)

    # ------------------------------------------------------------------ noi bo

    def _on_position(self, ms: int) -> None:
        seconds = ms / 1000.0
        if not self._seeking and self._duration > 0:
            self.slider.setValue(int(seconds / self._duration * 1000))
        self.lbl_time.setText(f"{format_display(seconds)} / {format_display(self._duration)}")
        self.positionChanged.emit(seconds)

    def _on_duration(self, ms: int) -> None:
        self._duration = ms / 1000.0
        self.durationChanged.emit(self._duration)

    def _on_native_size(self, size: QSizeF) -> None:
        if size.width() > 0 and size.height() > 0:
            self.scene.set_native_size(int(size.width()), int(size.height()))
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._sync_handle_scale()
            self._layout_subtitle()

    def _on_error(self, error, message: str) -> None:  # noqa: ARG002
        if error != QMediaPlayer.Error.NoError:
            self._placeholder.setText(
                f"Khong phat duoc video: {message or 'dinh dang khong ho tro'}"
            )
            self._placeholder.setVisible(True)
            self._update_placeholder()

    def _begin_seek(self) -> None:
        """Bat dau tua va tam dung de viec xem tung khung hinh on dinh hon."""
        self._seeking = True
        self._pending_seek_value = None
        self._resume_after_seek = (
            self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        if self._resume_after_seek:
            self.player.pause()

    def _queue_live_seek(self, value: int) -> None:
        """Cap nhat khung hinh lien tuc trong luc nguoi dung dang keo thanh tua."""
        if self._duration <= 0:
            return
        self._pending_seek_value = int(value)
        target = self._pending_seek_value / 1000 * self._duration
        self.lbl_time.setText(f"{format_display(target)} / {format_display(self._duration)}")

        # Lan di chuyen dau tien duoc xu ly ngay. Cac lan sau gom trong tung
        # khoang 40 ms de QMediaPlayer khong bi ngap lenh khi keo chuot nhanh.
        if not self._seek_timer.isActive():
            self._flush_live_seek()
            self._seek_timer.start()

    def _flush_live_seek(self) -> None:
        value = self._pending_seek_value
        self._pending_seek_value = None
        if value is not None and self._duration > 0:
            self.seek(value / 1000 * self._duration)
        if self._seeking and self._pending_seek_value is not None:
            self._seek_timer.start()

    def _seek_finished(self) -> None:
        self._seek_timer.stop()
        if self._duration > 0:
            self._pending_seek_value = self.slider.value()
            self._flush_live_seek()
        self._seeking = False
        if self._resume_after_seek:
            self.player.play()
        self._resume_after_seek = False

    def _update_placeholder(self) -> None:
        self._placeholder.setGeometry(
            0, 0, max(320, self.view.width()), max(40, self.view.height())
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._sync_handle_scale()
        self._update_placeholder()
