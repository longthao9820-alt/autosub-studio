"""Cac bang dieu khien cho tung buoc lam viec (B1 - B4 va Cau Hinh Chung)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import steps as pipeline_steps
from ..providers import asr, ocr, ocr_filter, separate, translate
from ..services.paths import human_size
from ..services.settings import Settings
from ..services.updater import ReleaseInfo
from ..version import APP_VERSION
from .style import BLUE, GREEN, MUTED
from .widgets import VerticalTabStrip, field_label


def _row(*widgets, spacing: int = 8, stretch_last: bool = False) -> QHBoxLayout:
    """Mot hang ngang gon gang."""
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for index, widget in enumerate(widgets):
        if isinstance(widget, str):
            layout.addWidget(field_label(widget))
        elif widget is None:
            layout.addStretch(1)
        else:
            grow = 1 if (stretch_last and index == len(widgets) - 1) else 0
            layout.addWidget(widget, grow)
    return layout


def _slider(minimum: int, maximum: int, value: int = 0) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    return slider


def _table(headers: list[str], height: int = 0) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    if height:
        table.setMinimumHeight(height)
    return table


# =========================================================== kich ban tu dong


class ScriptPanel(QWidget):
    """Cot 'Cac Buoc Chay Kich Ban AUTO' ben trai bang du an."""

    runRequested = Signal(list)
    scriptSaved = Signal(str, list)
    scriptDeleted = Signal(str)
    scriptSelected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        box = QGroupBox("Cac Buoc Chay Kich Ban AUTO")
        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        for name in pipeline_steps.ALL_STEPS:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list.addItem(item)

        self.combo = QComboBox()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Nhap ten kich ban moi...")
        self.btn_run = QPushButton("Chay Kich Ban")
        self.btn_save = QPushButton("Them")
        self.btn_save.setFixedWidth(72)
        self.btn_delete = QPushButton("Xoa")
        self.btn_delete.setObjectName("Danger")
        self.btn_delete.setFixedWidth(64)

        self.btn_run.clicked.connect(lambda: self.runRequested.emit(self.checked_steps()))
        self.btn_save.clicked.connect(self._save)
        self.btn_delete.clicked.connect(lambda: self.scriptDeleted.emit(self.combo.currentText()))
        self.combo.currentTextChanged.connect(self.scriptSelected)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(6, 12, 6, 6)
        inner.setSpacing(5)
        inner.addWidget(self.list, 1)
        inner.addLayout(_row("Kich Ban:", self.combo, self.btn_delete, stretch_last=False))
        inner.addLayout(_row(self.name_edit, self.btn_save))
        inner.addWidget(self.btn_run)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

    def checked_steps(self) -> list[str]:
        return [
            self.list.item(i).text()
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def set_checked_steps(self, names: list[str]) -> None:
        wanted = set(names)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in wanted else Qt.CheckState.Unchecked
            )

    def set_scripts(self, names: list[str], current: str = "") -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(names)
        if current and current in names:
            self.combo.setCurrentText(current)
        self.combo.blockSignals(False)

    def _save(self) -> None:
        name = self.name_edit.text().strip() or self.combo.currentText().strip()
        if name:
            self.scriptSaved.emit(name, self.checked_steps())
            self.name_edit.clear()


# =========================================================== B1: Tach Sub


class SubtitlePanel(QWidget):
    """B1 - lay phu de bang chu tren hinh (OCR) hoac bang giong noi (ASR)."""

    chooseVideo = Signal()
    importSubtitle = Signal()
    downloadVideo = Signal()
    runAsr = Signal()
    runOcr = Signal()
    markOcrRegion = Signal()
    measureOcr = Signal()
    checkMachine = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.strip = VerticalTabStrip(
            ["Tách Bằng Chữ (OCR)", "Tách Bằng Giọng (ASR)"], [GREEN, BLUE]
        )
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_ocr_page())
        self.stack.addWidget(self._build_asr_page())
        self.strip.currentChanged.connect(self.stack.setCurrentIndex)

        self.region_table = _table(["Time", "X", "Y", "Width", "Height"])
        region_box = QGroupBox("Danh Sách Vị Trí Vùng Sub Theo Thời Gian")
        region_layout = QVBoxLayout(region_box)
        region_layout.setContentsMargins(6, 12, 6, 6)
        region_layout.addWidget(self.region_table)
        region_box.setFixedWidth(465)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 4, 2)
        layout.setSpacing(6)
        layout.addWidget(self.strip)
        layout.addWidget(self.stack, 1)
        layout.addWidget(region_box)

    # ------------------------------------------------------------------ trang

    def _build_ocr_page(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Lấy Sub Chữ Trong Video V2")

        self.ocr_fps = QDoubleSpinBox()
        self.ocr_fps.setRange(0.5, 30.0)
        self.ocr_fps.setSingleStep(0.5)
        self.ocr_fps.setValue(6.0)
        self.ocr_fps.setFixedWidth(80)
        self.ocr_confidence = QDoubleSpinBox()
        self.ocr_confidence.setRange(0.0, 100.0)
        self.ocr_confidence.setSuffix(" %")
        self.ocr_confidence.setValue(50.0)
        self.ocr_confidence.setFixedWidth(90)
        self.ocr_similarity = _slider(50, 100, 82)
        self.lbl_similarity = QLabel("82%")
        self.lbl_similarity.setObjectName("Value")
        self.ocr_similarity.valueChanged.connect(lambda v: self.lbl_similarity.setText(f"{v}%"))
        self.ocr_min_duration = QDoubleSpinBox()
        self.ocr_min_duration.setRange(0.1, 5.0)
        self.ocr_min_duration.setSingleStep(0.1)
        self.ocr_min_duration.setValue(0.25)
        self.ocr_min_duration.setFixedWidth(80)
        self.ocr_continuous = QCheckBox("Cho time liền mạch")
        self.ocr_refine = QCheckBox("Đo lại mốc thời gian cho chính xác")
        self.ocr_refine.setToolTip(
            "Sau khi doc xong, phan mem lay them cac khung hinh sat nhau quanh dau\n"
            "va cuoi moi cau de tim dung luc chu hien ra va bien mat.\n"
            "Chinh xac hon nhieu, doi lai lau hon mot chut."
        )

        self.ocr_drop_chars = QLineEdit()
        self.ocr_drop_chars.setPlaceholderText(
            "Nhap cac ky tu rac can loc bo, cach nhau boi dau phay ','"
        )
        self.ocr_drop_words = QLineEdit()
        self.ocr_drop_words.setPlaceholderText(
            "Nhap cac tu, cum tu can xoa trong cau thoai, cach nhau boi dau phay ','"
        )

        self._build_ocr_precision()
        self.ocr_min_height.setValue(25)
        self.ocr_max_height.setValue(200)
        self.ocr_contrast.setValue(0)

        self.ocr_mode = QComboBox()
        self.ocr_mode.addItems(["Nhanh Như NTS", "Cân Bằng", "Chính Xác", "OCR AI"])
        self.ocr_mode.setCurrentText("Nhanh Như NTS")
        self.ocr_server = QComboBox()
        self.ocr_server.addItems(
            [
                "PP-OCRv4 Mobile (Nhanh Như NTS)",
                "PP-OCRv6 Small (Nhanh)",
                "PP-OCRv6 Medium (Chuẩn nhất)",
                "Server AI API",
            ]
        )
        self.ocr_ai_model = QComboBox()
        self.ocr_ai_model.addItems(["sub", "prime"])
        self.ocr_ai_model.setFixedWidth(70)
        self.ocr_language = QComboBox()
        self.ocr_language.addItems(["Simplified Chinese", "English", "Vietnamese", "Auto"])
        self.ocr_batch = QSpinBox()
        self.ocr_batch.setRange(1, 64)
        self.ocr_batch.setValue(6)
        self.ocr_count = QSpinBox()
        self.ocr_count.setRange(1, 5)
        self.ocr_count.setValue(3)
        self.ocr_count.setToolTip(
            "So khung hinh gan nhau phai cung ung ho mot ban doc. "
            "3 la muc chinh xac tot cho phu de tieng Trung."
        )
        self.ocr_mode.currentTextChanged.connect(self._apply_ocr_preset)

        self.btn_region = QPushButton("Xem Trước Vùng Cắt")
        self.btn_region.setToolTip(
            "Khung xanh luon hien san tren video o che do Screen Edit. "
            "Keo giua khung de di chuyen, keo 8 nut xanh o vien de doi kich thuoc. "
            "Nut nay dua khung ve vi tri goi y ban dau."
        )
        self.btn_test_ocr = QPushButton("Tách Thử")
        self.btn_ocr = QPushButton("START: Lấy Sub V2")
        self.btn_format_ocr = QPushButton("START: Format + Lấy Sub V2")
        self.status = QLabel()
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)

        self.btn_region.clicked.connect(self.markOcrRegion)
        self.btn_ocr.clicked.connect(self.runOcr)
        self.btn_test_ocr.clicked.connect(self.measureOcr)
        self.btn_format_ocr.clicked.connect(self.runOcr)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 14, 8, 8)
        inner.setSpacing(12)
        inner.addLayout(
            _row(
                "Loại Bỏ Chiều Cao Chữ: Nhỏ Hơn <",
                self.ocr_min_height,
                "Và Lớn Hơn >",
                self.ocr_max_height,
                "Server:",
                self.ocr_server,
                "AI Model:",
                self.ocr_ai_model,
                "Ngôn Ngữ Sub:",
                self.ocr_language,
                "Batch Size:",
                self.ocr_batch,
                "Số lượng:",
                self.ocr_count,
                None,
            )
        )
        inner.addLayout(
            _row(
                "Chế Độ Trích Xuất",
                self.ocr_mode,
                "Ngưỡng Bỏ Qua:",
                self.ocr_confidence,
                "Chỉnh sáng:",
                self.ocr_brightness,
                self.lbl_brightness,
                "Chỉnh tương phản:",
                self.ocr_contrast,
                self.lbl_contrast,
                "Hệ số trùng từ nhau:",
                self.ocr_similarity,
                self.lbl_similarity,
                self.ocr_continuous,
                None,
            )
        )
        inner.addLayout(
            _row(
                "Lọc Chữ Theo Màu Sub (Tùy Chọn):",
                self.color_dot,
                self.ocr_text_color,
                self.btn_pick_color,
                self.btn_clear_color,
                "Lọc Bỏ Ký Tự Rác:",
                self.ocr_drop_chars,
                "Xóa Các Từ Có Chứa Trong Văn Bản:",
                self.ocr_drop_words,
                None,
            )
        )
        inner.addStretch(1)
        inner.addLayout(
            _row(
                self.btn_measure,
                self.btn_region,
                self.btn_test_ocr,
                self.btn_ocr,
                self.btn_format_ocr,
            )
        )

        # Cac tham so ky thuat cu van duoc luu de tuong thich voi pipeline,
        # nhung khong lam roi man hinh dieu khien rut gon.
        for hidden in (
            self.ocr_fps,
            self.ocr_min_duration,
            self.ocr_refine,
            self.ocr_color_filter,
            self.ocr_tolerance,
            self.lbl_tolerance,
            self.ocr_drop_static,
        ):
            hidden.setParent(page)
            hidden.hide()

        server_box = QGroupBox("Kiểm Tra Cấu Hình Máy")
        server_row = QHBoxLayout(server_box)
        server_row.setContentsMargins(8, 13, 8, 7)
        machine_note = QLabel(
            "Tự động phát hiện card màn hình và chọn GPU/CPU phù hợp cho máy đang chạy."
        )
        machine_note.setObjectName("Muted")
        self.btn_check_machine = QPushButton("Kiểm Tra Cấu Hình Máy")
        self.btn_check_machine.setFixedWidth(230)
        self.machine_state = QLabel("Chưa kiểm tra cấu hình máy")
        self.machine_state.setObjectName("Muted")
        self.machine_state.setWordWrap(True)
        self.btn_check_machine.clicked.connect(self.checkMachine)
        server_row.addWidget(machine_note)
        server_row.addWidget(self.btn_check_machine)
        server_row.addWidget(self.machine_state, 1)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(box, 1)
        layout.addWidget(server_box)
        return page

    def _apply_ocr_preset(self, name: str) -> None:
        """Bien ba che do tren giao dien thanh tham so OCR thuc su."""
        if name == "OCR AI":
            self.ocr_server.setCurrentText("Server AI API")
            self.ocr_refine.setChecked(False)
            return
        presets = {
            "Nhanh Như NTS": (
                15.0,
                70.0,
                80,
                1,
                False,
                5,
                "PP-OCRv4 Mobile (Nhanh Như NTS)",
            ),
            "Cân Bằng": (4.0, 45.0, 80, 2, True, 6, "PP-OCRv6 Small (Nhanh)"),
            "Chính Xác": (
                6.0,
                50.0,
                82,
                3,
                True,
                6,
                "PP-OCRv6 Medium (Chuẩn nhất)",
            ),
        }
        fps, confidence, similarity, votes, refine, batch, server = presets.get(
            name, presets["Nhanh Như NTS"]
        )
        self.ocr_fps.setValue(fps)
        self.ocr_confidence.setValue(confidence)
        self.ocr_similarity.setValue(similarity)
        self.ocr_count.setValue(votes)
        self.ocr_refine.setChecked(refine)
        self.ocr_batch.setValue(batch)
        self.ocr_server.setCurrentText(server)

    def set_machine_checking(self, checking: bool) -> None:
        self.btn_check_machine.setEnabled(not checking)
        self.btn_check_machine.setText(
            "Đang Kiểm Tra..." if checking else "Kiểm Tra Cấu Hình Máy"
        )
        if checking:
            self.machine_state.setObjectName("Value")
            self.machine_state.setText("Đang dò card màn hình và thử bộ mã hóa phần cứng...")
            self.machine_state.style().unpolish(self.machine_state)
            self.machine_state.style().polish(self.machine_state)

    def set_machine_result(self, text: str, accelerated: bool) -> None:
        self.set_machine_checking(False)
        self.machine_state.setObjectName("Ok" if accelerated else "Muted")
        self.machine_state.setText(text.replace("\n", "  |  "))
        self.machine_state.style().unpolish(self.machine_state)
        self.machine_state.style().polish(self.machine_state)

    def set_machine_unchecked(self) -> None:
        self.set_machine_checking(False)
        self.machine_state.setObjectName("Muted")
        self.machine_state.setText("Chưa kiểm tra cấu hình máy")
        self.gpu_note.setText(
            "Chưa kiểm tra GPU trên máy này. Bấm 'Kiểm Tra Cấu Hình Máy' trước khi chạy."
        )
        self.machine_state.style().unpolish(self.machine_state)
        self.machine_state.style().polish(self.machine_state)

    def _build_ocr_precision(self) -> None:
        """Cac o dieu khien giup OCR bat dung dong phu de, khong bat nham chu khac."""
        self.ocr_min_height = QDoubleSpinBox()
        self.ocr_min_height.setRange(0.0, 2000.0)
        self.ocr_min_height.setDecimals(0)
        self.ocr_min_height.setSuffix(" px")
        self.ocr_min_height.setFixedWidth(90)
        self.ocr_min_height.setToolTip(
            "Bo cac dong chu thap hon muc nay. De 0 la de phan mem tu do tren video."
        )
        self.ocr_max_height = QDoubleSpinBox()
        self.ocr_max_height.setRange(0.0, 2000.0)
        self.ocr_max_height.setDecimals(0)
        self.ocr_max_height.setSuffix(" px")
        self.ocr_max_height.setFixedWidth(90)
        self.ocr_max_height.setToolTip(
            "Bo cac dong chu cao hon muc nay (chu tieu de, chu quang cao co to). "
            "De 0 la de phan mem tu do tren video."
        )
        self.btn_measure = QPushButton("Đo Chiều Cao Chữ")
        self.btn_measure.setToolTip(
            "Lay vai khung hinh rai deu trong video, doc thu roi dien san mau chu\n"
            "va khoang chieu cao chu cua phu de vao day."
        )
        self.btn_measure.clicked.connect(self.measureOcr)

        self.ocr_color_filter = QCheckBox("Loc Chu Theo Mau Sub (Tuy Chon):")
        self.ocr_color_filter.setToolTip(
            "Mac dinh NTS khong loc mau. Chi bat muc nay khi ban muon chi dinh\n"
            "mot mau phu de cu the de loai logo hoac chu quang cao khac mau."
        )
        self.color_dot = QLabel()
        self.color_dot.setFixedSize(24, 22)
        self.ocr_text_color = QLineEdit()
        self.ocr_text_color.setFixedWidth(100)
        self.ocr_text_color.setPlaceholderText("khong loc")
        self.ocr_text_color.setToolTip(
            "Mau chu phu de dang #RRGGBB. Chi co tac dung khi muc loc mau duoc bat."
        )
        self.btn_pick_color = QPushButton("Chon Mau...")
        self.btn_clear_color = QPushButton("Clear")
        self.btn_clear_color.setToolTip("Xoa mau da chon. OCR se dung anh goc, khong loc mau.")
        self.ocr_tolerance = _slider(4, 40, 15)
        self.ocr_tolerance.setToolTip(
            "Cho phep mau chu lech bao nhieu so voi mau da chon. Video net thi de\n"
            "thap cho chinh xac, video mo hoac chu bi vien mau thi tang len."
        )
        self.lbl_tolerance = QLabel("15%")
        self.lbl_tolerance.setObjectName("Value")
        self.ocr_brightness = _slider(-100, 100, 0)
        self.lbl_brightness = QLabel("0")
        self.lbl_brightness.setObjectName("Value")
        self.ocr_contrast = _slider(-100, 100, 0)
        self.lbl_contrast = QLabel("0")
        self.lbl_contrast.setObjectName("Value")
        self.ocr_drop_static = QCheckBox("Bo Chu Dung Yen Mot Cho (Tuy Chon)")
        self.ocr_drop_static.setToolTip(
            "Bo nhung dong chu nam y nguyen mot cho gan het video: logo kenh,\n"
            "watermark, dong chu dan san. Phu de thi doi lien tuc nen khong bi bo."
        )

        self.ocr_tolerance.valueChanged.connect(lambda v: self.lbl_tolerance.setText(f"{v}%"))
        self.ocr_brightness.valueChanged.connect(lambda v: self.lbl_brightness.setText(str(v)))
        self.ocr_contrast.valueChanged.connect(lambda v: self.lbl_contrast.setText(str(v)))
        self.ocr_text_color.textChanged.connect(self._show_color)
        self.btn_pick_color.clicked.connect(self._pick_color)
        self.btn_clear_color.clicked.connect(self._clear_color)
        self.ocr_color_filter.toggled.connect(self._toggle_color_filter)
        self._show_color()

    # ------------------------------------------------------------------ mau chu

    def _show_color(self) -> None:
        """Ve o mau nho ben canh de nhin la biet dang loc theo mau nao."""
        rgb = ocr_filter.parse_color(self.ocr_text_color.text())
        if rgb is None:
            self.color_dot.setStyleSheet(
                f"background:transparent;border:1px dashed {MUTED};border-radius:3px;"
            )
            self.color_dot.setToolTip("Chua chon mau: OCR dung anh goc, khong loc mau.")
            return
        self.color_dot.setStyleSheet(
            f"background:{ocr_filter.format_color(rgb)};border:1px solid #888;border-radius:3px;"
        )
        self.color_dot.setToolTip(f"Mau chu: {ocr_filter.color_name(rgb)}")

    def _pick_color(self) -> None:
        current = ocr_filter.parse_color(self.ocr_text_color.text()) or (255, 255, 255)
        chosen = QColorDialog.getColor(QColor(*current), self, "Chon mau chu phu de")
        if chosen.isValid():
            self.ocr_text_color.setText(
                ocr_filter.format_color((chosen.red(), chosen.green(), chosen.blue()))
            )
            self.ocr_color_filter.setChecked(True)

    def _clear_color(self) -> None:
        self.ocr_text_color.clear()
        self.ocr_color_filter.setChecked(False)

    def _toggle_color_filter(self, on: bool) -> None:
        # O checkbox ky thuat duoc an de giao dien giong NTS. Cac nut chon mau
        # luon dung duoc; chon/paste mau la hanh dong bat loc, Clear la tat.
        for widget in (
            self.color_dot,
            self.ocr_text_color,
            self.btn_pick_color,
            self.btn_clear_color,
            self.ocr_tolerance,
            self.lbl_tolerance,
        ):
            widget.setEnabled(True)

    def _build_asr_page(self) -> QWidget:
        page = QWidget()
        source_box = QGroupBox("Nguon")
        self.btn_video = QPushButton("Chon Video / Audio...")
        self.btn_import = QPushButton("Nhap Tep Phu De (SRT/VTT/ASS)...")
        self.btn_download = QPushButton("Tải Video Từ Link...")
        self.btn_video.clicked.connect(self.chooseVideo)
        self.btn_import.clicked.connect(self.importSubtitle)
        self.btn_download.clicked.connect(self.downloadVideo)
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(8, 14, 8, 8)
        source_layout.addWidget(self.btn_video)
        source_layout.addWidget(self.btn_import)
        source_layout.addWidget(self.btn_download)
        source_layout.addStretch(1)
        source_box.setFixedWidth(250)

        box = QGroupBox("Lay Sub Bang Giong Noi Trong Video")
        self.model = QComboBox()
        self.model.addItems(asr.MODEL_SIZES)
        self.model.setFixedWidth(120)
        self.language = QComboBox()
        for code, label in translate.LANGUAGES.items():
            self.language.addItem(label, code)
        self.language.setFixedWidth(160)
        self.device = QComboBox()
        self.device.addItems(asr.DEVICES)
        self.device.setFixedWidth(130)
        self.vad = QCheckBox("Bo qua doan im lang (VAD)")
        self.max_chars = QSpinBox()
        self.max_chars.setRange(10, 120)
        self.max_chars.setFixedWidth(70)
        self.max_lines = QSpinBox()
        self.max_lines.setRange(1, 4)
        self.max_lines.setFixedWidth(60)
        self.min_dur = QDoubleSpinBox()
        self.min_dur.setRange(0.1, 5.0)
        self.min_dur.setSingleStep(0.1)
        self.min_dur.setFixedWidth(80)
        self.max_dur = QDoubleSpinBox()
        self.max_dur.setRange(1.0, 20.0)
        self.max_dur.setSingleStep(0.5)
        self.max_dur.setFixedWidth(80)

        self.btn_asr = QPushButton("START: Lay Sub Bang Giong Noi")
        self.btn_asr.clicked.connect(self.runAsr)
        self.gpu_note = QLabel()
        self.gpu_note.setObjectName("Muted")
        self.gpu_note.setWordWrap(True)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 14, 8, 8)
        inner.setSpacing(9)
        inner.addLayout(
            _row(
                "Model:", self.model, "Ngon Ngu Sub:", self.language, "Thiet Bi:", self.device, None
            )
        )
        inner.addLayout(
            _row(
                "Ky Tu Toi Da / Dong:",
                self.max_chars,
                "So Dong Toi Da:",
                self.max_lines,
                "Do Dai Toi Thieu:",
                self.min_dur,
                "Toi Da:",
                self.max_dur,
                None,
            )
        )
        inner.addLayout(_row(self.vad, None))
        inner.addStretch(1)
        inner.addWidget(self.gpu_note)
        inner.addWidget(self.btn_asr)

        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(source_box)
        layout.addWidget(box, 1)
        return page

    # ------------------------------------------------------------------ du lieu

    def set_regions(self, ocr_region: list[int], blur_region: list[int]) -> None:
        rows = []
        if len(ocr_region) == 4:
            rows.append((1, ocr_region))
        if len(blur_region) == 4:
            rows.append((2, blur_region))
        self.region_table.setRowCount(len(rows))
        for r, (time_index, values) in enumerate(rows):
            self.region_table.setItem(r, 0, QTableWidgetItem(str(time_index)))
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.region_table.setItem(r, c + 1, item)

    def load(self, s: Settings) -> None:
        self.model.setCurrentText(s.asr_model)
        index = self.language.findData(s.asr_language)
        self.language.setCurrentIndex(index if index >= 0 else 0)
        self.vad.setChecked(s.asr_vad)
        self.max_chars.setValue(s.asr_max_chars)
        self.max_lines.setValue(s.asr_max_lines)
        self.min_dur.setValue(s.asr_min_duration)
        self.max_dur.setValue(s.asr_max_duration)
        self.device.setCurrentIndex(0 if s.use_gpu else 1)
        mode = "Nhanh Như NTS" if s.ocr_mode == "Nhanh" else s.ocr_mode
        self.ocr_mode.setCurrentText(mode)
        self.ocr_server.setCurrentText(s.ocr_server)
        if hasattr(s, "ocr_ai_model") and s.ocr_ai_model:
            self.ocr_ai_model.setCurrentText(s.ocr_ai_model)
        self.ocr_language.setCurrentText(s.ocr_language)
        self.ocr_batch.setValue(s.ocr_batch_size)
        self.ocr_count.setValue(s.ocr_consensus)
        self.ocr_fps.setValue(s.ocr_fps)
        self.ocr_confidence.setValue(s.ocr_confidence)
        self.ocr_similarity.setValue(int(s.ocr_similarity * 100))
        self.ocr_min_duration.setValue(s.ocr_min_duration)
        self.ocr_continuous.setChecked(s.ocr_continuous)
        self.ocr_refine.setChecked(s.ocr_refine)
        self.ocr_drop_chars.setText(s.ocr_drop_chars)
        self.ocr_drop_words.setText(s.ocr_drop_words)
        self.ocr_color_filter.setChecked(s.ocr_color_filter)
        self.ocr_text_color.setText(s.ocr_text_color)
        self.ocr_tolerance.setValue(int(round(s.ocr_color_tolerance)))
        self.ocr_min_height.setValue(s.ocr_min_height)
        self.ocr_max_height.setValue(s.ocr_max_height)
        self.ocr_brightness.setValue(s.ocr_brightness)
        self.ocr_contrast.setValue(s.ocr_contrast)
        self.ocr_drop_static.setChecked(s.ocr_drop_static)
        self._toggle_color_filter(s.ocr_color_filter)
        self._show_color()

        self.btn_asr.setEnabled(asr.is_available())
        self.btn_ocr.setEnabled(
            ocr.is_available()
            or self.ocr_mode.currentText() == "OCR AI"
            or self.ocr_server.currentText() == "Server AI API"
        )
        self.btn_measure.setEnabled(ocr.is_available() and ocr_filter.available())
        notes = []
        if not asr.is_available():
            notes.append(asr.install_hint())
        if not ocr.is_available():
            notes.append(ocr.install_hint())
        if ocr.is_available() and not ocr_filter.available():
            notes.append(
                "Thieu opencv-python nen khong loc duoc chu theo mau. "
                "Chay: pip install opencv-python"
            )
        if notes:
            self.status.setText("\n".join(notes))
        if s.hardware_signature:
            mode = "GPU" if s.use_gpu else "CPU"
            self.set_machine_result(
                f"Đã kiểm tra cấu hình trên máy này. Chế độ nhận dạng: {mode}. "
                "Bấm nút để kiểm tra lại.",
                s.use_gpu or s.use_gpu_encoder,
            )
            self.gpu_note.setText(f"Cấu hình đã lưu trên máy này: {mode}.")
        else:
            self.set_machine_unchecked()

    def apply(self, s: Settings) -> None:
        s.asr_model = self.model.currentText()
        s.asr_language = self.language.currentData() or "auto"
        s.asr_vad = self.vad.isChecked()
        s.asr_max_chars = self.max_chars.value()
        s.asr_max_lines = self.max_lines.value()
        s.asr_min_duration = self.min_dur.value()
        s.asr_max_duration = self.max_dur.value()
        s.use_gpu = self.device.currentIndex() != 1
        s.ocr_mode = self.ocr_mode.currentText()
        s.ocr_server = self.ocr_server.currentText()
        if hasattr(s, "ocr_ai_model"):
            s.ocr_ai_model = self.ocr_ai_model.currentText()
        s.ocr_language = self.ocr_language.currentText()
        s.ocr_batch_size = self.ocr_batch.value()
        s.ocr_consensus = self.ocr_count.value()
        s.ocr_fps = self.ocr_fps.value()
        s.ocr_confidence = self.ocr_confidence.value()
        s.ocr_similarity = self.ocr_similarity.value() / 100
        s.ocr_min_duration = self.ocr_min_duration.value()
        s.ocr_continuous = self.ocr_continuous.isChecked()
        s.ocr_refine = self.ocr_refine.isChecked()
        s.ocr_drop_chars = self.ocr_drop_chars.text().strip()
        s.ocr_drop_words = self.ocr_drop_words.text().strip()
        parsed_color = ocr_filter.parse_color(self.ocr_text_color.text())
        s.ocr_color_filter = parsed_color is not None
        s.ocr_text_color = ocr_filter.format_color(parsed_color)
        s.ocr_color_tolerance = float(self.ocr_tolerance.value())
        s.ocr_min_height = self.ocr_min_height.value()
        s.ocr_max_height = self.ocr_max_height.value()
        s.ocr_brightness = self.ocr_brightness.value()
        s.ocr_contrast = self.ocr_contrast.value()
        s.ocr_drop_static = self.ocr_drop_static.isChecked()


# =========================================================== B2: Dich Noi Dung


class TranslatePanel(QWidget):
    """B2 - dich phu de."""

    runTranslate = Signal()
    translateSelected = Signal()
    glossaryChanged = Signal(str)
    importTranslation = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.provider = QComboBox()
        self.provider.addItem("Server AI API", translate.PROVIDER_SERVER_AI)
        self.provider.addItem("Google Miễn Phí", translate.PROVIDER_GOOGLE)
        self.provider.addItem("Không Dịch", translate.PROVIDER_NONE)
        self.provider.setFixedWidth(190)
        self.source = QComboBox()
        self.target = QComboBox()
        for code, label in translate.LANGUAGES.items():
            shown = {
                "auto": "Tự Động Phát Hiện",
                "vi": "Vietnamese",
                "en": "English",
                "zh-CN": "Simplified Chinese",
            }.get(code, label)
            self.source.addItem(shown, code)
            if code != "auto":
                self.target.addItem(shown, code)
        self.model = QComboBox()
        self.model.addItem("sub", "sub")
        self.model.addItem("prime", "prime")
        self.model.setMinimumWidth(160)
        self.batch = QSpinBox()
        self.batch.setRange(1, 100)
        self.batch.setValue(8)
        self.batch.setFixedWidth(70)
        self.context = QSpinBox()
        self.context.setRange(0, 5)
        self.context.setFixedWidth(60)

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "Nếu bạn muốn sửa lại prompt dịch vui lòng viết thêm là cần dịch qua "
            "ngôn ngữ nào chứ tool không tự chọn Ngôn Ngữ ở Trên"
        )
        self.glossary = QPlainTextEdit()
        self.glossary.setPlaceholderText(
            "Moi dong mot cap:\nAnthropic = Anthropic\nNew York = New York"
        )
        self.glossary.setFixedWidth(300)
        self.glossary.textChanged.connect(
            lambda: self.glossaryChanged.emit(self.glossary.toPlainText())
        )

        self.speaker_mode = QComboBox()
        self.speaker_mode.addItems(["Không Nhận Dạng", "Nhận Dạng Nam/Nữ"])
        self.text_format = QComboBox()
        self.text_format.addItems(["Giữ Nguyên Bản", "Viết Hoa Đầu Câu", "Chuẩn Hóa"])
        self.auto_join = QCheckBox("Tự Động Gộp Dòng Dịch")
        self.max_gap = QSpinBox()
        self.max_gap.setRange(0, 5000)
        self.max_gap.setSuffix(" ms")
        self.max_gap.setEnabled(False)

        self.btn_all = QPushButton("START: Dịch Phụ Đề")
        self.btn_selected = QPushButton("Dịch Các Câu Đang Chọn")
        self.btn_import = QPushButton("Nhập Bản Dịch Từ File SRT")
        self.btn_import_txt = QPushButton("Nhập Bản Dịch Từ File TXT")
        self.status = QLabel()
        self.status.setObjectName("Muted")

        self.btn_all.clicked.connect(self.runTranslate)
        self.btn_selected.clicked.connect(self.translateSelected)
        self.btn_import.clicked.connect(self.importTranslation)
        self.btn_import_txt.clicked.connect(self.importTranslation)
        self.provider.currentTextChanged.connect(lambda _t: self.refresh_status())

        top = QWidget()
        top_col = QVBoxLayout(top)
        top_col.setContentsMargins(0, 0, 0, 0)
        top_col.setSpacing(9)
        top_col.addLayout(
            _row(
                "Chọn Server:",
                self.provider,
                "Ngôn Ngữ Gốc:",
                self.source,
                "Dịch Sang:",
                self.target,
                None,
            )
        )
        top_col.addLayout(
            _row(
                "Model AI:",
                self.model,
                "Nhận Dạng Người Nói:",
                self.speaker_mode,
                "Ngắt Đoạn Dịch:",
                self.batch,
                None,
            )
        )

        prompt_box = QGroupBox("Prompt Dịch Tùy Chỉnh")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.setContentsMargins(6, 12, 6, 6)
        prompt_layout.addWidget(self.prompt)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(8)
        layout.addWidget(top)
        layout.addWidget(prompt_box, 1)
        layout.addLayout(
            _row(
                "Định Dạng Lại Văn Bản Dịch",
                self.text_format,
                self.auto_join,
                "Khoảng Hở Tối Đa:",
                self.max_gap,
                None,
            )
        )
        layout.addLayout(_row(self.btn_all, self.btn_import, self.btn_import_txt))

        self.btn_selected.setParent(self)
        self.btn_selected.hide()
        self.glossary.setParent(self)
        self.glossary.hide()
        self.context.setParent(self)
        self.context.hide()
        self.status.setParent(self)
        self.status.hide()

    def load(self, s: Settings, api_key: str = "") -> None:
        provider_index = self.provider.findData(s.translate_provider)
        if provider_index < 0:
            provider_index = self.provider.findText(s.translate_provider)
        if provider_index < 0 and s.translate_provider in ("AI Gateway", "Server AI API"):
            provider_index = self.provider.findData(translate.PROVIDER_SERVER_AI)
            if provider_index < 0:
                provider_index = self.provider.findText("Server AI API")
        self.provider.setCurrentIndex(provider_index if provider_index >= 0 else 0)
        i = self.source.findData(s.source_language)
        self.source.setCurrentIndex(i if i >= 0 else 0)
        j = self.target.findData(s.target_language)
        self.target.setCurrentIndex(j if j >= 0 else 0)
        self.context.setValue(s.translate_context)
        self.batch.setValue(s.translate_batch)
        model_index = self.model.findData(s.llm_model)
        if model_index < 0:
            model_index = self.model.findText(s.llm_model)
        self.model.setCurrentIndex(model_index if model_index >= 0 else 0)
        self.prompt.setPlainText(s.translate_prompt)
        self.refresh_status()

    def refresh_status(self) -> None:
        ready, reason = translate.provider_ready(
            self.provider.currentData() or self.provider.currentText(),
            Settings.get_secret("ai_gateway_key"),
        )
        self.btn_all.setEnabled(ready)
        self.btn_selected.setEnabled(ready)
        self.status.setText(reason or "San sang.")

    def apply(self, s: Settings) -> None:
        s.translate_provider = self.provider.currentData() or self.provider.currentText()
        s.source_language = self.source.currentData() or "auto"
        s.target_language = self.target.currentData() or "vi"
        s.translate_context = self.context.value()
        s.translate_batch = self.batch.value()
        s.llm_model = self.model.currentData() or self.model.currentText() or "sub"
        s.translate_prompt = self.prompt.toPlainText().strip()

    def glossary_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in self.glossary.toPlainText().splitlines():
            if "=" not in line:
                continue
            src, _, dst = line.partition("=")
            if src.strip():
                out[src.strip()] = dst.strip() or src.strip()
        return out


# =========================================================== B3: Ghep Giong Doc


class _ReadOnlyVoiceCombo(QComboBox):
    """Combobox chi doc cho phep chon giong doc da tai."""

    def setCurrentText(self, text: str) -> None:
        if text and self.findText(text) < 0:
            self.addItem(text, text)
        super().setCurrentText(text)


class DubPanel(QWidget):
    """B3 - long tieng."""

    runDub = Signal()
    previewVoice = Signal()
    separateAudio = Signal()
    openVoiceLibrary = Signal()
    refreshVoices = Signal()
    runDiarize = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mode = QComboBox()
        self.mode.addItems(["Lồng Tiếng Vào Video", "Chỉ Xuất Tệp Tiếng"])
        self.mode.setFixedWidth(190)
        self.store_voice = QCheckBox("Lưu Trữ Voice")
        self.allow_overlap = QCheckBox("Chồng Tiếng Khi Khớp Thời Gian")
        self.keep_original = QCheckBox("Giữ Tiếng Gốc")
        self.min_gap = QSpinBox()
        self.min_gap.setRange(0, 5000)
        self.min_gap.setValue(300)
        self.min_gap.setSuffix(" ms")
        self.min_gap.setFixedWidth(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(
            _row(
                "Chế Độ Lồng Tiếng:",
                self.mode,
                self.store_voice,
                self.allow_overlap,
                "Check Sub Có Time Ngắn Hơn:",
                self.min_gap,
                None,
            )
        )

        body = QHBoxLayout()
        body.setSpacing(6)
        body.addWidget(self._build_voice_box(), 1)
        body.addWidget(self._build_right_column(), 1)
        layout.addLayout(body, 1)

    def _build_voice_box(self) -> QGroupBox:
        box = QGroupBox("Giọng Đọc Lồng Tiếng")
        self.language = QComboBox()
        self.language.addItem("Tiếng Việt", "vi-VN")
        self.language.addItem("English (United States)", "en-US")
        self.voice = _ReadOnlyVoiceCombo()
        self.voice.setEditable(False)
        self.voice.currentIndexChanged.connect(lambda _i: self.refresh_status())
        self.btn_library = QPushButton("Thư viện giọng")
        self.btn_preview = QPushButton("▶")
        self.btn_dictionary = QPushButton("Pronunciation Dictionary")
        self.btn_punctuation = QPushButton("Chỉnh Dấu Câu")
        self._dictionary_text = ""
        self._pause_period_ms = 300
        self._pause_comma_ms = 200
        self._pause_newline_ms = 400

        self.volume = _slider(0, 100, 100)
        self.lbl_volume = QLabel("100")
        self.rate = _slider(-10, 10, 0)
        self.rate.setRange(50, 200)
        self.rate.setValue(100)
        self.lbl_rate = QLabel("1.00x")
        self.original_volume = _slider(0, 100, 20)
        self.lbl_original_volume = QLabel("20%")
        self.bass = _slider(-12, 12, 0)
        self.lbl_bass = QLabel("0dB")
        self.mid = _slider(-12, 12, 0)
        self.lbl_mid = QLabel("0dB")
        self.treble = _slider(-12, 12, 0)
        self.lbl_treble = QLabel("0dB")
        for slider, value_label, suffix in (
            (self.volume, self.lbl_volume, ""),
            (self.original_volume, self.lbl_original_volume, "%"),
            (self.bass, self.lbl_bass, "dB"),
            (self.mid, self.lbl_mid, "dB"),
            (self.treble, self.lbl_treble, "dB"),
        ):
            value_label.setObjectName("Value")
            value_label.setFixedWidth(46)
            slider.valueChanged.connect(lambda v, lb=value_label, sf=suffix: lb.setText(f"{v}{sf}"))
        self.rate.valueChanged.connect(lambda v: self.lbl_rate.setText(f"{v / 100:.2f}x"))

        self.btn_library.clicked.connect(self.openVoiceLibrary)
        self.btn_preview.clicked.connect(self.previewVoice)
        self.btn_dictionary.clicked.connect(self._edit_dictionary)
        self.btn_punctuation.clicked.connect(self._edit_punctuation)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 14, 8, 8)
        inner.setSpacing(9)
        inner.addLayout(
            _row(
                "Chọn Giọng Đọc:",
                self.voice,
                self.btn_library,
                self.btn_preview,
                self.btn_dictionary,
                self.btn_punctuation,
            )
        )
        inner.addLayout(
            _row(
                "Âm Lượng Đọc:",
                self.volume,
                self.lbl_volume,
                "Tốc Độ Đọc:",
                self.rate,
                self.lbl_rate,
            )
        )
        inner.addLayout(
            _row(
                "Bass:",
                self.bass,
                self.lbl_bass,
                "Mid:",
                self.mid,
                self.lbl_mid,
                "Treble:",
                self.treble,
                self.lbl_treble,
            )
        )
        inner.addStretch(1)
        return box

    def _edit_dictionary(self) -> None:
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "English Pronunciation Dictionary",
            "Mỗi dòng: original=pronunciation",
            self._dictionary_text,
        )
        if accepted:
            self._dictionary_text = text.strip()

    def _edit_punctuation(self) -> None:
        current = (
            f"Dấu chấm={self._pause_period_ms}\n"
            f"Dấu phẩy={self._pause_comma_ms}\n"
            f"Xuống dòng={self._pause_newline_ms}"
        )
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "Chỉnh Dấu Câu",
            "Thời gian nghỉ (ms):",
            current,
        )
        if not accepted:
            return
        values = {}
        for line in text.splitlines():
            key, separator, value = line.partition("=")
            if not separator:
                continue
            try:
                values[key.strip().casefold()] = max(0, min(2000, int(value.strip())))
            except ValueError:
                continue
        self._pause_period_ms = values.get("dấu chấm", self._pause_period_ms)
        self._pause_comma_ms = values.get("dấu phẩy", self._pause_comma_ms)
        self._pause_newline_ms = values.get("xuống dòng", self._pause_newline_ms)

    def _build_right_column(self) -> QWidget:
        wrap = QWidget()
        column = QVBoxLayout(wrap)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        voices_box = QGroupBox("Danh Sách Giọng Đọc")
        self.voice_table = _table(
            ["Phím tắt", "Language", "Gender", "Volume", "Speed", "Bass", "Mid", "Treble"],
            height=150,
        )
        voices_layout = QVBoxLayout(voices_box)
        voices_layout.setContentsMargins(6, 12, 6, 6)
        voice_body = QHBoxLayout()
        icon_col = QVBoxLayout()
        self.btn_add_voice = QPushButton("＋")
        self.btn_remove_voice = QPushButton("−")
        self.btn_add_voice.setObjectName("RenderIcon")
        self.btn_remove_voice.setObjectName("RenderIcon")
        self.btn_voice_play = QPushButton("▶")
        self.btn_voice_play.setObjectName("Blue")
        self.btn_voice_start = QPushButton("START")
        self.profile_gender = QComboBox()
        self.profile_gender.addItems(["Mặc định", "Nam", "Nữ"])
        self.profile_gender.setFixedWidth(92)
        for button in (
            self.btn_add_voice,
            self.btn_remove_voice,
            self.btn_voice_play,
            self.btn_voice_start,
        ):
            button.setFixedSize(42, 32)
            icon_col.addWidget(button)
        icon_col.addWidget(self.profile_gender)
        icon_col.addStretch(1)
        voice_body.addLayout(icon_col)
        voice_body.addWidget(self.voice_table, 1)
        voices_layout.addLayout(voice_body)
        self._voice_profiles: list[dict[str, object]] = []
        self.btn_add_voice.clicked.connect(self._add_voice_profile)
        self.btn_remove_voice.clicked.connect(self._remove_voice_profile)
        self.btn_voice_play.clicked.connect(self._preview_profile)
        self.btn_voice_start.clicked.connect(self.runDub)

        speaker_box = QGroupBox("Cấu Hình Nhận Dạng Người Nói")
        self.btn_diarize = QPushButton("Phân Tách Giọng Nam / Nữ")
        self.btn_diarize.clicked.connect(self.runDiarize)
        self.diarize_server = QComboBox()
        self.diarize_server.addItems(["Nam/Nữ Local (Free)", "Không Nhận Dạng"])
        self.diarize_batch = QSpinBox()
        self.diarize_batch.setRange(1, 64)
        self.diarize_batch.setValue(5)
        speaker_layout = QHBoxLayout(speaker_box)
        speaker_layout.setContentsMargins(6, 12, 6, 6)
        speaker_layout.addWidget(field_label("Server Nhận Dạng"))
        speaker_layout.addWidget(self.diarize_server, 1)
        speaker_layout.addWidget(field_label("Batch Size"))
        speaker_layout.addWidget(self.diarize_batch)
        speaker_layout.addWidget(self.btn_diarize)

        dub_box = QGroupBox("Cấu Hình Lồng Tiếng")
        self.ducking = QCheckBox("Tự Hạ Tiếng Gốc Khi Đọc")
        self.btn_separate = QPushButton("Tách Nhạc Nền / Lời Thoại")
        self.btn_dub = QPushButton("START: Lồng Tiếng")
        self.render_original = QCheckBox("Render Bằng Video Gốc")
        self.render_new = QCheckBox("Render Kiểu Mới")
        self.render_mode = QComboBox()
        self.render_mode.addItems(["CANH THEO GIỌNG ĐỌC", "CANH THEO TIME SUB"])
        self.end_pause = QSpinBox()
        self.end_pause.setRange(0, 2000)
        self.end_pause.setSuffix(" ms")
        self.end_pause.setValue(250)
        self.use_original_video = QCheckBox("Render Bằng Video Gốc")
        self.timeline_cpu = QSpinBox()
        self.timeline_cpu.setRange(10, 100)
        self.timeline_cpu.setSuffix(" %")
        self.timeline_workers = QSpinBox()
        self.timeline_workers.setRange(1, 8)
        self.timeline_decode = QComboBox()
        self.timeline_decode.addItem("FFmpeg")
        self.status = QLabel()
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        self.btn_separate.clicked.connect(self.separateAudio)
        self.btn_dub.clicked.connect(self.runDub)
        dub_layout = QGridLayout(dub_box)
        dub_layout.setContentsMargins(6, 12, 6, 6)
        dub_layout.setSpacing(6)
        dub_layout.addWidget(self.keep_original, 0, 0)
        dub_layout.addWidget(self.use_original_video, 0, 1)
        dub_layout.addWidget(self.ducking, 0, 2)
        dub_layout.addWidget(field_label("Mode:"), 0, 3)
        dub_layout.addWidget(self.render_mode, 0, 4)
        dub_layout.addWidget(field_label("Nghỉ Cuối:"), 1, 0)
        dub_layout.addWidget(self.end_pause, 1, 1)
        dub_layout.addWidget(self.btn_separate, 1, 2, 1, 2)
        dub_layout.addWidget(self.btn_dub, 1, 4)
        dub_layout.addWidget(field_label("Giới Hạn CPU:"), 2, 0)
        dub_layout.addWidget(self.timeline_cpu, 2, 1)
        dub_layout.addWidget(field_label("Số Luồng Ghép:"), 2, 2)
        dub_layout.addWidget(self.timeline_workers, 2, 3)
        dub_layout.addWidget(self.timeline_decode, 2, 4)
        dub_layout.addWidget(self.status, 3, 0, 1, 5)

        column.addWidget(voices_box, 1)
        column.addWidget(speaker_box)
        column.addWidget(dub_box)
        return wrap

    def set_voices(self, voices: list[str]) -> None:
        current = self.voice.currentText()
        self.voice.clear()
        self.voice.addItems(voices)
        if current and current in voices:
            self.voice.setCurrentText(current)
        elif voices:
            self.voice.setCurrentIndex(0)
        self.refresh_status()

    def set_selected_voice(self, voice_id: str) -> None:
        idx = self.voice.findData(voice_id)
        if idx < 0:
            idx = self.voice.findText(voice_id)
        if idx >= 0:
            self.voice.setCurrentIndex(idx)
        else:
            self.voice.addItem(voice_id, voice_id)
            self.voice.setCurrentIndex(self.voice.count() - 1)
        self.refresh_status()

    def _add_voice_profile(self) -> None:
        voice = self.voice.currentText().strip()
        if not voice:
            return
        gender_text = self.profile_gender.currentText()
        gender = {"Nam": "nam", "Nữ": "nu"}.get(gender_text, "default")
        profile: dict[str, object] = {
            "voice": voice,
            "language": self.language.currentData() or "",
            "gender": gender,
            "volume": self.volume.value(),
            "speed": self.rate.value(),
            "bass": self.bass.value(),
            "mid": self.mid.value(),
            "treble": self.treble.value(),
        }
        self._voice_profiles = [
            item for item in self._voice_profiles if str(item.get("gender")) != gender
        ]
        self._voice_profiles.append(profile)
        self._render_voice_profiles()

    def _remove_voice_profile(self) -> None:
        row = self.voice_table.currentRow()
        if 0 <= row < len(self._voice_profiles):
            del self._voice_profiles[row]
            self._render_voice_profiles()

    def _preview_profile(self) -> None:
        row = self.voice_table.currentRow()
        if 0 <= row < len(self._voice_profiles):
            self.voice.setCurrentText(str(self._voice_profiles[row].get("voice", "")))
        self.previewVoice.emit()

    def _render_voice_profiles(self) -> None:
        self.voice_table.setRowCount(len(self._voice_profiles))
        labels = {"default": "Mặc định", "nam": "Nam", "nu": "Nữ"}
        for row, item in enumerate(self._voice_profiles):
            values = [
                str(row + 1),
                str(item.get("language", "")),
                labels.get(str(item.get("gender", "")), str(item.get("gender", ""))),
                str(item.get("volume", 100)),
                str(item.get("speed", 100)),
                str(item.get("bass", 0)),
                str(item.get("mid", 0)),
                str(item.get("treble", 0)),
            ]
            for column, value in enumerate(values):
                self.voice_table.setItem(row, column, QTableWidgetItem(value))

    def load(self, s: Settings) -> None:
        chosen = s.local_voice or s.tts_voice
        self.voice.clear()
        if chosen:
            self.voice.addItem(chosen, chosen)
            self.voice.setCurrentText(chosen)
        self.volume.setValue(s.tts_volume)
        self.rate.setValue(s.tts_speed_percent)
        self._dictionary_text = s.tts_dictionary
        self._pause_period_ms = s.tts_pause_period_ms
        self._pause_comma_ms = s.tts_pause_comma_ms
        self._pause_newline_ms = s.tts_pause_newline_ms
        self.original_volume.setValue(s.original_audio_volume)
        self.bass.setValue(s.eq_bass)
        self.mid.setValue(s.eq_mid)
        self.treble.setValue(s.eq_treble)
        self.mode.setCurrentIndex(0 if s.dub_output_mode == "video" else 1)
        self.store_voice.setChecked(s.tts_store_voice)
        self.allow_overlap.setChecked(s.tts_allow_overlap)
        self.min_gap.setValue(s.tts_short_threshold_ms)
        self.keep_original.setChecked(s.keep_original_audio)
        self.ducking.setChecked(s.ducking)
        self.end_pause.setValue(s.tts_end_pause_ms)
        self.use_original_video.setChecked(s.dub_use_original_video)
        self.render_mode.setCurrentIndex(0 if s.dub_timing_mode == "voice" else 1)
        self.timeline_cpu.setValue(s.timeline_cpu_percent)
        self.timeline_workers.setValue(s.timeline_workers)
        self.timeline_decode.setCurrentText(s.timeline_decode_library)
        language_index = self.language.findData(s.tts_language)
        self.language.setCurrentIndex(language_index if language_index >= 0 else 0)
        self._voice_profiles = [dict(item) for item in s.tts_voice_profiles]
        self._render_voice_profiles()
        self.refresh_status()

    def selected_provider(self) -> str:
        return "Local Voice"

    def refresh_status(self) -> None:
        from ..providers import local_voice as lv

        voice_id = str(self.voice.currentData() or self.voice.currentText() or "").strip()
        runtime_ok, runtime_msg = lv.piper_runtime_ready()
        if not runtime_ok:
            ready = False
            note = runtime_msg
        elif not voice_id:
            ready = False
            note = "Chưa chọn giọng đọc. Vui lòng mở Thư viện giọng để tải và chọn giọng."
        else:
            mgr = lv.get_default_manager()
            status = mgr.get_status(voice_id)
            if status == lv.STATUS_READY:
                ready = True
                note = "Sẵn sàng."
            elif status == lv.STATUS_DOWNLOADING:
                ready = False
                note = f"Giọng đọc '{voice_id}' đang tải về..."
            else:
                ready = False
                note = (
                    f"Giọng đọc '{voice_id}' chưa được tải về. "
                    "Vui lòng mở Thư viện giọng để tải."
                )
        self.btn_dub.setEnabled(ready)
        self.btn_preview.setEnabled(ready)
        if not separate.demucs_available():
            note += " Tách nhạc đang dùng FFmpeg (cơ bản)."
        self.status.setText(note)

    def apply(self, s: Settings) -> None:
        s.tts_provider = self.selected_provider()
        chosen = str(self.voice.currentData() or self.voice.currentText() or "").strip()
        s.local_voice = chosen
        s.tts_voice = chosen
        s.tts_volume = self.volume.value()
        s.tts_speed_percent = self.rate.value()
        s.tts_dictionary = self._dictionary_text
        s.tts_pause_period_ms = self._pause_period_ms
        s.tts_pause_comma_ms = self._pause_comma_ms
        s.tts_pause_newline_ms = self._pause_newline_ms
        s.original_audio_volume = self.original_volume.value()
        s.eq_bass = self.bass.value()
        s.eq_mid = self.mid.value()
        s.eq_treble = self.treble.value()
        s.tts_language = self.language.currentData() or ""
        s.dub_output_mode = "video" if self.mode.currentIndex() == 0 else "audio"
        s.tts_store_voice = self.store_voice.isChecked()
        s.tts_cache_enabled = True
        s.tts_allow_overlap = self.allow_overlap.isChecked()
        s.tts_short_threshold_ms = self.min_gap.value()
        s.tts_end_pause_ms = self.end_pause.value()
        s.dub_timing_mode = "voice" if self.render_mode.currentIndex() == 0 else "subtitle"
        s.timeline_cpu_percent = self.timeline_cpu.value()
        s.timeline_workers = self.timeline_workers.value()
        s.timeline_decode_library = self.timeline_decode.currentText()
        s.tts_fit_timing = s.dub_timing_mode == "subtitle"
        s.tts_voice_profiles = [dict(item) for item in self._voice_profiles]
        s.keep_original_audio = self.keep_original.isChecked()
        s.dub_use_original_video = self.use_original_video.isChecked()
        s.ducking = self.ducking.isChecked()


# =========================================================== B4: Render Va Xuat


class RenderPanel(QWidget):
    """B4 - kieu chu phu de, che mo, render va xuat."""

    runRender = Signal()
    runExport = Signal()
    runBlur = Signal()
    markBlurRegion = Signal()
    chooseLut = Signal()
    exportSubtitle = Signal(str)
    chooseOutputFolder = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._build_style_box(), 1)
        layout.addWidget(self._build_render_box(), 1)
        layout.addWidget(self._build_export_box())

    def _build_style_box(self) -> QGroupBox:
        box = QGroupBox("Kieu Chu Phu De Khi Render")
        self.font_name = QLineEdit()
        self.font_size = QSpinBox()
        self.font_size.setRange(10, 200)
        self.primary = QLineEdit()
        self.outline_color = QLineEdit()
        self.outline = QDoubleSpinBox()
        self.outline.setRange(0.0, 10.0)
        self.outline.setSingleStep(0.5)
        self.back_opacity = QSpinBox()
        self.back_opacity.setRange(0, 100)
        self.margin = QSpinBox()
        self.margin.setRange(0, 500)
        self.bold = QCheckBox("Chu dam")

        form = QFormLayout(box)
        form.setContentsMargins(8, 14, 8, 8)
        form.setSpacing(8)
        form.addRow(field_label("Font:"), self.font_name)
        form.addRow(field_label("Co Chu:"), self.font_size)
        form.addRow(field_label("Mau Chu (#RRGGBB):"), self.primary)
        form.addRow(field_label("Mau Vien:"), self.outline_color)
        form.addRow(field_label("Do Day Vien:"), self.outline)
        form.addRow(field_label("Do Mo Nen (%):"), self.back_opacity)
        form.addRow(field_label("Le Duoi:"), self.margin)
        form.addRow("", self.bold)
        return box

    def _build_render_box(self) -> QGroupBox:
        box = QGroupBox("Cau Hinh Render")
        self.crf = QSpinBox()
        self.crf.setRange(10, 35)
        self.preset = QComboBox()
        self.preset.addItems(["ultrafast", "veryfast", "fast", "medium", "slow"])
        self.lut_label = QLabel("Chua chon LUT")
        self.lut_label.setObjectName("Muted")
        self.lbl_blur = QLabel("Chua khoanh vung che mo")
        self.lbl_blur.setObjectName("Muted")
        self.lbl_blur.setWordWrap(True)

        self.btn_lut = QPushButton("Chon LUT Mau...")
        self.btn_blur_region = QPushButton("Dat Lai Vung Che Mo")
        self.btn_blur_region.setToolTip(
            "Khung che mo hien san khi ban chuyen sang che do Screen Render. "
            "Nut nay dua khung ve vi tri goi y ban dau."
        )
        self.btn_blur = QPushButton("Che Mo Vung Da Khoanh")
        self.btn_render = QPushButton("START: Render Video")

        self.btn_lut.clicked.connect(self.chooseLut)
        self.btn_blur_region.clicked.connect(self.markBlurRegion)
        self.btn_blur.clicked.connect(self.runBlur)
        self.btn_render.clicked.connect(self.runRender)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 14, 8, 8)
        inner.setSpacing(8)
        inner.addLayout(_row("Chat Luong (CRF thap = net):", self.crf, None))
        inner.addLayout(_row("Toc Do Ma Hoa:", self.preset, None))
        inner.addLayout(_row("LUT Mau:", self.lut_label, None))
        inner.addWidget(self.btn_lut)
        inner.addWidget(self.btn_blur_region)
        inner.addWidget(self.lbl_blur)
        inner.addWidget(self.btn_blur)
        inner.addStretch(1)
        inner.addWidget(self.btn_render)
        return box

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("Xuat")
        box.setFixedWidth(230)
        self.btn_srt = QPushButton("Xuat SRT")
        self.btn_ass = QPushButton("Xuat ASS")
        self.btn_vtt = QPushButton("Xuat VTT")
        self.btn_export = QPushButton("Xuat Goi Du An")
        self.btn_output_folder = QPushButton("Thư Mục Xuất...")
        self.btn_output_folder.setToolTip("Chọn thư mục lưu video render thành công")
        self.btn_srt.clicked.connect(lambda: self.exportSubtitle.emit(".srt"))
        self.btn_ass.clicked.connect(lambda: self.exportSubtitle.emit(".ass"))
        self.btn_vtt.clicked.connect(lambda: self.exportSubtitle.emit(".vtt"))
        self.btn_export.clicked.connect(self.runExport)
        self.btn_output_folder.clicked.connect(self.chooseOutputFolder.emit)

        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 14, 8, 8)
        inner.setSpacing(8)
        for button in (
            self.btn_srt,
            self.btn_ass,
            self.btn_vtt,
            self.btn_export,
            self.btn_output_folder,
        ):
            inner.addWidget(button)
        inner.addStretch(1)
        return box

    def load(self, s: Settings) -> None:
        st = s.style
        self.font_name.setText(st.font)
        self.font_size.setValue(st.font_size)
        self.primary.setText(st.primary_color)
        self.outline_color.setText(st.outline_color)
        self.outline.setValue(st.outline)
        self.margin.setValue(st.margin_v)
        self.bold.setChecked(st.bold)
        self.back_opacity.setValue(st.back_opacity)
        self.crf.setValue(s.render_crf)
        self.preset.setCurrentText(s.render_preset)

    def apply(self, s: Settings) -> None:
        st = s.style
        st.font = self.font_name.text().strip() or "Arial"
        st.font_size = self.font_size.value()
        st.primary_color = self.primary.text().strip() or "#FFFFFF"
        st.outline_color = self.outline_color.text().strip() or "#000000"
        st.outline = self.outline.value()
        st.margin_v = self.margin.value()
        st.bold = self.bold.isChecked()
        st.back_opacity = self.back_opacity.value()
        s.render_crf = self.crf.value()
        s.render_preset = self.preset.currentText()


# =========================================================== Cau Hinh Chung


class SettingsPanel(QWidget):
    """Cau hinh chung cua ung dung."""

    openAIGateway = Signal()
    chooseWorkspace = Signal()
    chooseFfmpeg = Signal()
    chooseModelDir = Signal()
    saveRequested = Signal()
    cleanTemp = Signal()
    presetSelected = Signal(str)
    createPresetRequested = Signal(str)
    deletePresetRequested = Signal(str)
    openWorkspace = Signal()
    importProject = Signal()
    exportProject = Signal()
    exportContent = Signal()
    chooseOutputFolder = Signal()
    checkUpdateRequested = Signal()
    applyUpdateRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 4, 5, 5)
        layout.setSpacing(6)
        layout.addWidget(self._build_app_box())
        layout.addWidget(self._build_render_box())
        layout.addWidget(self._build_edit_box())
        layout.addWidget(self._build_update_box())
        layout.addLayout(self._build_bottom_row())

    def _build_render_box(self) -> QGroupBox:
        box = QGroupBox("Cấu Hình Render")
        self.gpu = QCheckBox("Dùng GPU (NVIDIA)")
        self.gpu_encoder = QCheckBox("Nối Video Nhanh")
        self.keep_temp = QCheckBox("Chống Full 100% CPU (tốc độ sẽ chậm hơn)")
        self.scale = QComboBox()
        self.scale.addItems(["giu nguyen", "1920x1080", "1280x720", "854x480"])
        self.fps = QComboBox()
        self.fps.addItems(["giu nguyen", "60", "30", "25", "24"])
        self.gpu_status = QLabel()
        self.gpu_status.setObjectName("Muted")
        self.gpu_status.setWordWrap(True)

        self.add_music = QCheckBox("Thêm Nhạc Nền")
        self.smart_cut = QCheckBox("Cắt Thông Minh")
        grid = QGridLayout(box)
        grid.setContentsMargins(10, 14, 10, 9)
        grid.setHorizontalSpacing(30)
        grid.setVerticalSpacing(13)
        for col, widget in enumerate(
            (
                self.gpu,
                self.keep_temp,
                self.gpu_encoder,
                self.add_music,
                self.smart_cut,
            )
        ):
            grid.addWidget(widget, 0, col)
        self.btn_ai_gateway = QPushButton("AI Gateway  ⚙")
        self.btn_ai_gateway.setObjectName("Flat")
        self.btn_ai_gateway.setEnabled(True)
        self.btn_ai_gateway.setToolTip("Cấu hình kết nối AI Gateway")
        self.btn_ai_gateway.clicked.connect(self.openAIGateway.emit)
        self.btn_export_settings = QPushButton("Cài Đặt Xuất Video  ⚙")
        self.btn_export_settings.setObjectName("Flat")
        self.btn_export_settings.setToolTip("Chọn thư mục lưu video render thành công")
        self.btn_export_settings.clicked.connect(self.chooseOutputFolder.emit)
        grid.addWidget(self.btn_ai_gateway, 1, 0)
        grid.addWidget(self.btn_export_settings, 1, 1)
        grid.setColumnStretch(5, 1)

        for hidden in (self.scale, self.fps, self.gpu_status):
            hidden.setParent(box)
            hidden.hide()
        return box

    def _build_app_box(self) -> QGroupBox:
        box = QGroupBox("Cấu hình APP")
        self.workspace = QLineEdit()
        self.workspace.setReadOnly(True)
        self.ffmpeg = QLineEdit()
        self.ffmpeg.setPlaceholderText("De trong de tu tim trong thu muc ung dung")
        self.model_dir = QLineEdit()
        self.model_dir.setPlaceholderText("De trong de dung model di kem")
        self.workers = QSpinBox()
        self.workers.setRange(1, 8)
        self.workers.setFixedWidth(58)
        self.workers.setToolTip(
            "Số project OCR được chạy cùng lúc; các project còn lại sẽ nằm trong hàng chờ."
        )
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 1440)
        self.timeout.setFixedWidth(80)
        self.autosave = QSpinBox()
        self.autosave.setRange(15, 600)
        self.autosave.setFixedWidth(80)

        self.btn_workspace = QPushButton("Chon...")
        self.btn_ffmpeg = QPushButton("Chon...")
        self.btn_model = QPushButton("Chon...")
        self.btn_save = QPushButton("Tạo Mới")
        self.btn_update = QPushButton("Lưu")
        self.btn_clean = QPushButton("Xóa")
        self.btn_open = QPushButton("Mở Thư Mục Làm Việc")
        self.app_preset = QComboBox()
        self.app_preset.addItem("default")
        self.new_preset_name = QLineEdit()
        self.new_preset_name.setPlaceholderText("Nhập Tên Cấu Hình Mới")
        self.info = QLabel()
        self.info.setObjectName("Muted")
        self.info.setWordWrap(True)

        for button in (self.btn_workspace, self.btn_ffmpeg, self.btn_model):
            button.setFixedWidth(70)
        self.btn_workspace.clicked.connect(self.chooseWorkspace)
        self.btn_ffmpeg.clicked.connect(self.chooseFfmpeg)
        self.btn_model.clicked.connect(self.chooseModelDir)
        self.btn_save.clicked.connect(self._request_create_preset)
        self.btn_update.clicked.connect(self.saveRequested)
        self.btn_clean.clicked.connect(
            lambda: self.deletePresetRequested.emit(self.app_preset.currentText())
        )
        self.btn_open.clicked.connect(self.openWorkspace)

        self.app_preset.currentTextChanged.connect(self.presetSelected)
        inner = QHBoxLayout(box)
        inner.setContentsMargins(8, 13, 8, 8)
        inner.setSpacing(7)
        inner.addWidget(field_label("Cấu Hình Tùy Chỉnh"))
        inner.addWidget(self.app_preset, 1)
        inner.addWidget(field_label("Nhập Tên Cấu Hình Mới"))
        inner.addWidget(self.new_preset_name, 1)
        inner.addWidget(field_label("Project OCR Cùng Lúc"))
        inner.addWidget(self.workers)
        inner.addWidget(self.btn_update)
        inner.addWidget(self.btn_save)
        inner.addWidget(self.btn_clean)

        for hidden in (
            self.workspace,
            self.ffmpeg,
            self.model_dir,
            self.timeout,
            self.autosave,
            self.btn_workspace,
            self.btn_ffmpeg,
            self.btn_model,
            self.btn_open,
            self.info,
        ):
            hidden.setParent(box)
            hidden.hide()
        return box

    def _request_create_preset(self) -> None:
        name = self.new_preset_name.text().strip()
        if name:
            self.createPresetRequested.emit(name)

    def set_config_profiles(self, names: list[str], active: str) -> None:
        cleaned = sorted({str(name).strip() for name in names if str(name).strip()})
        if "default" not in cleaned:
            cleaned.insert(0, "default")
        self.app_preset.blockSignals(True)
        self.app_preset.clear()
        self.app_preset.addItems(cleaned)
        self.app_preset.setCurrentText(active if active in cleaned else "default")
        self.app_preset.blockSignals(False)

    def _build_edit_box(self) -> QGroupBox:
        box = QGroupBox("Cấu Hình Edit Sub")
        self.play_on_edit = QCheckBox("Phát Video Khi Edit")
        self.edit_volume = _slider(0, 100, 50)
        self.edit_volume.setMaximumWidth(140)
        self.edit_volume_value = QLabel("0")
        self.edit_volume.valueChanged.connect(
            lambda value: self.edit_volume_value.setText(str(value))
        )
        self.enter_newline = QCheckBox("Enter Để Qua Dòng Mới")
        self.left_screen = QCheckBox("Màn Hình Bên Trái")
        self.edit_font_size = QSpinBox()
        self.edit_font_size.setRange(8, 120)
        self.edit_color = QLineEdit("#e0e196")
        self.edit_color.setFixedWidth(130)
        grid = QHBoxLayout(box)
        grid.setContentsMargins(10, 13, 10, 8)
        grid.setSpacing(10)
        grid.addWidget(self.play_on_edit)
        grid.addWidget(field_label("Âm Lượng Phát:"))
        grid.addWidget(self.edit_volume)
        grid.addWidget(self.edit_volume_value)
        grid.addWidget(self.enter_newline)
        grid.addWidget(self.left_screen)
        grid.addWidget(field_label("Font Size Sub:"))
        grid.addWidget(self.edit_font_size, 1)
        grid.addWidget(field_label("Màu Chữ Bảng Sub:"))
        grid.addWidget(self.edit_color)
        return box

    def _build_update_box(self) -> QGroupBox:
        box = QGroupBox("Cập Nhật Ứng Dụng")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(10, 10, 10, 8)
        vbox.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        self.lbl_current_version = QLabel(f"Phiên bản hiện tại: {APP_VERSION}")
        self.lbl_channel = QLabel("Kênh: Stable")
        self.lbl_channel.setObjectName("Muted")
        self.auto_check_update = QCheckBox("Tự động kiểm tra bản cập nhật khi khởi động")
        self.btn_check_update = QPushButton("Kiểm Tra Cập Nhật")
        self.btn_check_update.clicked.connect(self.checkUpdateRequested.emit)

        top_row.addWidget(self.lbl_current_version)
        top_row.addWidget(self.lbl_channel)
        top_row.addWidget(self.auto_check_update)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_check_update)
        vbox.addLayout(top_row)

        self.lbl_update_status = QLabel("Chưa kiểm tra")
        self.lbl_update_status.setObjectName("Muted")
        vbox.addWidget(self.lbl_update_status)

        self.update_details = QWidget()
        details_layout = QVBoxLayout(self.update_details)
        details_layout.setContentsMargins(0, 4, 0, 0)
        details_layout.setSpacing(6)

        info_row = QHBoxLayout()
        self.lbl_new_version = QLabel("Phiên bản mới: -")
        self.lbl_new_version.setStyleSheet("font-weight: bold; color: #4CAF50;")
        self.lbl_update_size = QLabel("Dung lượng: -")
        self.lbl_update_size.setObjectName("Muted")
        self.btn_update_now = QPushButton("Cập Nhật Ngay")
        self.btn_update_now.clicked.connect(self._on_update_now_clicked)

        info_row.addWidget(self.lbl_new_version)
        info_row.addWidget(self.lbl_update_size)
        info_row.addStretch(1)
        info_row.addWidget(self.btn_update_now)
        details_layout.addLayout(info_row)

        self.txt_changelog = QTextEdit()
        self.txt_changelog.setReadOnly(True)
        self.txt_changelog.setMaximumHeight(85)
        self.txt_changelog.setPlaceholderText("Thông tin cập nhật / changelog...")
        details_layout.addWidget(self.txt_changelog)

        self.update_progress = QProgressBar()
        self.update_progress.setRange(0, 100)
        self.update_progress.setValue(0)
        self.update_progress.hide()
        details_layout.addWidget(self.update_progress)

        self.update_details.hide()
        vbox.addWidget(self.update_details)
        return box

    def _on_update_now_clicked(self) -> None:
        ver = getattr(self, "_available_version", "")
        reply = QMessageBox.question(
            self,
            "Xác nhận cập nhật",
            f"Bạn có chắc muốn tải về và cài đặt bản cập nhật {ver} không?\n\n"
            "Ứng dụng sẽ tải gói cập nhật, xác thực mã băm SHA256 an toàn và tự động "
            "khởi động lại sau khi hoàn tất.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.applyUpdateRequested.emit()

    def show_update_info(self, release: ReleaseInfo) -> None:
        self._available_version = release.version
        self.lbl_new_version.setText(f"Phiên bản mới: v{release.version}")
        size_str = human_size(release.asset_size) if release.asset_size > 0 else "Không rõ"
        self.lbl_update_size.setText(f"Dung lượng: {size_str}")
        self.txt_changelog.setPlainText(release.changelog or "Không có ghi chú phát hành.")
        self.lbl_update_status.setText(f"Đã tìm thấy bản cập nhật mới v{release.version}!")
        self.btn_update_now.setEnabled(True)
        self.update_progress.hide()
        self.update_details.show()

    def set_update_progress(self, downloaded: int, total: int) -> None:
        self.update_progress.show()
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.update_progress.setValue(percent)
            cur = human_size(downloaded)
            tot = human_size(total)
            self.lbl_update_status.setText(f"Đang tải bản cập nhật: {percent}% ({cur} / {tot})")
        else:
            self.lbl_update_status.setText(f"Đang tải bản cập nhật: {human_size(downloaded)}")

    def _build_bottom_row(self) -> QHBoxLayout:
        self.btn_import_project = QPushButton("Nhập Project")
        self.btn_export_project = QPushButton("Xuất Project")
        self.btn_export_content = QPushButton("Xuất Nội Dung")
        self.btn_import_project.clicked.connect(self.importProject)
        self.btn_export_project.clicked.connect(self.exportProject)
        self.btn_export_content.clicked.connect(self.exportContent)
        return _row(self.btn_import_project, self.btn_export_project, self.btn_export_content)

    def load(self, s: Settings, api_key: str, ffmpeg_version: str, ffmpeg_path: str = "") -> None:
        self.set_config_profiles(list(s.config_profiles), s.active_config_profile)
        self.workspace.setText(s.workspace)
        self.ffmpeg.setText(s.ffmpeg_path)
        self.model_dir.setText(s.model_dir)
        self.workers.setValue(s.max_workers)
        self.timeout.setValue(s.task_timeout_minutes)
        self.autosave.setValue(s.autosave_seconds)
        self.gpu.setChecked(s.use_gpu)
        self.gpu_encoder.setChecked(s.use_gpu_encoder)
        self.keep_temp.setChecked(s.limit_cpu)
        self.add_music.setChecked(s.add_background_music)
        self.smart_cut.setChecked(s.smart_cut)
        self.scale.setCurrentText(s.render_scale)
        self.fps.setCurrentText(s.render_fps)
        self.edit_font_size.setValue(s.style.font_size)
        self.edit_color.setText(s.style.primary_color)
        self.play_on_edit.setChecked(s.play_on_edit)
        self.edit_volume.setValue(s.edit_volume)
        self.enter_newline.setChecked(s.enter_newline)
        self.left_screen.setChecked(s.left_screen)
        self.auto_check_update.setChecked(bool(s.auto_check_update))
        self.gpu_status.setText(
            "Đã kiểm tra cấu hình máy"
            if s.hardware_signature
            else "Chưa kiểm tra cấu hình máy"
        )
        parts = [
            "Nhan dang giong noi: " + ("da cai" if asr.is_available() else "chua cai"),
            "Doc chu tren hinh: " + ("da cai" if ocr.is_available() else "chua cai"),
            "Tach nhac chat luong cao: "
            + ("da cai" if separate.demucs_available() else "chua cai"),
        ]
        self.info.setText(
            (ffmpeg_version or "Chua tim thay FFmpeg.") + "\n" + "   |   ".join(parts)
        )

    def apply(self, s: Settings) -> None:
        s.ffmpeg_path = self.ffmpeg.text().strip()
        s.model_dir = self.model_dir.text().strip()
        s.max_workers = self.workers.value()
        s.task_timeout_minutes = self.timeout.value()
        s.autosave_seconds = self.autosave.value()
        s.use_gpu = self.gpu.isChecked()
        s.use_gpu_encoder = self.gpu_encoder.isChecked()
        s.limit_cpu = self.keep_temp.isChecked()
        s.add_background_music = self.add_music.isChecked()
        s.smart_cut = self.smart_cut.isChecked()
        s.render_scale = self.scale.currentText()
        s.render_fps = self.fps.currentText()
        s.style.font_size = self.edit_font_size.value()
        s.style.primary_color = self.edit_color.text().strip() or "#FFFFFF"
        s.play_on_edit = self.play_on_edit.isChecked()
        s.edit_volume = self.edit_volume.value()
        s.enter_newline = self.enter_newline.isChecked()
        s.left_screen = self.left_screen.isChecked()
        s.auto_check_update = self.auto_check_update.isChecked()


__all__ = [
    "MUTED",
    "DubPanel",
    "RenderPanel",
    "ScriptPanel",
    "SettingsPanel",
    "SubtitlePanel",
    "TranslatePanel",
]
