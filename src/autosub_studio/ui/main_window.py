"""Cua so chinh cua AutoSub Studio."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, APP_VERSION
from ..core import editing, formats
from ..core.models import SubtitleDoc
from ..core.timecode import TimecodeError, format_display, parse_timecode
from ..data.db import AutoScript, Database, Project, TaskLog
from ..data.project import ProjectData, ProjectFormatError, ProjectStore
from ..pipeline import steps as P
from ..providers import download, nts_import
from ..providers.local_voice import (
    STATUS_READY,
    get_default_manager,
    get_voice_info,
    piper_runtime_ready,
    synthesize_piper,
)
from ..services import gpu
from ..services.ffmpeg import CancelToken, FFmpeg, FFmpegError
from ..services.paths import app_root, bundled_dir, ensure_workspace, migrate_v1_models, safe_name
from ..services.presets import PresetManager
from ..services.settings import Settings, SubtitleStyle
from ..services.tasks import CANCELLED, DONE, PENDING, RUNNING, TaskContext, TaskManager
from ..services.updater import (
    ReleaseInfo,
    download_release_asset,
    extract_update_archive,
    fetch_latest_release,
    get_staging_dir,
    launch_updater_helper,
    verify_package_layout,
)
from .cue_table import CueTableModel, CueTableView
from .dialogs import (
    AIGatewayDialog,
    IssueDialog,
    LogDialog,
    ShiftDialog,
    VoiceLibraryDialog,
)
from .panels import (
    DubPanel,
    RenderPanel,
    ScriptPanel,
    SettingsPanel,
    SubtitlePanel,
    TranslatePanel,
)
from .player import MODE_BLUR, MODE_OCR, VideoPlayer, clamp_region
from .project_table import (
    COL_ACTIONS,
    COL_STT,
    ProjectFilterProxy,
    ProjectRow,
    ProjectTableModel,
    ProjectTableView,
)
from .style import BLUE, GREEN, OLIVE, get_qss, init_app_font
from .widgets import (
    ACTION_FOLDER,
    ACTION_RUN,
    ACTION_SRT,
    ACTION_STOP,
    PillTabBar,
    RowActionDelegate,
    VerticalTabStrip,
    field_label,
)


def _small(text: str) -> QPushButton:
    """Nut nho dang vien mo, dung cho cac thao tac bien tap cau."""
    button = QPushButton(text)
    button.setObjectName("Flat")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


VIDEO_FILTER = (
    "Video/Audio (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.ts *.m4v *.mp3 *.wav *.m4a "
    "*.aac *.flac);;Tat ca (*.*)"
)
SUB_FILTER = "Phu de (*.srt *.vtt *.ass *.ssa *.txt);;Tat ca (*.*)"


class _CheckUpdateWorker(QThread):
    finished = Signal(object, str)  # ReleaseInfo | None, error message

    def run(self) -> None:
        try:
            rel = fetch_latest_release(timeout=15.0)
            self.finished.emit(rel, "")
        except Exception as exc:
            self.finished.emit(None, str(exc))


class _DownloadUpdateWorker(QThread):
    progress = Signal(int, int)  # downloaded, total
    finished = Signal(bool, object, str)  # success, payload_dir, error message

    def __init__(self, release: ReleaseInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.release = release
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            staging_dir = get_staging_dir(self.release.version)

            def on_progress(cur: int, tot: int) -> None:
                self.progress.emit(cur, tot)

            def cancel_flag() -> bool:
                return self._cancelled

            zip_path = download_release_asset(
                self.release,
                staging_dir,
                on_progress=on_progress,
                cancel_flag=cancel_flag,
            )
            payload_dir = extract_update_archive(zip_path, staging_dir)
            if not verify_package_layout(payload_dir):
                raise ValueError("Cấu trúc gói cập nhật tải về không hợp lệ.")
            self.finished.emit(True, payload_dir, "")
        except Exception as exc:
            self.finished.emit(False, None, str(exc))


class MainWindow(QMainWindow):
    """Cua so chinh: trinh phat, bang phu de, danh sach du an va cac tab quy trinh."""

    logAppended = Signal(str)
    machineCheckFinished = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        existing = QApplication.instance()
        if isinstance(existing, QApplication):
            init_app_font(existing)
            if not existing.styleSheet():
                existing.setStyleSheet(get_qss())
        self.settings = Settings.load()
        migrated_models = migrate_v1_models()
        ensure_workspace(self.settings.workspace)
        self.db = Database(Path(self.settings.workspace) / "db" / "app.db")
        self.db.ensure_default_scripts()
        interrupted_projects = self.db.recover_interrupted_projects()
        self.store = ProjectStore(self.settings.workspace)
        self.preset_manager = PresetManager()
        stale_ocr_freed = self.store.clean_stale_ocr_temp()
        self.ff = FFmpeg(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        self.tasks = TaskManager(self.settings.max_workers, self)
        self.project: ProjectData | None = None
        self.project_id: int = 0
        self._busy_project: int = 0
        self._logs: dict[str, list[str]] = {}
        self._download_task_ids: set[str] = set()
        self._preview_task_ids: set[str] = set()
        self._preview_player = QMediaPlayer(self)
        self._preview_audio_output = QAudioOutput(self)
        self._preview_player.setAudioOutput(self._preview_audio_output)
        self._session_log: list[str] = []
        if migrated_models:
            names = ", ".join(m.name for m in migrated_models)
            self._session_log.append(
                f"Đã tự động chuyển {len(migrated_models)} mô hình ASR ({names}) "
                "sang Data/models/asr/."
            )
        if stale_ocr_freed:
            self._session_log.append(
                f"Đã tự động dọn {stale_ocr_freed / (1024 * 1024):.1f} MB "
                "ảnh tạm OCR còn sót lại."
            )
        if interrupted_projects:
            self._session_log.append(
                f"Đã phục hồi trạng thái của {interrupted_projects} project bị dừng đột ngột."
            )
        self._dirty = False
        self._editor_history_session: tuple[int, bool] | None = None
        self._voices_loaded = False
        self._render_preview = False
        self._render_subtitles_visible = True
        self._machine_check_running = False
        self._batch_task_ids: set[str] = set()
        self._batch_pending_projects: list[ProjectData] = []
        self._batch_total = 0
        self._batch_done = 0
        self._batch_failed = 0
        self._check_update_worker: _CheckUpdateWorker | None = None
        self._download_update_worker: _DownloadUpdateWorker | None = None
        self._latest_release_info: ReleaseInfo | None = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1500, 940)
        self.setMinimumSize(1180, 760)

        self._build_ui()
        self._connect()
        self._load_settings_into_ui()
        self.reload_projects()
        self._refresh_scripts()
        self._open_last_project()
        self._check_environment()

        if self.settings.auto_check_update:
            QTimer.singleShot(1500, lambda: self._check_for_updates(silent=True))

        self._autosave = QTimer(self)
        self._autosave.timeout.connect(self._autosave_tick)
        self._autosave.start(max(15, self.settings.autosave_seconds) * 1000)

    # ------------------------------------------------------------------ giao dien

    def _build_ui(self) -> None:
        self._build_top()
        self._build_bottom()
        self._assemble()
        self._build_shortcuts()

    # ------------------------------------------------------------------ phan tren

    def _build_top(self) -> None:
        """Trinh phat ben trai, bang phu de va lich su ben phai."""
        self.player = VideoPlayer()
        self.view_strip = VerticalTabStrip(["Screen Edit", "Screen Render"], [GREEN, BLUE])
        self.view_strip.currentChanged.connect(self._on_view_mode)

        self.render_toolbar = QFrame()
        self.render_toolbar.setObjectName("RenderToolbar")
        preset_row = QHBoxLayout(self.render_toolbar)
        preset_row.setContentsMargins(8, 5, 8, 5)
        preset_row.setSpacing(7)
        preset_row.addWidget(field_label("Chọn Preset"))
        self.render_preset_combo = QComboBox()
        self.render_preset_combo.setFixedWidth(185)
        self.render_preset_name = QLineEdit()
        self.render_preset_name.setPlaceholderText("Tên preset mới")
        self.render_preset_name.setFixedWidth(145)
        preset_row.addWidget(self.render_preset_combo)
        preset_row.addWidget(self.render_preset_name)
        self.btn_preset_default = QPushButton("🔗")
        self.btn_preset_default.setObjectName("Flat")
        self.btn_preset_default.setFixedWidth(32)
        self.btn_preset_default.setToolTip("Đặt preset này làm mặc định cho các project mới")
        preset_row.addWidget(self.btn_preset_default)
        self.btn_preset_create = QPushButton("Tạo Mới")
        self.btn_preset_create.setFixedWidth(74)
        self.btn_preset_apply = QPushButton("Áp Dụng")
        self.btn_preset_apply.setFixedWidth(74)
        self.btn_preset_update = QPushButton("Cập Nhật")
        self.btn_preset_update.setFixedWidth(74)
        self.btn_preset_delete = QPushButton("Xóa")
        self.btn_preset_delete.setFixedWidth(74)
        preset_row.addWidget(self.btn_preset_create)
        preset_row.addWidget(self.btn_preset_apply)
        preset_row.addWidget(self.btn_preset_update)
        preset_row.addWidget(self.btn_preset_delete)
        preset_row.addStretch(1)

        self.render_tools = QFrame()
        self.render_tools.setObjectName("RenderTools")
        render_tools_box = QVBoxLayout(self.render_tools)
        render_tools_box.setContentsMargins(4, 5, 4, 5)
        render_tools_box.setSpacing(2)
        self.render_tool_buttons: dict[str, QPushButton] = {}
        for key, icon, tip in (
            ("add", "T+", "Thêm dòng subtitle tại vị trí video"),
            ("bold", "▣", "Bật/tắt chữ đậm"),
            ("larger", "◉", "Tăng cỡ chữ"),
            ("smaller", "◔", "Giảm cỡ chữ"),
            ("subtitles", "CC", "Ẩn/hiện sub dịch"),
            ("up", "☷", "Đưa dòng subtitle lên"),
            ("down", "▤", "Đưa dòng subtitle xuống"),
        ):
            button = QPushButton(icon)
            button.setObjectName("RenderIcon")
            button.setToolTip(tip)
            button.setFixedSize(36, 28)
            if key == "subtitles":
                button.setCheckable(True)
                button.setChecked(True)
            self.render_tool_buttons[key] = button
            render_tools_box.addWidget(button)
        render_tools_box.addStretch(1)
        self.btn_render_reset = QPushButton("16:9\nFHD")
        self.btn_render_reset.setObjectName("Flat")
        self.btn_render_reset.setToolTip("Căn lại khung hình và vị trí subtitle mặc định")
        self.btn_render_reset.setFixedSize(42, 36)
        render_tools_box.addWidget(self.btn_render_reset)

        player_stage = QWidget()
        player_stage_row = QHBoxLayout(player_stage)
        player_stage_row.setContentsMargins(0, 0, 0, 0)
        player_stage_row.setSpacing(3)
        player_stage_row.addWidget(self.player, 1)
        player_stage_row.addWidget(self.render_tools)

        player_column = QVBoxLayout()
        player_column.setContentsMargins(0, 0, 0, 0)
        player_column.setSpacing(3)
        player_column.addWidget(self.render_toolbar)
        player_column.addWidget(player_stage, 1)

        player_wrap = QWidget()
        player_row = QHBoxLayout(player_wrap)
        player_row.setContentsMargins(0, 0, 0, 0)
        player_row.setSpacing(2)
        player_row.addWidget(self.view_strip)
        player_row.addLayout(player_column, 1)
        self.render_toolbar.hide()
        self.render_tools.hide()

        self.cue_model = CueTableModel(self)
        self.cue_view = CueTableView()
        self.cue_view.setModel(self.cue_model)
        self.cue_view.apply_column_widths()
        self.cue_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cue_view.customContextMenuRequested.connect(self._cue_menu)

        self.history = QListWidget()
        history_wrap = QFrame()
        history_wrap.setObjectName("Card")
        history_wrap.setFixedWidth(202)
        history_box = QVBoxLayout(history_wrap)
        history_box.setContentsMargins(0, 0, 0, 0)
        history_box.setSpacing(0)
        history_title = QLabel("Lịch Sử Chỉnh Sửa")
        history_title.setStyleSheet(
            f"background: {OLIVE}; color: #ffffff; font-weight: 700; padding: 5px 8px;"
        )
        history_box.addWidget(history_title)
        history_box.addWidget(self.history, 1)

        table_col = QVBoxLayout()
        table_col.setContentsMargins(0, 0, 0, 0)
        table_col.setSpacing(0)
        table_col.addWidget(self.cue_view, 1)

        # Cac nut bien tap van ton tai de giu phim tat va chuc nang, nhung giao
        # dien mau thao tac truc tiep bang menu chuot phai tren bang phu de.
        self._build_edit_tools()

        table_row = QHBoxLayout()
        table_row.setContentsMargins(0, 0, 0, 0)
        table_row.setSpacing(4)
        table_row.addLayout(table_col, 1)
        table_row.addWidget(history_wrap)

        self._build_sub_controls()
        self._build_text_boxes()

        right_wrap = QWidget()
        right_col = QVBoxLayout(right_wrap)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(5)
        right_col.addLayout(table_row, 1)
        right_col.addLayout(self.sub_controls)
        right_col.addLayout(self.text_row)

        self.top_split = QSplitter(Qt.Orientation.Horizontal)
        self.top_split.setHandleWidth(32)
        self.top_split.setMinimumHeight(420)
        self.top_split.addWidget(player_wrap)
        self.top_split.addWidget(right_wrap)
        self.top_split.setStretchFactor(0, 44)
        self.top_split.setStretchFactor(1, 56)
        self.top_split.setSizes([840, 1080])

    def _build_sub_controls(self) -> None:
        """Hang dieu khien ngay duoi bang phu de."""
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Sub Gốc", "Sub Dịch", "Cả Hai"])
        self.display_mode.setCurrentIndex(0)
        self.display_mode.setFixedWidth(96)

        self.dub_source = QComboBox()
        self.dub_source.addItems(["Sub Gốc", "Sub Dịch"])
        self.dub_source.setCurrentIndex(0)
        self.dub_source.setFixedWidth(96)

        self.start_edit = QLineEdit()
        self.end_edit = QLineEdit()
        for field in (self.start_edit, self.end_edit):
            field.setFixedWidth(122)
            field.setPlaceholderText("00:00:00,000")

        self.line_spin = QSpinBox()
        self.line_spin.setRange(0, 999999)
        self.line_spin.setFixedWidth(80)
        self.line_spin.setToolTip("Nhay toi cau theo so thu tu")

        self.btn_check = QPushButton("Check Lệch Time")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        row.addWidget(field_label("Sub Hiển Thị"))
        row.addWidget(self.display_mode)
        row.addSpacing(8)
        row.addWidget(field_label("Sub Lồng Tiếng"))
        row.addWidget(self.dub_source)
        row.addSpacing(8)
        row.addWidget(field_label("START"))
        row.addWidget(self.start_edit)
        row.addWidget(field_label("END"))
        row.addWidget(self.end_edit)
        row.addSpacing(8)
        row.addWidget(field_label("Số Dòng:"))
        row.addWidget(self.line_spin)
        row.addStretch(1)
        row.addWidget(self.btn_check)
        self.sub_controls = row

    def _build_edit_tools(self) -> None:
        """Hang nut bien tap cau, dat ngay tren bang phu de."""
        self.btn_add = _small("Them Cau")
        self.btn_split = _small("Tach Cau")
        self.btn_merge = _small("Gop Cau")
        self.btn_del = _small("Xoa Cau")
        self.btn_shift = _small("Doi Time")
        self.btn_set_start = _small("Bat Dau = Video")
        self.btn_set_end = _small("Ket Thuc = Video")
        self.btn_undo = _small("Hoan Tac")
        self.btn_redo = _small("Lam Lai")
        self.lbl_count = QLabel("0 cau")
        self.lbl_count.setObjectName("Muted")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for button in (
            self.btn_add,
            self.btn_split,
            self.btn_merge,
            self.btn_del,
            self.btn_shift,
            self.btn_set_start,
            self.btn_set_end,
            self.btn_undo,
            self.btn_redo,
        ):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self.lbl_count)
        self.edit_tools = row

    def _build_text_boxes(self) -> None:
        self.edit_origin = QPlainTextEdit()
        self.edit_origin.setPlaceholderText("Noi dung goc cua cau dang chon")
        self.edit_trans = QPlainTextEdit()
        self.edit_trans.setPlaceholderText("Ban dich cua cau dang chon")
        for box in (self.edit_origin, self.edit_trans):
            box.setFixedHeight(62)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(field_label("TEXT ORIGIN"))
        row.addWidget(self.edit_origin, 1)
        row.addWidget(field_label("TEXT TRANS"))
        row.addWidget(self.edit_trans, 1)
        self.text_row = row

    # ------------------------------------------------------------------ cu (bo)

    def _build_bottom(self) -> None:
        """Hang tab cong viec va vung lam viec ben duoi."""
        self.script_panel = ScriptPanel()
        self.project_model = ProjectTableModel(self)
        self.project_proxy = ProjectFilterProxy(self)
        self.project_proxy.setSourceModel(self.project_model)
        self.project_view = ProjectTableView()
        self.project_view.setModel(self.project_proxy)
        self.row_actions = RowActionDelegate(self.project_view)
        self.row_actions.actionClicked.connect(self._on_row_action)
        self.project_view.setItemDelegateForColumn(COL_ACTIONS, self.row_actions)
        self.project_view.setMouseTracking(True)
        self.project_view.apply_column_widths()

        self.search = QLineEdit()
        self.search.setPlaceholderText("Tim theo ten hoac ID du an...")
        self.search.setFixedWidth(230)
        self.btn_new = QPushButton("Tạo Dự Án Mới")
        self.btn_new.setFixedWidth(150)
        self.btn_batch_ocr = QPushButton("Tách Sub Nhiều Video")
        self.btn_batch_ocr.setObjectName("Blue")
        self.btn_batch_ocr.setFixedWidth(190)
        self.btn_batch_ocr.setToolTip(
            "Chọn nhiều video; tool sẽ tạo project, đưa vào hàng đợi "
            "và tách sub theo giới hạn bên cạnh."
        )
        self.batch_workers = QSpinBox()
        self.batch_workers.setRange(1, 8)
        self.batch_workers.setValue(max(1, min(8, self.settings.max_workers)))
        self.batch_workers.setFixedWidth(58)
        self.batch_workers.setToolTip(
            "Số project được tách sub đồng thời. Các project còn lại sẽ tự động chờ."
        )
        self.batch_queue_status = QLabel("")
        self.batch_queue_status.setObjectName("Empty")
        self.btn_open = QPushButton("Mở Dự Án")
        self.btn_folder = QPushButton("Mở Thư Mục")
        self.btn_stop = QPushButton("Dừng Tác Vụ")
        self.btn_stop.setObjectName("Danger")
        self.btn_log = QPushButton("Xem Nhật Ký")
        self.btn_delete = QPushButton("Xóa Dự Án")
        self.btn_delete.setObjectName("Danger")

        self.empty_projects = QLabel(
            "Chưa có dự án nào. Bấm “Tạo Dự Án Mới” để chọn video."
        )
        self.empty_projects.setObjectName("Empty")
        self.empty_projects.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        project_toolbar = QHBoxLayout()
        project_toolbar.setContentsMargins(2, 0, 2, 0)
        project_toolbar.setSpacing(7)
        project_toolbar.addWidget(self.btn_new)
        project_toolbar.addWidget(self.btn_batch_ocr)
        project_toolbar.addWidget(field_label("Project OCR cùng lúc:"))
        project_toolbar.addWidget(self.batch_workers)
        project_toolbar.addWidget(self.batch_queue_status)
        project_toolbar.addWidget(self.empty_projects)
        project_toolbar.addStretch(1)

        projects_page = QWidget()
        page_col = QVBoxLayout(projects_page)
        page_col.setContentsMargins(6, 5, 6, 6)
        page_col.setSpacing(3)
        page_col.addLayout(project_toolbar)
        page_col.addWidget(self.project_view, 1)

        self.subtitle_panel = SubtitlePanel()
        self.translate_panel = TranslatePanel()
        self.dub_panel = DubPanel()
        self.render_panel = RenderPanel()
        self.settings_panel = SettingsPanel()

        self.tab_bar = PillTabBar(
            [
                "Danh Sách Dự Án",
                "B1: Tách Sub",
                "B2: Dịch Nội Dung",
                "B3: Ghép Giọng Đọc",
                "B4: Render & Xuất Video",
                "Cấu Hình Chung",
            ]
        )
        self.tab_stack = QStackedWidget()
        self._panel_scroll_areas: dict[QWidget, QScrollArea] = {}
        self.tab_stack.addWidget(projects_page)
        for name, panel in (
            ("B1Scroll", self.subtitle_panel),
            ("B2Scroll", self.translate_panel),
            ("B3Scroll", self.dub_panel),
            ("B4Scroll", self.render_panel),
            ("SettingsScroll", self.settings_panel),
        ):
            scroll = QScrollArea()
            scroll.setObjectName(name)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(panel)
            self._panel_scroll_areas[panel] = scroll
            self.tab_stack.addWidget(scroll)

        self.tab_bar.currentChanged.connect(self.tab_stack.setCurrentIndex)
        self.tab_bar.currentChanged.connect(self._on_tab_changed)

        # Nhat ky giao dien da duoc bo theo mau tham chieu. Giu cac dong gan
        # nhat trong bo nho Python thay vi tao mot QPlainTextEdit bi an; widget
        # khong co cha co the bi Qt xoa va gay loi khi tao du an.

    def _assemble(self) -> None:
        """Ghep phan tren, hang tab va phan duoi vao cua so."""
        bottom_wrap = QWidget()
        bottom_col = QVBoxLayout(bottom_wrap)
        bottom_col.setContentsMargins(0, 0, 0, 0)
        bottom_col.setSpacing(2)
        bottom_col.addWidget(self.tab_bar)
        bottom_col.addWidget(self.tab_stack, 1)

        self.main_split = QSplitter(Qt.Orientation.Vertical)
        self.main_split.addWidget(self.top_split)
        self.main_split.addWidget(bottom_wrap)
        self.main_split.setStretchFactor(0, 5)
        self.main_split.setStretchFactor(1, 5)
        self.main_split.setSizes([455, 465])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 5, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(self.main_split)
        self.setCentralWidget(container)

        self.status_project = QLabel("Chua mo du an")
        self.status_task = QLabel("San sang")
        self.statusBar().addWidget(self.status_project, 1)
        self.statusBar().addPermanentWidget(self.status_task)
        self.statusBar().hide()

    def _build_shortcuts(self) -> None:
        def add(text: str, seq: str, slot) -> None:
            action = QAction(text, self)
            action.setShortcut(QKeySequence(seq))
            action.triggered.connect(slot)
            self.addAction(action)

        add("Phat/Dung", "Space", self.player.toggle_play)
        add("Luu du an", "Ctrl+S", self._save_or_project_render)
        add("Tach cau", "Ctrl+K", self._split_cue)
        add("Gop cau", "Ctrl+M", self._merge_cues)
        add("Them cau", "Ctrl+N", self._add_or_clone_project)
        add("Xoa cau", "Ctrl+Del", self._delete_cues)
        self._build_subtitle_shortcuts()
        self._build_project_shortcuts()

    def _build_subtitle_shortcuts(self) -> None:
        """Ctrl+Z/Y rieng cho bang va cac o noi dung subtitle."""
        self._subtitle_shortcut_actions: list[QAction] = []
        for text, sequence, callback in (
            ("Hoàn tác subtitle", "Ctrl+Z", self._undo),
            ("Làm lại subtitle", "Ctrl+Y", self._redo),
        ):
            action = QAction(text, self.cue_view)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.cue_view.addAction(action)
            self._subtitle_shortcut_actions.append(action)
        # QPlainTextEdit co bo undo noi bo nen can chan phim tai day de Ctrl+Z
        # quay lai tai lieu subtitle, khong chi quay lai ky tu trong widget.
        for editor in (self.edit_origin, self.edit_trans):
            editor.installEventFilter(self)

    def _build_project_shortcuts(self) -> None:
        """Dang ky phim tat ben vung project de van dung duoc sau khi dong menu."""
        self._project_shortcut_actions: list[QAction] = []

        def add(text: str, sequence: str, callback) -> None:
            action = QAction(text, self.project_view)
            action.setShortcut(QKeySequence(sequence))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda _checked=False: callback())
            self.project_view.addAction(action)
            self._project_shortcut_actions.append(action)

        # Ctrl+S van dung dispatcher cap cua so. Ctrl+Z cua project la QAction
        # rieng trong bang, tach hoan toan voi Ctrl+Z cua khu subtitle.
        shared = {"Ctrl+S"}
        for label, step in self.MENU_STEPS:
            sequence = self.MENU_SHORTCUTS[label]
            if sequence not in shared:
                add(label, sequence, partial(self._run_selected_project_step, step))
        add("START: CHẠY KỊCH BẢN TỰ ĐỘNG", "Alt+A", self._run_selected_project_script)
        add("STOP", "Ctrl+Q", self._stop_selected_project)
        add("Làm Mới Dữ Liệu", "F5", self.reload_projects)
        add("Xuất Nội Dung Phụ Đề Ra", "F1", self._export_selected_project)
        add("Copy Vị Trí Phụ Đề", "Ctrl+C", self._copy_selected_project_region)
        add("Paste Vị Trí Phụ Đề", "Ctrl+V", self._paste_selected_project_region)
        add("Delete", "Ctrl+D", self._delete_project)

    def _project_table_has_focus(self) -> bool:
        focus = QApplication.focusWidget()
        return focus is not None and (
            focus is self.project_view or self.project_view.isAncestorOf(focus)
        )

    def _save_or_project_render(self) -> None:
        if self._project_table_has_focus():
            self._run_selected_project_step(P.STEP_RENDER)
        else:
            self._save_project()

    def _add_or_clone_project(self) -> None:
        if self._project_table_has_focus():
            row = self._selected_project_row()
            if row is not None:
                self._clone_project(row)
        else:
            self._add_cue()

    def _run_selected_project_step(self, step: str) -> None:
        rows = self._selected_project_rows()
        if step == P.STEP_OCR and len(rows) > 1:
            self._queue_selected_projects_ocr()
        elif rows:
            self._run_step_for(rows[0], step)

    def _run_selected_project_script(self) -> None:
        row = self._selected_project_row()
        if row is not None:
            self._run_script_for(row)

    def _stop_selected_project(self) -> None:
        row = self._selected_project_row()
        if row is not None:
            self._stop_tasks_for(row)

    def _export_selected_project(self) -> None:
        row = self._selected_project_row()
        if row is not None:
            self._export_subtitle_for(row)

    def _copy_selected_project_region(self) -> None:
        row = self._selected_project_row()
        if row is not None:
            self._copy_ocr_region_for(row)

    def _paste_selected_project_region(self) -> None:
        row = self._selected_project_row()
        if row is not None:
            self._paste_ocr_region_for(row)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched in (self.edit_origin, self.edit_trans):
            if event.type() == QEvent.Type.FocusOut:
                self._editor_history_session = None
            elif event.type() == QEvent.Type.KeyPress:
                if event.matches(QKeySequence.StandardKey.Undo):
                    self._undo()
                    return True
                if event.matches(QKeySequence.StandardKey.Redo):
                    self._redo()
                    return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------ ket noi

    def _connect(self) -> None:
        self.player.positionChanged.connect(self._on_position)
        self.player.regionSelected.connect(self._on_region_selected)
        self.player.regionPreview.connect(self._on_region_preview)
        self.player.subtitlePositionChanged.connect(self._set_render_subtitle_margin)
        self.player.btn_prev.clicked.connect(lambda: self._jump_cue(-1))
        self.player.btn_next.clicked.connect(lambda: self._jump_cue(1))

        # Dung `pressed` de dua video toi cau ngay khi bam chuot, truoc khi
        # QTableView mo o nhap. Khong bat `clicked/doubleClicked` o day vi cap
        # nhat video sau khi editor da mo co the lam editor bi dong ngay.
        self.cue_view.pressed.connect(self._on_cue_activated)
        selection = self.cue_view.selectionModel()
        if selection is not None:
            selection.currentRowChanged.connect(self._on_cue_activated)
        self.cue_model.documentChanged.connect(self._on_document_changed)
        self.cue_model.historyChanged.connect(self._refresh_history)

        self.edit_origin.textChanged.connect(lambda: self._apply_editor(False))
        self.edit_trans.textChanged.connect(lambda: self._apply_editor(True))
        self.start_edit.editingFinished.connect(lambda: self._apply_time(True))
        self.end_edit.editingFinished.connect(lambda: self._apply_time(False))
        self.display_mode.currentIndexChanged.connect(
            lambda: self._on_position(self.player.position)
        )

        self.dub_source.currentIndexChanged.connect(self._on_dub_source)
        self.line_spin.valueChanged.connect(self._jump_to_line)
        self.btn_check.clicked.connect(self._check_timing)
        self.btn_split.clicked.connect(self._split_cue)
        self.btn_merge.clicked.connect(self._merge_cues)
        self.btn_add.clicked.connect(self._add_cue)
        self.btn_del.clicked.connect(self._delete_cues)
        self.btn_shift.clicked.connect(self._shift_cues)
        self.btn_set_start.clicked.connect(lambda: self._set_time_from_player(True))
        self.btn_set_end.clicked.connect(lambda: self._set_time_from_player(False))
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo.clicked.connect(self._redo)

        self.search.textChanged.connect(self.project_proxy.set_keyword)
        self.btn_new.clicked.connect(self._create_project)
        self.btn_batch_ocr.clicked.connect(self._choose_batch_videos)
        self.batch_workers.valueChanged.connect(self._set_batch_worker_limit)
        self.btn_open.clicked.connect(self._open_selected_project)
        # Mot lan bam vao dong la mo ngay project. Truoc day chi nghe doubleClicked
        # nen lan dau chi doi lua chon, lan thu hai moi nap project.
        self.project_view.clicked.connect(self._open_project_from_index)
        self.project_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.project_view.customContextMenuRequested.connect(self._project_menu)
        self.btn_folder.clicked.connect(self._open_project_folder)
        self.btn_stop.clicked.connect(self._stop_tasks)
        self.btn_delete.clicked.connect(self._delete_project)
        self.btn_log.clicked.connect(self._show_project_log)

        self.render_tool_buttons["add"].clicked.connect(self._add_render_subtitle)
        self.render_tool_buttons["bold"].clicked.connect(self._toggle_render_bold)
        self.render_tool_buttons["larger"].clicked.connect(lambda: self._change_render_font(2))
        self.render_tool_buttons["smaller"].clicked.connect(
            lambda: self._change_render_font(-2)
        )
        self.render_tool_buttons["subtitles"].toggled.connect(
            self._toggle_render_subtitles
        )
        self.render_tool_buttons["up"].clicked.connect(lambda: self._move_render_subtitle(20))
        self.render_tool_buttons["down"].clicked.connect(
            lambda: self._move_render_subtitle(-20)
        )
        self.btn_render_reset.clicked.connect(self._reset_render_preview)
        self.btn_preset_default.clicked.connect(self._set_default_render_preset)
        self.btn_preset_create.clicked.connect(self._create_render_preset)
        self.btn_preset_apply.clicked.connect(self._apply_render_preset)
        self.btn_preset_update.clicked.connect(self._update_render_preset)
        self.btn_preset_delete.clicked.connect(self._delete_render_preset)

        self.script_panel.runRequested.connect(self._run_script)
        self.script_panel.scriptSaved.connect(self._save_script)
        self.script_panel.scriptDeleted.connect(self._delete_script)
        self.script_panel.scriptSelected.connect(self._select_script)

        self.subtitle_panel.chooseVideo.connect(self._choose_video)
        self.subtitle_panel.importSubtitle.connect(self._import_subtitle)
        self.subtitle_panel.downloadVideo.connect(self._download_video)
        self.subtitle_panel.runAsr.connect(lambda: self._run_step(P.STEP_ASR))
        self.subtitle_panel.runOcr.connect(lambda: self._run_step(P.STEP_OCR))
        self.subtitle_panel.markOcrRegion.connect(lambda: self._reset_region(MODE_OCR))
        self.subtitle_panel.measureOcr.connect(lambda: self._run_step(P.STEP_OCR_MEASURE))
        self.subtitle_panel.checkMachine.connect(self._check_machine_configuration)

        self.translate_panel.runTranslate.connect(lambda: self._run_step(P.STEP_TRANSLATE))
        self.translate_panel.translateSelected.connect(self._translate_selected)
        self.translate_panel.importTranslation.connect(self._import_translation)

        self.dub_panel.runDub.connect(self._start_dub)
        self.dub_panel.separateAudio.connect(lambda: self._run_step(P.STEP_KEEP_VOICE))
        self.dub_panel.previewVoice.connect(self._preview_voice)
        self.dub_panel.openVoiceLibrary.connect(self._open_voice_library)
        self.dub_panel.runDiarize.connect(lambda: self._run_step(P.STEP_DIARIZE))

        self.render_panel.runRender.connect(lambda: self._run_step(P.STEP_RENDER))
        self.render_panel.runExport.connect(lambda: self._run_step(P.STEP_EXPORT))
        self.render_panel.runBlur.connect(lambda: self._run_step(P.STEP_BLUR))
        self.render_panel.markBlurRegion.connect(lambda: self._reset_region(MODE_BLUR))
        self.render_panel.chooseLut.connect(self._choose_lut)
        self.render_panel.exportSubtitle.connect(self._export_subtitle)
        self.render_panel.chooseOutputFolder.connect(self._choose_output_folder)

        self.settings_panel.chooseWorkspace.connect(self._choose_workspace)
        self.settings_panel.openAIGateway.connect(self._open_ai_gateway_dialog)
        self.settings_panel.chooseFfmpeg.connect(self._choose_ffmpeg)
        self.settings_panel.chooseModelDir.connect(self._choose_model_dir)
        self.settings_panel.saveRequested.connect(self._save_settings)
        self.settings_panel.cleanTemp.connect(self._clean_temp)
        self.settings_panel.presetSelected.connect(self._switch_config_profile)
        self.settings_panel.createPresetRequested.connect(self._create_config_profile)
        self.settings_panel.deletePresetRequested.connect(self._delete_config_profile)
        self.settings_panel.openWorkspace.connect(lambda: self._open_path(self.settings.workspace))
        self.settings_panel.importProject.connect(self._import_project_package)
        self.settings_panel.exportProject.connect(lambda: self._run_step(P.STEP_EXPORT))
        self.settings_panel.exportContent.connect(lambda: self._export_subtitle(".srt"))
        self.settings_panel.edit_volume.valueChanged.connect(self.player.volume.setValue)
        self.settings_panel.chooseOutputFolder.connect(self._choose_output_folder)
        self.settings_panel.checkUpdateRequested.connect(
            lambda: self._check_for_updates(silent=False)
        )
        self.settings_panel.applyUpdateRequested.connect(self._start_update_download)

        self.tasks.task_started.connect(self._on_task_started)
        self.tasks.task_progress.connect(self._on_task_progress)
        self.tasks.task_log.connect(self._on_task_log)
        self.tasks.task_finished.connect(self._on_task_finished)
        self.tasks.queue_changed.connect(self._update_batch_queue_status)
        self.logAppended.connect(self._append_log)
        self.machineCheckFinished.connect(self._apply_machine_check)

    # ------------------------------------------------------------------ cai dat

    def _open_ai_gateway_dialog(self) -> None:
        dialog = AIGatewayDialog(self.settings, self)
        if dialog.exec():
            self._load_settings_into_ui()
            self._log("Đã lưu cấu hình AI Gateway.")

    # ------------------------------------------------------------------ cap nhat

    def _check_for_updates(self, silent: bool = False) -> None:
        if self._check_update_worker and self._check_update_worker.isRunning():
            return
        if not silent:
            self.settings_panel.lbl_update_status.setText("Đang kiểm tra bản cập nhật mới...")
            self.settings_panel.btn_check_update.setEnabled(False)

        self._check_update_worker = _CheckUpdateWorker(self)
        self._check_update_worker.finished.connect(
            lambda rel, err: self._on_check_update_finished(rel, err, silent)
        )
        self._check_update_worker.start()

    def _on_check_update_finished(
        self, release: ReleaseInfo | None, error: str, silent: bool
    ) -> None:
        self.settings_panel.btn_check_update.setEnabled(True)
        if error:
            if not silent:
                self.settings_panel.lbl_update_status.setText(f"Lỗi kiểm tra cập nhật: {error}")
            return

        if not release:
            if not silent:
                self.settings_panel.lbl_update_status.setText(
                    "Không tìm thấy bản cập nhật mới trên kênh Stable."
                )
            return

        if release.is_newer:
            self._latest_release_info = release
            self.settings_panel.show_update_info(release)
            if not silent:
                self._log(f"Đã tìm thấy bản cập nhật mới: v{release.version}")
        else:
            self.settings_panel.lbl_update_status.setText(
                f"Bạn đang sử dụng phiên bản mới nhất ({APP_VERSION})."
            )
            self.settings_panel.update_details.hide()

    def _start_update_download(self) -> None:
        if not self._latest_release_info:
            return
        if self._download_update_worker and self._download_update_worker.isRunning():
            return

        self.settings_panel.btn_update_now.setEnabled(False)
        self.settings_panel.lbl_update_status.setText("Đang chuẩn bị tải bản cập nhật...")
        self.settings_panel.update_progress.show()
        self.settings_panel.update_progress.setValue(0)

        self._download_update_worker = _DownloadUpdateWorker(self._latest_release_info, self)
        self._download_update_worker.progress.connect(self._on_download_update_progress)
        self._download_update_worker.finished.connect(self._on_download_update_finished)
        self._download_update_worker.start()

    def _on_download_update_progress(self, downloaded: int, total: int) -> None:
        self.settings_panel.set_update_progress(downloaded, total)

    def _on_download_update_finished(self, success: bool, payload_dir: Any, error: str) -> None:
        if not success or not payload_dir:
            self.settings_panel.lbl_update_status.setText(f"Tải cập nhật thất bại: {error}")
            self.settings_panel.btn_update_now.setEnabled(True)
            self.settings_panel.update_progress.hide()
            QMessageBox.critical(self, "Lỗi tải cập nhật", f"Không thể tải bản cập nhật:\n{error}")
            return

        self.settings_panel.lbl_update_status.setText(
            "Đã tải xong. Khởi chạy cập nhật và đóng ứng dụng..."
        )
        try:
            launch_updater_helper(app_root(), Path(payload_dir), restart=True)
            QApplication.quit()
        except Exception as exc:
            self.settings_panel.lbl_update_status.setText(f"Lỗi khởi chạy cập nhật: {exc}")
            self.settings_panel.btn_update_now.setEnabled(True)
            QMessageBox.critical(self, "Lỗi cập nhật", f"Không thể khởi chạy cập nhật:\n{exc}")

    def _load_settings_into_ui(self) -> None:
        api_key = Settings.get_secret("ai_gateway_key")
        self.subtitle_panel.load(self.settings)
        self.translate_panel.load(self.settings, api_key)
        self.dub_panel.load(self.settings)
        self.render_panel.load(self.settings)
        self.settings_panel.load(self.settings, api_key, self.ff.version(), self.ff.ffmpeg)
        self.dub_source.blockSignals(True)
        self.dub_source.setCurrentIndex(0 if self.settings.dub_source == "original" else 1)
        self.dub_source.blockSignals(False)
        self.batch_workers.blockSignals(True)
        self.batch_workers.setValue(max(1, min(8, self.settings.max_workers)))
        self.batch_workers.blockSignals(False)
        self.player.set_subtitle_style(self.settings.style)
        self.player.volume.setValue(max(0, min(100, self.settings.edit_volume)))
        self._refresh_preset_combo()

    def _set_batch_worker_limit(self, count: int) -> None:
        """Ap dung ngay gioi han project OCR chay dong thoi va luu cho lan sau."""
        limit = max(1, min(8, int(count)))
        self.settings.max_workers = limit
        self.tasks.set_max_workers(limit)
        self.settings_panel.workers.setValue(limit)
        self.settings.save()
        self._start_next_batch_projects()
        self._update_batch_queue_status()
        self.statusBar().showMessage(
            f"Tối đa {limit} project OCR chạy cùng lúc; các project còn lại sẽ chờ.", 5000
        )

    def _save_settings(self) -> None:
        self._collect_settings()
        self.settings.remember_active_profile()
        self.settings.save()
        self.ff = FFmpeg(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        self.tasks.set_max_workers(self.settings.max_workers)
        self.player.set_subtitle_style(self.settings.style)
        self._autosave.setInterval(max(15, self.settings.autosave_seconds) * 1000)
        self._load_settings_into_ui()
        self._log("Da luu cai dat.")
        self.statusBar().showMessage("Da luu cai dat.", 4000)

    def _collect_settings(self) -> None:
        """Lay gia tri tren giao dien vao doi tuong cai dat truoc khi chay tac vu."""
        self.subtitle_panel.apply(self.settings)
        self.translate_panel.apply(self.settings)
        self.dub_panel.apply(self.settings)
        self.render_panel.apply(self.settings)
        self.settings_panel.apply(self.settings)

    def _switch_config_profile(self, name: str) -> None:
        target = str(name).strip()
        if not target or target == self.settings.active_config_profile:
            return
        self._collect_settings()
        self.settings.remember_active_profile()
        if not self.settings.apply_config_profile(target):
            return
        self.settings.save()
        self.ff = FFmpeg(self.settings.ffmpeg_path, self.settings.ffprobe_path)
        self.tasks.set_max_workers(self.settings.max_workers)
        self._load_settings_into_ui()
        self._log(f"Da ap dung cau hinh: {target}.")
        self.statusBar().showMessage(f"Đã áp dụng cấu hình {target}.", 4000)

    def _create_config_profile(self, name: str) -> None:
        target = str(name).strip()
        if not target:
            return
        if any(
            existing.casefold() == target.casefold()
            for existing in self.settings.config_profiles
        ):
            QMessageBox.information(self, "Cấu Hình Đã Có", f"Đã tồn tại cấu hình '{target}'.")
            return
        self._collect_settings()
        self.settings.remember_active_profile()
        self.settings.active_config_profile = target
        self.settings.remember_active_profile()
        self.settings.save()
        self.settings_panel.new_preset_name.clear()
        self._load_settings_into_ui()
        self._log(f"Da tao cau hinh: {target}.")

    def _delete_config_profile(self, name: str) -> None:
        target = str(name).strip()
        if not target or target == "default":
            QMessageBox.information(self, "Không Thể Xóa", "Cấu hình default luôn được giữ lại.")
            return
        if target not in self.settings.config_profiles:
            return
        answer = QMessageBox.question(
            self,
            "Xóa Cấu Hình",
            f"Xóa cấu hình '{target}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        del self.settings.config_profiles[target]
        self.settings.apply_config_profile("default")
        self.settings.save()
        self._load_settings_into_ui()
        self._log(f"Da xoa cau hinh: {target}.")

    def _check_environment(self) -> None:
        if not self.ff.available:
            QMessageBox.warning(
                self,
                "Chua co FFmpeg",
                "Khong tim thay FFmpeg tren may. Hau het cac chuc nang can no.\n\n"
                "Cach xu ly: vao tab 'Cai dat chung', bam 'Chon ffmpeg.exe...' "
                "va tro toi tep ffmpeg.exe da tai ve.",
            )

    def _check_machine_configuration(self) -> None:
        """Kiem tra theo yeu cau tu nut trong B1 va luu lua chon tang toc."""
        self._start_machine_check(notify=True)

    def _start_machine_check(self, *, notify: bool) -> None:
        if self._machine_check_running:
            return
        self._machine_check_running = True
        self.subtitle_panel.set_machine_checking(True)
        ffmpeg_path = self.ff.ffmpeg

        def worker() -> None:
            try:
                gpu.refresh_detection()
                payload = {
                    "notify": notify,
                    "profile": gpu.acceleration_profile(ffmpeg_path),
                }
            except Exception as exc:  # bao loi tren UI, khong lam sap ung dung
                payload = {"error": str(exc), "notify": notify}
            self.machineCheckFinished.emit(payload)

        threading.Thread(target=worker, name="machine-check", daemon=True).start()

    def _apply_machine_check(self, payload: object) -> None:
        self._machine_check_running = False
        data = payload if isinstance(payload, dict) else {"error": "Kết quả không hợp lệ"}
        error = str(data.get("error", ""))
        if error:
            self.settings.use_gpu = False
            self.settings.use_gpu_encoder = False
            self.settings.hardware_signature = ""
            self.settings.hardware_machine_id = gpu.machine_id()
            self.settings.save()
            self.subtitle_panel.device.setCurrentIndex(1)
            self.settings_panel.gpu.setChecked(False)
            self.settings_panel.gpu_encoder.setChecked(False)
            self.subtitle_panel.set_machine_result(f"Không kiểm tra được: {error}", False)
            self.subtitle_panel.gpu_note.setText(
                "Không kiểm tra được GPU; tool sẽ dùng CPU để tránh áp dụng sai cấu hình."
            )
            if data.get("notify"):
                QMessageBox.warning(self, "Kiểm Tra Cấu Hình Máy", error)
            return

        profile = data.get("profile")
        if not isinstance(profile, gpu.AccelerationProfile):
            self.subtitle_panel.set_machine_result("Không nhận được kết quả kiểm tra.", False)
            return

        self.settings.use_gpu = profile.use_gpu
        self.settings.use_gpu_encoder = profile.use_gpu_encoder
        self.settings.hardware_signature = profile.signature
        self.settings.hardware_machine_id = gpu.machine_id()
        self.settings.save()

        self.subtitle_panel.device.setCurrentIndex(0 if profile.use_gpu else 1)
        self.settings_panel.gpu.setChecked(profile.use_gpu)
        self.settings_panel.gpu_encoder.setChecked(profile.use_gpu_encoder)
        accelerated = profile.use_gpu or profile.use_gpu_encoder
        self.subtitle_panel.set_machine_result(profile.summary(), accelerated)
        self.subtitle_panel.gpu_note.setText(
            "OCR: dùng GPU NVIDIA (CUDA)."
            if profile.ocr_cuda_ready
            else f"OCR: chạy CPU. {profile.ocr_cuda_reason}."
        )
        self.settings_panel.gpu_status.setText(profile.summary())
        self._log("Kiểm tra cấu hình máy: " + profile.summary().replace("\n", " | "))

        if data.get("notify"):
            detail = profile.summary()
            if not profile.cuda_ready:
                detail += f"\n\nGhi chú CUDA: {profile.cuda_reason}"
            QMessageBox.information(self, "Kiểm Tra Cấu Hình Máy", detail)

    # ------------------------------------------------------------------ du an

    def reload_projects(self) -> None:
        rows: list[ProjectRow] = []
        with self.db.session() as s:
            for p in s.query(Project).order_by(Project.id.desc()).all():
                old_folder = p.folder
                current_folder = self.store.resolve_project_folder(old_folder)
                if current_folder.is_dir() and Path(old_folder) != current_folder:
                    p.folder = str(current_folder)
                    p.video_path = self.store.relocate_project_path(
                        p.video_path, old_folder, current_folder
                    )
                rows.append(
                    ProjectRow(
                        id=p.id,
                        name=p.name,
                        folder=p.folder,
                        video_path=p.video_path,
                        language=p.language,
                        target_language=p.target_language,
                        cue_count=p.cue_count,
                        char_count=p.char_count,
                        has_subtitle=p.has_subtitle,
                        has_translation=p.has_translation,
                        has_dub=p.has_dub,
                        has_render=p.has_render,
                        exported=p.exported,
                        current_task=p.current_task,
                        progress=p.progress,
                        status=p.status,
                        duration=p.duration,
                    )
                )
        self.project_model.set_rows(rows)
        self.project_view.apply_column_widths()
        self.empty_projects.setVisible(not rows)
        # Luon hien bang va tieu de cot, ke ca khi chua co du an. Neu an bang
        # luc rong, trang Danh Sach Du An chi con mot khoang trong kho hieu.
        self.project_view.setVisible(True)

    def _open_last_project(self) -> None:
        """Mo lai du an moi nhat de nguoi dung khong phai tim lai tu dau."""
        rows = self.project_model.rows()
        if not rows:
            return
        target = rows[0]
        if target.folder and Path(target.folder).is_dir():
            self._load_project(target.id, target.folder)
            self.project_view.selectRow(0)

    def _selected_project_row(self) -> ProjectRow | None:
        indexes = self.project_view.selectionModel().selectedRows()
        if not indexes:
            return None
        source = self.project_proxy.mapToSource(indexes[0])
        return self.project_model.row_at(source.row())

    def _selected_project_rows(self) -> list[ProjectRow]:
        """Tra project dang boi den theo thu tu duoi len: project cu chay truoc."""
        rows: list[ProjectRow] = []
        seen: set[int] = set()
        indexes = sorted(
            self.project_view.selectionModel().selectedRows(),
            key=lambda index: index.row(),
            reverse=True,
        )
        for index in indexes:
            source = self.project_proxy.mapToSource(index)
            row = self.project_model.row_at(source.row())
            if row is not None and row.id not in seen:
                rows.append(row)
                seen.add(row.id)
        return rows

    def _create_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chon video cho du an moi", "", VIDEO_FILTER)
        if not path:
            return
        try:
            data = self._create_project_from_video(path)
        except (FFmpegError, OSError) as exc:
            QMessageBox.critical(self, "Khong doc duoc video", str(exc))
            return
        self.reload_projects()
        self._load_project(data.project_id, data.folder)

    def _create_project_from_video(
        self,
        path: str,
        *,
        region_template: list[int] | None = None,
        template_size: tuple[int, int] = (0, 0),
    ) -> ProjectData:
        """Tao project hoan chinh tu video ma khong can mo no tren trinh phat."""
        info = self.ff.probe(path)
        name = Path(path).stem
        with self.db.session() as s:
            created = Project(
                name=name,
                status="Mới tạo",
                video_path=path,
                duration=info.duration,
                language=self.settings.ocr_language,
                target_language=self.settings.target_language,
            )
            s.add(created)
            s.flush()
            project_id = created.id
        try:
            data = self.store.create(project_id, name)
        except Exception:
            with self.db.session() as s:
                stored = s.get(Project, project_id)
                if stored is not None:
                    s.delete(stored)
            raise
        data.video_path = path
        data.original_video = path
        data.duration = info.duration
        data.width = info.width
        data.height = info.height
        data.doc.language = self.settings.ocr_language
        data.doc.target_language = self.settings.target_language
        default_preset_name = self.settings.default_preset or "DEFAULT"
        preset = self.preset_manager.get(default_preset_name)
        data.render_preset = preset.name
        data.render_preset_snapshot = preset.to_dict()
        if preset.lut_path and not data.lut_path:
            data.lut_path = preset.lut_path
        data.ocr_region = self._scaled_batch_region(
            region_template or [], template_size, info.width, info.height
        )
        self.store.save(data)
        with self.db.session() as s:
            stored = s.get(Project, project_id)
            if stored is not None:
                stored.folder = data.folder
                stored.video_path = path
                stored.duration = info.duration
        return data

    @staticmethod
    def _scaled_batch_region(
        region: list[int], source_size: tuple[int, int], width: int, height: int
    ) -> list[int]:
        """Co gian vung OCR mau; neu chua co thi lay vung phu de phia duoi video."""
        if width <= 0 or height <= 0:
            return []
        source_w, source_h = source_size
        if len(region) == 4 and source_w > 0 and source_h > 0:
            scaled = [
                round(region[0] * width / source_w),
                round(region[1] * height / source_h),
                round(region[2] * width / source_w),
                round(region[3] * height / source_h),
            ]
        else:
            # Du rong de bat duoc ca phu de mot hoac hai dong, nhung tranh logo o tren.
            scaled = [
                round(width * 0.05),
                round(height * 0.70),
                round(width * 0.90),
                round(height * 0.28),
            ]
        return clamp_region(scaled, width, height)

    def _choose_batch_videos(self) -> None:
        """Chon nhieu video, tao project va dua ngay vao hang doi OCR."""
        if not self.ff.available:
            QMessageBox.critical(
                self, "Thiếu FFmpeg", "Chưa tìm thấy FFmpeg nên chưa thể tách sub."
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Chọn nhiều video để tách sub", "", VIDEO_FILTER
        )
        if not paths:
            return
        self._collect_settings()
        template = list(self.project.ocr_region) if self.project is not None else []
        template_size = (
            (self.project.width, self.project.height) if self.project is not None else (0, 0)
        )
        made: list[ProjectData] = []
        errors: list[str] = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for path in paths:
                QApplication.processEvents()
                try:
                    made.append(
                        self._create_project_from_video(
                            path, region_template=template, template_size=template_size
                        )
                    )
                except (FFmpegError, OSError, ValueError) as exc:
                    errors.append(f"{Path(path).name}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self.reload_projects()
        if made:
            self._queue_batch_projects(made)
        if errors:
            QMessageBox.warning(
                self,
                "Một số video không thêm được",
                "\n".join(errors[:12]),
            )

    def _queue_selected_projects_ocr(self) -> None:
        """Tach sub cho nhieu dong project da boi den trong bang."""
        rows = self._selected_project_rows()
        if not rows:
            return
        if self.project is not None and self.project_id in {row.id for row in rows} and self._dirty:
            self._save_project()
        projects: list[ProjectData] = []
        errors: list[str] = []
        pending_ids = {data.project_id for data in self._batch_pending_projects}
        for row in rows:
            if self.tasks.running_for_project(row.id) is not None or row.id in pending_ids:
                continue
            try:
                data = self.store.load(row.folder)
            except ProjectFormatError as exc:
                errors.append(f"{row.name}: {exc}")
                continue
            if not Path(data.video_path or data.original_video).is_file():
                errors.append(f"{row.name}: không tìm thấy video")
                continue
            if len(data.ocr_region) != 4:
                data.ocr_region = self._scaled_batch_region(
                    [], (0, 0), data.width, data.height
                )
                self.store.save(data)
            projects.append(data)
        if projects:
            self._collect_settings()
            self._queue_batch_projects(projects)
        if errors:
            QMessageBox.warning(self, "Không thể xếp hàng", "\n".join(errors[:12]))

    def _queue_batch_projects(self, projects: list[ProjectData]) -> None:
        """Noi project vao hang FIFO; chi dua dung so luong cho phep sang QThreadPool."""
        if not projects:
            return
        if not self._batch_task_ids and not self._batch_pending_projects:
            self._batch_total = 0
            self._batch_done = 0
            self._batch_failed = 0
        active_ids = {
            record.project_id
            for task_id in self._batch_task_ids
            if (record := self.tasks.record(task_id)) is not None
        }
        pending_ids = {data.project_id for data in self._batch_pending_projects}
        for data in projects:
            if data.project_id in active_ids or data.project_id in pending_ids:
                continue
            self._batch_pending_projects.append(data)
            pending_ids.add(data.project_id)
            self._batch_total += 1
            self._set_project_status(
                data.project_id,
                task="Tách Sub Bằng Chữ",
                status="Đang đợi...",
                progress=0,
            )
        self._start_next_batch_projects()
        self._update_batch_queue_status()

    def _start_next_batch_projects(self) -> None:
        """Lay tu dau hang FIFO cho den khi du so project OCR dang hoat dong."""
        limit = max(1, min(8, int(self.settings.max_workers)))
        self.tasks.set_max_workers(limit)
        while self._batch_pending_projects and len(self._batch_task_ids) < limit:
            data = self._batch_pending_projects.pop(0)
            if self.tasks.running_for_project(data.project_id) is not None:
                continue
            self._queue_batch_project(data)

    def _queue_batch_project(self, data: ProjectData) -> None:
        project_id = data.project_id
        settings = copy.deepcopy(self.settings)
        ff = FFmpeg(settings.ffmpeg_path, settings.ffprobe_path)
        store = ProjectStore(settings.workspace)

        def job(ctx: TaskContext) -> str:
            project = store.load(data.folder)
            pc = P.PipelineContext(
                ff=ff,
                settings=settings,
                store=store,
                project=project,
                task=ctx,
                api_key=Settings.get_secret("ai_gateway_key"),
                glossary={},
            )
            return P.run_script([P.STEP_OCR], pc)

        task_id = self.tasks.submit(
            P.STEP_OCR,
            job,
            project_id=project_id,
            timeout=max(60, settings.task_timeout_minutes * 60),
        )
        self._logs[task_id] = []
        self._batch_task_ids.add(task_id)

    def _remove_pending_batch_projects(self, project_ids: set[int]) -> int:
        """Bo project chua chay khoi hang doi, dung khi STOP hoac xoa project."""
        removed = [
            data for data in self._batch_pending_projects if data.project_id in project_ids
        ]
        if not removed:
            return 0
        self._batch_pending_projects = [
            data for data in self._batch_pending_projects if data.project_id not in project_ids
        ]
        self._batch_failed += len(removed)
        for data in removed:
            self._set_project_status(
                data.project_id, task="", status="Đã hủy", progress=0
            )
        self._update_batch_queue_status()
        return len(removed)

    def _open_selected_project(self) -> None:
        row = self._selected_project_row()
        if row is None:
            QMessageBox.information(
                self, "Chua chon du an", "Hay bam chon mot dong trong bang du an truoc."
            )
            return
        self._load_project(row.id, row.folder)

    def _open_project_from_index(self, index) -> None:
        """Mo dung dong vua bam, ke ca khi dang chon nhieu dong qua Ctrl/Shift."""
        if not index.isValid():
            return
        source = self.project_proxy.mapToSource(index.siblingAtColumn(COL_STT))
        row = self.project_model.row_at(source.row())
        if row is not None and self.project_id != row.id:
            self._load_project(row.id, row.folder)

    def _load_project(self, project_id: int, folder: str) -> None:
        if self._dirty and self.project is not None:
            self._save_project()
        try:
            data = self.store.load(folder)
        except ProjectFormatError as exc:
            QMessageBox.critical(self, "Khong mo duoc du an", str(exc))
            return
        folder = data.folder
        recovery = self.store.recovery_path(folder)
        if recovery is not None:
            answer = QMessageBox.question(
                self,
                "Co ban phuc hoi",
                "Du an nay co ban tu luu chua duoc ghi. Ban co muon khoi phuc khong?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    import json

                    data = ProjectData.from_dict(json.loads(recovery.read_text(encoding="utf-8")))
                    data.folder = folder
                except (OSError, ValueError, ProjectFormatError) as exc:
                    QMessageBox.warning(self, "Khong doc duoc ban phuc hoi", str(exc))
            self.store.clear_recovery(data)

        self.project = data
        self.project_id = project_id
        self.cue_model.set_document(data.doc)
        self.player.clear()
        if data.video_path and Path(data.video_path).is_file():
            self._apply_frame_size(data)
            self.player.load(data.video_path)
        else:
            self.player.show_region(None)
        if self.view_strip.current() == 1:
            self.player.hide_region()
        else:
            self._show_region_editor(MODE_OCR)
        if len(data.ocr_region) == 4:
            self._update_region_label(MODE_OCR, *data.ocr_region)
        else:
            self.subtitle_panel.status.setText(
                "Keo khung xanh tren video o Screen Edit den dong chu can doc."
            )
        if len(data.blur_region) == 4:
            self._update_region_label(MODE_BLUR, *data.blur_region)
        else:
            self.render_panel.lbl_blur.setText("Chua khoanh vung che mo")
        self.subtitle_panel.set_regions(data.ocr_region, data.blur_region)
        self.render_panel.lut_label.setText(
            Path(data.lut_path).name if data.lut_path else "Chua chon LUT"
        )
        if data.render_preset:
            self.render_preset_combo.blockSignals(True)
            self.render_preset_combo.setCurrentText(data.render_preset)
            self.render_preset_combo.blockSignals(False)
        snap = data.render_preset_snapshot
        if snap:
            if "style" in snap and isinstance(snap["style"], dict):
                self.settings.style = SubtitleStyle.from_dict(snap["style"])
                self.render_panel.load(self.settings)
            if "subtitle_visible" in snap:
                self._render_subtitles_visible = bool(snap["subtitle_visible"])
                if "subtitles" in self.render_tool_buttons:
                    self.render_tool_buttons["subtitles"].setChecked(self._render_subtitles_visible)
            if "render_crf" in snap:
                self.settings.render_crf = int(snap["render_crf"])
                self.render_panel.crf.setValue(self.settings.render_crf)
            if "render_preset" in snap:
                self.settings.render_preset = str(snap["render_preset"])
                self.render_panel.preset.setCurrentText(self.settings.render_preset)
            if "render_scale" in snap:
                self.settings.render_scale = str(snap["render_scale"])
            if "render_fps" in snap:
                self.settings.render_fps = str(snap["render_fps"])
            self.player.set_subtitle_style(self.settings.style)
            self._on_position(self.player.position)
        self._dirty = False
        self.status_project.setText(f"Du an: {data.name}  |  {data.folder}")
        self._update_counts()
        self._log(f"Da mo du an: {data.name}")

    def _apply_frame_size(self, data: ProjectData) -> None:
        """Bao kich thuoc khung hinh cho trinh phat truoc khi video kip hien hinh."""
        if not (data.width and data.height):
            try:
                info = self.ff.probe(data.video_path)
            except (FFmpegError, OSError):
                return
            data.width, data.height = info.width, info.height
            if data.width and data.height:
                self.store.save(data)
        if not (data.width and data.height):
            return
        self.player.set_frame_size(data.width, data.height)
        # Video co the da doi kich thuoc, nen ep lai cac vung da luu cho vua khung hinh.
        changed = False
        for field in ("ocr_region", "blur_region"):
            stored = list(getattr(data, field))
            if len(stored) != 4:
                continue
            fixed = clamp_region(stored, data.width, data.height)
            if fixed and fixed != stored:
                setattr(data, field, fixed)
                changed = True
        if changed:
            self.store.save(data)

    def _attach_video(self, path: str) -> None:
        if self.project is None:
            return
        try:
            info = self.ff.probe(path)
        except (FFmpegError, OSError) as exc:
            QMessageBox.critical(self, "Khong doc duoc video", str(exc))
            return
        self.project.video_path = path
        self.project.original_video = path
        self.project.duration = info.duration
        self.project.width = info.width
        self.project.height = info.height
        self.store.save(self.project)
        if info.width and info.height:
            self.player.set_frame_size(info.width, info.height)
        self.player.load(path)
        self._sync_project_record()
        self._log(
            f"Da gan video: {Path(path).name} ({info.resolution}, {format_display(info.duration)})"
        )

    def _choose_video(self) -> None:
        if not self._require_project():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Chon video hoac audio", "", VIDEO_FILTER)
        if path:
            self._attach_video(path)

    def _download_video(self) -> None:
        if not download.is_available():
            QMessageBox.warning(self, "Thiếu yt-dlp", "Bản tool này chưa có yt-dlp đi kèm.")
            return
        url, accepted = QInputDialog.getText(self, "Tải Video", "Dán link video:")
        if not accepted or not url.strip():
            return
        default_folder = self.settings.download_folder or str(Path.home() / "Downloads")
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu video", default_folder)
        if not folder:
            return
        self.settings.download_folder = folder
        self.settings.save()

        def job(ctx: TaskContext) -> str:
            output = download.download_video(
                url,
                folder,
                proxy=self.settings.download_proxy,
                token=ctx.token,
                on_progress=ctx.progress,
                on_log=ctx.log,
            )
            return str(output)

        task_id = self.tasks.submit(
            "Tải video từ link",
            job,
            timeout=max(300, self.settings.task_timeout_minutes * 60),
        )
        self._download_task_ids.add(task_id)
        self._logs[task_id] = []
        self.status_task.setText("Đang tải video...")

    def _delete_project(self) -> None:
        rows = self._selected_project_rows()
        if not rows:
            return
        if len(rows) == 1:
            detail = f"Xóa dự án '{rows[0].name}' và toàn bộ tệp trong thư mục dự án?"
        else:
            names = "\n".join(f"  • {row.name}" for row in rows[:10])
            more = f"\n  • ... và {len(rows) - 10} dự án khác" if len(rows) > 10 else ""
            detail = f"Xóa {len(rows)} dự án đã chọn và toàn bộ tệp của chúng?\n\n{names}{more}"
        answer = QMessageBox.question(
            self,
            "Xóa dự án",
            detail + "\n\nThao tác này không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._delete_projects(rows)

    def _delete_projects(self, rows: list[ProjectRow]) -> None:
        """Xoa toan bo cac project da xac nhan, ke ca project dang mo."""
        ids = {row.id for row in rows}
        self._remove_pending_batch_projects(ids)
        for row in rows:
            self.tasks.cancel_project(row.id)
            self.store.delete(row.folder)
        with self.db.session() as s:
            for project_id in ids:
                record = s.get(Project, project_id)
                if record is not None:
                    s.delete(record)
        if self.project_id in ids:
            self.project = None
            self.project_id = 0
            self.cue_model.set_document(SubtitleDoc())
            self.player.clear()
            self.status_project.setText("Chua mo du an")
        self.reload_projects()
        self._log(f"Đã xóa {len(rows)} dự án: " + ", ".join(row.name for row in rows))

    def _open_project_folder(self) -> None:
        row = self._selected_project_row()
        folder = row.folder if row else (self.project.folder if self.project else "")
        if folder:
            self._open_path(folder)

    def _save_project(self) -> None:
        if self.project is None:
            return
        self.project.doc = self.cue_model.doc
        self.store.save(self.project)
        self._sync_project_record()
        self._dirty = False
        self.statusBar().showMessage("Da luu du an.", 3000)

    def _sync_project_record(self) -> None:
        if self.project is None or not self.project_id:
            return
        self.project.doc = self.cue_model.doc
        self._sync_project_data(self.project_id, self.project)

    def _sync_project_data(self, project_id: int, data: ProjectData) -> None:
        """Cap nhat thong tin project vua xu ly ma khong lam mat lua chon trong bang."""
        doc = data.doc
        with self.db.session() as s:
            record = s.get(Project, project_id)
            if record is None:
                return
            record.folder = data.folder
            record.video_path = data.video_path
            record.duration = data.duration
            record.cue_count = len(doc.cues)
            record.char_count = doc.total_chars
            record.has_subtitle = bool(doc.cues)
            record.has_translation = doc.translated_count > 0
            record.has_dub = bool(data.dub_path)
            record.has_render = bool(data.render_path)
            record.language = doc.language
            record.target_language = doc.target_language or self.settings.target_language
        self.project_model.update_row(
            project_id,
            video_path=data.video_path,
            duration=data.duration,
            cue_count=len(doc.cues),
            char_count=doc.total_chars,
            has_subtitle=bool(doc.cues),
            has_translation=doc.translated_count > 0,
            has_dub=bool(data.dub_path),
            has_render=bool(data.render_path),
            language=doc.language,
            target_language=doc.target_language or self.settings.target_language,
        )

    def _autosave_tick(self) -> None:
        if self.project is None or not self._dirty:
            return
        self.project.doc = self.cue_model.doc
        self.store.save_recovery(self.project)

    # ------------------------------------------------------------------ bang phu de

    def _current_row(self) -> int:
        rows = self.cue_view.selected_rows()
        return rows[0] if rows else -1

    def _on_cue_activated(self, index, *_unused) -> None:
        row = index.row()
        cues = self.cue_model.doc.cues
        if not 0 <= row < len(cues):
            return
        cue = cues[row]
        self.player.show_frame(cue.start)
        if getattr(getattr(self, "settings", None), "play_on_edit", False):
            self.player.play()
        self._fill_editors(row)

    def _fill_editors(self, row: int) -> None:
        cues = self.cue_model.doc.cues
        if not 0 <= row < len(cues):
            return
        cue = cues[row]
        for widget, value in ((self.edit_origin, cue.text), (self.edit_trans, cue.translation)):
            widget.blockSignals(True)
            widget.setPlainText(value)
            widget.blockSignals(False)
        self.start_edit.setText(format_display(cue.start))
        self.end_edit.setText(format_display(cue.end))
        self._editor_history_session = None

    def _apply_editor(self, translation: bool) -> None:
        row = self._current_row()
        cues = self.cue_model.doc.cues
        if not 0 <= row < len(cues):
            return
        cue = cues[row]
        text = (self.edit_trans if translation else self.edit_origin).toPlainText()
        current = cue.translation if translation else cue.text
        if current == text:
            return
        session = (row, translation)
        if self._editor_history_session != session:
            label = "Sửa bản dịch" if translation else "Sửa nội dung gốc"
            self.cue_model.push_history(f"{label} câu {row + 1}")
            self._editor_history_session = session
        if translation:
            cue.translation = text
        else:
            cue.text = text
        self.cue_model.refresh_row(row)
        self._mark_dirty()
        if translation and self._render_preview:
            self._on_position(self.player.position)

    def _apply_time(self, is_start: bool) -> None:
        row = self._current_row()
        cues = self.cue_model.doc.cues
        if not 0 <= row < len(cues):
            return
        widget = self.start_edit if is_start else self.end_edit
        try:
            seconds = parse_timecode(widget.text())
        except TimecodeError:
            QMessageBox.warning(
                self,
                "Sai dinh dang thoi gian",
                "Nhap theo dang 00:00:05,250 (gio:phut:giay,mili giay).",
            )
            self._fill_editors(row)
            return
        cue = cues[row]
        self.cue_model.push_history(f"Sua thoi gian cau {row + 1}")
        if is_start:
            cue.start = min(seconds, max(0.0, cue.end - 0.05))
        else:
            cue.end = max(seconds, cue.start + 0.05)
        self.cue_model.refresh_row(row)
        self._mark_dirty()

    def _set_time_from_player(self, is_start: bool) -> None:
        row = self._current_row()
        if row < 0:
            return
        widget = self.start_edit if is_start else self.end_edit
        widget.setText(format_display(self.player.position))
        self._apply_time(is_start)

    def _split_cue(self) -> None:
        row = self._current_row()
        if row < 0:
            QMessageBox.information(self, "Chua chon cau", "Hay chon mot cau de tach.")
            return
        position = self.player.position
        try:
            self.cue_model.push_history(f"Tach cau {row + 1}")
            self.cue_model.beginResetModel()
            new_index = editing.split_cue(self.cue_model.doc, row, position)
            self.cue_model.endResetModel()
        except (ValueError, IndexError) as exc:
            self.cue_model.undo()
            QMessageBox.warning(self, "Khong tach duoc", str(exc))
            return
        self.cue_view.selectRow(new_index)
        self._mark_dirty()

    def _merge_cues(self) -> None:
        rows = self.cue_view.selected_rows()
        if len(rows) < 2:
            QMessageBox.information(
                self, "Chon it nhat 2 cau", "Giu Ctrl hoac Shift de chon nhieu cau lien tiep."
            )
            return
        try:
            self.cue_model.push_history(f"Gop {len(rows)} cau")
            self.cue_model.beginResetModel()
            index = editing.merge_cues(self.cue_model.doc, rows)
            self.cue_model.endResetModel()
        except (ValueError, IndexError) as exc:
            self.cue_model.undo()
            QMessageBox.warning(self, "Khong gop duoc", str(exc))
            return
        self.cue_view.selectRow(index)
        self._mark_dirty()

    def _add_cue(self) -> None:
        if self.cue_model.doc is None:
            return
        index = self.cue_model.insert_cue(self.player.position)
        self.cue_view.selectRow(index)
        self._fill_editors(index)
        self._mark_dirty()

    def _delete_cues(self) -> None:
        rows = self.cue_view.selected_rows()
        if not rows:
            return
        if self.cue_model.remove_rows(rows):
            self._mark_dirty()

    def _shift_cues(self) -> None:
        dialog = ShiftDialog(self)
        if dialog.exec() != ShiftDialog.DialogCode.Accepted:
            return
        offset = dialog.value()
        if abs(offset) < 1e-6:
            return
        rows = self.cue_view.selected_rows()
        target = rows if rows else None
        self.cue_model.apply_change(
            f"Doi thoi gian {offset:+.2f}s",
            lambda doc: editing.shift_cues(doc, offset, target),
        )
        self._mark_dirty()

    def _check_timing(self) -> None:
        issues = editing.check_timing(self.cue_model.doc)
        dialog = IssueDialog(issues, self)
        dialog.jumpRequested.connect(self._jump_to_index)
        dialog.exec()

    def _jump_to_index(self, index: int) -> None:
        if 0 <= index < len(self.cue_model.doc.cues):
            self.cue_view.selectRow(index)
            self.player.seek(self.cue_model.doc.cues[index].start)
            self._fill_editors(index)

    def _jump_to_line(self, number: int) -> None:
        """Nhay toi cau theo so thu tu nhap o o 'So Dong'."""
        if number > 0:
            self._jump_to_index(number - 1)

    def _on_dub_source(self, index: int) -> None:
        """Chon lay ban goc hay ban dich lam loi doc khi long tieng."""
        self.settings.dub_source = "original" if index == 0 else "translation"

    def _jump_cue(self, direction: int) -> None:
        cues = self.cue_model.doc.cues
        if not cues:
            return
        current = self.cue_model.doc.index_at(self.player.position)
        if current < 0:
            current = 0 if direction > 0 else len(cues) - 1
        else:
            current = max(0, min(len(cues) - 1, current + direction))
        self._jump_to_index(current)

    def _undo(self) -> None:
        row = self._current_row()
        label = self.cue_model.undo()
        if label:
            self._editor_history_session = None
            if 0 <= row < len(self.cue_model.doc.cues):
                self._fill_editors(row)
            self._on_position(self.player.position)
            self._log(f"Hoan tac: {label}")
            self._mark_dirty()

    def _redo(self) -> None:
        row = self._current_row()
        label = self.cue_model.redo()
        if label:
            self._editor_history_session = None
            if 0 <= row < len(self.cue_model.doc.cues):
                self._fill_editors(row)
            self._on_position(self.player.position)
            self._log(f"Lam lai: {label}")
            self._mark_dirty()

    def _refresh_history(self) -> None:
        self.history.clear()
        self.history.addItems(self.cue_model.history_labels())
        self.btn_undo.setEnabled(self.cue_model.can_undo())
        self.btn_redo.setEnabled(self.cue_model.can_redo())

    def _on_document_changed(self) -> None:
        self._update_counts()
        self._mark_dirty()

    def _update_counts(self) -> None:
        doc = self.cue_model.doc
        self.line_spin.blockSignals(True)
        self.line_spin.setMaximum(max(0, len(doc.cues)))
        self.line_spin.blockSignals(False)
        self.lbl_count.setText(
            f"{len(doc.cues)} cau | {doc.translated_count} da dich | {doc.total_chars} ky tu"
        )

    def _mark_dirty(self) -> None:
        self._dirty = True
        if self.project is not None:
            self.project.doc = self.cue_model.doc

    def _on_position(self, seconds: float) -> None:
        doc = self.cue_model.doc
        index = doc.index_at(seconds)
        self.cue_model.set_current(index)
        # Screen Edit dung de khoanh/cat va sua noi dung nen khong chen chu len
        # video. Screen Render chi xem dung ban dich se duoc dua vao thanh pham.
        self.player.show_subtitle(
            self._preview_subtitle_text(
                doc, index, self._render_preview, self._render_subtitles_visible
            )
        )

    @staticmethod
    def _preview_subtitle_text(
        doc: SubtitleDoc, index: int, render_mode: bool, visible: bool
    ) -> str:
        if not render_mode or not visible or not 0 <= index < len(doc.cues):
            return ""
        return doc.cues[index].translation.strip()

    # ------------------------------------------------------------------ khoanh vung

    def _show_region_editor(self, mode: str) -> None:
        """Hien khung keo duoc cho che do dang xem. Khung luon co san."""
        if self.project is None:
            return
        x, y, w, h = self.player.begin_region(mode, self._region_for(mode))
        if self._region_for(mode) != [x, y, w, h]:
            self._save_region(mode, [x, y, w, h])
        else:
            self._update_region_label(mode, x, y, w, h)

    def _reset_region(self, mode: str) -> None:
        """Dat lai khung ve vi tri goi y o phan duoi khung hinh."""
        if not self._require_project() or self.project is None:
            return
        if mode == MODE_OCR:
            self.project.ocr_region = []
            self.view_strip.set_current(0)  # khung doc chu chi lam viec o Screen Edit
        else:
            self.project.blur_region = []
            self.view_strip.set_current(1)  # khung che mo thuoc ve Screen Render
        self.player.pause()
        x, y, w, h = self.player.begin_region(mode, None)
        self._save_region(mode, [x, y, w, h])
        self.statusBar().showMessage(
            f"Da dat lai vung ve mac dinh: rong {w}, cao {h} tai ({x}, {y}). "
            "Keo giua khung de di chuyen, keo 8 nut xanh de doi kich thuoc.",
            10000,
        )

    def _region_for(self, mode: str) -> list[int]:
        if self.project is None:
            return []
        return list(self.project.ocr_region if mode == MODE_OCR else self.project.blur_region)

    def _save_region(self, mode: str, region: list[int]) -> None:
        if self.project is None or len(region) != 4:
            return
        if mode == MODE_OCR:
            self.project.ocr_region = region
        elif mode == MODE_BLUR:
            self.project.blur_region = region
        else:
            return
        self.store.save(self.project)
        self._update_region_label(mode, *region)
        self.subtitle_panel.set_regions(self.project.ocr_region, self.project.blur_region)

    def _update_region_label(self, mode: str, x: int, y: int, w: int, h: int) -> None:
        if mode == MODE_OCR:
            self.subtitle_panel.status.setText(f"Vung doc chu: rong {w}, cao {h} - tai ({x}, {y})")
        else:
            self.render_panel.lbl_blur.setText(f"Vung che mo: {w} x {h} tai ({x}, {y})")

    def _on_region_selected(self, mode: str, x: int, y: int, w: int, h: int) -> None:
        self._save_region(mode, [x, y, w, h])

    def _on_region_preview(self, mode: str, x: int, y: int, w: int, h: int) -> None:
        self._update_region_label(mode, x, y, w, h)

    def _choose_lut(self) -> None:
        if not self._require_project() or self.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chon tep LUT",
            str(bundled_dir("lut") or ""),
            "LUT (*.cube *.CUBE);;Tat ca (*.*)",
        )
        if not path:
            return
        self.project.lut_path = path
        self.store.save(self.project)
        self.render_panel.lut_label.setText(Path(path).name)

    # ------------------------------------------------------------------ nhap/xuat

    def _import_subtitle(self, path: str = "") -> None:
        if not self._require_project():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Chon tep phu de", "", SUB_FILTER)
        if not path:
            return
        try:
            result = formats.load_subtitle(path)
        except formats.SubtitleFormatError as exc:
            QMessageBox.critical(self, "Khong doc duoc tep phu de", str(exc))
            return
        if not result.doc.cues:
            QMessageBox.warning(self, "Tep rong", "Tep nay khong chua cau phu de nao doc duoc.")
            return
        self.cue_model.set_document(result.doc)
        self._mark_dirty()
        self._save_project()
        first = self.cue_model.index(0, 0)
        self.cue_view.selectRow(0)
        self.cue_view.scrollTo(first)
        self._on_cue_activated(first)
        if result.warnings:
            self._log("Canh bao khi doc tep phu de:")
            for warn in result.warnings[:20]:
                self._log(f"  - {warn}")
        self._log(f"Da nhap {len(result.doc.cues)} cau tu {Path(path).name}")

    def _import_translation(self) -> None:
        """Nap ban dich tu mot tep phu de khac vao cot ban dich."""
        if not self._require_project() or self.project is None:
            return
        cues = self.cue_model.doc.cues
        if not cues:
            QMessageBox.information(
                self, "Chua co phu de goc", "Hay lay phu de goc truoc, roi moi nap ban dich vao."
            )
            return
        path, _ = QFileDialog.getOpenFileName(self, "Chon tep ban dich", "", SUB_FILTER)
        if not path:
            return
        try:
            result = formats.load_subtitle(path)
        except formats.SubtitleFormatError as exc:
            QMessageBox.critical(self, "Khong doc duoc tep", str(exc))
            return
        source = result.doc.cues
        if not source:
            QMessageBox.warning(self, "Tep rong", "Tep nay khong co cau nao.")
            return
        self.cue_model.push_history("Nap ban dich tu tep")
        matched = 0
        for cue in cues:
            middle = (cue.start + cue.end) / 2
            best = next(
                (c for c in source if c.start <= middle < c.end),
                None,
            )
            if best is None:
                best = min(source, key=lambda c: abs((c.start + c.end) / 2 - middle))
                if abs((best.start + best.end) / 2 - middle) > 2.0:
                    continue
            cue.translation = best.text
            matched += 1
        self.cue_model.refresh_all()
        self._mark_dirty()
        self._save_project()
        self._log(f"Da nap ban dich cho {matched}/{len(cues)} cau tu {Path(path).name}")
        QMessageBox.information(
            self, "Da nap ban dich", f"Da gan ban dich cho {matched} tren {len(cues)} cau."
        )

    def _export_subtitle(self, suffix: str) -> None:
        if not self._require_project() or self.project is None:
            return
        doc = self.cue_model.doc
        if not doc.cues:
            QMessageBox.information(self, "Chua co phu de", "Du an chua co cau nao.")
            return
        default = str(
            Path(self.project.folder) / "exports" / f"{safe_name(self.project.name)}{suffix}"
        )
        path, _ = QFileDialog.getSaveFileName(self, "Luu phu de", default, f"Phu de (*{suffix})")
        if not path:
            return
        mode = "translation" if doc.translated_count else "original"
        try:
            if suffix == ".ass":
                self._collect_settings()
                content = P.build_ass(self._context_for_export(), text_mode=mode)
                Path(path).write_text(content.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                formats.save_subtitle(path, doc, text_mode=mode)
        except (OSError, P.StepError) as exc:
            QMessageBox.critical(self, "Khong luu duoc", str(exc))
            return
        self._log(f"Da xuat phu de: {path}")
        self.statusBar().showMessage(f"Da xuat: {path}", 5000)

    def _context_for_export(self) -> P.PipelineContext:
        """Ngu canh nhe dung cho cac thao tac chay ngay tren giao dien."""
        assert self.project is not None
        task = TaskContext(
            token=CancelToken(),
            _progress=lambda _p: None,
            _log=self._log,
        )
        return P.PipelineContext(
            ff=self.ff,
            settings=self.settings,
            store=self.store,
            project=self.project,
            task=task,
        )

    # ------------------------------------------------------------------ tac vu

    def _require_project(self) -> bool:
        if self.project is None:
            QMessageBox.information(
                self,
                "Chua mo du an",
                "Hay tao du an moi hoac mo mot du an trong tab 'Danh sach du an'.",
            )
            return False
        return True

    def _require_idle(self) -> bool:
        record = self.tasks.running_for_project(self.project_id)
        if record is not None:
            QMessageBox.information(
                self,
                "Du an dang chay",
                f"Du an nay dang chay: {record.name}. Hay doi hoac bam 'Dung tac vu'.",
            )
            return False
        return True

    def _require_voice_model_ready(self) -> bool:
        voice_id = (self.settings.local_voice or self.settings.tts_voice).strip()
        if not voice_id:
            QMessageBox.warning(
                self,
                "Chưa chọn giọng đọc",
                "Chưa chọn giọng đọc nào. Vui lòng mở Thư viện giọng để tải và chọn giọng đọc.",
            )
            return False
        runtime_ok, runtime_msg = piper_runtime_ready()
        if not runtime_ok:
            QMessageBox.warning(self, "Piper chưa sẵn sàng", runtime_msg)
            return False
        mgr = get_default_manager()
        status = mgr.get_status(voice_id)
        if status != STATUS_READY:
            QMessageBox.warning(
                self,
                "Giọng đọc chưa sẵn sàng",
                (
                    f"Giọng đọc '{voice_id}' chưa được tải về hoặc bị lỗi. "
                    "Vui lòng mở Thư viện giọng để tải về."
                ),
            )
            return False
        return True

    def _start_dub(self) -> None:
        self.dub_panel.apply(self.settings)
        if not self._require_voice_model_ready():
            return
        self._run_step(P.STEP_DUB)

    def _run_step(self, name: str) -> None:
        if name == P.STEP_DUB and not self._require_voice_model_ready():
            return
        if not self._require_project() or not self._require_idle():
            return
        if not self.ff.available:
            QMessageBox.critical(
                self, "Thieu FFmpeg", "Chua tim thay FFmpeg. Vao tab 'Cai dat chung' de chon."
            )
            return
        self._collect_settings()
        self._save_project()
        self._submit(name, [name])

    def _run_script(self, names: list[str]) -> None:
        if not names:
            QMessageBox.information(
                self, "Chua chon buoc", "Hay tich chon it nhat mot buoc trong danh sach."
            )
            return
        if P.STEP_DUB in names and not self._require_voice_model_ready():
            return
        if not self._require_project() or not self._require_idle():
            return
        self._collect_settings()
        self._save_project()
        self._submit(f"Kich ban ({len(names)} buoc)", names)

    def _submit(self, label: str, names: list[str]) -> None:
        assert self.project is not None
        project = self.project
        project_id = self.project_id
        settings = self.settings
        store = self.store
        ff = self.ff
        api_key = Settings.get_secret("ai_gateway_key")
        glossary = self.translate_panel.glossary_dict()

        def job(ctx: TaskContext) -> str:
            pc = P.PipelineContext(
                ff=ff,
                settings=settings,
                store=store,
                project=project,
                task=ctx,
                api_key=api_key,
                glossary=glossary,
            )
            return P.run_script(names, pc)

        timeout = max(60, settings.task_timeout_minutes * 60)
        task_id = self.tasks.submit(label, job, project_id=project_id, timeout=timeout)
        self._logs[task_id] = []
        self._busy_project = project_id
        self.cue_view.setEnabled(False)
        self.status_task.setText(f"Dang chay: {label}")
        self._set_project_status(project_id, task=label, status="Đang đợi...", progress=0)
        self._log(f"Xếp hàng: {label}")

    def _on_task_started(self, task_id: str) -> None:
        record = self.tasks.record(task_id)
        if record is None:
            return
        if task_id in self._batch_task_ids:
            task = "Tách Sub Bằng Chữ"
            status = "Đang Trích Xuất..."
        else:
            task = record.name
            status = "Đang chạy..."
        self._set_project_status(record.project_id, task=task, status=status, progress=0)
        self._log(f"Bắt đầu: {record.name}")
        self._update_batch_queue_status()

    def _on_task_progress(self, task_id: str, percent: int) -> None:
        record = self.tasks.record(task_id)
        if record is None:
            return
        self.project_model.update_row(record.project_id, progress=percent)
        self.status_task.setText(f"{record.name}: {percent}%")

    def _on_task_log(self, task_id: str, message: str) -> None:
        self._logs.setdefault(task_id, []).append(message)
        self.logAppended.emit(message)
        record = self.tasks.record(task_id)
        if record is None or record.name != P.STEP_OCR:
            return
        detail = message.casefold()
        if "doc chu bang card do hoa" in detail:
            self._set_project_status(
                record.project_id,
                task="Tách Sub Bằng Chữ",
                status="OCR: GPU CUDA",
                progress=record.percent,
            )
        elif "doc chu bang cpu" in detail or "chuyen cac frame con lai sang cpu" in detail:
            self._set_project_status(
                record.project_id,
                task="Tách Sub Bằng Chữ",
                status="OCR: CPU",
                progress=record.percent,
            )

    def _on_task_finished(self, task_id: str, status: str, result: object, message: str) -> None:
        record = self.tasks.record(task_id)
        project_id = record.project_id if record else 0
        name = record.name if record else task_id
        is_batch = task_id in self._batch_task_ids
        is_download = task_id in self._download_task_ids
        is_preview = task_id in self._preview_task_ids

        if status == DONE:
            text = str(result or message)
            if is_preview:
                self._preview_task_ids.discard(task_id)
                audio_path = Path(text)
                if audio_path.is_file():
                    self._preview_player.stop()
                    self._preview_player.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
                    self._preview_player.play()
                    self._log(f"Đang phát nghe thử: {audio_path.name}")
            if is_download:
                try:
                    downloaded = Path(text)
                    download_data = self._create_project_from_video(str(downloaded))
                    self.reload_projects()
                    self._load_project(download_data.project_id, download_data.folder)
                    text = f"Đã tải và tạo project: {downloaded.name}"
                except (FFmpegError, OSError) as exc:
                    QMessageBox.critical(self, "Không Tạo Được Project", str(exc))
            if name == P.STEP_OCR_MEASURE:
                self.subtitle_panel.load(self.settings)  # hien so vua do duoc len giao dien
            self._log(f"Hoan tat: {name} - {text}")
            self._set_project_status(project_id, task="", status="Xong", progress=100)
            self.statusBar().showMessage(text, 8000)
        elif status == CANCELLED:
            if is_preview:
                self._preview_task_ids.discard(task_id)
            self._log(f"Da huy: {name}")
            self._set_project_status(project_id, task="", status="Da huy", progress=0)
        else:
            if is_preview:
                self._preview_task_ids.discard(task_id)
            self._log(f"LOI: {name} - {message}")
            self._set_project_status(project_id, task="", status=f"Loi: {message[:60]}", progress=0)
            if not is_batch and not is_preview:
                QMessageBox.critical(self, "Tac vu that bai", message)
            elif is_preview:
                QMessageBox.critical(self, "Không tạo được giọng đọc", message)

        detail_lines = self._logs.pop(task_id, [])
        stored_message = message
        if detail_lines:
            stored_message += "\n\n" + "\n".join(detail_lines)
        if project_id:
            with self.db.session() as s:
                s.add(
                    TaskLog(
                        project_id=project_id,
                        name=name,
                        status=status,
                        message=stored_message[:50000],
                    )
                )

        finished_data: ProjectData | None = None
        if project_id:
            row = next((item for item in self.project_model.rows() if item.id == project_id), None)
            folder = row.folder if row is not None else ""
            if folder:
                try:
                    finished_data = self.store.load(folder)
                except ProjectFormatError:
                    finished_data = None
            if finished_data is not None:
                self._sync_project_data(project_id, finished_data)

        if project_id and project_id == self.project_id and self.project is not None:
            try:
                data = self.store.load(self.project.folder)
            except ProjectFormatError:
                data = None
            if data is not None:
                self.project = data
                self.cue_model.set_document(data.doc, reset_history=False)
                if data.video_path and Path(data.video_path).is_file():
                    self.player.load(data.video_path)
                self._update_counts()
                self._sync_project_data(project_id, data)

        if is_batch:
            self._batch_task_ids.discard(task_id)
            if status == DONE:
                self._batch_done += 1
            else:
                self._batch_failed += 1
            self._start_next_batch_projects()
        if is_download:
            self._download_task_ids.discard(task_id)
        active_current = self.tasks.running_for_project(self.project_id)
        self.cue_view.setEnabled(active_current is None)
        self._busy_project = self.project_id if active_current is not None else 0
        self.status_task.setText(
            "San sang" if self.tasks.active_count() == 0 else "Dang xu ly hang doi"
        )
        self._update_batch_queue_status()

    def _update_batch_queue_status(self) -> None:
        records = [
            record
            for task_id in self._batch_task_ids
            if (record := self.tasks.record(task_id)) is not None
        ]
        running = sum(record.status == RUNNING for record in records)
        waiting = sum(record.status == PENDING for record in records) + len(
            self._batch_pending_projects
        )
        if running or waiting:
            limit = max(1, min(8, int(self.settings.max_workers)))
            self.batch_queue_status.setText(
                f"Đang chạy: {running}/{limit}  •  Đang đợi: {waiting}"
            )
        elif self._batch_total:
            failed = f"  •  Lỗi/hủy: {self._batch_failed}" if self._batch_failed else ""
            self.batch_queue_status.setText(
                f"Đã xong: {self._batch_done}/{self._batch_total}{failed}"
            )

    def _set_project_status(
        self, project_id: int, *, task: str, status: str, progress: int
    ) -> None:
        if not project_id:
            return
        with self.db.session() as s:
            record = s.get(Project, project_id)
            if record is not None:
                record.current_task = task
                record.status = status
                record.progress = progress
        self.project_model.update_row(
            project_id, current_task=task, status=status, progress=progress
        )

    def _stop_tasks(self) -> None:
        row = self._selected_project_row()
        target = row.id if row else self.project_id
        count = self.tasks.cancel_project(target) if target else self.tasks.cancel_all()
        self._log(f"Da yeu cau dung {count} tac vu.")

    def _show_project_log(self) -> None:
        row = self._selected_project_row()
        project_id = row.id if row else self.project_id
        lines: list[str] = []
        with self.db.session() as s:
            query = s.query(TaskLog).order_by(TaskLog.id.desc()).limit(200)
            if project_id:
                query = query.filter(TaskLog.project_id == project_id)
            for entry in query.all():
                lines.append(
                    f"[{entry.created_at:%Y-%m-%d %H:%M:%S}] {entry.name} - "
                    f"{entry.status}: {entry.message}"
                )
        LogDialog("Nhat ky tac vu", "\n".join(lines) or "Chua co nhat ky.", self).exec()

    # ------------------------------------------------------------------ kich ban

    def _refresh_scripts(self) -> None:
        with self.db.session() as s:
            scripts = s.query(AutoScript).order_by(AutoScript.name).all()
            names = [a.name for a in scripts]
            steps = scripts[0].steps if scripts else []
        self.script_panel.set_scripts(names, names[0] if names else "")
        if steps:
            self.script_panel.set_checked_steps(steps)

    def _select_script(self, name: str) -> None:
        if not name:
            return
        with self.db.session() as s:
            record = s.query(AutoScript).filter(AutoScript.name == name).first()
            steps = record.steps if record else []
        self.script_panel.set_checked_steps(steps)

    def _save_script(self, name: str, steps: list[str]) -> None:
        with self.db.session() as s:
            record = s.query(AutoScript).filter(AutoScript.name == name).first()
            if record is None:
                record = AutoScript(name=name)
                s.add(record)
            record.steps = steps
        self._refresh_scripts()
        self.script_panel.set_scripts(self._script_names(), name)
        self.script_panel.set_checked_steps(steps)
        self._log(f"Da luu kich ban '{name}' voi {len(steps)} buoc.")

    def _script_names(self) -> list[str]:
        with self.db.session() as s:
            return [a.name for a in s.query(AutoScript).order_by(AutoScript.name).all()]

    def _delete_script(self, name: str) -> None:
        if not name:
            return
        with self.db.session() as s:
            record = s.query(AutoScript).filter(AutoScript.name == name).first()
            if record is not None:
                s.delete(record)
        self._refresh_scripts()
        self._log(f"Da xoa kich ban '{name}'.")

    # ------------------------------------------------------------------ dich/long tieng

    def _translate_selected(self) -> None:
        if not self._require_project() or not self._require_idle():
            return
        rows = self.cue_view.selected_rows()
        if not rows:
            QMessageBox.information(
                self, "Chua chon cau", "Hay chon cac cau can dich trong bang phu de."
            )
            return
        self._collect_settings()
        self._save_project()
        cues = [self.cue_model.doc.cues[r] for r in rows]
        texts = [c.text for c in cues]
        provider = self.settings.translate_provider
        api_key = Settings.get_secret("ai_gateway_key")
        settings = self.settings
        glossary = self.translate_panel.glossary_dict()
        from ..providers import translate as tr

        def job(ctx: TaskContext) -> str:
            done = 0
            batches = tr.chunk(texts, 20)
            offset = 0
            for i, batch in enumerate(batches):
                ctx.check_cancel()
                request = tr.TranslationRequest(
                    texts=batch,
                    source=settings.source_language,
                    target=settings.target_language,
                    glossary=glossary,
                )
                out = tr.translate_batch(
                    provider, request, api_key=api_key, model=settings.llm_model
                )
                for j, text in enumerate(out):
                    if text.strip():
                        cues[offset + j].translation = text.strip()
                        done += 1
                offset += len(batch)
                ctx.progress(int((i + 1) / len(batches) * 100))
            return f"Da dich {done} cau dang chon."

        task_id = self.tasks.submit(
            "Dich cac cau dang chon",
            job,
            project_id=self.project_id,
            timeout=max(60, settings.task_timeout_minutes * 60),
        )
        self._logs[task_id] = []
        self.status_task.setText("Dang dich cac cau dang chon...")

    def _on_view_mode(self, index: int) -> None:
        """Chuyen giua che do sua (Screen Edit) va che do xem truoc (Screen Render)."""
        render_mode = index == 1
        self._render_preview = render_mode
        self.render_toolbar.setVisible(render_mode)
        self.render_tools.setVisible(render_mode)
        self.player.set_subtitle_movable(render_mode)
        if render_mode:
            # Screen Render hien sub dich va cho keo dong sub len/xuong.
            self.player.hide_region()
        elif self.project is not None:
            self._show_region_editor(MODE_OCR)
        self._on_position(self.player.position)
        self.statusBar().showMessage(
            "Screen Render: xem truoc phu de nhu luc render xong."
            if render_mode
            else "Screen Edit: keo khung xanh de chon vung chu can doc (OCR).",
            6000,
        )

    def _save_render_preview_style(self) -> None:
        """Dong bo thay doi tren thanh Screen Render vao UI, cau hinh va preview."""
        style = self.settings.style
        self.render_panel.font_size.setValue(style.font_size)
        self.render_panel.bold.setChecked(style.bold)
        self.render_panel.margin.setValue(style.margin_v)
        self.player.set_subtitle_style(style)
        self.settings.save()
        self._on_position(self.player.position)

    def _set_render_subtitle_margin(self, margin: int) -> None:
        """Luu vi tri dong sub sau khi nguoi dung tha chuot."""
        frame_height = self.player.frame_size[1] or 1080
        self.settings.style.margin_v = max(0, min(int(margin), frame_height - 10))
        self._save_render_preview_style()
        self.statusBar().showMessage(
            f"Đã lưu vị trí subtitle: lề dưới {self.settings.style.margin_v}px.", 4000
        )

    def _move_render_subtitle(self, delta: int) -> None:
        self._set_render_subtitle_margin(self.settings.style.margin_v + delta)

    def _change_render_font(self, delta: int) -> None:
        self.settings.style.font_size = max(
            10, min(200, self.settings.style.font_size + delta)
        )
        self._save_render_preview_style()

    def _toggle_render_bold(self) -> None:
        self.settings.style.bold = not self.settings.style.bold
        self._save_render_preview_style()

    def _toggle_render_subtitles(self, visible: bool) -> None:
        self._render_subtitles_visible = bool(visible)
        self._on_position(self.player.position)

    def _add_render_subtitle(self) -> None:
        if not self._require_project():
            return
        self._add_cue()
        self.edit_trans.setFocus()
        self.statusBar().showMessage(
            "Đã thêm dòng subtitle. Nhập nội dung vào ô TEXT TRANS.", 5000
        )

    def _reset_render_preview(self) -> None:
        self.player.fit_view()
        self._set_render_subtitle_margin(60)

    def _refresh_preset_combo(self, select_name: str = "") -> None:
        current = (
            select_name
            or self.render_preset_combo.currentText()
            or self.settings.default_preset
            or "DEFAULT"
        )
        names = self.preset_manager.list_names()
        self.render_preset_combo.blockSignals(True)
        self.render_preset_combo.clear()
        self.render_preset_combo.addItems(names)
        if current in names:
            self.render_preset_combo.setCurrentText(current)
        elif "DEFAULT" in names:
            self.render_preset_combo.setCurrentText("DEFAULT")
        self.render_preset_combo.blockSignals(False)

    def _set_default_render_preset(self) -> None:
        name = self.render_preset_combo.currentText().strip()
        if not name:
            return
        self.settings.default_preset = name
        self.settings.save()
        self.statusBar().showMessage(
            f"Đã đặt '{name}' làm mặc định cho các project mới.", 5000
        )

    def _create_render_preset(self) -> None:
        name = self.render_preset_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Tên Preset", "Vui lòng nhập tên preset mới.")
            return
        self.render_panel.apply(self.settings)
        style = copy.deepcopy(self.settings.style)
        subtitle_visible = self._render_subtitles_visible
        render_scale = self.settings.render_scale
        render_fps = self.settings.render_fps
        render_crf = self.settings.render_crf
        render_preset = self.settings.render_preset
        lut_path = self.project.lut_path if self.project else ""
        try:
            created = self.preset_manager.create(
                name,
                style=style,
                subtitle_visible=subtitle_visible,
                render_scale=render_scale,
                render_fps=render_fps,
                render_crf=render_crf,
                render_preset=render_preset,
                lut_path=lut_path,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Không Tạo Được Preset", str(exc))
            return
        self._refresh_preset_combo(select_name=created.name)
        self.render_preset_name.setText("")
        self.statusBar().showMessage(f"Đã tạo preset mới: {created.name}", 5000)

    def _apply_render_preset(self) -> None:
        name = self.render_preset_combo.currentText().strip()
        if not name:
            return
        preset = self.preset_manager.get(name)
        if self.project is not None:
            self.project.render_preset = preset.name
            self.project.render_preset_snapshot = preset.to_dict()
            if preset.lut_path:
                self.project.lut_path = preset.lut_path
            self.store.save(self.project)

        self.settings.style = copy.deepcopy(preset.style)
        self.render_panel.load(self.settings)
        self._render_subtitles_visible = preset.subtitle_visible
        if "subtitles" in self.render_tool_buttons:
            self.render_tool_buttons["subtitles"].setChecked(preset.subtitle_visible)
        self.settings.render_scale = preset.render_scale
        self.settings.render_fps = preset.render_fps
        self.settings.render_crf = preset.render_crf
        self.settings.render_preset = preset.render_preset
        self.render_panel.preset.setCurrentText(preset.render_preset)
        self.render_panel.crf.setValue(preset.render_crf)
        self._save_render_preview_style()
        self.statusBar().showMessage(
            f"Đã áp dụng preset '{preset.name}' cho dự án hiện tại.", 5000
        )

    def _update_render_preset(self) -> None:
        name = self.render_preset_combo.currentText().strip()
        if not name:
            return
        preset = self.preset_manager.get(name)
        if preset.builtin:
            QMessageBox.information(
                self,
                "Preset Mặc Định",
                "Không thể ghi đè preset mặc định của hệ thống. "
                "Hãy nhập tên mới vào ô bên cạnh rồi bấm 'Tạo Mới'.",
            )
            return
        self.render_panel.apply(self.settings)
        style = copy.deepcopy(self.settings.style)
        subtitle_visible = self._render_subtitles_visible
        render_scale = self.settings.render_scale
        render_fps = self.settings.render_fps
        render_crf = self.settings.render_crf
        render_preset = self.settings.render_preset
        lut_path = self.project.lut_path if self.project else ""
        try:
            updated = self.preset_manager.update(
                name,
                style=style,
                subtitle_visible=subtitle_visible,
                render_scale=render_scale,
                render_fps=render_fps,
                render_crf=render_crf,
                render_preset=render_preset,
                lut_path=lut_path,
            )
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Không Cập Nhật Được Preset", str(exc))
            return
        if self.project is not None and self.project.render_preset == name:
            self.project.render_preset_snapshot = updated.to_dict()
            self.store.save(self.project)
        self.statusBar().showMessage(f"Đã cập nhật preset: {name}", 5000)

    def _delete_render_preset(self) -> None:
        name = self.render_preset_combo.currentText().strip()
        if not name:
            return
        preset = self.preset_manager.get(name)
        if preset.builtin:
            QMessageBox.information(
                self, "Không Thể Xóa", "Không thể xóa preset mặc định của hệ thống."
            )
            return
        if len(self.preset_manager.list_presets()) <= 1:
            QMessageBox.information(
                self, "Không Thể Xóa", "Không thể xóa preset cuối cùng."
            )
            return
        ans = QMessageBox.question(
            self,
            "Xóa Preset",
            f"Bạn có chắc muốn xóa preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            self.preset_manager.delete(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Không Thể Xóa", str(exc))
            return
        if self.settings.default_preset == name:
            self.settings.default_preset = "DEFAULT"
            self.settings.save()
        self._refresh_preset_combo(select_name="DEFAULT")
        self.statusBar().showMessage(f"Đã xóa preset: {name}", 5000)

    def _choose_output_folder(self) -> None:
        default_folder = self.settings.output_folder or str(Path.home() / "Videos")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Thư mục lưu video render thành công",
            default_folder,
        )
        if folder:
            self.settings.output_folder = folder
            self.settings.save()
            self.statusBar().showMessage(
                f"Đã chọn thư mục xuất video: {folder}", 5000
            )

    def _cue_menu(self, point) -> None:
        """Menu chuot phai tren bang phu de."""
        menu = QMenu(self)
        menu.addAction("Them cau moi", self._add_cue)
        menu.addAction("Tach cau tai vi tri video", self._split_cue)
        menu.addAction("Gop cac cau dang chon", self._merge_cues)
        menu.addSeparator()
        menu.addAction("Dat bat dau = vi tri video", lambda: self._set_time_from_player(True))
        menu.addAction("Dat ket thuc = vi tri video", lambda: self._set_time_from_player(False))
        menu.addAction("Doi thoi gian...", self._shift_cues)
        menu.addSeparator()
        menu.addAction("Xoa cac cau dang chon", self._delete_cues)
        menu.exec(self.cue_view.viewport().mapToGlobal(point))

    # Cac viec lam duoc ngay tren menu chuot phai cua bang du an.
    MENU_STEPS: tuple[tuple[str, str], ...] = (
        ("START: Format Lại Video Gốc", P.STEP_NORMALIZE),
        ("START: Lấy Sub Bằng Chữ", P.STEP_OCR),
        ("START: Lấy Sub Bằng Giọng Nói", P.STEP_ASR),
        ("START: Dịch Phụ Đề", P.STEP_TRANSLATE),
        ("START: Phân Tách Giọng Nam Nữ Tự Động", P.STEP_DIARIZE),
        ("START: Xóa Thoại Gốc Giữ Âm Thanh Nền", P.STEP_KEEP_MUSIC),
        ("START: Xóa Âm Thanh Nền Giữ Thoại Gốc", P.STEP_KEEP_VOICE),
        ("START: Che Mờ Sub Gốc Theo Timeline", P.STEP_BLUR),
        ("START: RENDER LỒNG TIẾNG BẰNG TOOL", P.STEP_RENDER),
        ("START: Xuất Gói Dự Án", P.STEP_EXPORT),
    )
    MENU_SHORTCUTS: dict[str, str] = {
        "START: Format Lại Video Gốc": "Ctrl+B",
        "START: Lấy Sub Bằng Chữ": "Ctrl+R",
        "START: Lấy Sub Bằng Giọng Nói": "Alt+X",
        "START: Dịch Phụ Đề": "Ctrl+T",
        "START: Phân Tách Giọng Nam Nữ Tự Động": "Ctrl+E",
        "START: Xóa Thoại Gốc Giữ Âm Thanh Nền": "Ctrl+G",
        "START: Xóa Âm Thanh Nền Giữ Thoại Gốc": "Ctrl+H",
        "START: Che Mờ Sub Gốc Theo Timeline": "Ctrl+Z",
        "START: RENDER LỒNG TIẾNG BẰNG TOOL": "Ctrl+S",
        "START: Xuất Gói Dự Án": "Ctrl+X",
    }

    @staticmethod
    def _add_project_menu_action(
        menu: QMenu, text: str, callback, shortcut: str = ""
    ) -> QAction:
        action = QAction(text, menu)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutVisibleInContextMenu(True)
        action.triggered.connect(lambda _checked=False: callback())
        menu.addAction(action)
        return action

    def _style_project_menu(self, menu: QMenu) -> None:
        """Mau toi, chu vang va khoang cach giong menu trong anh mau."""
        menu.setObjectName("ProjectContextMenu")
        menu.setStyleSheet(
            "QMenu#ProjectContextMenu { background: #222629; border: 1px solid #3b4247; "
            "padding: 5px; }"
            "QMenu#ProjectContextMenu::item { color: #ffeb3b; font-weight: 700; "
            "padding: 8px 14px 8px 8px; min-width: 310px; }"
            "QMenu#ProjectContextMenu::item:selected { background: #0aa77d; color: white; }"
            "QMenu#ProjectContextMenu::separator { height: 1px; background: #465057; "
            "margin: 5px 7px; }"
        )

    def _project_menu(self, point) -> None:
        """Menu chuot phai tren bang du an: lam duoc cac viec chinh ngay tai day."""
        menu = QMenu(self)
        index = self.project_view.indexAt(point)
        row: ProjectRow | None = None
        if index.isValid():
            selected_proxy_rows = {
                item.row() for item in self.project_view.selectionModel().selectedRows()
            }
            if index.row() not in selected_proxy_rows:
                self.project_view.selectRow(index.row())
            source = self.project_proxy.mapToSource(self.project_proxy.index(index.row(), 0))
            row = self.project_model.row_at(source.row())
        if row is None:
            self._style_project_menu(menu)
            self._add_project_menu_action(menu, "Tạo Dự Án Mới...", self._create_project)
            self._add_project_menu_action(
                menu, "Tách Sub Nhiều Video...", self._choose_batch_videos
            )
            menu.exec(self.project_view.viewport().mapToGlobal(point))
            return

        selected_rows = self._selected_project_rows()
        if len(selected_rows) > 1:
            self._add_project_menu_action(
                menu,
                f"START: TÁCH SUB {len(selected_rows)} PROJECT ĐÃ CHỌN",
                self._queue_selected_projects_ocr,
            )
            menu.addSeparator()

        for label, step in self.MENU_STEPS:
            callback = (
                self._queue_selected_projects_ocr
                if step == P.STEP_OCR and len(selected_rows) > 1
                else partial(self._run_step_for, row, step)
            )
            self._add_project_menu_action(
                menu,
                label,
                callback,
                self.MENU_SHORTCUTS[label],
            )
        self._add_project_menu_action(
            menu,
            "START: CHẠY KỊCH BẢN TỰ ĐỘNG",
            partial(self._run_script_for, row),
            "Alt+A",
        )
        self._add_project_menu_action(
            menu, "STOP", partial(self._stop_tasks_for, row), "Ctrl+Q"
        )
        menu.addSeparator()
        self._add_project_menu_action(
            menu, "Làm Mới Dữ Liệu", self.reload_projects, "F5"
        )
        self._add_project_menu_action(
            menu,
            "Xuất Nội Dung Phụ Đề Ra",
            partial(self._export_subtitle_for, row),
            "F1",
        )
        self._add_project_menu_action(
            menu,
            "Copy Vị Trí Phụ Đề",
            partial(self._copy_ocr_region_for, row),
            "Ctrl+C",
        )
        self._add_project_menu_action(
            menu,
            "Paste Vị Trí Phụ Đề",
            partial(self._paste_ocr_region_for, row),
            "Ctrl+V",
        )
        self._add_project_menu_action(
            menu, "Clone Dự Án", partial(self._clone_project, row), "Ctrl+N"
        )
        self._add_project_menu_action(
            menu, "Delete", self._delete_project, "Ctrl+D"
        )
        self._style_project_menu(menu)
        menu.exec(self.project_view.viewport().mapToGlobal(point))

    def _use_project(self, row: ProjectRow) -> bool:
        """Bao dam du an trong dong dang chon da duoc mo truoc khi chay viec gi do."""
        if self.project_id != row.id:
            self._load_project(row.id, row.folder)
        return self.project is not None and self.project_id == row.id

    def _run_step_for(self, row: ProjectRow, step: str) -> None:
        if self._use_project(row):
            self._run_step(step)

    def _run_script_for(self, row: ProjectRow) -> None:
        if self._use_project(row):
            self._run_script(self.script_panel.checked_steps())

    def _export_subtitle_for(self, row: ProjectRow) -> None:
        if self._use_project(row):
            self._export_subtitle(".srt")

    def _import_subtitle_for(self, row: ProjectRow) -> None:
        """Mo dung project cua hang va nap tep SRT vao bang phu de."""
        if self._use_project(row):
            self._import_subtitle()

    def _stop_tasks_for(self, row: ProjectRow) -> None:
        count = self.tasks.cancel_project(row.id)
        count += self._remove_pending_batch_projects({row.id})
        self._log(f"Đã yêu cầu dừng {count} tác vụ của dự án {row.name}.")

    def _copy_ocr_region_for(self, row: ProjectRow) -> None:
        if not self._use_project(row) or self.project is None:
            return
        region = list(self.project.ocr_region)
        if len(region) != 4:
            QMessageBox.information(
                self, "Chưa có vị trí phụ đề", "Dự án này chưa khoanh vùng phụ đề."
            )
            return
        QApplication.clipboard().setText("AutoSubStudio OCR " + json.dumps(region))
        self.statusBar().showMessage(f"Đã copy vị trí phụ đề: {region}", 5000)

    @staticmethod
    def _parse_copied_region(text: str) -> list[int]:
        raw = str(text or "").strip()
        if raw.startswith("AutoSubStudio OCR "):
            raw = raw.removeprefix("AutoSubStudio OCR ").strip()
        try:
            values = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            values = [item.strip() for item in raw.strip("[]() ").split(",")]
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            return []
        try:
            region = [int(float(value)) for value in values]
        except (TypeError, ValueError):
            return []
        return region if region[2] > 0 and region[3] > 0 else []

    def _paste_ocr_region_for(self, row: ProjectRow) -> None:
        if not self._use_project(row) or self.project is None:
            return
        region = self._parse_copied_region(QApplication.clipboard().text())
        if not region:
            QMessageBox.warning(
                self,
                "Không có vị trí hợp lệ",
                "Clipboard không chứa vị trí phụ đề đã copy từ AutoSub Studio.",
            )
            return
        if self.project.width and self.project.height:
            region = clamp_region(region, self.project.width, self.project.height)
        if len(region) != 4:
            return
        self.project.ocr_region = region
        self.store.save(self.project)
        self.subtitle_panel.set_regions(region, self.project.blur_region)
        self._update_region_label(MODE_OCR, *region)
        if not self._render_preview:
            self._show_region_editor(MODE_OCR)
        self.statusBar().showMessage(f"Đã paste vị trí phụ đề: {region}", 5000)

    @staticmethod
    def _remap_cloned_path(value: str, old_root: Path, new_root: Path) -> str:
        if not value:
            return ""
        path = Path(value)
        try:
            relative = path.resolve().relative_to(old_root.resolve())
        except (OSError, ValueError):
            return value
        return str(new_root / relative)

    def _clone_project(self, row: ProjectRow) -> None:
        """Nhân bản dữ liệu, phụ đề và các tệp nằm trong thư mục dự án."""
        try:
            source = self.store.load(row.folder)
        except ProjectFormatError as exc:
            QMessageBox.critical(self, "Không clone được dự án", str(exc))
            return
        clone_name = f"{row.name} - Copy"
        with self.db.session() as session:
            record = Project(name=clone_name, status="Đã clone")
            session.add(record)
            session.flush()
            clone_id = record.id
        clone = self.store.create(clone_id, clone_name)
        old_root, new_root = Path(source.folder), Path(clone.folder)
        try:
            shutil.copytree(old_root, new_root, dirs_exist_ok=True)
            clone = ProjectData.from_dict(source.to_dict())
            clone.project_id = clone_id
            clone.name = clone_name
            clone.folder = str(new_root)
            clone.auto_srt_path = ""
            for field in (
                "audio_path",
                "music_path",
                "voice_path",
                "dub_path",
                "render_path",
                "lut_path",
            ):
                setattr(
                    clone,
                    field,
                    self._remap_cloned_path(getattr(clone, field), old_root, new_root),
                )
            self.store.save(clone)
        except (OSError, ValueError) as exc:
            self.store.delete(new_root)
            with self.db.session() as session:
                failed = session.get(Project, clone_id)
                if failed is not None:
                    session.delete(failed)
            QMessageBox.critical(self, "Không clone được dự án", str(exc))
            return

        with self.db.session() as session:
            cloned_record = session.get(Project, clone_id)
            if cloned_record is not None:
                cloned_record.folder = clone.folder
                cloned_record.video_path = clone.video_path
                cloned_record.duration = clone.duration
                cloned_record.language = clone.doc.language
                cloned_record.target_language = (
                    clone.doc.target_language or self.settings.target_language
                )
                cloned_record.cue_count = len(clone.doc.cues)
                cloned_record.char_count = clone.doc.total_chars
                cloned_record.has_subtitle = bool(clone.doc.cues)
                record.has_translation = clone.doc.translated_count > 0
                record.has_dub = bool(clone.dub_path)
                record.has_render = bool(clone.render_path)
        self.reload_projects()
        self._load_project(clone_id, clone.folder)
        self._log(f"Đã clone dự án '{row.name}' thành '{clone_name}'.")

    def _on_row_action(self, proxy_row: int, action: str) -> None:
        """Xu ly khi bam mot nut chuc nang tren dong cua bang du an."""
        source = self.project_proxy.mapToSource(self.project_proxy.index(proxy_row, 0))
        row = self.project_model.row_at(source.row())
        if row is None:
            return
        self.project_view.selectRow(proxy_row)
        if action == ACTION_STOP:
            count = self.tasks.cancel_project(row.id)
            self._log(f"Da yeu cau dung {count} tac vu cua du an {row.name}.")
        elif action == ACTION_FOLDER:
            self._open_path(row.folder)
        elif action == ACTION_SRT:
            self._import_subtitle_for(row)
        elif action == ACTION_RUN:
            if self.project_id != row.id:
                self._load_project(row.id, row.folder)
            self._run_script(self.script_panel.checked_steps())

    def _on_tab_changed(self, index: int) -> None:
        current = self.tab_stack.widget(index)
        dub_page = self._panel_scroll_areas.get(self.dub_panel, self.dub_panel)
        if current is dub_page and not self._voices_loaded:
            self._voices_loaded = True
            self.dub_panel.load(self.settings)
            self.dub_panel.refresh_status()

    def _open_voice_library(self) -> None:
        current = (self.settings.local_voice or self.settings.tts_voice).strip()
        dlg = VoiceLibraryDialog(parent=self, current_voice=current)
        dlg.selectedVoice.connect(self._on_voice_selected)
        if dlg.exec() and dlg.selected_voice:
            self._on_voice_selected(dlg.selected_voice)

    def _on_voice_selected(self, voice_id: str) -> None:
        voice_id = voice_id.strip()
        if not voice_id:
            return
        self.settings.local_voice = voice_id
        self.settings.tts_voice = voice_id
        self.settings.tts_provider = "Local Voice"
        self.settings.save()
        self.dub_panel.load(self.settings)
        self.dub_panel.refresh_status()
        self._log(f"Đã chọn giọng đọc: {voice_id}")

    def _preview_voice(self) -> None:
        self.dub_panel.apply(self.settings)
        if not self._require_voice_model_ready():
            return

        voice_id = (self.settings.local_voice or self.settings.tts_voice).strip()

        # Text to synthesize: use selected cue text if project, else localized sample
        text = ""
        if self.project is not None and hasattr(self, "cue_model") and self.cue_model is not None:
            row = self._current_row()
            doc = getattr(self.cue_model, "doc", None)
            cues = doc.cues if doc and hasattr(doc, "cues") else []
            if 0 <= row < len(cues):
                cue = cues[row]
                text = cue.translation.strip() or cue.text.strip()
        if not text:
            info = get_voice_info(voice_id)
            lang = info.language if info else ""
            if lang.startswith("vi"):
                text = "Xin chào, đây là giọng đọc thử nghiệm."
            elif lang.startswith("zh"):
                text = "你好，这是测试语音。"
            else:
                text = "Hello, this is a test voice."

        speed = max(0.25, min(4.0, self.settings.tts_speed_percent / 100.0))

        if self.project is not None:
            out_dir = Path(self.project.folder) / "temp"
            project_id = self.project_id
        else:
            import tempfile

            out_dir = Path(tempfile.gettempdir()) / "autosub_preview"
            project_id = 0
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"preview_{safe_name(voice_id)}.wav"

        def job(ctx: TaskContext) -> str:
            ctx.progress(10)
            ctx.log("Đang tạo giọng đọc nghe thử...")
            produced = synthesize_piper(
                text=text.replace("\n", " "),
                out_path=out,
                voice_id=voice_id,
                speed=speed,
            )
            ctx.progress(100)
            ctx.log("Đã tạo giọng đọc nghe thử.")
            return str(produced)

        task_id = self.tasks.submit("Nghe thử giọng đọc", job, project_id=project_id, timeout=60)
        self._preview_task_ids.add(task_id)
        self._log(f"Đang tạo nghe thử cho giọng '{voice_id}'...")

    # ------------------------------------------------------------------ cai dat duong dan

    def _choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Chon thu muc lam viec", self.settings.workspace
        )
        if not path:
            return
        self.settings.workspace = path
        ensure_workspace(path)
        self.settings.save()
        self.db = Database(Path(path) / "db" / "app.db")
        self.db.ensure_default_scripts()
        self.store = ProjectStore(path)
        self.project = None
        self.project_id = 0
        self.cue_model.set_document(SubtitleDoc())
        self.player.clear()
        self.reload_projects()
        self._refresh_scripts()
        self._load_settings_into_ui()
        self._log(f"Da doi thu muc lam viec: {path}")

    def _import_project_package(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Nhập Project AutoSub",
            "",
            "Project (project_export.json app.db *.json *.db);;NTS Database (*.db);;JSON (*.json)",
        )
        if not path:
            return
        try:
            if Path(path).suffix.casefold() == ".db":
                names = nts_import.project_names(path)
                if not names:
                    raise ValueError("Database NTS không có project nào.")
                name, accepted = QInputDialog.getItem(
                    self,
                    "Chọn Project NTS",
                    "Project cần nhập:",
                    names,
                    0,
                    False,
                )
                if not accepted:
                    return
                video_value, document, nts_config = nts_import.load_project(path, name)
                video = Path(video_value)
                if not video.is_file():
                    raise ValueError(f"Video của project NTS không còn tồn tại: {video}")
                data = self._create_project_from_video(str(video))
                data.doc = document
                nts_import.apply_global_settings(
                    self.settings, nts_import.load_global_settings(path)
                )
                nts_import.apply_project_settings(self.settings, nts_config)
                self.settings.remember_active_profile()
                self.settings.save()
                self.store.save(data)
                self.reload_projects()
                self._load_project(data.project_id, data.folder)
                self._load_settings_into_ui()
                self._log(f"Da nhap project NTS '{name}' tu {path}")
                return
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            video = Path(str(raw.get("video") or ""))
            if not video.is_file():
                raise ValueError("Video trong project không còn tồn tại.")
            data = self._create_project_from_video(str(video))
            data.doc = SubtitleDoc.from_dict(
                {
                    "language": raw.get("language", ""),
                    "target_language": raw.get("target_language", "vi"),
                    "cues": raw.get("cues") or [],
                }
            )
            dub_audio = Path(str(raw.get("dub_audio") or ""))
            if dub_audio.is_file():
                data.dub_path = str(dub_audio)
            self.store.save(data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Không Nhập Được Project", str(exc))
            return
        self.reload_projects()
        self._load_project(data.project_id, data.folder)
        self._log(f"Da nhap project: {path}")

    def _choose_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chon ffmpeg.exe", "", "FFmpeg (ffmpeg.exe ffmpeg);;Tat ca (*.*)"
        )
        if not path:
            return
        self.settings_panel.ffmpeg.setText(path)
        self.settings.ffmpeg_path = path
        self.settings.save()
        self.ff = FFmpeg(path, self.settings.ffprobe_path)
        self._load_settings_into_ui()

    def _choose_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Chon thu muc model", self.settings.model_dir or ""
        )
        if path:
            self.settings_panel.model_dir.setText(path)

    def _clean_temp(self) -> None:
        if self.project is None:
            QMessageBox.information(
                self, "Chua mo du an", "Hay mo mot du an de don tep trung gian cua no."
            )
            return
        freed = self.store.clean_temp(self.project)
        self._log(f"Da don {freed / 1024 / 1024:.1f} MB tep trung gian.")
        self.statusBar().showMessage(f"Da giai phong {freed / 1024 / 1024:.1f} MB.", 5000)

    def _open_path(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "Khong tim thay", f"Khong ton tai: {path}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(p))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except OSError as exc:
            QMessageBox.warning(self, "Khong mo duoc", str(exc))

    # ------------------------------------------------------------------ nhat ky

    def _log(self, message: str) -> None:
        self.logAppended.emit(message)

    def _append_log(self, message: str) -> None:
        self._session_log.append(message)
        if len(self._session_log) > 3000:
            del self._session_log[: len(self._session_log) - 3000]

    # ------------------------------------------------------------------ dong ung dung

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.tasks.active_count():
            answer = QMessageBox.question(
                self,
                "Con tac vu dang chay",
                "Van con tac vu chua xong. Dong ung dung se dung chung lai. Tiep tuc?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.tasks.cancel_all()
            self.tasks.wait_for_done(8000)
        if self.project is not None and self._dirty:
            self._save_project()
        with contextlib.suppress(OSError):
            self._collect_settings()
            self.settings.remember_active_profile()
            self.settings.save()
        event.accept()


def run() -> int:
    """Diem vao cua ung dung."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    init_app_font(app)
    app.setStyle("Fusion")
    app.setStyleSheet(get_qss())
    window = MainWindow()
    window.showMaximized()
    return app.exec()
