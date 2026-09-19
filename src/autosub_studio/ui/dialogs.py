"""Cac hop thoai phu tro."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.editing import Issue
from ..services import ai_gateway
from ..services.settings import Settings


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


class AIGatewayDialog(QDialog):
    """Hop thoai cau hinh ket noi AI Gateway duy nhat cua ung dung."""

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Cấu hình AI Gateway")
        self.resize(580, 480)

        # 1. Endpoint & Khoa API
        conn_box = QGroupBox("Kết Nối Server AI (Chuẩn OpenAI)")
        conn_layout = QFormLayout(conn_box)
        conn_layout.setContentsMargins(10, 14, 10, 10)
        conn_layout.setSpacing(8)

        self.endpoint = QLineEdit()
        self.endpoint.setPlaceholderText("https://api.openai.com/v1 hoặc http://localhost:8000/v1")
        self.endpoint.setText(settings.ai_endpoint)
        conn_layout.addRow("Endpoint:", self.endpoint)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Khóa API (lưu mã hóa bảo mật trên máy)")
        self.api_key.setText(Settings.get_secret("ai_gateway_key"))

        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key, 1)
        self.show_key = QCheckBox("Hiện")
        self.show_key.toggled.connect(
            lambda checked: self.api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(self.show_key)
        conn_layout.addRow("Khóa API:", key_row)

        self.btn_test_conn = QPushButton("Kiểm Tra Kết Nối")
        self.btn_test_conn.clicked.connect(self._test_connection)
        conn_layout.addRow("", self.btn_test_conn)

        # 2. Cau hinh Model Sub & Prime
        model_box = QGroupBox("Cấu Hình Model & Mức Độ Suy Nghĩ (Thinking)")
        model_layout = QFormLayout(model_box)
        model_layout.setContentsMargins(10, 14, 10, 10)
        model_layout.setSpacing(8)

        # Sub
        self.model_sub = QLineEdit()
        self.model_sub.setPlaceholderText("Tên model thực tế (mặc định: sub)")
        self.model_sub.setText(settings.ai_model_sub or "sub")

        self.thinking_sub = QComboBox()
        self.thinking_sub.addItems(["none", "low", "medium", "high"])
        self.thinking_sub.setCurrentText(settings.ai_thinking_sub or "low")

        self.btn_test_sub = QPushButton("Test Model Sub")
        self.btn_test_sub.clicked.connect(self._test_sub_model)

        sub_row = QHBoxLayout()
        sub_row.addWidget(self.model_sub, 1)
        sub_row.addWidget(QLabel("Thinking:"))
        sub_row.addWidget(self.thinking_sub)
        sub_row.addWidget(self.btn_test_sub)
        model_layout.addRow("Alias 'sub':", sub_row)

        # Prime
        self.model_prime = QLineEdit()
        self.model_prime.setPlaceholderText("Tên model thực tế (mặc định: prime)")
        self.model_prime.setText(settings.ai_model_prime or "prime")

        self.thinking_prime = QComboBox()
        self.thinking_prime.addItems(["none", "low", "medium", "high"])
        self.thinking_prime.setCurrentText(settings.ai_thinking_prime or "medium")

        self.btn_test_prime = QPushButton("Test Model Prime")
        self.btn_test_prime.clicked.connect(self._test_prime_model)

        prime_row = QHBoxLayout()
        prime_row.addWidget(self.model_prime, 1)
        prime_row.addWidget(QLabel("Thinking:"))
        prime_row.addWidget(self.thinking_prime)
        prime_row.addWidget(self.btn_test_prime)
        model_layout.addRow("Alias 'prime':", prime_row)

        # 3. Trang thai ket qua kiem tra
        status_box = QGroupBox("Trạng Thái")
        status_layout = QVBoxLayout(status_box)
        self.status = QLabel("Nhập thông tin kết nối và bấm kiểm tra.")
        self.status.setWordWrap(True)
        self.status.setObjectName("Muted")
        status_layout.addWidget(self.status)

        # 4. Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(conn_box)
        layout.addWidget(model_box)
        layout.addWidget(status_box)
        layout.addWidget(buttons)

    def _set_status(self, text: str, mode: str = "Muted") -> None:
        self.status.setText(text)
        self.status.setObjectName(mode)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _test_connection(self) -> None:
        ep = self.endpoint.text().strip()
        key = self.api_key.text().strip()
        if not ep:
            self._set_status("Lỗi: Chưa nhập Endpoint.", "Error")
            return
        self._set_status("Đang kiểm tra kết nối...", "Muted")
        ok, msg = ai_gateway.test_connection(ep, key)
        self._set_status(msg, "Ok" if ok else "Error")

    def _test_sub_model(self) -> None:
        ep = self.endpoint.text().strip()
        key = self.api_key.text().strip()
        model = self.model_sub.text().strip() or "sub"
        thinking = self.thinking_sub.currentText()
        if not ep:
            self._set_status("Lỗi: Chưa nhập Endpoint.", "Error")
            return
        self._set_status(f"Đang kiểm tra model '{model}'...", "Muted")
        ok, msg = ai_gateway.test_model(ep, key, model=model, thinking=thinking)
        self._set_status(msg, "Ok" if ok else "Error")

    def _test_prime_model(self) -> None:
        ep = self.endpoint.text().strip()
        key = self.api_key.text().strip()
        model = self.model_prime.text().strip() or "prime"
        thinking = self.thinking_prime.currentText()
        if not ep:
            self._set_status("Lỗi: Chưa nhập Endpoint.", "Error")
            return
        self._set_status(f"Đang kiểm tra model '{model}'...", "Muted")
        ok, msg = ai_gateway.test_model(ep, key, model=model, thinking=thinking)
        self._set_status(msg, "Ok" if ok else "Error")

    def _on_save(self) -> None:
        self.settings.ai_endpoint = self.endpoint.text().strip()
        self.settings.ai_model_sub = self.model_sub.text().strip() or "sub"
        self.settings.ai_thinking_sub = self.thinking_sub.currentText()
        self.settings.ai_model_prime = self.model_prime.text().strip() or "prime"
        self.settings.ai_thinking_prime = self.thinking_prime.currentText()
        Settings.set_secret("ai_gateway_key", self.api_key.text().strip())
        self.settings.save()
        self.accept()


class _VoiceDownloadWorker(QThread):
    progress = Signal(str, int, int)
    finished = Signal(str, bool, str)

    def __init__(self, manager: Any, voice_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.voice_id = voice_id

    def run(self) -> None:
        try:
            def on_progress(cur: int, total: int) -> None:
                self.progress.emit(self.voice_id, cur, total)

            self.manager.download_voice(self.voice_id, on_progress=on_progress)
            self.finished.emit(self.voice_id, True, "Tải thành công")
        except Exception as exc:
            self.finished.emit(self.voice_id, False, str(exc))


class _VoicePreviewWorker(QThread):
    finished = Signal(str, bool, str, object)

    def __init__(
        self,
        synth_factory: Any,
        voice_id: str,
        text: str,
        out_path: Path,
        manager: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.synth_factory = synth_factory
        self.voice_id = voice_id
        self.text = text
        self.out_path = out_path
        self.manager = manager

    def run(self) -> None:
        try:
            try:
                produced = self.synth_factory(
                    self.text,
                    self.out_path,
                    voice_id=self.voice_id,
                    manager=self.manager,
                )
            except TypeError:
                try:
                    produced = self.synth_factory(
                        self.text,
                        self.out_path,
                        voice_id=self.voice_id,
                    )
                except TypeError:
                    produced = self.synth_factory(self.text, self.out_path)
            self.finished.emit(self.voice_id, True, "", produced)
        except Exception as exc:
            self.finished.emit(self.voice_id, False, str(exc), None)


class VoiceLibraryDialog(QDialog):
    """Thư viện giọng đọc Piper Local."""

    selectedVoice = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        manager: Any | None = None,
        synth_factory: Any | None = None,
        player_factory: Any | None = None,
        current_voice: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Thư viện giọng đọc")
        self.resize(800, 500)

        from ..providers import local_voice as lv

        self._manager = manager if manager is not None else lv.get_default_manager()
        self._synth_factory = synth_factory if synth_factory is not None else lv.synthesize_piper
        self._player_factory = player_factory
        self._selected_voice = current_voice or ""
        self._download_workers: dict[str, _VoiceDownloadWorker] = {}
        self._preview_worker: _VoicePreviewWorker | None = None
        self._catalog: list[Any] = []
        self._filtered: list[Any] = []

        if player_factory is not None:
            self._player = player_factory()
        else:
            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

                self._player = QMediaPlayer(self)
                self._audio_output = QAudioOutput(self)
                self._player.setAudioOutput(self._audio_output)
            except Exception:
                self._player = None

        # Bo loc tim kiem va ngon ngu
        filter_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Tìm kiếm giọng đọc theo tên, mã...")
        self.search_edit.textChanged.connect(self._apply_filter)

        self.language_combo = QComboBox()
        self.language_combo.addItem("Tất cả ngôn ngữ", "")
        self.language_combo.currentIndexChanged.connect(self._apply_filter)

        filter_layout.addWidget(QLabel("Tìm kiếm:"))
        filter_layout.addWidget(self.search_edit, 1)
        filter_layout.addWidget(QLabel("Ngôn ngữ:"))
        filter_layout.addWidget(self.language_combo)

        # Bang danh muc giong doc
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Giọng đọc",
            "Ngôn ngữ",
            "Engine",
            "Giấy phép",
            "Trạng thái",
            "Thao tác",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)

        # Cac nut thao tac cua hop thoai
        self.btn_download = QPushButton("Tải về")
        self.btn_delete = QPushButton("Xóa")
        self.btn_use = QPushButton("Sử dụng")
        self.btn_preview = QPushButton("Nghe thử")
        self.btn_close = QPushButton("Đóng")

        self.btn_download.clicked.connect(self._download_selected)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_use.clicked.connect(self._use_selected)
        self.btn_preview.clicked.connect(self._preview_selected)
        self.btn_close.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_download)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_use)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_close)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("Muted")

        layout = QVBoxLayout(self)
        layout.addLayout(filter_layout)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.lbl_status)
        layout.addLayout(btn_layout)

        self._load_catalog()
        self._on_selection_changed()

    @property
    def selected_voice(self) -> str:
        return self._selected_voice

    def _load_catalog(self) -> None:
        from ..providers import local_voice as lv

        if hasattr(self._manager, "list_catalog"):
            catalog = self._manager.list_catalog()
        elif hasattr(self._manager, "catalog"):
            c = self._manager.catalog
            catalog = list(c.values()) if isinstance(c, dict) else list(c)
        else:
            catalog = lv.list_catalog()

        self._catalog = list(catalog)

        languages: dict[str, str] = {}
        for voice in self._catalog:
            code = getattr(voice, "language", "")
            name = getattr(voice, "language_name", "") or code
            if code and code not in languages:
                languages[code] = name

        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        self.language_combo.addItem("Tất cả ngôn ngữ", "")
        for code, name in sorted(languages.items(), key=lambda x: x[1]):
            self.language_combo.addItem(f"{name} ({code})", code)
        self.language_combo.blockSignals(False)

        self._render_table()

    def _apply_filter(self) -> None:
        self._render_table()

    def _status_vietnamese(self, voice_id: str) -> str:
        from ..providers import local_voice as lv

        if voice_id in self._download_workers:
            return "Đang tải"
        try:
            st = self._manager.get_status(voice_id)
        except Exception:
            return "Chưa tải"
        if st == lv.STATUS_READY:
            return "Đã tải"
        elif st == lv.STATUS_DOWNLOADING:
            return "Đang tải"
        return "Chưa tải"

    def _render_table(self) -> None:
        search = self.search_edit.text().strip().casefold()
        selected_lang = self.language_combo.currentData() or ""

        self._filtered = []
        for voice in self._catalog:
            v_lang = getattr(voice, "language", "")
            v_lang_name = getattr(voice, "language_name", "")
            v_name = getattr(voice, "name", "")
            v_id = getattr(voice, "id", "")

            if selected_lang and v_lang != selected_lang:
                continue
            if search and (
                search not in v_name.casefold()
                and search not in v_id.casefold()
                and search not in v_lang.casefold()
                and search not in v_lang_name.casefold()
            ):
                continue
            self._filtered.append(voice)

        self.table.setRowCount(len(self._filtered))
        target_row = -1
        for row, voice in enumerate(self._filtered):
            v_id = getattr(voice, "id", "")
            v_name = getattr(voice, "name", "")
            v_lang = getattr(voice, "language_name", "") or getattr(voice, "language", "")
            v_license = getattr(voice, "license", "")

            item_voice = QTableWidgetItem(v_name)
            item_voice.setData(Qt.ItemDataRole.UserRole, v_id)
            self.table.setItem(row, 0, item_voice)
            self.table.setItem(row, 1, QTableWidgetItem(v_lang))
            self.table.setItem(row, 2, QTableWidgetItem("Piper"))
            self.table.setItem(row, 3, QTableWidgetItem(v_license))

            st_text = self._status_vietnamese(v_id)
            self.table.setItem(row, 4, QTableWidgetItem(st_text))

            action_widget = self._create_row_action_widget(v_id)
            self.table.setCellWidget(row, 5, action_widget)

            if v_id == self._selected_voice:
                target_row = row

        if target_row >= 0:
            self.table.selectRow(target_row)
        elif self.table.rowCount() > 0 and self.table.currentRow() < 0:
            self.table.selectRow(0)

        self._update_button_states()

    def _create_row_action_widget(self, voice_id: str) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        status = self._status_vietnamese(voice_id)
        if status == "Đã tải":
            btn_use = QPushButton("Sử dụng")
            btn_use.setFixedHeight(24)
            btn_use.clicked.connect(lambda _, v=voice_id: self._use_voice(v))
            layout.addWidget(btn_use)

            btn_prev = QPushButton("Nghe thử")
            btn_prev.setFixedHeight(24)
            btn_prev.clicked.connect(lambda _, v=voice_id: self._preview_voice(v))
            layout.addWidget(btn_prev)

            btn_del = QPushButton("Xóa")
            btn_del.setFixedHeight(24)
            btn_del.clicked.connect(lambda _, v=voice_id: self._delete_voice(v))
            layout.addWidget(btn_del)
        elif status == "Đang tải":
            lbl = QLabel("Đang tải...")
            lbl.setObjectName("Muted")
            layout.addWidget(lbl)
        else:
            btn_dl = QPushButton("Tải về")
            btn_dl.setFixedHeight(24)
            btn_dl.clicked.connect(lambda _, v=voice_id: self._download_voice(v))
            layout.addWidget(btn_dl)

        layout.addStretch(1)
        return widget

    def _selected_voice_id_from_table(self) -> str:
        row = self.table.currentRow()
        if row >= 0:
            item = self.table.item(row, 0)
            if item is not None:
                return str(item.data(Qt.ItemDataRole.UserRole) or item.text())
        return ""

    def _on_selection_changed(self) -> None:
        self._update_button_states()

    def _update_button_states(self) -> None:
        voice_id = self._selected_voice_id_from_table()
        if not voice_id:
            self.btn_download.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_use.setEnabled(False)
            self.btn_preview.setEnabled(False)
            return

        status = self._status_vietnamese(voice_id)
        if status == "Đã tải":
            self.btn_download.setEnabled(False)
            self.btn_delete.setEnabled(True)
            self.btn_use.setEnabled(True)
            self.btn_preview.setEnabled(True)
        elif status == "Đang tải":
            self.btn_download.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_use.setEnabled(False)
            self.btn_preview.setEnabled(False)
        else:
            self.btn_download.setEnabled(True)
            self.btn_delete.setEnabled(False)
            self.btn_use.setEnabled(False)
            self.btn_preview.setEnabled(False)

    def _download_selected(self) -> None:
        vid = self._selected_voice_id_from_table()
        if vid:
            self._download_voice(vid)

    def _delete_selected(self) -> None:
        vid = self._selected_voice_id_from_table()
        if vid:
            self._delete_voice(vid)

    def _use_selected(self) -> None:
        vid = self._selected_voice_id_from_table()
        if vid:
            self._use_voice(vid)

    def _preview_selected(self) -> None:
        vid = self._selected_voice_id_from_table()
        if vid:
            self._preview_voice(vid)

    def _on_row_double_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        vid = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
        if self._status_vietnamese(vid) == "Đã tải":
            self._use_voice(vid)
        elif self._status_vietnamese(vid) == "Chưa tải":
            self._download_voice(vid)

    def _download_voice(self, voice_id: str) -> None:
        if voice_id in self._download_workers:
            return
        worker = _VoiceDownloadWorker(self._manager, voice_id, parent=self)
        self._download_workers[voice_id] = worker
        worker.progress.connect(self._on_download_progress)
        worker.finished.connect(self._on_download_finished)
        self.lbl_status.setText(f"Đang tải {voice_id}...")
        self._render_table()
        worker.start()

    def _on_download_progress(self, voice_id: str, cur: int, total: int) -> None:
        if total > 0:
            pct = int(cur * 100 / total)
            self.lbl_status.setText(f"Đang tải {voice_id}: {pct}%")
        else:
            self.lbl_status.setText(f"Đang tải {voice_id}...")

    def _on_download_finished(self, voice_id: str, success: bool, msg: str) -> None:
        self._download_workers.pop(voice_id, None)
        if success:
            self.lbl_status.setText(f"Tải thành công: {voice_id}")
        else:
            self.lbl_status.setText(f"Lỗi tải {voice_id}: {msg}")
            QMessageBox.critical(self, "Lỗi tải giọng đọc", f"Không thể tải {voice_id}:\n{msg}")
        self._render_table()

    def _delete_voice(self, voice_id: str) -> None:
        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa giọng đọc '{voice_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._manager.delete_voice(voice_id)
            if self._selected_voice == voice_id:
                self._selected_voice = ""
            self.lbl_status.setText(f"Đã xóa {voice_id}")
        except Exception as exc:
            self.lbl_status.setText(f"Lỗi xóa {voice_id}: {exc}")
            QMessageBox.critical(self, "Lỗi xóa giọng", str(exc))
        self._render_table()

    def _use_voice(self, voice_id: str) -> None:
        self._selected_voice = voice_id
        self.selectedVoice.emit(voice_id)
        self.accept()

    def _preview_voice(self, voice_id: str) -> None:
        import tempfile

        from ..providers import local_voice as lv

        voice_info = lv.get_voice_info(voice_id)
        lang = voice_info.language if voice_info else ""
        if lang.startswith("vi"):
            text = "Xin chào, đây là giọng đọc thử nghiệm."
        elif lang.startswith("zh"):
            text = "你好，这是测试语音。"
        else:
            text = "Hello, this is a test voice."

        temp_dir = Path(tempfile.gettempdir()) / "autosub_preview"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = temp_dir / f"preview_{voice_id}.wav"

        self.lbl_status.setText(f"Đang tạo giọng đọc nghe thử cho {voice_id}...")
        self.btn_preview.setEnabled(False)

        worker = _VoicePreviewWorker(
            self._synth_factory,
            voice_id,
            text,
            out_path,
            manager=self._manager,
            parent=self,
        )
        self._preview_worker = worker
        worker.finished.connect(self._on_preview_finished)
        worker.start()

    def _on_preview_finished(
        self, voice_id: str, success: bool, msg: str, out_path: object
    ) -> None:
        self.btn_preview.setEnabled(True)
        if not success or out_path is None:
            self.lbl_status.setText(f"Lỗi nghe thử {voice_id}: {msg}")
            QMessageBox.critical(self, "Lỗi nghe thử", f"Không thể tạo âm thanh nghe thử:\n{msg}")
            return

        self.lbl_status.setText(f"Đang phát {voice_id}...")
        if self._player is not None:
            try:
                from PySide6.QtCore import QUrl

                p = Path(str(out_path))
                if hasattr(self._player, "setSource"):
                    self._player.setSource(QUrl.fromLocalFile(str(p.resolve())))
                if hasattr(self._player, "play"):
                    self._player.play()
            except Exception as exc:
                self.lbl_status.setText(f"Lỗi phát âm thanh: {exc}")
