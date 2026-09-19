"""Kiem thu giao dien thu vien giong doc Piper Local V2 (Contract v2-p3-voice-ui-r2c)."""

from __future__ import annotations

from unittest.mock import MagicMock

from PySide6.QtWidgets import QMessageBox

from autosub_studio.providers.local_voice import (
    STATUS_DOWNLOADING,
    STATUS_NOT_DOWNLOADED,
    STATUS_READY,
    VoiceArtifact,
    VoiceInfo,
)
from autosub_studio.services.settings import Settings
from autosub_studio.ui.dialogs import VoiceLibraryDialog
from autosub_studio.ui.panels import DubPanel


def _make_dummy_catalog() -> list[VoiceInfo]:
    return [
        VoiceInfo(
            id="vi_VN-vais1000-medium",
            name="VAIS 1000 (Tiếng Việt)",
            language="vi-VN",
            language_name="Tiếng Việt",
            license="CC BY 4.0",
            artifacts=(
                VoiceArtifact("model.onnx", "http://x/m.onnx", "hash1", 1000),
                VoiceArtifact("model.onnx.json", "http://x/m.json", "hash2", 100),
            ),
        ),
        VoiceInfo(
            id="en_US-bryce-medium",
            name="Bryce (English US)",
            language="en-US",
            language_name="English (US)",
            license="Public Domain",
            artifacts=(
                VoiceArtifact("model.onnx", "http://x/m.onnx", "hash3", 1000),
                VoiceArtifact("model.onnx.json", "http://x/m.json", "hash4", 100),
            ),
        ),
        VoiceInfo(
            id="zh_CN-chaowen-medium",
            name="Chaowen (Chinese)",
            language="zh-CN",
            language_name="Chinese",
            license="CC0",
            artifacts=(
                VoiceArtifact("model.onnx", "http://x/m.onnx", "hash5", 1000),
                VoiceArtifact("model.onnx.json", "http://x/m.json", "hash6", 100),
            ),
        ),
    ]


class DummyVoiceManager:
    """Mock PiperVoiceManager cho unit test."""

    def __init__(self, catalog: list[VoiceInfo] | None = None) -> None:
        self._catalog = catalog if catalog is not None else _make_dummy_catalog()
        self.statuses: dict[str, str] = {
            "vi_VN-vais1000-medium": STATUS_READY,
            "en_US-bryce-medium": STATUS_NOT_DOWNLOADED,
            "zh_CN-chaowen-medium": STATUS_DOWNLOADING,
        }
        self.downloaded_calls: list[str] = []
        self.deleted_calls: list[str] = []

    def list_catalog(self, language: str = "") -> list[VoiceInfo]:
        if not language:
            return list(self._catalog)
        return [v for v in self._catalog if v.language == language]

    def get_status(self, voice_id: str, verify_checksum: bool = True) -> str:
        return self.statuses.get(voice_id, STATUS_NOT_DOWNLOADED)

    def download_voice(self, voice_id: str, on_progress=None, cancel_token=None) -> None:
        self.downloaded_calls.append(voice_id)
        if on_progress:
            on_progress(50, 100)
            on_progress(100, 100)
        self.statuses[voice_id] = STATUS_READY

    def delete_voice(self, voice_id: str) -> bool:
        self.deleted_calls.append(voice_id)
        self.statuses[voice_id] = STATUS_NOT_DOWNLOADED
        return True


class DummyPlayer:
    """Mock media player."""

    def __init__(self) -> None:
        self.source = None
        self.played = False
        self.stopped = False

    def setSource(self, src) -> None:
        self.source = src

    def play(self) -> None:
        self.played = True

    def stop(self) -> None:
        self.stopped = True


class TestVoiceLibraryDialog:
    """Kiem tra hop thoai thu vien giong doc."""

    def test_catalog_columns_and_statuses(self, qapp):
        mgr = DummyVoiceManager()
        dlg = VoiceLibraryDialog(manager=mgr)
        table = dlg.table

        # 6 cot: voice/language/engine/license/status/action
        assert table.columnCount() == 6
        assert table.rowCount() == 3

        # Row 0: VAIS (vi-VN) - status: "Đã tải"
        assert table.item(0, 0).text() == "VAIS 1000 (Tiếng Việt)"
        assert table.item(0, 1).text() == "Tiếng Việt"
        assert table.item(0, 2).text() == "Piper"
        assert table.item(0, 3).text() == "CC BY 4.0"
        assert table.item(0, 4).text() == "Đã tải"

        # Row 1: Bryce (en-US) - status: "Chưa tải"
        assert table.item(1, 0).text() == "Bryce (English US)"
        assert table.item(1, 4).text() == "Chưa tải"

        # Row 2: Chaowen (zh-CN) - status: "Đang tải"
        assert table.item(2, 4).text() == "Đang tải"
        dlg.deleteLater()

    def test_search_and_language_filter(self, qapp):
        mgr = DummyVoiceManager()
        dlg = VoiceLibraryDialog(manager=mgr)

        # Tim kiem theo ten
        dlg.search_edit.setText("Bryce")
        assert dlg.table.rowCount() == 1
        assert dlg.table.item(0, 0).text() == "Bryce (English US)"

        # Xoa tim kiem
        dlg.search_edit.setText("")
        assert dlg.table.rowCount() == 3

        # Loc theo ngon ngu
        idx = dlg.language_combo.findData("vi-VN")
        dlg.language_combo.setCurrentIndex(idx)
        assert dlg.table.rowCount() == 1
        assert dlg.table.item(0, 0).text() == "VAIS 1000 (Tiếng Việt)"

        dlg.deleteLater()

    def test_use_button_and_selected_voice(self, qapp):
        mgr = DummyVoiceManager()
        dlg = VoiceLibraryDialog(manager=mgr)

        selected = []
        dlg.selectedVoice.connect(selected.append)

        # Chon row 0 (VAIS)
        dlg.table.selectRow(0)
        assert dlg.btn_use.isEnabled()
        dlg.btn_use.click()

        assert dlg.selected_voice == "vi_VN-vais1000-medium"
        assert selected == ["vi_VN-vais1000-medium"]
        dlg.deleteLater()

    def test_download_worker(self, qapp):
        mgr = DummyVoiceManager()
        dlg = VoiceLibraryDialog(manager=mgr)

        # Row 1 (Bryce) chua tai
        dlg.table.selectRow(1)
        assert dlg.btn_download.isEnabled()

        dlg.btn_download.click()
        # Cho worker hoan thanh
        if "en_US-bryce-medium" in dlg._download_workers:
            dlg._download_workers["en_US-bryce-medium"].wait(5000)
            qapp.processEvents()

        assert "en_US-bryce-medium" in mgr.downloaded_calls
        assert mgr.get_status("en_US-bryce-medium") == STATUS_READY
        dlg.deleteLater()

    def test_delete_voice(self, qapp, monkeypatch):
        mgr = DummyVoiceManager()
        dlg = VoiceLibraryDialog(manager=mgr)

        # Mock confirm dialog
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes
        )

        dlg.table.selectRow(0)
        assert dlg.btn_delete.isEnabled()
        dlg.btn_delete.click()

        assert "vi_VN-vais1000-medium" in mgr.deleted_calls
        assert mgr.get_status("vi_VN-vais1000-medium") == STATUS_NOT_DOWNLOADED
        dlg.deleteLater()

    def test_preview_factory_injection(self, qapp, tmp_path):
        mgr = DummyVoiceManager()
        fake_wav = tmp_path / "fake.wav"
        fake_wav.write_bytes(b"RIFFdummywav")

        mock_synth = MagicMock(return_value=fake_wav)
        dummy_player = DummyPlayer()

        dlg = VoiceLibraryDialog(
            manager=mgr,
            synth_factory=mock_synth,
            player_factory=lambda: dummy_player,
        )

        dlg.table.selectRow(0)
        dlg.btn_preview.click()

        if dlg._preview_worker is not None:
            dlg._preview_worker.wait(5000)
            qapp.processEvents()

        assert mock_synth.called
        assert dummy_player.played is True
        dlg.deleteLater()


class TestDubPanelRefactor:
    """Kiem tra DubPanel loai bo pitch/provider va them thu vien giong."""

    def test_dub_panel_has_no_provider_combo_or_pitch(self, qapp):
        panel = DubPanel()

        assert not hasattr(panel, "provider")
        assert not hasattr(panel, "pitch")
        assert not hasattr(panel, "lbl_pitch")
        assert not hasattr(panel, "btn_refresh")

        # Co nut thu vien giong va signal
        assert hasattr(panel, "btn_library")
        assert panel.btn_library.text() == "Thư viện giọng"
        assert hasattr(panel, "openVoiceLibrary")

        # voice combo la read-only
        assert not panel.voice.isEditable()
        panel.deleteLater()

    def test_voice_table_has_no_pitch_column(self, qapp):
        panel = DubPanel()
        headers = [
            panel.voice_table.horizontalHeaderItem(i).text()
            for i in range(panel.voice_table.columnCount())
        ]
        assert "Pitch" not in headers
        assert "Cao Độ" not in headers

        panel.voice.setCurrentText("vi_VN-vais1000-medium")
        panel.profile_gender.setCurrentText("Nam")
        panel._add_voice_profile()

        prof = panel._voice_profiles[0]
        assert "pitch" not in prof
        assert prof["voice"] == "vi_VN-vais1000-medium"
        assert prof["gender"] == "nam"
        panel.deleteLater()

    def test_load_and_apply_persists_local_voice(self, qapp):
        panel = DubPanel()
        s = Settings()
        s.local_voice = "vi_VN-vais1000-medium"
        s.tts_voice = "vi_VN-vais1000-medium"
        s.tts_speed_percent = 120

        panel.load(s)
        assert panel.voice.currentText() == "vi_VN-vais1000-medium"
        assert panel.rate.value() == 120

        # Change settings via panel
        panel.rate.setValue(110)
        s2 = Settings()
        panel.apply(s2)

        assert s2.tts_provider == "Local Voice"
        assert s2.local_voice == "vi_VN-vais1000-medium"
        assert s2.tts_voice == "vi_VN-vais1000-medium"
        assert s2.tts_speed_percent == 110
        panel.deleteLater()

    def test_status_message_when_unready(self, qapp):
        panel = DubPanel()
        panel.voice.clear()
        panel.refresh_status()

        assert not panel.btn_dub.isEnabled()
        assert not panel.btn_preview.isEnabled()
        assert "Chưa chọn giọng đọc" in panel.status.text()
        panel.deleteLater()


class TestMainWindowVoiceWiring:
    """Kiem tra ket noi giua MainWindow, DubPanel va VoiceLibrary."""

    def test_main_window_has_single_media_player(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        from autosub_studio.ui.main_window import MainWindow

        win = MainWindow()
        assert hasattr(win, "_preview_player")
        assert hasattr(win, "_preview_audio_output")
        assert win._preview_player is not None

        # DubPanel signal openVoiceLibrary duoc connect
        assert hasattr(win, "_open_voice_library")
        assert hasattr(win, "_on_voice_selected")

        win.close()
        qapp.processEvents()

    def test_on_voice_selected_updates_settings_and_dub_panel(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        from autosub_studio.ui.main_window import MainWindow

        win = MainWindow()
        win._on_voice_selected("vi_VN-vais1000-medium")

        assert win.settings.local_voice == "vi_VN-vais1000-medium"
        assert win.settings.tts_voice == "vi_VN-vais1000-medium"
        assert win.settings.tts_provider == "Local Voice"
        assert win.dub_panel.voice.currentText() == "vi_VN-vais1000-medium"

        win.close()
        qapp.processEvents()

    def test_preview_voice_without_project_uses_sample(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        from autosub_studio.ui.main_window import MainWindow

        win = MainWindow()
        win.settings.local_voice = "vi_VN-vais1000-medium"
        win.settings.tts_voice = "vi_VN-vais1000-medium"

        # Mock model ready guard
        monkeypatch.setattr(win, "_require_voice_model_ready", lambda: True)

        submitted = []

        def mock_submit(name, job, project_id=0, timeout=60):
            submitted.append((name, project_id))
            return "task-preview-123"

        monkeypatch.setattr(win.tasks, "submit", mock_submit)

        # Preview khong can project
        win.project = None
        win._preview_voice()

        assert len(submitted) == 1
        assert submitted[0][0] == "Nghe thử giọng đọc"
        assert submitted[0][1] == 0
        assert "task-preview-123" in win._preview_task_ids

        win.close()
        qapp.processEvents()

    def test_dub_guard_warns_when_model_not_ready(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        from autosub_studio.ui.main_window import MainWindow

        win = MainWindow()
        win.settings.local_voice = ""
        win.settings.tts_voice = ""

        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a: warned.append(a))

        run_called = []
        monkeypatch.setattr(win, "_run_step", lambda s: run_called.append(s))

        win._start_dub()

        assert len(warned) == 1
        assert "Chưa chọn giọng đọc" in warned[0][1]
        assert len(run_called) == 0

        win.close()
        qapp.processEvents()
