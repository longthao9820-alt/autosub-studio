"""Kiem thu Render Presets va xuat video (Contract v2-p6-presets-output-r2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QFileDialog

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.data.project import ProjectStore
from autosub_studio.pipeline.steps import PipelineContext, step_render
from autosub_studio.services.ffmpeg import CancelToken, FFmpeg, FFmpegError
from autosub_studio.services.presets import PresetManager
from autosub_studio.services.settings import Settings, SubtitleStyle
from autosub_studio.services.tasks import TaskContext
from autosub_studio.ui.main_window import MainWindow
from autosub_studio.ui.panels import RenderPanel, SettingsPanel


class TestPresetCRUDAndFallbacks:
    """Kiem tra day du CRUD, bao ve builtin va fallback cua PresetManager."""

    def test_builtin_presets_exist(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        pm = PresetManager(config_file=cfg)
        names = pm.list_names()
        for expected in ("DEFAULT", "REVIEW ENG", "REVIEW VI", "BODYCAM", "ANIMAL STORY"):
            assert expected in names
            preset = pm.get(expected)
            assert preset.name == expected
            assert preset.builtin is True
            assert isinstance(preset.style, SubtitleStyle)
            assert preset.subtitle_visible is True
            assert preset.render_preset in {"medium", "fast"}

    def test_custom_preset_crud(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        pm = PresetManager(config_file=cfg)

        # Create
        custom_style = SubtitleStyle(
            font="Tahoma",
            font_size=24,
            bold=False,
            primary_color="#123456",
            outline_color="#654321",
            margin_v=80,
        )
        created = pm.create(
            "MY_CUSTOM",
            style=custom_style,
            subtitle_visible=False,
            render_scale="1280x720",
            render_fps="30",
            render_crf=23,
            render_preset="ultrafast",
            lut_path="lut.cube",
        )
        assert created.name == "MY_CUSTOM"
        assert created.builtin is False
        assert created.style.font == "Tahoma"
        assert created.subtitle_visible is False
        assert created.render_scale == "1280x720"
        assert created.render_crf == 23

        # Read / Get
        fetched = pm.get("MY_CUSTOM")
        assert fetched.name == "MY_CUSTOM"
        assert fetched.style.primary_color == "#123456"

        # Update
        updated = pm.update(
            "MY_CUSTOM",
            render_crf=16,
            render_scale="1920x1080",
            subtitle_visible=True,
        )
        assert updated.render_crf == 16
        assert updated.render_scale == "1920x1080"
        assert updated.subtitle_visible is True

        # Reload from disk (persistence check)
        pm2 = PresetManager(config_file=cfg)
        persisted = pm2.get("MY_CUSTOM")
        assert persisted.render_crf == 16
        assert persisted.render_scale == "1920x1080"
        assert persisted.subtitle_visible is True

        # Delete
        assert pm2.delete("MY_CUSTOM") is True
        assert "MY_CUSTOM" not in pm2.list_names()

    def test_builtin_cannot_be_deleted(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        pm = PresetManager(config_file=cfg)
        for name in ("DEFAULT", "REVIEW ENG", "REVIEW VI", "BODYCAM", "ANIMAL STORY"):
            with pytest.raises(ValueError, match="Khong the xoa preset mac dinh"):
                pm.delete(name)

    def test_builtin_cannot_be_updated_directly(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        pm = PresetManager(config_file=cfg)
        with pytest.raises(ValueError, match="Khong the cap nhat preset mac dinh"):
            pm.update("DEFAULT", render_crf=10)

    def test_cannot_create_with_builtin_name(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        pm = PresetManager(config_file=cfg)
        with pytest.raises(ValueError, match="trung voi preset mac dinh"):
            pm.create("DEFAULT")

    def test_fallback_to_default_on_unknown(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        pm = PresetManager(config_file=cfg)
        fallback = pm.get("NON_EXISTENT_PRESET_12345")
        assert fallback.name == "DEFAULT"
        assert fallback.builtin is True

    def test_corrupt_json_recovery(self, tmp_path):
        cfg = tmp_path / "render_presets.json"
        cfg.write_text("CORRUPTED_NOT_JSON", encoding="utf-8")
        pm = PresetManager(config_file=cfg)
        assert "DEFAULT" in pm.list_names()
        assert pm.get("DEFAULT").name == "DEFAULT"


class TestProjectPresetIntegration:
    """Kiem tra snapshot preset tren ProjectData va tac dong mac dinh cho new vs old project."""

    def test_default_preset_affects_new_projects_only(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        window = MainWindow()
        store = window.store

        # Old project created with DEFAULT
        p1 = store.create(1, "OldProject")
        p1.render_preset = "DEFAULT"
        p1.render_preset_snapshot = window.preset_manager.get("DEFAULT").to_dict()
        store.save(p1)

        # Set default to BODYCAM
        window.render_preset_combo.setCurrentText("BODYCAM")
        window._set_default_render_preset()
        assert window.settings.default_preset == "BODYCAM"

        # Verify old project is UNCHANGED
        p1_loaded = store.load(p1.folder)
        assert p1_loaded.render_preset == "DEFAULT"
        assert p1_loaded.render_preset_snapshot.get("name") == "DEFAULT"

        # Mock probe for new project creation
        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.width = 1920
        mock_info.height = 1080
        dummy_video = tmp_path / "dummy.mp4"
        dummy_video.write_bytes(b"dummy")

        with patch.object(window.ff, "probe", return_value=mock_info):
            new_p = window._create_project_from_video(str(dummy_video))

        assert new_p.render_preset == "BODYCAM"
        assert new_p.render_preset_snapshot.get("name") == "BODYCAM"

        window.close()
        qapp.processEvents()

    def test_apply_preset_updates_current_project_only(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        window = MainWindow()
        store = window.store

        p1 = store.create(10, "ProjectA")
        p1.render_preset = "DEFAULT"
        p1.render_preset_snapshot = window.preset_manager.get("DEFAULT").to_dict()
        store.save(p1)

        p2 = store.create(20, "ProjectB")
        p2.render_preset = "BODYCAM"
        p2.render_preset_snapshot = window.preset_manager.get("BODYCAM").to_dict()
        store.save(p2)

        # Load Project A
        window.project = p1
        window.project_id = p1.project_id

        # Apply REVIEW VI to Project A
        window.render_preset_combo.setCurrentText("REVIEW VI")
        window._apply_render_preset()

        # Project A updated
        assert window.project.render_preset == "REVIEW VI"
        assert window.project.render_preset_snapshot.get("name") == "REVIEW VI"

        # Project B untouched
        p2_loaded = store.load(p2.folder)
        assert p2_loaded.render_preset == "BODYCAM"
        assert p2_loaded.render_preset_snapshot.get("name") == "BODYCAM"

        window.close()
        qapp.processEvents()

    def test_project_data_snapshot_roundtrip(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(1, "test")
        snapshot = {
            "name": "MY_PRESET",
            "style": {"font": "Verdana", "font_size": 22},
            "subtitle_visible": False,
            "render_crf": 18,
            "render_preset": "fast",
        }
        data.render_preset = "MY_PRESET"
        data.render_preset_snapshot = snapshot
        store.save(data)

        loaded = store.load(data.folder)
        assert loaded.render_preset == "MY_PRESET"
        assert loaded.render_preset_snapshot["name"] == "MY_PRESET"
        assert loaded.render_preset_snapshot["style"]["font"] == "Verdana"
        assert loaded.render_preset_snapshot["subtitle_visible"] is False


class TestUIControlsAndExportDialog:
    """Kiem tra cac nut giao dien that, tooltip exact va dialog xuat."""

    def test_render_toolbar_controls_and_tooltip(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        window = MainWindow()

        # Tooltip exact
        assert hasattr(window, "btn_preset_default")
        assert (
            window.btn_preset_default.toolTip()
            == "Đặt preset này làm mặc định cho các project mới"
        )

        # Action buttons exist
        assert hasattr(window, "btn_preset_create")
        assert hasattr(window, "btn_preset_apply")
        assert hasattr(window, "btn_preset_update")
        assert hasattr(window, "btn_preset_delete")

        window.close()
        qapp.processEvents()

    def test_render_panel_and_settings_panel_export_button(self, qapp):
        render_panel = RenderPanel()
        assert hasattr(render_panel, "btn_output_folder")
        assert hasattr(render_panel, "chooseOutputFolder")

        settings_panel = SettingsPanel()
        assert hasattr(settings_panel, "btn_export_settings")
        assert hasattr(settings_panel, "chooseOutputFolder")

        render_panel.deleteLater()
        settings_panel.deleteLater()

    def test_output_folder_separate_from_download_folder(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        window = MainWindow()
        window.settings.download_folder = str(tmp_path / "downloads")
        window.settings.output_folder = ""

        # Mock QFileDialog to return output folder
        target_output = str(tmp_path / "renders")
        with patch.object(
            QFileDialog,
            "getExistingDirectory",
            return_value=target_output,
        ) as mock_dialog:
            window._choose_output_folder()
            # Dialog title exact
            assert mock_dialog.call_args[0][1] == "Thư mục lưu video render thành công"

        assert window.settings.output_folder == target_output
        assert window.settings.download_folder == str(tmp_path / "downloads")

        window.close()
        qapp.processEvents()


class TestPipelineRenderPublication:
    """Kiem tra render atomic publish, blank vs custom output folder, failure cleanup."""

    def _make_context(self, tmp_path, output_folder="") -> tuple[PipelineContext, Path, Path]:
        store = ProjectStore(tmp_path / "workspace")
        project = store.create(1, "sample_vid")
        project.doc = SubtitleDoc(
            cues=[Cue(start=0.0, end=2.0, text="Hello", translation="Xin chao")]
        )

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy video content")
        project.video_path = str(video_path)
        project.duration = 2.0
        store.save(project)

        settings = Settings()
        settings.output_folder = output_folder

        ff = MagicMock(spec=FFmpeg)
        ff.probe.return_value = MagicMock(duration=2.0, width=1920, height=1080)

        task = TaskContext(CancelToken(), lambda _p: None, lambda _m: None)
        pc = PipelineContext(
            ff=ff,
            settings=settings,
            store=store,
            project=project,
            task=task,
        )
        return pc, video_path, Path(project.folder)

    def test_render_with_blank_output_folder_publishes_to_project_exports(self, tmp_path):
        pc, video_path, project_folder = self._make_context(tmp_path, output_folder="")

        def fake_burn(ff, video, ass, out_path, **kwargs):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"rendered video bytes")
            return Path(out_path)

        with patch("autosub_studio.services.media.burn_subtitles", side_effect=fake_burn):
            result = step_render(pc)

        expected_file = project_folder / "exports" / "sample_vid_sub.mp4"
        assert expected_file.is_file()
        assert expected_file.read_bytes() == b"rendered video bytes"

        # Consistent paths
        assert pc.project.render_path == str(expected_file)
        assert pc.project.output_video_path == str(expected_file)
        assert "Da render xong" in result

    def test_render_with_custom_output_folder_atomically_publishes(self, tmp_path):
        custom_out = tmp_path / "custom_output"
        custom_out.mkdir()
        pc, video_path, project_folder = self._make_context(
            tmp_path, output_folder=str(custom_out)
        )

        def fake_burn(ff, video, ass, out_path, **kwargs):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"rendered video bytes")
            return Path(out_path)

        with patch("autosub_studio.services.media.burn_subtitles", side_effect=fake_burn):
            result = step_render(pc)

        published_file = custom_out / "sample_vid_sub.mp4"
        assert published_file.is_file()
        assert published_file.read_bytes() == b"rendered video bytes"

        # Consistent paths
        assert pc.project.render_path == str(published_file)
        assert pc.project.output_video_path == str(published_file)
        assert "Da render xong" in result

    def test_render_failure_leaves_no_partial_final(self, tmp_path):
        custom_out = tmp_path / "custom_output_fail"
        custom_out.mkdir()
        pc, video_path, project_folder = self._make_context(
            tmp_path, output_folder=str(custom_out)
        )

        def failing_burn(ff, video, ass, out_path, **kwargs):
            # Create a partial file and then fail
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"partial corrupted bytes")
            raise FFmpegError("Render crashed unexpectedly")

        with (
            patch("autosub_studio.services.media.burn_subtitles", side_effect=failing_burn),
            pytest.raises(FFmpegError),
        ):
            step_render(pc)

        # Final output folder MUST NOT have any partial file
        assert not (custom_out / "sample_vid_sub.mp4").exists()
        assert list(custom_out.iterdir()) == []

        # Temp staging file cleaned up
        temp_dir = project_folder / "temp"
        assert not any(f.name.startswith("render_sample_vid") for f in temp_dir.iterdir())

    def test_preset_subtitle_visibility_false_burns_empty_ass(self, tmp_path):
        pc, video_path, project_folder = self._make_context(tmp_path)
        pc.project.render_preset_snapshot = {
            "name": "NO_SUB",
            "subtitle_visible": False,
            "render_crf": 25,
            "render_preset": "ultrafast",
        }

        captured_ass = []

        def inspect_burn(ff, video, ass, out_path, **kwargs):
            captured_ass.append(Path(ass).read_text(encoding="utf-8"))
            Path(out_path).write_bytes(b"video")
            return Path(out_path)

        with patch("autosub_studio.services.media.burn_subtitles", side_effect=inspect_burn):
            step_render(pc)

        assert len(captured_ass) == 1
        ass_content = captured_ass[0]
        # Dialogue events should be empty when subtitle_visible is False
        assert "Dialogue:" not in ass_content
