"""Kiem thu nen tang AutoSub Studio V2 (Contract v2-p1-foundation-r2)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from PySide6.QtWidgets import QPushButton, QScrollArea

import autosub_studio
from autosub_studio.data.project import PROJECT_SCHEMA_VERSION, ProjectStore
from autosub_studio.services.settings import SCHEMA_VERSION, Settings
from autosub_studio.ui.main_window import MainWindow
from autosub_studio.ui.panels import RenderPanel, SettingsPanel, TranslatePanel
from autosub_studio.version import APP_NAME, APP_VERSION, ORG_NAME, __version__


class TestVersionConsistency:
    """Kiem tra tinh nhat quan cua phien ban trung tam 2.0.1."""

    def test_version_constants(self):
        assert APP_VERSION == "2.0.1"
        assert __version__ == "2.0.1"
        assert autosub_studio.APP_VERSION == "2.0.1"
        assert autosub_studio.__version__ == "2.0.1"
        assert APP_NAME == "AutoSub Studio"
        assert ORG_NAME == "AutoSubStudio"

    def test_pyproject_dynamic_version(self):
        root = Path(__file__).resolve().parent.parent
        pyproject_path = root / "pyproject.toml"
        assert pyproject_path.is_file()
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        assert "version" not in project  # version is dynamic
        assert "version" in project.get("dynamic", [])
        dynamic = data.get("tool", {}).get("setuptools", {}).get("dynamic", {})
        assert dynamic.get("version", {}).get("attr") == "autosub_studio.version.APP_VERSION"


class TestSettingsMigration:
    """Kiem tra migration Settings tu schema 11 sang schema 12."""

    def test_settings_schema_11_to_12_migration(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        v1_data = {
            "schema_version": 11,
            "asr_model": "base",
            "tts_provider": "VoiceStudio Local (English US)",
            "translate_provider": "Claude (can khoa API)",
            "ocr_mode": "Nhanh Như NTS",
            "capcut_path": r"C:\Users\test\AppData\Local\CapCut",
            "capcut_template_draft": "test_draft",
            "format_capcut": True,
            "capcut_unknown_legacy": "safely_kept",
            "unrelated_custom_field": "keep_me_safe",
            "plugin_settings": {"enabled": True, "timeout": 42},
        }
        Settings.config_path().write_text(json.dumps(v1_data), encoding="utf-8")

        loaded = Settings.load()

        assert loaded.schema_version == 12
        assert loaded.schema_version == SCHEMA_VERSION
        assert loaded.asr_model == "base"
        assert loaded.tts_provider == "Local Voice"
        assert loaded.translate_provider == "AI Gateway"
        assert loaded.default_preset == "DEFAULT"
        assert loaded.ai_model_sub == "sub"
        assert loaded.ai_model_prime == "prime"
        assert loaded.ai_thinking_sub == "low"
        assert loaded.ai_thinking_prime == "medium"
        assert loaded.auto_check_update is True
        assert loaded.output_folder == ""

        # Deprecated legacy fields safely migrated into extra without active attributes
        assert not hasattr(loaded, "capcut_path")
        assert not hasattr(loaded, "capcut_template_draft")
        assert not hasattr(loaded, "format_capcut")
        assert loaded.extra.get("capcut_path") == r"C:\Users\test\AppData\Local\CapCut"
        assert loaded.extra.get("capcut_template_draft") == "test_draft"
        assert loaded.extra.get("format_capcut") is True
        assert loaded.extra.get("capcut_unknown_legacy") == "safely_kept"

        # Khong mat field khong lien quan
        assert loaded.extra.get("unrelated_custom_field") == "keep_me_safe"
        assert loaded.extra.get("plugin_settings") == {"enabled": True, "timeout": 42}

        # Profile active duoc cap nhat provider moi
        active_prof = loaded.config_profiles.get(loaded.active_config_profile, {})
        assert active_prof.get("tts_provider") == "Local Voice"
        assert active_prof.get("translate_provider") == "AI Gateway"

        # Deterministic / repeat-safe round-trip
        saved_path = loaded.save()
        saved_raw = json.loads(saved_path.read_text(encoding="utf-8"))
        assert saved_raw["schema_version"] == 12
        assert saved_raw["tts_provider"] == "Local Voice"
        assert saved_raw["translate_provider"] == "AI Gateway"
        assert saved_raw["capcut_path"] == r"C:\Users\test\AppData\Local\CapCut"
        assert saved_raw["capcut_template_draft"] == "test_draft"
        assert saved_raw["format_capcut"] is True
        assert saved_raw["capcut_unknown_legacy"] == "safely_kept"
        assert saved_raw["unrelated_custom_field"] == "keep_me_safe"
        assert saved_raw["plugin_settings"] == {"enabled": True, "timeout": 42}

        reloaded = Settings.load()
        assert reloaded.schema_version == 12
        assert reloaded.tts_provider == "Local Voice"
        assert reloaded.translate_provider == "AI Gateway"
        assert not hasattr(reloaded, "capcut_path")
        assert reloaded.extra.get("capcut_path") == r"C:\Users\test\AppData\Local\CapCut"
        assert reloaded.extra.get("unrelated_custom_field") == "keep_me_safe"
        assert reloaded.extra.get("capcut_unknown_legacy") == "safely_kept"

    def test_settings_schema_11_sapi_and_edge_migrate_to_local_voice(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        for old_provider in ("Windows SAPI (offline)", "Edge TTS (can Internet)"):
            v1_data = {
                "schema_version": 11,
                "tts_provider": old_provider,
                "translate_provider": "Claude",
            }
            Settings.config_path().write_text(json.dumps(v1_data), encoding="utf-8")
            loaded = Settings.load()
            assert loaded.schema_version == 12
            assert loaded.tts_provider == "Local Voice"
            assert loaded.translate_provider == "AI Gateway"


class TestProjectMigration:
    """Kiem tra migration ProjectData tu schema 3 sang schema 4."""

    def test_project_schema_3_to_4_migration(self, tmp_path):
        store = ProjectStore(tmp_path / "workspace")
        project_folder = store.projects_dir / "101-test-v1"
        project_folder.mkdir(parents=True)
        for sub in ("video", "audio", "subtitles", "temp", "exports"):
            (project_folder / sub).mkdir()

        v1_project = {
            "schema_version": 3,
            "project_id": 101,
            "name": "test-v1",
            "folder": str(project_folder),
            "video_path": str(project_folder / "video" / "test.mp4"),
            "duration": 45.2,
            "width": 1920,
            "height": 1080,
            "unrelated_project_meta": "important_value",
            "custom_tags": ["action", "recap"],
        }
        (project_folder / "project.json").write_text(json.dumps(v1_project), encoding="utf-8")

        loaded = store.load(project_folder)

        assert loaded.schema_version == 4
        assert loaded.schema_version == PROJECT_SCHEMA_VERSION
        assert loaded.project_id == 101
        assert loaded.name == "test-v1"
        assert loaded.render_preset == ""
        assert loaded.output_video_path == ""

        # Khong mat field khong lien quan
        assert loaded.extra.get("unrelated_project_meta") == "important_value"
        assert loaded.extra.get("custom_tags") == ["action", "recap"]

        # Deterministic / repeat-safe round-trip
        store.save(loaded)
        saved_raw = json.loads((project_folder / "project.json").read_text(encoding="utf-8"))
        assert saved_raw["schema_version"] == 4
        assert saved_raw["unrelated_project_meta"] == "important_value"
        assert saved_raw["custom_tags"] == ["action", "recap"]

        reloaded = store.load(project_folder)
        assert reloaded.schema_version == 4
        assert reloaded.extra.get("unrelated_project_meta") == "important_value"


class TestUIChanges:
    """Kiem tra B4 la RenderPanel va cac dead controls da bi loai bo."""

    def test_b4_is_render_panel_and_dead_controls_absent(self, qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text("{}", encoding="utf-8")

        window = MainWindow()

        # RenderPanel B4 tren tab
        assert window.tab_bar.buttons[4].text() == "B4: Render & Xuất Video"
        b4_page = window.tab_stack.widget(4)
        assert isinstance(b4_page, QScrollArea)
        assert b4_page.widget() is window.render_panel
        assert isinstance(window.render_panel, RenderPanel)

        # CapCut panel khong con ton tai
        assert not hasattr(window, "capcut_panel")

        # dub_admin_bar khong con ton tai
        assert not hasattr(window, "dub_admin_bar")

        # btn_reformat khong con ton tai trong TranslatePanel
        assert not hasattr(window.translate_panel, "btn_reformat")

        # Placeholder buttons trong SettingsPanel da bi xoa
        settings_buttons = window.settings_panel.findChildren(QPushButton)
        button_texts = [b.text() for b in settings_buttons]
        assert not any("Cấu Hình Xóa Thoại Và Nhạc Nền" in t for t in button_texts)
        assert not any("Intro & Outro" in t for t in button_texts)
        assert not any("Hiệu Ứng Video" in t for t in button_texts)

        # AI Gateway button ton tai, enabled
        assert hasattr(window.settings_panel, "btn_ai_gateway")
        ai_btn = window.settings_panel.btn_ai_gateway
        assert "AI Gateway" in ai_btn.text()
        assert ai_btn.isEnabled()

        window.close()
        qapp.processEvents()

    def test_translate_panel_does_not_have_btn_reformat(self, qapp):
        panel = TranslatePanel()
        assert not hasattr(panel, "btn_reformat")
        panel.deleteLater()

    def test_settings_panel_ai_gateway_and_removed_placeholders(self, qapp):
        panel = SettingsPanel()
        assert hasattr(panel, "btn_ai_gateway")
        assert panel.btn_ai_gateway.isEnabled()
        assert "AI Gateway" in panel.btn_ai_gateway.text()

        buttons = [b.text() for b in panel.findChildren(QPushButton)]
        assert not any("Cấu Hình Xóa Thoại" in t for t in buttons)
        assert not any("Intro & Outro" in t for t in buttons)
        assert not any("Hiệu Ứng Video" in t for t in buttons)
        panel.deleteLater()
