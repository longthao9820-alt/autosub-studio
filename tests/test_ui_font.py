"""Kiem thu co che dang ky font Unicode cho giao dien va co che du phong."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.services.paths import fonts_dir
from autosub_studio.services.settings import Settings
from autosub_studio.ui.cue_table import CueTableModel
from autosub_studio.ui.project_table import ProjectRow, ProjectTableModel
from autosub_studio.ui.style import (
    _reset_font_state,
    find_ui_font_file,
    get_fallback_font_family,
    get_qss,
    get_ui_font_family,
    get_ui_font_path,
    init_app_font,
)


class TestUiFontRegistration:
    def test_fonts_dir_locates_fonts(self):
        fdir = fonts_dir()
        assert fdir is not None
        assert fdir.is_dir()
        assert (fdir / "NotoSans-Regular.ttf").is_file()

    def test_find_ui_font_file(self):
        font_file = find_ui_font_file()
        assert font_file is not None
        assert font_file.is_file()
        assert font_file.name == "NotoSans-Regular.ttf"

    def test_init_app_font_registers_and_sets_family(self, qapp):
        _reset_font_state()
        family = init_app_font(qapp)
        assert family == "Noto Sans"
        path = get_ui_font_path()
        assert path is not None
        assert path.name == "NotoSans-Regular.ttf"
        assert get_ui_font_family() == "Noto Sans"
        assert qapp.font().family() == "Noto Sans"
        qss = get_qss()
        assert 'font-family: "Noto Sans"' in qss

    def test_init_app_font_fallback_when_forced(self, qapp):
        _reset_font_state()
        fallback_fam = get_fallback_font_family(qapp)
        family = init_app_font(qapp, force_fallback=True)
        assert family == fallback_fam
        assert get_ui_font_path() is None
        # Must not be hardcoded absent font
        assert family != "AbsentFont12345"
        assert qapp.font().family() == fallback_fam

        # Re-initialize to Noto Sans for subsequent tests
        _reset_font_state()
        init_app_font(qapp)

    def test_cue_table_uses_registered_font_family(self, qapp):
        init_app_font(qapp)
        doc = SubtitleDoc(cues=[Cue(0, 1000, "Xin chào", "Hello")])
        model = CueTableModel()
        model.set_document(doc)
        idx = model.index(0, 5)  # text column
        font = model.data(idx, Qt.ItemDataRole.FontRole)
        assert isinstance(font, QFont)
        assert font.family() == "Noto Sans"

    def test_project_table_uses_registered_font_family(self, qapp):
        init_app_font(qapp)
        model = ProjectTableModel()
        model.set_rows([ProjectRow(id=1, name="Dự án 1")])
        idx = model.index(0, 3)  # name column
        font = model.data(idx, Qt.ItemDataRole.FontRole)
        assert isinstance(font, QFont)
        assert font.family() == "Noto Sans"

    def test_subtitle_render_font_setting_not_overwritten(self):
        settings = Settings.load()
        assert settings.style.font != ""
        assert settings.style.font_size > 0
