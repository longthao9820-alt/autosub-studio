"""Kiem thu cuon cac panel V2 tren man hinh 1366x768."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

from autosub_studio.services.settings import Settings
from autosub_studio.ui.main_window import MainWindow


def test_workflow_panels_use_vertical_scroll_areas(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    Settings.config_path().write_text("{}", encoding="utf-8")

    window = MainWindow()
    window.resize(1366, 768)
    window.show()
    qapp.processEvents()

    for index, panel in enumerate(
        (
            window.subtitle_panel,
            window.translate_panel,
            window.dub_panel,
            window.render_panel,
            window.settings_panel,
        ),
        start=1,
    ):
        page = window.tab_stack.widget(index)
        assert isinstance(page, QScrollArea)
        assert page.widget() is panel
        assert page.widgetResizable()
        assert (
            page.horizontalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    window.tab_stack.setCurrentIndex(3)
    qapp.processEvents()
    dub_scroll = window.tab_stack.widget(3)
    assert isinstance(dub_scroll, QScrollArea)
    dub_scroll.ensureWidgetVisible(window.dub_panel.btn_dub)
    qapp.processEvents()
    assert dub_scroll.verticalScrollBar().value() >= 0

    window.tab_stack.setCurrentIndex(5)
    qapp.processEvents()
    settings_scroll = window.tab_stack.widget(5)
    assert isinstance(settings_scroll, QScrollArea)
    settings_scroll.ensureWidgetVisible(window.settings_panel.btn_check_update)
    qapp.processEvents()
    assert settings_scroll.verticalScrollBar().value() >= 0

    window.close()
