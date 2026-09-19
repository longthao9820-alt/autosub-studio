"""Kiem thu khung khoanh vung tren video va menu chuot phai cua bang du an."""

from __future__ import annotations

import pytest

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.ui.player import MODE_BLUR, MODE_NONE, MODE_OCR, VideoPlayer


@pytest.fixture
def player(qapp):
    widget = VideoPlayer()
    widget.set_frame_size(1280, 720)
    yield widget
    widget.deleteLater()


class TestRegionVisibility:
    def test_screen_edit_shows_a_draggable_box(self, player):
        player.begin_region(MODE_OCR, None)
        item = player.scene.region_item
        assert item.isVisible()
        assert item.isEnabled()
        assert player.region_mode == MODE_OCR

    def test_hide_region_clears_it_for_screen_render(self, player):
        player.begin_region(MODE_OCR, None)
        player.hide_region()
        assert not player.scene.region_item.isVisible()
        assert player.region_mode == MODE_NONE

    def test_blur_box_can_still_be_summoned(self, player):
        player.hide_region()
        x, y, w, h = player.begin_region(MODE_BLUR, None)
        assert player.scene.region_item.isVisible()
        assert player.region_mode == MODE_BLUR
        assert w > 0 and h > 0

    def test_box_stays_inside_the_frame(self, player):
        x, y, w, h = player.begin_region(MODE_OCR, [-50, -50, 5000, 5000])
        assert x >= 0 and y >= 0
        assert x + w <= 1280
        assert y + h <= 720


class TestRenderSubtitlePreview:
    def test_screen_edit_never_overlays_subtitles(self):
        from autosub_studio.ui.main_window import MainWindow

        doc = SubtitleDoc([Cue(0, 2, "中文字幕", "Phụ đề dịch")])
        assert MainWindow._preview_subtitle_text(doc, 0, False, True) == ""

    def test_screen_render_uses_translation_only(self):
        from autosub_studio.ui.main_window import MainWindow

        doc = SubtitleDoc([Cue(0, 2, "中文字幕", "Phụ đề dịch")])
        assert MainWindow._preview_subtitle_text(doc, 0, True, True) == "Phụ đề dịch"

    def test_subtitle_can_be_moved_vertically(self, player):
        player.show_subtitle("Phụ đề dịch")
        player.set_subtitle_movable(True)
        player.scene.subtitle_item.setPos(10, 220)

        item = player.scene.subtitle_item
        expected_x = (player.scene.sceneRect().width() - item.boundingRect().width()) / 2
        assert item.movable
        assert item.x() == pytest.approx(expected_x)
        assert item.y() == pytest.approx(220)


class TestVideoNavigation:
    def test_dragging_seek_updates_before_mouse_release(self, player, monkeypatch):
        positions: list[float] = []
        player._duration = 100.0
        monkeypatch.setattr(player, "seek", positions.append)

        player._begin_seek()
        player.slider.setValue(250)
        player._queue_live_seek(250)

        assert positions == [25.0]
        assert player._seeking

    def test_show_frame_pauses_and_seeks_to_cue_start(self, player, monkeypatch):
        calls: list[object] = []
        player._duration = 100.0
        monkeypatch.setattr(player, "pause", lambda: calls.append("pause"))
        monkeypatch.setattr(player, "seek", lambda value: calls.append(value))

        player.show_frame(45.92)

        assert calls == ["pause", 45.92]

    def test_selecting_cue_displays_its_video_frame(self, qapp):
        from autosub_studio.ui.cue_table import CueTableModel
        from autosub_studio.ui.main_window import MainWindow

        model = CueTableModel()
        model.set_document(
            SubtitleDoc(
                [
                    Cue(1.2, 2.0, "第一句"),
                    Cue(45.92, 47.32, "八笔"),
                ]
            )
        )
        frames: list[float] = []
        editors: list[int] = []

        class PlayerStub:
            @staticmethod
            def show_frame(seconds: float) -> None:
                frames.append(seconds)

        class WindowLike:
            cue_model = model
            player = PlayerStub()

            @staticmethod
            def _fill_editors(row: int) -> None:
                editors.append(row)

        MainWindow._on_cue_activated(WindowLike(), model.index(1, 0))

        assert frames == [45.92]
        assert editors == [1]


class TestProjectMenu:
    def test_ctrl_shift_selection_returns_every_project(self, qapp):
        from PySide6.QtCore import QItemSelectionModel

        from autosub_studio.ui.main_window import MainWindow
        from autosub_studio.ui.project_table import (
            ProjectFilterProxy,
            ProjectRow,
            ProjectTableModel,
            ProjectTableView,
        )

        model = ProjectTableModel()
        model.set_rows([ProjectRow(id=i, name=f"Project {i}") for i in range(1, 6)])
        proxy = ProjectFilterProxy()
        proxy.setSourceModel(model)
        view = ProjectTableView()
        view.setModel(proxy)
        selection = view.selectionModel()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for row in (0, 2, 4):
            selection.select(proxy.index(row, 0), flags)

        class WindowLike:
            project_view = view
            project_proxy = proxy
            project_model = model

        selected = MainWindow._selected_project_rows(WindowLike())

        assert [row.id for row in selected] == [5, 3, 1]

    def test_ctrl_r_queues_every_selected_project_for_ocr(self):
        from autosub_studio.pipeline import steps
        from autosub_studio.ui.main_window import MainWindow
        from autosub_studio.ui.project_table import ProjectRow

        queued: list[str] = []
        single: list[int] = []

        class WindowLike:
            @staticmethod
            def _selected_project_rows() -> list[ProjectRow]:
                return [ProjectRow(id=1), ProjectRow(id=2), ProjectRow(id=3)]

            @staticmethod
            def _queue_selected_projects_ocr() -> None:
                queued.append("ocr")

            @staticmethod
            def _run_step_for(row: ProjectRow, _step: str) -> None:
                single.append(row.id)

        MainWindow._run_selected_project_step(WindowLike(), steps.STEP_OCR)

        assert queued == ["ocr"]
        assert single == []

    def test_delete_receives_all_selected_projects(self, monkeypatch):
        from PySide6.QtWidgets import QMessageBox

        from autosub_studio.ui.main_window import MainWindow
        from autosub_studio.ui.project_table import ProjectRow

        rows = [ProjectRow(id=i, name=f"Project {i}") for i in range(1, 5)]
        deleted: list[int] = []
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
        )

        class WindowLike:
            @staticmethod
            def _selected_project_rows() -> list[ProjectRow]:
                return rows

            @staticmethod
            def _delete_projects(items: list[ProjectRow]) -> None:
                deleted.extend(item.id for item in items)

        MainWindow._delete_project(WindowLike())

        assert deleted == [1, 2, 3, 4]

    def test_batch_queue_uses_configured_parallel_limit(self):
        from types import SimpleNamespace

        from autosub_studio.ui.main_window import MainWindow

        limits: list[int] = []
        queued: list[object] = []
        statuses: list[int] = []

        class TasksStub:
            @staticmethod
            def set_max_workers(count: int) -> None:
                limits.append(count)

            @staticmethod
            def record(_task_id: str):
                return None

            @staticmethod
            def running_for_project(_project_id: int):
                return None

        class WindowLike:
            settings = SimpleNamespace(max_workers=4)
            tasks = TasksStub()
            _batch_task_ids: set[str] = set()
            _batch_pending_projects: list[object] = []
            _batch_total = 12
            _batch_done = 9
            _batch_failed = 1

            def _queue_batch_project(self, project) -> None:
                queued.append(project)
                self._batch_task_ids.add(f"T{project.project_id}")

            def _start_next_batch_projects(self) -> None:
                MainWindow._start_next_batch_projects(self)

            @staticmethod
            def _set_project_status(project_id: int, **_values) -> None:
                statuses.append(project_id)

            @staticmethod
            def _update_batch_queue_status() -> None:
                pass

        projects = [SimpleNamespace(project_id=i) for i in range(1, 11)]
        target = WindowLike()
        MainWindow._queue_batch_projects(target, projects)

        assert limits == [4]
        assert queued == projects[:4]
        assert target._batch_pending_projects == projects[4:]
        assert statuses == list(range(1, 11))
        assert target._batch_total == 10
        assert target._batch_done == 0
        assert target._batch_failed == 0

        target._batch_task_ids.remove("T1")
        MainWindow._start_next_batch_projects(target)
        assert queued == projects[:5]
        assert target._batch_pending_projects == projects[5:]

    def test_project_shortcuts_are_persistent_actions(self, qapp):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QTableView

        from autosub_studio.pipeline import steps
        from autosub_studio.ui.main_window import MainWindow

        calls: list[str] = []

        class WindowLike:
            MENU_STEPS = MainWindow.MENU_STEPS
            MENU_SHORTCUTS = MainWindow.MENU_SHORTCUTS
            project_view = QTableView()

            @staticmethod
            def _run_selected_project_step(step: str) -> None:
                calls.append(step)

            @staticmethod
            def _run_selected_project_script() -> None:
                pass

            @staticmethod
            def _stop_selected_project() -> None:
                pass

            @staticmethod
            def reload_projects() -> None:
                pass

            @staticmethod
            def _export_selected_project() -> None:
                pass

            @staticmethod
            def _copy_selected_project_region() -> None:
                pass

            @staticmethod
            def _paste_selected_project_region() -> None:
                pass

            @staticmethod
            def _delete_project() -> None:
                pass

        target = WindowLike()
        MainWindow._build_project_shortcuts(target)
        ocr_action = next(
            action
            for action in target._project_shortcut_actions
            if action.shortcut().toString() == "Ctrl+R"
        )
        assert ocr_action.shortcutContext() == Qt.ShortcutContext.WidgetWithChildrenShortcut
        ocr_action.trigger()
        project_undo = next(
            action
            for action in target._project_shortcut_actions
            if action.shortcut().toString() == "Ctrl+Z"
        )
        project_undo.trigger()
        assert calls == [steps.STEP_OCR, steps.STEP_BLUR]

    def test_single_click_opens_the_clicked_project(self, qapp):
        from autosub_studio.ui.main_window import MainWindow
        from autosub_studio.ui.project_table import (
            ProjectFilterProxy,
            ProjectRow,
            ProjectTableModel,
        )

        model = ProjectTableModel()
        model.set_rows(
            [
                ProjectRow(id=2, name="Project 2", folder="two"),
                ProjectRow(id=1, name="Project 1", folder="one"),
            ]
        )
        proxy = ProjectFilterProxy()
        proxy.setSourceModel(model)
        opened: list[tuple[int, str]] = []

        class WindowLike:
            project_id = 2
            project_proxy = proxy
            project_model = model

            @staticmethod
            def _load_project(project_id: int, folder: str) -> None:
                opened.append((project_id, folder))

        MainWindow._open_project_from_index(WindowLike(), proxy.index(1, 3))
        assert opened == [(1, "one")]

    def test_srt_row_button_imports_into_clicked_project(self, qapp):
        from PySide6.QtWidgets import QTableView

        from autosub_studio.ui.main_window import MainWindow
        from autosub_studio.ui.project_table import (
            ProjectFilterProxy,
            ProjectRow,
            ProjectTableModel,
        )
        from autosub_studio.ui.widgets import ACTION_SRT

        model = ProjectTableModel()
        target_row = ProjectRow(id=7, name="Project 7", folder="seven")
        model.set_rows([target_row])
        proxy = ProjectFilterProxy()
        proxy.setSourceModel(model)
        view = QTableView()
        view.setModel(proxy)
        imported: list[int] = []

        class WindowLike:
            project_proxy = proxy
            project_model = model
            project_view = view

            @staticmethod
            def _import_subtitle_for(row: ProjectRow) -> None:
                imported.append(row.id)

        MainWindow._on_row_action(WindowLike(), 0, ACTION_SRT)

        assert imported == [7]

    def test_imported_srt_replaces_the_visible_cue_table(self, qapp, tmp_path):
        from autosub_studio.ui.cue_table import CueTableModel, CueTableView
        from autosub_studio.ui.main_window import MainWindow

        srt = tmp_path / "chinese.srt"
        srt.write_text(
            "1\n00:00:00,400 --> 00:00:01,600\n我问你几个问题\n\n"
            "2\n00:00:02,000 --> 00:00:02,800\n你说\n",
            encoding="utf-8",
        )
        model = CueTableModel()
        view = CueTableView()
        view.setModel(model)
        activated: list[int] = []
        logs: list[str] = []

        class WindowLike:
            cue_model = model
            cue_view = view

            @staticmethod
            def _require_project() -> bool:
                return True

            @staticmethod
            def _mark_dirty() -> None:
                pass

            @staticmethod
            def _save_project() -> None:
                pass

            @staticmethod
            def _on_cue_activated(index) -> None:
                activated.append(index.row())

            @staticmethod
            def _log(message: str) -> None:
                logs.append(message)

        MainWindow._import_subtitle(WindowLike(), str(srt))

        assert [cue.text for cue in model.doc.cues] == ["我问你几个问题", "你说"]
        assert model.rowCount() == 2
        assert activated == [0]
        assert "2 cau" in logs[-1]

    def test_batch_region_scales_from_current_project(self):
        from autosub_studio.ui.main_window import MainWindow

        region = MainWindow._scaled_batch_region([100, 700, 800, 200], (1000, 1000), 2000, 500)
        assert region == [200, 350, 1600, 100]

    def test_batch_region_defaults_to_lower_part_of_video(self):
        from autosub_studio.ui.main_window import MainWindow

        region = MainWindow._scaled_batch_region([], (0, 0), 1920, 1080)
        assert region == [96, 756, 1728, 302]

    def test_hidden_session_log_does_not_require_a_qt_widget(self):
        from autosub_studio.ui.main_window import MainWindow

        class WindowLike:
            _session_log: list[str] = []

        target = WindowLike()
        MainWindow._append_log(target, "Da mo du an")
        assert target._session_log == ["Da mo du an"]

    def test_menu_lists_the_main_jobs(self):
        from autosub_studio.pipeline import steps
        from autosub_studio.ui.main_window import MainWindow

        labels = [label for label, _step in MainWindow.MENU_STEPS]
        names = [step for _label, step in MainWindow.MENU_STEPS]
        assert "START: Lấy Sub Bằng Chữ" in labels
        assert "START: Lấy Sub Bằng Giọng Nói" in labels
        assert steps.STEP_OCR in names
        assert steps.STEP_ASR in names
        assert steps.STEP_TRANSLATE in names
        assert steps.STEP_RENDER in names

    def test_every_menu_entry_is_a_real_step(self):
        from autosub_studio.pipeline import steps
        from autosub_studio.ui.main_window import MainWindow

        for _label, step in MainWindow.MENU_STEPS:
            assert step in steps.ALL_STEPS

    def test_menu_matches_reference_order_and_shortcuts(self):
        from autosub_studio.ui.main_window import MainWindow

        assert [label for label, _step in MainWindow.MENU_STEPS][:4] == [
            "START: Format Lại Video Gốc",
            "START: Lấy Sub Bằng Chữ",
            "START: Lấy Sub Bằng Giọng Nói",
            "START: Dịch Phụ Đề",
        ]
        assert MainWindow.MENU_SHORTCUTS["START: Lấy Sub Bằng Chữ"] == "Ctrl+R"
        assert MainWindow.MENU_SHORTCUTS["START: RENDER LỒNG TIẾNG BẰNG TOOL"] == "Ctrl+S"

    def test_copied_subtitle_region_parser(self):
        from autosub_studio.ui.main_window import MainWindow

        assert MainWindow._parse_copied_region("AutoSubStudio OCR [192, 946, 1536, 98]") == [
            192,
            946,
            1536,
            98,
        ]
        assert MainWindow._parse_copied_region("1, 2, 300, 40") == [1, 2, 300, 40]
        assert MainWindow._parse_copied_region("not a region") == []
