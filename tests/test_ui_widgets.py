"""Kiem thu cho cac thanh phan giao dien va ham loc ket qua OCR."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QLineEdit

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.providers import ocr
from autosub_studio.ui.cue_table import (
    COL_INDEX,
    COL_START,
    COL_TEXT,
    CueTableModel,
    CueTableView,
)
from autosub_studio.ui.project_table import (
    COL_NAME,
    COL_PROGRESS,
    COL_STATUS,
    ProjectRow,
    ProjectTableModel,
)
from autosub_studio.ui.widgets import (
    ROW_ACTIONS,
    PillTabBar,
    RowActionDelegate,
    VerticalTabStrip,
)


class TestOcrCleaning:
    def test_drops_unwanted_characters(self):
        out = ocr.clean_cues([Cue(0, 1, "Xin |chao| the gioi")], drop_chars="|")
        assert out[0].text == "Xin chao the gioi"

    def test_drops_unwanted_words(self):
        out = ocr.clean_cues([Cue(0, 1, "Dang ky kenh de xem them")], drop_words="Dang ky kenh")
        assert "Dang ky kenh" not in out[0].text

    def test_word_removal_ignores_case(self):
        out = ocr.clean_cues([Cue(0, 1, "SUBSCRIBE ngay hom nay")], drop_words="subscribe")
        assert "SUBSCRIBE" not in out[0].text

    def test_removes_cue_that_becomes_empty(self):
        out = ocr.clean_cues([Cue(0, 1, "###"), Cue(1, 2, "con lai")], drop_chars="#")
        assert len(out) == 1
        assert out[0].text == "con lai"

    def test_keeps_everything_without_filters(self):
        cues = [Cue(0, 1, "giu nguyen")]
        assert ocr.clean_cues(cues)[0].text == "giu nguyen"

    def test_multiple_filters_at_once(self):
        out = ocr.clean_cues(
            [Cue(0, 1, "[Nhac] xin chao ban")], drop_chars="[,]", drop_words="Nhac"
        )
        assert out[0].text == "xin chao ban"

    def test_continuous_closes_small_gaps(self):
        cues = [Cue(0, 1.0, "a"), Cue(1.4, 2.0, "b")]
        ocr.make_continuous(cues)
        assert cues[0].end == 1.4

    def test_continuous_keeps_big_gaps(self):
        cues = [Cue(0, 1.0, "a"), Cue(9.0, 10.0, "b")]
        ocr.make_continuous(cues)
        assert cues[0].end == 1.0

    def test_continuous_handles_single_cue(self):
        cues = [Cue(0, 1.0, "a")]
        assert ocr.make_continuous(cues)[0].end == 1.0


class TestReferenceTableAppearance:
    def test_time_column_displays_start_and_end(self, qapp):
        model = CueTableModel()
        model.set_document(SubtitleDoc([Cue(0.133, 2.6, "Mới 31 tuổi", speaker="1")]))

        assert model.data(model.index(0, COL_INDEX)) == "No. 1"
        assert model.data(model.index(0, COL_START)) == "00:00:00,133 --> 00:00:02,600"
        assert (
            model.data(model.index(0, COL_START), Qt.ItemDataRole.EditRole)
            == "00:00:00,133 --> 00:00:02,600"
        )

    def test_time_column_edits_start_and_end_together(self, qapp):
        model = CueTableModel()
        model.set_document(SubtitleDoc([Cue(0.133, 2.6, "中文字幕")]))

        changed = model.setData(
            model.index(0, COL_START),
            "00:00:01,000 --> 00:00:04,250",
            Qt.ItemDataRole.EditRole,
        )

        assert changed
        assert model.doc.cues[0].start == 1.0
        assert model.doc.cues[0].end == 4.25

    def test_cue_colors_match_reference(self, qapp):
        model = CueTableModel()
        model.set_document(SubtitleDoc([Cue(0.133, 2.6, "中文字幕", speaker="1")]))

        time_brush = model.data(model.index(0, COL_START), Qt.ItemDataRole.ForegroundRole)
        text_brush = model.data(model.index(0, COL_TEXT), Qt.ItemDataRole.ForegroundRole)
        assert time_brush.color().name() == "#ffeb00"
        assert text_brush.color().name() == "#f5f0a0"

    def test_project_status_uses_reference_cyan(self, qapp):
        model = ProjectTableModel()
        model.set_rows([ProjectRow(id=1, name="Demo", status="STOP trích xuất")])

        brush = model.data(model.index(0, COL_STATUS), Qt.ItemDataRole.ForegroundRole)
        assert brush.color().name() == "#67ddfd"

    def test_batch_status_does_not_paint_a_false_selection_background(self, qapp):
        model = ProjectTableModel()
        model.set_rows(
            [
                ProjectRow(id=1, name="Đang chạy", status="Đang Trích Xuất..."),
                ProjectRow(id=2, name="Hàng chờ", status="Đang đợi..."),
            ]
        )

        for row in range(model.rowCount()):
            assert (
                model.data(model.index(row, COL_NAME), Qt.ItemDataRole.BackgroundRole)
                is None
            )
            status_brush = model.data(
                model.index(row, COL_STATUS), Qt.ItemDataRole.ForegroundRole
            )
            assert status_brush.color().name() == "#67ddfd"
            assert (
                model.data(model.index(row, COL_NAME), Qt.ItemDataRole.ForegroundRole)
                is None
            )

    def test_project_state_and_progress_colors_match_reference(self, qapp):
        model = ProjectTableModel()
        model.set_rows(
            [
                ProjectRow(
                    id=1,
                    name="Demo",
                    has_subtitle=True,
                    has_translation=False,
                    progress=4,
                )
            ]
        )

        subtitle = model.data(model.index(0, 4), Qt.ItemDataRole.ForegroundRole)
        translation = model.data(model.index(0, 5), Qt.ItemDataRole.ForegroundRole)
        progress = model.data(
            model.index(0, COL_PROGRESS), Qt.ItemDataRole.ForegroundRole
        )
        assert subtitle.color().name() == "#06ff6f"
        assert translation.color().name() == "#f81919"
        assert progress.color().name() == "#ffe400"

    def test_undo_restores_the_previous_subtitle_edit(self, qapp):
        model = CueTableModel()
        model.set_document(SubtitleDoc([Cue(0, 2, "Nội dung ban đầu")]))
        model.push_history("Sửa nội dung")
        model.doc.cues[0].text = "Nội dung nhập nhầm"

        assert model.undo() == "Sửa nội dung"
        assert model.doc.cues[0].text == "Nội dung ban đầu"
        assert model.redo() == "Sửa nội dung"
        assert model.doc.cues[0].text == "Nội dung nhập nhầm"

    def test_selected_subtitle_cell_can_be_clicked_again_to_edit(self, qapp):
        model = CueTableModel()
        model.set_document(SubtitleDoc([Cue(0.4, 1.6, "中文字幕")]))
        view = CueTableView()
        view.setModel(model)
        view.resize(900, 180)
        view.show()
        qapp.processEvents()

        view.pressed.connect(lambda index: model.set_current(index.row()))
        selection = view.selectionModel()
        selection.currentRowChanged.connect(lambda index, _old: model.set_current(index.row()))
        index = model.index(0, COL_TEXT)
        point = view.visualRect(index).center()

        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.qWait(qapp.doubleClickInterval() + 20)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.qWait(qapp.doubleClickInterval() + 20)

        editors = [editor for editor in view.findChildren(QLineEdit) if editor.isVisible()]
        assert view.state() == QAbstractItemView.State.EditingState
        assert len(editors) == 1
        view.close()


class TestPillTabBar:
    def test_first_tab_active_by_default(self, qapp):
        bar = PillTabBar(["Mot", "Hai", "Ba"])
        assert bar.current() == 0
        assert bar.buttons[0].isChecked()

    def test_switching_updates_state(self, qapp):
        bar = PillTabBar(["Mot", "Hai"])
        seen: list[int] = []
        bar.currentChanged.connect(seen.append)
        bar.set_current(1)
        assert bar.current() == 1
        assert seen == [1]
        assert not bar.buttons[0].isChecked()

    def test_ignores_bad_index(self, qapp):
        bar = PillTabBar(["Mot", "Hai"])
        bar.set_current(99)
        assert bar.current() == 0

    def test_labels_are_kept(self, qapp):
        bar = PillTabBar(["Danh Sach", "B1"])
        assert [b.text() for b in bar.buttons] == ["Danh Sach", "B1"]


class TestVerticalTabStrip:
    def test_default_selection(self, qapp):
        strip = VerticalTabStrip(["Screen Edit", "Screen Render"])
        assert strip.current() == 0

    def test_switch_emits_signal(self, qapp):
        strip = VerticalTabStrip(["A", "B"])
        seen: list[int] = []
        strip.currentChanged.connect(seen.append)
        strip.set_current(1)
        assert strip.current() == 1
        assert seen == [1]

    def test_has_one_button_per_label(self, qapp):
        strip = VerticalTabStrip(["A", "B", "C"])
        assert len(strip.buttons) == 3


class TestBatchWorkerSetting:
    def test_main_window_can_create_save_switch_and_delete_config_profile(
        self, qapp, tmp_path, monkeypatch
    ):
        from PySide6.QtWidgets import QMessageBox

        from autosub_studio.services.settings import Settings
        from autosub_studio.ui.main_window import MainWindow

        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        settings = Settings(workspace=str(tmp_path / "workspace"))
        settings.remember_active_profile()
        settings.save()
        window = MainWindow()
        window.dub_panel.volume.setValue(77)
        window._create_config_profile("American Review")
        assert window.settings.active_config_profile == "American Review"
        window.dub_panel.volume.setValue(65)
        window._save_settings()

        window._switch_config_profile("default")
        assert window.dub_panel.volume.value() == 77
        window._switch_config_profile("American Review")
        assert window.dub_panel.volume.value() == 65

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
        )
        window._delete_config_profile("American Review")
        assert "American Review" not in window.settings.config_profiles
        assert window.settings.active_config_profile == "default"
        window.close()
        qapp.processEvents()

    def test_general_settings_has_real_save_create_and_delete_buttons(self, qapp):
        from autosub_studio.ui.panels import SettingsPanel

        panel = SettingsPanel()
        saved = []
        created = []
        deleted = []
        panel.saveRequested.connect(lambda: saved.append(True))
        panel.createPresetRequested.connect(created.append)
        panel.deletePresetRequested.connect(deleted.append)
        panel.btn_update.click()
        panel.new_preset_name.setText("American Review")
        panel.btn_save.click()
        panel.app_preset.addItem("Delete Me")
        panel.app_preset.setCurrentText("Delete Me")
        panel.btn_clean.click()

        assert saved == [True]
        assert created == ["American Review"]
        assert deleted == ["Delete Me"]
        panel.deleteLater()

    def test_dub_nts_controls_are_persisted(self, qapp):
        from autosub_studio.services.settings import Settings
        from autosub_studio.ui.panels import DubPanel

        panel = DubPanel()
        settings = Settings()
        panel.load(settings)
        panel.voice.setCurrentText("vi-VN-HoaiMyNeural")
        panel.profile_gender.setCurrentText("Nữ")
        panel._add_voice_profile()
        panel.mode.setCurrentText("Lồng Tiếng Vào Video")
        panel.rate.setValue(115)
        panel.original_volume.setValue(20)
        panel.end_pause.setValue(250)
        panel.apply(settings)

        assert settings.dub_output_mode == "video"
        assert settings.tts_speed_percent == 115
        assert settings.original_audio_volume == 20
        assert settings.tts_end_pause_ms == 250
        assert settings.tts_voice_profiles[0]["gender"] == "nu"
        panel.deleteLater()

    def test_new_machine_stays_unchecked_until_button_is_pressed(self, qapp):
        from autosub_studio.services.settings import Settings
        from autosub_studio.ui.panels import SubtitlePanel

        settings = Settings()
        settings.hardware_signature = ""
        settings.use_gpu = False
        panel = SubtitlePanel()
        panel.load(settings)

        assert panel.machine_state.text() == "Chưa kiểm tra cấu hình máy"
        assert "Bấm 'Kiểm Tra Cấu Hình Máy'" in panel.gpu_note.text()
        panel.deleteLater()

    def test_parallel_ocr_limit_is_visible_in_general_settings(self, qapp):
        from autosub_studio.ui.panels import SettingsPanel

        panel = SettingsPanel()

        assert not panel.workers.isHidden()
        assert panel.workers.minimum() == 1
        assert panel.workers.maximum() == 8
        assert "hàng chờ" in panel.workers.toolTip()
        panel.deleteLater()

    def test_project_rows_do_not_auto_sort_when_status_changes(self, qapp):
        from autosub_studio.ui.project_table import ProjectTableView

        view = ProjectTableView()

        assert not view.isSortingEnabled()
        view.deleteLater()

    def test_plain_click_replaces_a_multi_project_selection(self, qapp):
        from autosub_studio.ui.project_table import ProjectTableView

        model = ProjectTableModel()
        model.set_rows(
            [ProjectRow(id=index, name=f"Project {index}") for index in range(1, 5)]
        )
        view = ProjectTableView()
        view.setModel(model)
        view.resize(900, 260)
        view.show()
        qapp.processEvents()

        first = view.visualRect(model.index(0, COL_NAME)).center()
        second = view.visualRect(model.index(1, COL_NAME)).center()
        third = view.visualRect(model.index(2, COL_NAME)).center()
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            pos=second,
        )
        assert sorted(index.row() for index in view.selectionModel().selectedRows()) == [0, 1]

        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=third)

        assert [index.row() for index in view.selectionModel().selectedRows()] == [2]
        view.close()


class TestRowActionDelegate:
    def test_four_actions(self):
        assert len(ROW_ACTIONS) == 4

    def test_icon_rects_stay_inside_cell(self, qapp):
        from PySide6.QtCore import QRect

        delegate = RowActionDelegate()
        cell = QRect(0, 0, 140, 30)
        rects = delegate._icon_rects(cell)
        assert len(rects) == 4
        assert rects[0].left() >= cell.left()
        assert rects[-1].right() <= cell.right() + 2

    def test_hit_test_finds_each_icon(self, qapp):
        from PySide6.QtCore import QRect

        delegate = RowActionDelegate()
        cell = QRect(0, 0, 140, 30)
        for index, box in enumerate(delegate._icon_rects(cell)):
            assert delegate._hit(box.center(), cell) == index

    def test_hit_test_misses_empty_space(self, qapp):
        from PySide6.QtCore import QPoint, QRect

        delegate = RowActionDelegate()
        cell = QRect(0, 0, 300, 30)
        assert delegate._hit(QPoint(295, 15), cell) == -1
