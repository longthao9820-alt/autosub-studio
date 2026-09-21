"""Kiem thu giao dien SubtitlePanel cho AI Gateway Vision OCR (Contract ai-ocr-ui-panel-r4)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QLineEdit, QWidget

from autosub_studio.providers import ocr
from autosub_studio.services.settings import Settings
from autosub_studio.ui.panels import SubtitlePanel


class TestSubtitlePanelAiOcrUi:
    def test_server_combo_items_exactly_local_3_plus_ai_gateway(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            items = [panel.ocr_server.itemText(i) for i in range(panel.ocr_server.count())]
            assert items == [
                "PP-OCRv4 Mobile (Nhanh Như NTS)",
                "PP-OCRv6 Small (Nhanh)",
                "PP-OCRv6 Medium (Chuẩn nhất)",
                "AI Gateway",
            ]
        finally:
            panel.deleteLater()

    def test_ai_settings_box_controls_exist_and_defaults(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            assert isinstance(panel.ai_settings_box, QGroupBox)

            # Extraction task fixes the Gateway role to sub.
            assert [panel.ocr_ai_model.itemText(i) for i in range(panel.ocr_ai_model.count())] == [
                "sub",
            ]
            assert not panel.ocr_ai_model.isEnabled()

            # Batch: 1-64, default 16
            assert panel.ocr_ai_batch.minimum() == 1
            assert panel.ocr_ai_batch.maximum() == 64
            assert panel.ocr_ai_batch.value() == 16

            # Concurrency: 1-16, default 4
            assert panel.ocr_ai_concurrency.minimum() == 1
            assert panel.ocr_ai_concurrency.maximum() == 16
            assert panel.ocr_ai_concurrency.value() == 4

            # Timeout: 5-600, default 60, suffix sec
            assert panel.ocr_ai_timeout.minimum() == 5
            assert panel.ocr_ai_timeout.maximum() == 600
            assert panel.ocr_ai_timeout.value() == 60
            assert "sec" in panel.ocr_ai_timeout.suffix()

            # Retries: 0-8, default 3
            assert panel.ocr_ai_retries.minimum() == 0
            assert panel.ocr_ai_retries.maximum() == 8
            assert panel.ocr_ai_retries.value() == 3

            # Quality: 30-100, default 88
            assert panel.ocr_ai_quality.minimum() == 30
            assert panel.ocr_ai_quality.maximum() == 100
            assert panel.ocr_ai_quality.value() == 88

            # Diff: QDouble 0.1-50, step 0.5, default 30.0
            assert isinstance(panel.ocr_ai_diff, QDoubleSpinBox)
            assert panel.ocr_ai_diff.minimum() == 0.1
            assert panel.ocr_ai_diff.maximum() == 50.0
            assert panel.ocr_ai_diff.singleStep() == 0.5
            assert panel.ocr_ai_diff.value() == 30.0

            # Mode combo: Nhanh (data fast), Chinh xac (data accuracy)
            assert panel.ocr_ai_mode.count() == 2
            assert panel.ocr_ai_mode.itemText(0) == "Nhanh"
            assert panel.ocr_ai_mode.itemData(0) == "fast"
            assert panel.ocr_ai_mode.itemText(1) == "Chính xác"
            assert panel.ocr_ai_mode.itemData(1) == "accuracy"

            # Button: Kiem Tra Vision AI
            assert panel.btn_test_ai_vision.text() == "Kiểm Tra Vision AI"
        finally:
            panel.deleteLater()

    def test_combo_toggle_visibility_and_control_enablement(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            # Initially local server selected
            assert not panel.is_ai_selected()
            assert not panel.ai_settings_box.isVisible()
            assert panel.ocr_confidence.isEnabled()
            assert panel.ocr_min_height.isEnabled()
            assert panel.ocr_max_height.isEnabled()
            assert panel.ocr_batch.isEnabled()
            assert panel.ocr_count.isEnabled()
            assert panel.ocr_refine.isEnabled()

            # Switch to AI Gateway
            panel.ocr_server.setCurrentText("AI Gateway")
            assert panel.is_ai_selected()
            assert panel.ai_settings_box.isVisible()
            assert panel.ai_settings_box.isEnabled()
            assert panel.ocr_mode.currentText() == "OCR AI"

            # Local-only controls disabled
            assert not panel.ocr_confidence.isEnabled()
            assert not panel.ocr_min_height.isEnabled()
            assert not panel.ocr_max_height.isEnabled()
            assert not panel.ocr_batch.isEnabled()
            assert not panel.ocr_count.isEnabled()
            assert not panel.ocr_refine.isEnabled()
            assert not panel.color_dot.isEnabled()
            assert not panel.ocr_text_color.isEnabled()
            assert not panel.btn_pick_color.isEnabled()
            assert not panel.btn_clear_color.isEnabled()
            assert not panel.btn_measure.isEnabled()
            assert not panel.btn_test_ocr.isEnabled()

            # Recognition-neutral remain enabled
            assert panel.ocr_similarity.isEnabled()
            assert panel.ocr_min_duration.isEnabled()
            assert panel.ocr_drop_words.isEnabled()
            assert panel.ocr_drop_chars.isEnabled()
            assert panel.ocr_continuous.isEnabled()

            # Switch back to local server
            panel.ocr_server.setCurrentText("PP-OCRv4 Mobile (Nhanh Như NTS)")
            assert not panel.is_ai_selected()
            assert not panel.ai_settings_box.isVisible()
            assert panel.ocr_mode.currentText() == "Nhanh Như NTS"

            # Local controls re-enabled
            assert panel.ocr_confidence.isEnabled()
            assert panel.ocr_min_height.isEnabled()
            assert panel.ocr_max_height.isEnabled()
            assert panel.ocr_batch.isEnabled()
            assert panel.ocr_count.isEnabled()
            assert panel.ocr_refine.isEnabled()

            # Switching ocr_mode to "OCR AI" sets server to "AI Gateway"
            panel.ocr_mode.setCurrentText("OCR AI")
            assert panel.ocr_server.currentText() == "AI Gateway"
            assert panel.is_ai_selected()
            assert panel.ai_settings_box.isVisible()

            # Switching ocr_mode to "Cân Bằng" sets server to Small
            panel.ocr_mode.setCurrentText("Cân Bằng")
            assert panel.ocr_server.currentText() == "PP-OCRv6 Small (Nhanh)"
            assert not panel.is_ai_selected()
            assert not panel.ai_settings_box.isVisible()
        finally:
            panel.deleteLater()

    def test_all_fields_save_load_roundtrip(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            s = Settings(
                ocr_server="AI Gateway",
                ocr_mode="OCR AI",
                ocr_ai_model="prime",
                ocr_ai_batch_size=12,
                ocr_ai_max_concurrency=6,
                ocr_ai_timeout=120,
                ocr_ai_max_retries=5,
                ocr_ai_image_quality=92,
                ocr_ai_diff_threshold=5.5,
                ocr_ai_consensus_mode="accuracy",
                ocr_ai_consensus_frames=3,
                ocr_batch_size=10,
            )
            panel.load(s)

            assert panel.ocr_server.currentText() == "AI Gateway"
            assert panel.is_ai_selected()
            assert panel.ocr_ai_model.currentText() == "sub"
            assert panel.ocr_ai_batch.value() == 12
            assert panel.ocr_ai_concurrency.value() == 6
            assert panel.ocr_ai_timeout.value() == 120
            assert panel.ocr_ai_retries.value() == 5
            assert panel.ocr_ai_quality.value() == 92
            assert panel.ocr_ai_diff.value() == 5.5
            assert panel.ocr_ai_mode.currentData() == "accuracy"
            assert panel.ocr_batch.value() == 10

            # Modify all fields in panel
            panel.ocr_ai_model.setCurrentText("sub")
            panel.ocr_ai_batch.setValue(16)
            panel.ocr_ai_concurrency.setValue(8)
            panel.ocr_ai_timeout.setValue(45)
            panel.ocr_ai_retries.setValue(2)
            panel.ocr_ai_quality.setValue(80)
            panel.ocr_ai_diff.setValue(2.5)
            panel.ocr_ai_mode.setCurrentIndex(panel.ocr_ai_mode.findData("fast"))
            panel.ocr_batch.setValue(7)

            s_out = Settings()
            panel.apply(s_out)

            assert s_out.ocr_server == "AI Gateway"
            assert s_out.ocr_ai_model == "sub"
            assert s_out.ocr_ai_batch_size == 16
            assert s_out.ocr_ai_max_concurrency == 8
            assert s_out.ocr_ai_timeout == 45
            assert s_out.ocr_ai_max_retries == 2
            assert s_out.ocr_ai_image_quality == 80
            assert s_out.ocr_ai_diff_threshold == 2.5
            assert s_out.ocr_ai_consensus_mode == "fast"
            assert s_out.ocr_ai_consensus_frames == 1
            assert s_out.ocr_batch_size == 7
        finally:
            panel.deleteLater()

    def test_migration_legacy_server_and_concurrency_config(
        self, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "config.json"
        monkeypatch.setattr(Settings, "config_path", staticmethod(lambda *_args: cfg_file))

        legacy_data = {
            "schema_version": 12,
            "ocr_mode": "OCR AI",
            "ocr_server": "Server AI API",
            "ai_ocr_concurrency": 7,
            "ocr_batch_size": 9,
        }
        cfg_file.write_text(json.dumps(legacy_data, ensure_ascii=False), encoding="utf-8")

        loaded = Settings.load()
        assert loaded.ocr_server == "AI Gateway"
        assert loaded.ocr_ai_max_concurrency == 7
        assert loaded.ocr_batch_size == 9
        assert "ai_ocr_concurrency" not in loaded.extra

        panel = SubtitlePanel()
        panel.show()
        try:
            # Loading legacy Settings instance directly also migrates server label
            legacy_s = Settings(ocr_server="Server AI API", ocr_ai_max_concurrency=7)
            panel.load(legacy_s)
            assert panel.ocr_server.currentText() == "AI Gateway"
            assert panel.is_ai_selected()
            assert panel.ocr_ai_concurrency.value() == 7
        finally:
            panel.deleteLater()

    def test_ai_start_independent_of_local_engine_local_requires_engine(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Case 1: local ocr is NOT available
        monkeypatch.setattr(ocr, "is_available", lambda: False)
        panel = SubtitlePanel()
        panel.show()
        try:
            # Local server -> START disabled
            panel.ocr_server.setCurrentText("PP-OCRv4 Mobile (Nhanh Như NTS)")
            panel._update_ai_mode_ui()
            assert not panel.btn_ocr.isEnabled()
            assert not panel.btn_format_ocr.isEnabled()

            # AI Gateway -> START enabled even though local engine missing
            panel.ocr_server.setCurrentText("AI Gateway")
            assert panel.btn_ocr.isEnabled()
            assert panel.btn_format_ocr.isEnabled()

            # Switch back to local server -> START disabled again
            panel.ocr_server.setCurrentText("PP-OCRv4 Mobile (Nhanh Như NTS)")
            assert not panel.btn_ocr.isEnabled()
            assert not panel.btn_format_ocr.isEnabled()

            # Case 2: local ocr IS available
            monkeypatch.setattr(ocr, "is_available", lambda: True)
            panel._update_ai_mode_ui()

            # Local server -> START enabled
            assert panel.btn_ocr.isEnabled()
            assert panel.btn_format_ocr.isEnabled()

            # AI Gateway -> START still enabled
            panel.ocr_server.setCurrentText("AI Gateway")
            assert panel.btn_ocr.isEnabled()
            assert panel.btn_format_ocr.isEnabled()
        finally:
            panel.deleteLater()

    def test_ai_vision_test_signal_emits(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            panel.ocr_server.setCurrentText("AI Gateway")
            spy = MagicMock()
            panel.testAiVision.connect(spy)
            panel.btn_test_ai_vision.click()
            assert spy.call_count == 1
        finally:
            panel.deleteLater()

    def test_region_signals_and_table_unchanged(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            region_spy = MagicMock()
            panel.markOcrRegion.connect(region_spy)
            panel.btn_region.click()
            assert region_spy.call_count == 1

            panel.set_regions([10, 20, 100, 50], [0, 0, 50, 50])
            assert panel.region_table.rowCount() == 2
            item_1 = panel.region_table.item(0, 1)
            item_2 = panel.region_table.item(0, 2)
            item_3 = panel.region_table.item(0, 3)
            item_4 = panel.region_table.item(0, 4)
            assert item_1 is not None
            assert item_2 is not None
            assert item_3 is not None
            assert item_4 is not None
            assert item_1.text() == "10"
            assert item_2.text() == "20"
            assert item_3.text() == "100"
            assert item_4.text() == "50"
        finally:
            panel.deleteLater()

    def test_no_widgets_contain_key(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            for child in panel.findChildren(QWidget):
                # No widget should be an API key input
                assert "api_key" not in child.objectName().lower()
                assert "apikey" not in child.objectName().lower()
                if isinstance(child, QLineEdit):
                    assert "key" not in child.placeholderText().lower()
                    assert "khóa" not in child.placeholderText().lower()
                    assert "khoa" not in child.placeholderText().lower()
        finally:
            panel.deleteLater()

    def test_tooltips_clear_no_local_fallback(self, qapp) -> None:
        panel = SubtitlePanel()
        panel.show()
        try:
            panel.ocr_server.setCurrentText("AI Gateway")
            tooltips = [
                panel.ocr_server.toolTip(),
                panel.ai_settings_box.toolTip(),
                panel.ocr_ai_model.toolTip(),
                panel.ocr_ai_batch.toolTip(),
                panel.ocr_ai_concurrency.toolTip(),
                panel.ocr_ai_timeout.toolTip(),
                panel.ocr_ai_retries.toolTip(),
                panel.ocr_ai_mode.toolTip(),
                panel.btn_ocr.toolTip(),
                panel.btn_format_ocr.toolTip(),
                panel.btn_test_ai_vision.toolTip(),
            ]
            for tt in tooltips:
                assert "fallback" in tt.lower()
        finally:
            panel.deleteLater()
