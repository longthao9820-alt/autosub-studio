"""Kiem thu giao dien Vision AI cho AIGatewayDialog va MainWindow."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QMessageBox, QPushButton

from autosub_studio.services import ai_gateway
from autosub_studio.services.settings import Settings
from autosub_studio.services.tasks import DONE, FAILED, TaskContext
from autosub_studio.ui.dialogs import (
    _ORPHAN_VISION_WORKERS,
    AIGatewayDialog,
    _VisionTestWorker,
)
from autosub_studio.ui.main_window import MainWindow


class TestAIGatewayDialogVision:
    def test_dialog_vision_buttons_exist(self, qapp) -> None:
        s = Settings()
        dialog = AIGatewayDialog(s)
        try:
            assert isinstance(dialog.btn_test_vision_sub, QPushButton)
            assert isinstance(dialog.btn_test_vision_prime, QPushButton)
            assert dialog.btn_test_vision_sub.text() == "Test Vision Sub"
            assert dialog.btn_test_vision_prime.text() == "Test Dịch Prime"
        finally:
            dialog.deleteLater()

    def test_dialog_missing_endpoint_shows_error(self, qapp) -> None:
        s = Settings(ai_endpoint="")
        dialog = AIGatewayDialog(s)
        try:
            dialog.endpoint.setText("")
            dialog.model_sub.setText("sub")
            dialog.btn_test_vision_sub.click()
            assert "Chưa nhập Endpoint" in dialog.status.text()
            assert len(dialog._vision_workers) == 0

            dialog.btn_test_vision_prime.click()
            assert "Chưa nhập Endpoint" in dialog.status.text()
            assert len(dialog._vision_workers) == 0
        finally:
            dialog.deleteLater()

    def test_dialog_missing_model_shows_error(self, qapp) -> None:
        s = Settings(ai_endpoint="https://api.openai.com/v1")
        dialog = AIGatewayDialog(s)
        try:
            dialog.endpoint.setText("https://api.openai.com/v1")
            dialog.model_sub.setText("")
            dialog.btn_test_vision_sub.click()
            assert "Chưa chỉ định tên model" in dialog.status.text()
            assert len(dialog._vision_workers) == 0

            dialog.model_prime.setText("")
            dialog.btn_test_vision_prime.click()
            assert "Chưa chỉ định tên model" in dialog.status.text()
            assert len(dialog._vision_workers) == 0
        finally:
            dialog.deleteLater()

    def test_dialog_unsaved_values_passed_to_worker(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = Settings(
            ai_endpoint="https://old-endpoint.com/v1",
            ai_model_sub="old-sub",
            ai_thinking_sub="low",
        )
        dialog = AIGatewayDialog(s)
        try:
            # Type unsaved values into dialog fields
            dialog.endpoint.setText("https://unsaved-endpoint.com/v1")
            dialog.api_key.setText("sk-unsaved-secret-99999")
            dialog.model_sub.setText("new-custom-sub")
            dialog.thinking_sub.setCurrentText("high")

            captured_worker_args = {}

            def fake_start(worker_self: _VisionTestWorker) -> None:
                captured_worker_args["endpoint"] = worker_self.endpoint
                captured_worker_args["api_key"] = worker_self.api_key
                captured_worker_args["model"] = worker_self.model
                captured_worker_args["thinking"] = worker_self.thinking
                # Synchronously emit finished
                worker_self.finished.emit(
                    True, "Vision model 'new-custom-sub' hoạt động tốt.", 150.0
                )

            monkeypatch.setattr(_VisionTestWorker, "start", fake_start)

            dialog.btn_test_vision_sub.click()

            assert captured_worker_args["endpoint"] == "https://unsaved-endpoint.com/v1"
            assert captured_worker_args["api_key"] == "sk-unsaved-secret-99999"
            assert captured_worker_args["model"] == "new-custom-sub"
            assert captured_worker_args["thinking"] == "high"

            # Verify API key never leaks in status text
            assert "sk-unsaved-secret" not in dialog.status.text()
            assert "99999" not in dialog.status.text()
        finally:
            dialog.deleteLater()

    def test_dialog_success_exact_multiline_output(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = Settings(ai_endpoint="https://api.openai.com/v1")
        dialog = AIGatewayDialog(s)
        try:
            dialog.endpoint.setText("https://api.openai.com/v1")
            dialog.model_sub.setText("gpt-4o-mini")

            def fake_start(worker_self: _VisionTestWorker) -> None:
                worker_self.finished.emit(
                    True,
                    "Vision model 'gpt-4o-mini' hoạt động tốt. Nhận diện: 'TEST'.",
                    42.6,
                )

            monkeypatch.setattr(_VisionTestWorker, "start", fake_start)

            dialog.btn_test_vision_sub.click()

            status_text = dialog.status.text()
            expected_lines = [
                "AI Gateway: Connected",
                "Authentication: Valid",
                "Model: gpt-4o-mini",
                "Vision OCR: OK",
                "Latency: 43 ms",
            ]
            for line in expected_lines:
                assert line in status_text

            assert dialog.btn_test_vision_sub.isEnabled()
        finally:
            dialog.deleteLater()

    def test_dialog_failure_sanitized_output(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = Settings(ai_endpoint="https://api.openai.com/v1")
        dialog = AIGatewayDialog(s)
        try:
            dialog.endpoint.setText("https://api.openai.com/v1")
            dialog.model_prime.setText("gpt-4o")

            monkeypatch.setattr(
                ai_gateway,
                "test_translation_role",
                lambda *_args, **_kwargs: (
                    False,
                    "HTTP 401: Unauthorized with key sk-secret1234567890",
                ),
            )

            dialog.btn_test_vision_prime.click()

            status_text = dialog.status.text()
            # AI Gateway: Connected must ONLY appear on success
            assert "AI Gateway: Connected" not in status_text
            # Secret key must be sanitized
            assert "sk-secret1234567890" not in status_text
            assert "sk-***" in status_text or "Unauthorized" in status_text
            assert dialog.btn_test_vision_prime.isEnabled()
        finally:
            dialog.deleteLater()

    def test_dialog_close_worker_cleanup_and_lifetime(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = Settings(ai_endpoint="https://api.openai.com/v1")
        dialog = AIGatewayDialog(s)
        try:
            dialog.endpoint.setText("https://api.openai.com/v1")
            dialog.model_sub.setText("sub")

            worker_inst = None

            def fake_start(worker_self: _VisionTestWorker) -> None:
                nonlocal worker_inst
                worker_inst = worker_self
                # Do not emit finished immediately; simulate running worker

            monkeypatch.setattr(_VisionTestWorker, "start", fake_start)

            dialog.btn_test_vision_sub.click()
            assert worker_inst is not None
            assert worker_inst in dialog._vision_workers

            # Simulate worker is running
            monkeypatch.setattr(worker_inst, "isRunning", lambda: True)
            interruption_requested = False

            def fake_interrupt() -> None:
                nonlocal interruption_requested
                interruption_requested = True

            monkeypatch.setattr(worker_inst, "requestInterruption", fake_interrupt)
            waited_ms: list[int] = []
            monkeypatch.setattr(worker_inst, "wait", lambda ms: waited_ms.append(ms))

            # Close dialog
            dialog.reject()

            assert interruption_requested is True
            assert 1000 in waited_ms
            # Worker moved to _ORPHAN_VISION_WORKERS to prevent GC crash
            assert worker_inst in _ORPHAN_VISION_WORKERS
            assert len(dialog._vision_workers) == 0

            # When worker finally finishes, it cleans up from _ORPHAN_VISION_WORKERS
            worker_inst.finished.emit(True, "Done", 10.0)
            assert worker_inst not in _ORPHAN_VISION_WORKERS
        finally:
            dialog.deleteLater()


class TestMainWindowVisionIntegration:
    def test_main_window_signal_connection_exists(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        win = MainWindow()
        try:
            assert hasattr(win, "_test_ai_vision")
            assert hasattr(win, "_ai_vision_task_ids")
            # Triggering signal invokes _test_ai_vision
            spy = MagicMock()
            monkeypatch.setattr(win, "_test_ai_vision", spy)
            win.subtitle_panel.testAiVision.emit()
            assert spy.call_count == 1
        finally:
            win.close()
            qapp.processEvents()

    def test_main_window_missing_endpoint_shows_error(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        win = MainWindow()
        try:
            win.settings.ai_endpoint = ""
            submit_spy = MagicMock()
            monkeypatch.setattr(win.tasks, "submit", submit_spy)

            win._test_ai_vision()

            assert "Chưa nhập Endpoint AI Gateway" in win.subtitle_panel.status.text()
            assert submit_spy.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_main_window_alias_resolution_and_task_submission(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        win = MainWindow()
        try:
            win.settings.ai_endpoint = "https://gateway.ai/v1"
            win.settings.ocr_ai_model = "prime"
            win.settings.ai_model_sub = "qwen-vl-sub"
            win.settings.ai_thinking_sub = "low"
            win.subtitle_panel.ocr_ai_timeout.setValue(25)
            monkeypatch.setattr(
                Settings, "get_secret", lambda name: "sk-test-secret-key-123"
            )

            submitted_jobs = []

            def fake_submit(label: str, job, project_id: int = 0, timeout: int = 60) -> str:
                submitted_jobs.append((label, job, timeout))
                return "task-ai-vision-999"

            monkeypatch.setattr(win.tasks, "submit", fake_submit)

            win._test_ai_vision()

            assert len(submitted_jobs) == 1
            label, job, timeout = submitted_jobs[0]
            assert "qwen-vl-sub" in label
            assert timeout == 35  # timeout + 10
            assert "task-ai-vision-999" in win._ai_vision_task_ids

            # Test the submitted job execution
            mock_test_vision = MagicMock(return_value=(True, "Nhận diện: 'TEST'"))
            from autosub_studio.providers import ocr_ai_provider

            monkeypatch.setattr(ocr_ai_provider, "test_vision", mock_test_vision)

            ctx = MagicMock(spec=TaskContext)
            result = job(ctx)

            assert "Vision OCR: OK" in result
            assert "Latency:" in result
            assert mock_test_vision.call_count == 1
            call_kwargs = mock_test_vision.call_args[1]
            assert call_kwargs["model"] == "qwen-vl-sub"
            assert call_kwargs["thinking"] == "low"
            assert call_kwargs["timeout"] == 25.0
        finally:
            win.close()
            qapp.processEvents()

    def test_main_window_task_job_failure_raises_sanitized(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        win = MainWindow()
        try:
            win.settings.ai_endpoint = "https://gateway.ai/v1"
            win.settings.ocr_ai_model = "sub"
            win.settings.ai_model_sub = "gpt-4o"
            monkeypatch.setattr(Settings, "get_secret", lambda name: "sk-secret-999")

            submitted_jobs = []

            def fake_submit(label: str, job, project_id: int = 0, timeout: int = 60) -> str:
                submitted_jobs.append(job)
                return "task-1"

            monkeypatch.setattr(win.tasks, "submit", fake_submit)

            win._test_ai_vision()
            job = submitted_jobs[0]

            from autosub_studio.providers import ocr_ai_provider

            monkeypatch.setattr(
                ocr_ai_provider,
                "test_vision",
                lambda *args, **kwargs: (False, "Error with key sk-secret-999"),
            )

            ctx = MagicMock(spec=TaskContext)
            with pytest.raises(RuntimeError) as exc_info:
                job(ctx)

            err_msg = str(exc_info.value)
            assert "sk-secret-999" not in err_msg
            assert "sk-***" in err_msg
        finally:
            win.close()
            qapp.processEvents()

    def test_main_window_on_task_finished_success_and_failure(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        win = MainWindow()
        try:
            # Case 1: Success
            win._ai_vision_task_ids.add("task-success")
            win._on_task_finished(
                "task-success",
                DONE,
                "Vision OCR: OK (Latency: 50 ms) - Good",
                "",
            )
            assert "task-success" not in win._ai_vision_task_ids
            assert win.subtitle_panel.status.text() == "Vision OCR: OK (Latency: 50 ms) - Good"

            # Case 2: Failure - must NOT show QMessageBox.critical
            critical_spy = MagicMock()
            monkeypatch.setattr(QMessageBox, "critical", critical_spy)

            win._ai_vision_task_ids.add("task-fail")
            win._on_task_finished(
                "task-fail",
                FAILED,
                None,
                "Connection error sk-leak12345678",
            )
            assert "task-fail" not in win._ai_vision_task_ids
            assert "Lỗi:" in win.subtitle_panel.status.text()
            assert "sk-leak12345678" not in win.subtitle_panel.status.text()
            assert critical_spy.call_count == 0
        finally:
            win.close()
            qapp.processEvents()

    def test_main_window_on_task_log_metrics_display(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        win = MainWindow()
        try:
            metrics_msg = (
                "OCR AI Metrics: sampled=10, rejected=2, segments=5, "
                "avg_batch=4.0, latency=0.25s, elapsed=1.50s"
            )
            log_spy = MagicMock()
            win.logAppended.connect(log_spy)

            win._on_task_log("task-ocr", metrics_msg)

            # Log signal still emitted
            assert log_spy.call_count == 1
            assert log_spy.call_args[0][0] == metrics_msg

            # SubtitlePanel status shows heading followed by line-separated fields
            status_text = win.subtitle_panel.status.text()
            assert status_text.startswith("OCR AI Metrics:")
            assert "sampled=10\n" in status_text
            assert "rejected=2\n" in status_text
            assert "avg_batch=4.0\n" in status_text

            # Unrelated message does NOT alter SubtitlePanel status
            win._on_task_log("task-other", "Starting audio extraction...")
            assert win.subtitle_panel.status.text() == status_text
        finally:
            win.close()
            qapp.processEvents()

    def test_no_local_ocr_called_during_vision_test(
        self, qapp, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        Settings.config_path().write_text(
            json.dumps({"workspace": str(tmp_path / "workspace")}), encoding="utf-8"
        )

        # Mock out local ocr provider functions to ensure they are NEVER called
        from autosub_studio.providers import ocr

        mock_local_ocr = MagicMock()
        monkeypatch.setattr(ocr, "read_frames", mock_local_ocr)
        monkeypatch.setattr(ocr, "read_frame", mock_local_ocr)
        monkeypatch.setattr(ocr, "probe_frames", mock_local_ocr)
        monkeypatch.setattr(ocr, "_load_engine", mock_local_ocr)

        win = MainWindow()
        try:
            win.settings.ai_endpoint = "https://gateway.ai/v1"
            win.settings.ocr_ai_model = "sub"
            monkeypatch.setattr(Settings, "get_secret", lambda name: "sk-secret")

            from autosub_studio.providers import ocr_ai_provider

            monkeypatch.setattr(
                ocr_ai_provider, "test_vision", lambda *a, **kw: (True, "Vision OK")
            )

            # Run MainWindow vision test
            submitted_jobs = []

            def fake_submit(label: str, job, project_id: int = 0, timeout: int = 60) -> str:
                submitted_jobs.append(job)
                return "t1"

            monkeypatch.setattr(win.tasks, "submit", fake_submit)
            win._test_ai_vision()
            ctx = MagicMock(spec=TaskContext)
            submitted_jobs[0](ctx)

            # Run Dialog vision test worker
            worker = _VisionTestWorker(
                "https://gateway.ai/v1",
                "sk-secret",
                model="sub",
            )
            worker.run()

            # Local OCR must not be called
            assert mock_local_ocr.call_count == 0
        finally:
            win.close()
            qapp.processEvents()
