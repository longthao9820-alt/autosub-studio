"""Cac bai kiem thu toan dien cho AI Gateway, dich B2, doc chu OCR AI va hop thoai UI."""

from __future__ import annotations

import io
import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from autosub_studio.core.models import Cue
from autosub_studio.data.project import ProjectStore
from autosub_studio.pipeline import steps as P
from autosub_studio.providers import ocr_ai, translate
from autosub_studio.services import ai_gateway
from autosub_studio.services.ai_gateway import (
    AIGatewayError,
    chat_completion,
    normalize_chat_endpoint,
    normalize_models_endpoint,
    resolve_model,
)
from autosub_studio.services.ffmpeg import FFmpeg
from autosub_studio.services.settings import Settings
from autosub_studio.ui.dialogs import AIGatewayDialog
from autosub_studio.ui.panels import SettingsPanel, TranslatePanel


class FakeResponse:
    def __init__(self, data: dict[str, Any] | str, status: int = 200) -> None:
        self.status = status
        if isinstance(data, dict):
            self._raw = json.dumps(data).encode("utf-8")
        else:
            self._raw = data.encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# =========================================================================== 1. Secrets


class TestSecretManagement:
    def test_secret_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("autosub_studio.services.settings.config_dir", lambda: tmp_path)
        monkeypatch.setattr("autosub_studio.services.paths.config_dir", lambda: tmp_path)

        secret_value = "sk-super-secret-key-12345"
        Settings.set_secret("ai_gateway_key", secret_value)

        loaded = Settings.get_secret("ai_gateway_key")
        assert loaded == secret_value

        # Luu cau hinh chung va dam bao khoa khong bao gio nam trong config.json
        settings = Settings()
        settings.ai_endpoint = "http://localhost:8000/v1"
        settings.save()

        cfg_text = (tmp_path / "config.json").read_text(encoding="utf-8")
        assert secret_value not in cfg_text
        assert "ai_gateway_key" not in cfg_text

        # Kiem tra tep secrets.dat duoc ma hoa
        secrets_file = tmp_path / "secrets.dat"
        assert secrets_file.is_file()
        raw_bytes = secrets_file.read_bytes()
        assert raw_bytes.startswith(b"DPAPI") or raw_bytes.startswith(b"B64__")

    def test_legacy_claude_secret_migration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("autosub_studio.services.settings.config_dir", lambda: tmp_path)
        monkeypatch.setattr("autosub_studio.services.paths.config_dir", lambda: tmp_path)

        Settings.set_secret("claude_api_key", "sk-legacy-claude-key")
        assert Settings.get_secret("ai_gateway_key") == ""

        # Load settings se tu dong chuyen claude_api_key sang ai_gateway_key
        loaded_settings = Settings.load()
        assert loaded_settings is not None
        assert Settings.get_secret("ai_gateway_key") == "sk-legacy-claude-key"
        assert Settings.get_secret("claude_api_key") == ""


# =========================================================================== 2. Endpoints & Payload


class TestEndpointAndPayload:
    def test_endpoint_normalization(self) -> None:
        assert normalize_chat_endpoint("http://localhost:8000") == "http://localhost:8000/v1/chat/completions"
        assert normalize_chat_endpoint("http://localhost:8000/") == "http://localhost:8000/v1/chat/completions"
        assert normalize_chat_endpoint("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"
        assert normalize_chat_endpoint("http://localhost:8000/v1/") == "http://localhost:8000/v1/chat/completions"
        assert (
            normalize_chat_endpoint("http://localhost:8000/v1/chat/completions")
            == "http://localhost:8000/v1/chat/completions"
        )
        assert (
            normalize_chat_endpoint("http://localhost:8000/chat/completions")
            == "http://localhost:8000/chat/completions"
        )
        assert normalize_chat_endpoint("") == ""

        assert normalize_models_endpoint("http://localhost:8000/v1") == "http://localhost:8000/v1/models"
        assert normalize_models_endpoint("http://localhost:8000") == "http://localhost:8000/v1/models"

    def test_resolve_model_alias(self) -> None:
        settings = Settings(
            ai_model_sub="custom-sub-v1",
            ai_thinking_sub="low",
            ai_model_prime="custom-prime-v2",
            ai_thinking_prime="high",
        )
        m_sub, t_sub = resolve_model("sub", settings)
        assert m_sub == "custom-sub-v1"
        assert t_sub == "low"

        m_prime, t_prime = resolve_model("prime", settings)
        assert m_prime == "custom-prime-v2"
        assert t_prime == "high"

        # Default fallback
        m_def, t_def = resolve_model("unknown", settings)
        assert m_def == "custom-sub-v1"
        assert t_def == "low"

    def test_chat_completion_payload_and_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_requests: list[urllib.request.Request] = []

        def mock_urlopen(req: urllib.request.Request, timeout: float = 60.0):
            captured_requests.append(req)
            assert timeout == 45.0
            return FakeResponse({
                "choices": [{"message": {"content": "Test response"}}]
            })

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        reply = chat_completion(
            "http://localhost:8000/v1",
            "sk-test-key",
            model="my-reasoning-model",
            messages=[{"role": "user", "content": "Hello"}],
            thinking="medium",
            timeout=45.0,
        )
        assert reply == "Test response"
        assert len(captured_requests) == 1
        req = captured_requests[0]

        assert req.full_url == "http://localhost:8000/v1/chat/completions"
        assert req.get_header("Authorization") == "Bearer sk-test-key"
        assert req.get_header("Content-type") == "application/json"

        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "my-reasoning-model"
        assert body["reasoning_effort"] == "medium"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_chat_completion_no_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_requests: list[urllib.request.Request] = []

        def mock_urlopen(req: urllib.request.Request, timeout: float = 60.0):
            captured_requests.append(req)
            return FakeResponse({
                "choices": [{"message": {"content": "ok"}}]
            })

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        chat_completion(
            "http://localhost:8000",
            "",
            model="fast-model",
            messages=[{"role": "user", "content": "hi"}],
            thinking="none",
        )
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.get_header("Authorization") is None

        body = json.loads(req.data.decode("utf-8"))
        assert "reasoning_effort" not in body


# ========================================================== 3. Connection & Model Checks


class TestConnectionAndModelChecks:
    def test_connection_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            return FakeResponse({"data": [{"id": "sub"}]})

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = ai_gateway.test_connection("http://localhost:8000/v1", "key123")
        assert ok is True
        assert "thành công" in msg.lower()

    def test_connection_fallback_to_ping_when_no_models_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            nonlocal call_count
            call_count += 1
            if "/models" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b"{}"))
            return FakeResponse({"choices": [{"message": {"content": "pong"}}]})

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = ai_gateway.test_connection("http://localhost:8000", "key123")
        assert ok is True
        assert "thành công" in msg.lower()
        assert call_count == 2

    def test_connection_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            err_body = io.BytesIO(b'{"error": "Invalid API key"}')
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, err_body)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = ai_gateway.test_connection("http://localhost:8000", "bad_key")
        assert ok is False
        assert "401" in msg

    def test_model_check_real_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            body = json.loads(req.data.decode("utf-8"))
            assert body["model"] == "my-sub-model"
            assert body["reasoning_effort"] == "low"
            return FakeResponse({"choices": [{"message": {"content": "OK"}}]})

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = ai_gateway.test_model(
            "http://localhost:8000", "k", model="my-sub-model", thinking="low"
        )
        assert ok is True
        assert "hoạt động tốt" in msg
        assert "OK" in msg


# =============================================== 4. B2 Translation (Count mismatch & Timestamps)


class TestTranslationB2:
    def test_translation_count_mismatch_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings(ai_endpoint="http://localhost:8000/v1")
        monkeypatch.setattr(Settings, "load", lambda: settings)
        monkeypatch.setattr(Settings, "get_secret", lambda name: "valid-key")

        def mock_chat_completion(*args, **kwargs):
            # Trả về 2 câu trong khi đầu vào có 3 câu
            return json.dumps({
                "translations": [
                    {"id": 0, "text": "Câu một"},
                    {"id": 1, "text": "Câu hai"},
                ]
            })

        monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

        req = translate.TranslationRequest(texts=["One", "Two", "Three"])
        with pytest.raises(translate.TranslationError) as excinfo:
            translate.translate_batch("Server AI API", req)

        assert "không khớp" in str(excinfo.value)

    def test_b2_timestamps_strictly_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_folder = tmp_path / "proj"
        project_folder.mkdir(parents=True)
        store = ProjectStore(tmp_path / "work")
        project = store.create(str(project_folder), "test_proj")

        c1 = Cue(start=1.234, end=3.456, text="Hello world")
        c2 = Cue(start=4.000, end=7.890, text="Good morning")
        project.doc.cues = [c1, c2]
        store.save(project)

        settings = Settings(
            translate_provider="Server AI API",
            ai_endpoint="http://localhost:8000/v1",
            llm_model="sub",
        )
        monkeypatch.setattr(Settings, "load", lambda: settings)
        monkeypatch.setattr(Settings, "get_secret", lambda name: "test-key")

        def mock_chat_completion(*args, **kwargs):
            return json.dumps({
                "translations": [
                    {"id": 0, "text": "Xin chào thế giới"},
                    {"id": 1, "text": "Chào buổi sáng"},
                ]
            })

        monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

        pc = P.PipelineContext(
            ff=FFmpeg(),
            settings=settings,
            store=store,
            project=project,
            task=MagicMock(),
            api_key="test-key",
            glossary={},
        )

        res = P.step_translate(pc)
        assert "Da dich 2 cau" in res

        assert len(project.doc.cues) == 2
        cue0, cue1 = project.doc.cues[0], project.doc.cues[1]

        # Timestamp phai giu nguyen tuyet doi
        assert cue0.start == 1.234
        assert cue0.end == 3.456
        assert cue0.text == "Hello world"
        assert cue0.translation == "Xin chào thế giới"

        assert cue1.start == 4.000
        assert cue1.end == 7.890
        assert cue1.text == "Good morning"
        assert cue1.translation == "Chào buổi sáng"


# =============================================== 5. OCR AI Keyframe Dedupe & Cache


class TestOcrAiDedupeAndCache:
    def test_keyframe_dedupe_and_sqlite_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(ai_endpoint="http://localhost:8000/v1", ocr_ai_model="sub")
        monkeypatch.setattr(Settings, "load", lambda: settings)
        monkeypatch.setattr(Settings, "get_secret", lambda name: "test-key")

        # Tao 4 anh test
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Frame 0 & 1: text trang tren nen den
        img1 = Image.new("RGB", (120, 40), color=(0, 0, 0))
        for x in range(20, 80):
            for y in range(15, 25):
                img1.putpixel((x, y), (255, 255, 255))
        f0 = frames_dir / "f0.jpg"
        f1 = frames_dir / "f1.jpg"
        img1.save(f0)
        img1.save(f1)  # f1 giong het f0 -> dedupe keyframe cuc bo

        # Frame 2: text khac
        img2 = Image.new("RGB", (120, 40), color=(0, 0, 0))
        for x in range(10, 40):
            for y in range(5, 35):
                img2.putpixel((x, y), (255, 255, 255))
        f2 = frames_dir / "f2.jpg"
        img2.save(f2)

        # Frame 3: anh den hoan toan (blank)
        img3 = Image.new("RGB", (120, 40), color=(0, 0, 0))
        f3 = frames_dir / "f3.jpg"
        img3.save(f3)

        ai_calls: list[bytes] = []

        def mock_ocr_image(endpoint, api_key, *, model, image_data, thinking="", timeout=60.0):
            ai_calls.append(bytes(image_data))
            if len(ai_calls) == 1:
                return "HELLO WORLD"
            return "WELCOME EVERYONE"

        monkeypatch.setattr(ai_gateway, "ocr_image_with_ai", mock_ocr_image)

        cache_db = tmp_path / "ocr_cache.sqlite3"
        cues = ocr_ai.read_frames_ai(
            [f0, f1, f2, f3],
            fps=1.0,
            stamps=[0.0, 1.0, 2.0, 3.0],
            min_duration=0.2,
            cache_path=cache_db,
            cache_key="test_key_v1",
        )

        # AI chi duoc goi cho f0 va f2 (2 lan), f1 dedupe cuc bo, f3 la blank
        assert len(ai_calls) == 2

        # Kiem tra ket qua timeline merge
        assert len(cues) >= 2
        assert cues[0].text == "HELLO WORLD"
        assert cues[0].start == 0.0
        assert cues[0].end == 2.0  # f0 va f1 gop lai thanh 2 giay

        assert cues[1].text == "WELCOME EVERYONE"
        assert cues[1].start == 2.0
        assert cues[1].end == 3.0

        # Kiem tra SQLite cache da luu 4 frame
        assert cache_db.is_file()
        with sqlite3.connect(cache_db) as conn:
            query = "SELECT count(*) FROM frames WHERE cache_key='test_key_v1'"
            cnt = conn.execute(query).fetchone()[0]
            assert cnt == 4

        # Chay lai lan 2 voi cache -> khong goi AI lan nao nua
        ai_calls.clear()
        cues_cached = ocr_ai.read_frames_ai(
            [f0, f1, f2, f3],
            fps=1.0,
            stamps=[0.0, 1.0, 2.0, 3.0],
            min_duration=0.2,
            cache_path=cache_db,
            cache_key="test_key_v1",
        )
        assert len(ai_calls) == 0
        assert len(cues_cached) == len(cues)

    def test_ocr_ai_fails_clearly_when_api_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(ai_endpoint="http://localhost:8000/v1")
        monkeypatch.setattr(Settings, "load", lambda: settings)
        monkeypatch.setattr(Settings, "get_secret", lambda name: "test-key")

        # Anh co noi dung ro rang (khong bi coi la blank)
        img = Image.new("RGB", (100, 40), color=(0, 0, 0))
        for x in range(10, 90):
            for y in range(5, 35):
                img.putpixel((x, y), (255, 255, 255))
        f0 = tmp_path / "f0.jpg"
        img.save(f0)

        def mock_ocr_image(*args, **kwargs):
            raise AIGatewayError("Gateway connection timed out")

        monkeypatch.setattr(ai_gateway, "ocr_image_with_ai", mock_ocr_image)

        with pytest.raises(AIGatewayError) as excinfo:
            ocr_ai.read_frames_ai([f0], fps=1.0)
        assert "timed out" in str(excinfo.value)


# =========================================================================== 6. UI Dialog & Wiring


class TestUIDialogAndWiring:
    def test_ai_gateway_dialog_inputs_and_save(
        self, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("autosub_studio.services.settings.config_dir", lambda: tmp_path)
        monkeypatch.setattr("autosub_studio.services.paths.config_dir", lambda: tmp_path)

        settings = Settings()
        settings.ai_endpoint = "http://initial.host/v1"
        settings.ai_model_sub = "sub_orig"
        settings.ai_thinking_sub = "low"
        settings.ai_model_prime = "prime_orig"
        settings.ai_thinking_prime = "medium"
        Settings.set_secret("ai_gateway_key", "init-secret")

        dialog = AIGatewayDialog(settings)
        assert dialog.endpoint.text() == "http://initial.host/v1"
        assert dialog.api_key.text() == "init-secret"
        assert dialog.model_sub.text() == "sub_orig"
        assert dialog.thinking_sub.currentText() == "low"
        assert dialog.model_prime.text() == "prime_orig"
        assert dialog.thinking_prime.currentText() == "medium"

        # Doi gia tri va Save
        dialog.endpoint.setText("http://new.host:8080/v1")
        dialog.api_key.setText("new-secret-999")
        dialog.model_sub.setText("sub_v2")
        dialog.thinking_sub.setCurrentText("high")
        dialog.model_prime.setText("prime_v3")
        dialog.thinking_prime.setCurrentText("none")

        dialog._on_save()

        assert settings.ai_endpoint == "http://new.host:8080/v1"
        assert settings.ai_model_sub == "sub_v2"
        assert settings.ai_thinking_sub == "high"
        assert settings.ai_model_prime == "prime_v3"
        assert settings.ai_thinking_prime == "none"
        assert Settings.get_secret("ai_gateway_key") == "new-secret-999"

        dialog.deleteLater()

    def test_settings_panel_ai_gateway_button_and_no_claude_field(self, qapp) -> None:
        panel = SettingsPanel()
        assert hasattr(panel, "btn_ai_gateway")
        assert panel.btn_ai_gateway.isEnabled()
        assert not hasattr(panel, "api_key")

        opened = False

        def on_open():
            nonlocal opened
            opened = True

        panel.openAIGateway.connect(on_open)
        panel.btn_ai_gateway.click()
        assert opened is True

        panel.deleteLater()

    def test_translate_panel_options(self, qapp) -> None:
        panel = TranslatePanel()
        model_items = [panel.model.itemText(i) for i in range(panel.model.count())]
        assert "sub" in model_items
        assert "prime" in model_items
        assert not any("claude" in m.lower() for m in model_items)

        provider_items = [panel.provider.itemText(i) for i in range(panel.provider.count())]
        assert "Server AI API" in provider_items

        panel.deleteLater()
