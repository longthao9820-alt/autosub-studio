"""Kiem thu toan dien cho OCR AI Provider, taxonomy loi HTTP va chia doi dong."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, cast

import pytest

from autosub_studio.providers.ocr_ai_provider import (
    AI_OCR_PROMPT,
    AI_OCR_PROMPT_VERSION,
    AIOCRProvider,
    BatchItem,
    BatchResult,
    OCRValidationError,
    build_batch_messages,
    estimate_batch_payload_size,
    execute_batch_with_split,
    generate_marker_image,
    ocr_batch,
    test_vision,
    validate_ocr_response,
)
from autosub_studio.providers.ocr_selector import PreparedImage
from autosub_studio.services.ai_gateway import (
    AIGatewayAuthError,
    AIGatewayConnectionError,
    AIGatewayInvalidRequestError,
    AIGatewayPayloadTooLargeError,
    AIGatewayRateLimitError,
    AIGatewayServerError,
    AIGatewayTimeoutError,
    chat_completion,
)


class FakeHTTPResponse:
    def __init__(
        self,
        data: dict[str, Any] | str | bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        if isinstance(data, dict):
            self._raw = json.dumps(data).encode("utf-8")
        elif isinstance(data, str):
            self._raw = data.encode("utf-8")
        else:
            self._raw = data

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ========================================================== 1. Multimodal ID Binding
class TestMultimodalIDBinding:
    def test_messages_structure_and_id_binding(self) -> None:
        assert AI_OCR_PROMPT_VERSION == "v1"
        img1_bytes = b"fake_jpeg_1"
        img2_bytes = b"fake_jpeg_2"

        items = [
            BatchItem(id="seg_001", image=img1_bytes),
            BatchItem(id="seg_002", image=img2_bytes),
        ]

        messages = build_batch_messages(items)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)

        # First block is prompt
        assert content[0]["type"] == "text"
        prompt_text = content[0]["text"]
        assert "cropped subtitle strips" in prompt_text
        assert "transcribe only visible" in prompt_text.lower()
        assert "no translation" in prompt_text.lower()
        assert "punctuation" in prompt_text.lower()
        assert "uncertain" in prompt_text.lower()
        assert "structured json only" in prompt_text.lower()
        assert "every stable id" in prompt_text.lower()

        # Followed by items: each ID text immediately before image_url
        assert content[1] == {"type": "text", "text": "ID: seg_001"}
        assert content[2]["type"] == "image_url"
        assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        assert content[3] == {"type": "text", "text": "ID: seg_002"}
        assert content[4]["type"] == "image_url"
        assert content[4]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_batch_item_with_prepared_image(self) -> None:
        prep = PreparedImage(
            payload=b"prepared_jpeg_bytes",
            width=100,
            height=30,
            timestamp=1.23,
        )
        item = BatchItem(id="seg_prep", image=prep)
        assert item.get_image_bytes() == b"prepared_jpeg_bytes"

        messages = build_batch_messages([item])
        assert len(messages[0]["content"]) == 3
        assert messages[0]["content"][1]["text"] == "ID: seg_prep"

    def test_estimate_batch_payload_size(self) -> None:
        items = [
            BatchItem(id="seg_1", image=b"1234567890"),
            BatchItem(id="seg_2", image=b"abcdefghij"),
        ]
        size = estimate_batch_payload_size(items)
        assert size > 300
        assert size > len(AI_OCR_PROMPT)


# ========================================================== 2. Strict Validator Cases
class TestStrictValidator:
    def test_valid_results_dict_schema(self) -> None:
        raw = json.dumps({
            "results": [
                {"id": "id1", "text": "Line 1", "confidence": 0.95, "uncertain": False},
                {"id": "id2", "text": "Line 2", "confidence": 0.88, "uncertain": True},
            ]
        })
        res = validate_ocr_response(raw, ["id1", "id2"])
        assert len(res) == 2
        assert res[0].id == "id1"
        assert res[0].text == "Line 1"
        assert res[0].confidence == 0.95
        assert res[0].uncertain is False
        assert res[1].id == "id2"
        assert res[1].text == "Line 2"
        assert res[1].confidence == 0.88
        assert res[1].uncertain is True

    def test_valid_top_array_schema(self) -> None:
        raw = json.dumps([
            {"id": "id1", "text": "Top array text"}
        ])
        res = validate_ocr_response(raw, ["id1"])
        assert len(res) == 1
        assert res[0].id == "id1"
        assert res[0].text == "Top array text"
        assert res[0].confidence == 1.0

    def test_harmless_json_fences_normalized(self) -> None:
        inner = json.dumps({"results": [{"id": "id1", "text": "Fenced text"}]})
        fenced_1 = f"```json\n{inner}\n```"
        res1 = validate_ocr_response(fenced_1, ["id1"])
        assert res1[0].text == "Fenced text"

        fenced_2 = f"```\n{inner}\n```"
        res2 = validate_ocr_response(fenced_2, ["id1"])
        assert res2[0].text == "Fenced text"

    def test_empty_text_allowed(self) -> None:
        raw = json.dumps({"results": [{"id": "blank_id", "text": "", "uncertain": True}]})
        res = validate_ocr_response(raw, ["blank_id"])
        assert res[0].text == ""
        assert res[0].uncertain is True

    def test_confidence_bounded_and_optional(self) -> None:
        raw = json.dumps({
            "results": [
                {"id": "id1", "text": "A", "confidence": 0.85},
                {"id": "id2", "text": "B"},  # missing -> 1.0
                {"id": "id3", "text": "C", "confidence": 1.5},  # clamped to 1.0
                {"id": "id4", "text": "D", "confidence": -0.5},  # clamped to 0.0
            ]
        })
        res = validate_ocr_response(raw, ["id1", "id2", "id3", "id4"])
        assert res[0].confidence == 0.85
        assert res[1].confidence == 1.0
        assert res[2].confidence == 1.0
        assert res[3].confidence == 0.0

    def test_missing_ids_rejected(self) -> None:
        raw = json.dumps({"results": [{"id": "id1", "text": "Only id1"}]})
        with pytest.raises(OCRValidationError) as excinfo:
            validate_ocr_response(raw, ["id1", "id2"])
        assert "Thiếu các ID" in str(excinfo.value)
        assert "id2" in str(excinfo.value)

    def test_duplicate_ids_rejected(self) -> None:
        raw = json.dumps({
            "results": [
                {"id": "id1", "text": "First"},
                {"id": "id1", "text": "Second"},
            ]
        })
        with pytest.raises(OCRValidationError) as excinfo:
            validate_ocr_response(raw, ["id1"])
        assert "trùng lặp" in str(excinfo.value)

    def test_unexpected_ids_rejected(self) -> None:
        raw = json.dumps({
            "results": [
                {"id": "id1", "text": "A"},
                {"id": "id_extra", "text": "B"},
            ]
        })
        with pytest.raises(OCRValidationError) as excinfo:
            validate_ocr_response(raw, ["id1"])
        assert "không mong muốn" in str(excinfo.value)

    def test_non_string_text_rejected(self) -> None:
        raw = json.dumps({"results": [{"id": "id1", "text": 12345}]})
        with pytest.raises(OCRValidationError) as excinfo:
            validate_ocr_response(raw, ["id1"])
        assert "phải là chuỗi" in str(excinfo.value)

    def test_non_string_id_rejected(self) -> None:
        raw = json.dumps({"results": [{"id": 1, "text": "Valid text"}]})
        with pytest.raises(OCRValidationError) as excinfo:
            validate_ocr_response(raw, ["1"])
        assert "phải là chuỗi" in str(excinfo.value)

    def test_commentary_rejected(self) -> None:
        inner = json.dumps({"results": [{"id": "id1", "text": "Valid text"}]})
        with_commentary = f"Here is the OCR output:\n```json\n{inner}\n```\nHope it helps!"
        with pytest.raises(OCRValidationError):
            validate_ocr_response(with_commentary, ["id1"])

    def test_truncated_json_rejected(self) -> None:
        truncated = '{"results": [{"id": "id1", "text": "Hel'
        with pytest.raises(OCRValidationError) as excinfo:
            validate_ocr_response(truncated, ["id1"])
        assert "không đúng định dạng" in str(excinfo.value)

    def test_preserves_expected_id_order(self) -> None:
        raw = json.dumps({
            "results": [
                {"id": "id_b", "text": "B"},
                {"id": "id_a", "text": "A"},
                {"id": "id_c", "text": "C"},
            ]
        })
        res = validate_ocr_response(raw, ["id_a", "id_b", "id_c"])
        assert [r.id for r in res] == ["id_a", "id_b", "id_c"]


# ========================================================== 3. Live SSE Compatibility
class TestLiveSSECompatibility:
    def test_live_sse_stream_assembled_and_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sse_payload = (
            'data: {"choices":[{"delta":{"content":"{\\"results\\": ["}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"{\\"id\\": \\"seg_1\\", \\"text\\": '
            '\\"Sub 1\\"}"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"]}"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def mock_urlopen(req: urllib.request.Request, timeout: float = 60.0):
            return FakeHTTPResponse(
                sse_payload,
                headers={"Content-Type": "text/event-stream"},
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        provider = AIOCRProvider(
            endpoint="http://localhost:8000/v1",
            api_key="test-key",
            model="sub",
        )
        item = BatchItem(id="seg_1", image=b"img_data")
        results = provider.execute_batch([item], allow_split=False)
        assert len(results) == 1
        assert results[0].id == "seg_1"
        assert results[0].text == "Sub 1"


# ========================================================== 4. Dynamic Split Pure Method
class TestDynamicSplit:
    def test_batch_16_split_to_8_then_4(self) -> None:
        items = [BatchItem(id=f"item_{i:02d}", image=b"x") for i in range(16)]
        call_history: list[list[str]] = []

        def mock_call(sub_items: Sequence[BatchItem]) -> list[BatchResult]:
            ids = [it.id for it in sub_items]
            call_history.append(ids)

            # 16 fails with 413
            if len(sub_items) == 16:
                raise AIGatewayPayloadTooLargeError("Batch 16 too large", status_code=413)

            # First 8 (item 0..7) fails with 413
            if len(sub_items) == 8 and sub_items[0].id == "item_00":
                raise AIGatewayPayloadTooLargeError("Batch 8 too large", status_code=413)

            # Sub-batches of 4 or second 8 succeed
            return [BatchResult(id=it.id, text=f"Text {it.id}") for it in sub_items]

        results = execute_batch_with_split(items, mock_call)

        assert len(results) == 16
        assert [r.id for r in results] == [f"item_{i:02d}" for i in range(16)]

        # Call sizes: 16 -> 8 -> 4, 4 -> 8
        call_sizes = [len(c) for c in call_history]
        assert call_sizes == [16, 8, 4, 4, 8]

    def test_single_item_failure_raises(self) -> None:
        single_item = [BatchItem(id="single", image=b"huge")]

        def failing_call(_items: Sequence[BatchItem]) -> list[BatchResult]:
            raise AIGatewayPayloadTooLargeError("Single item too large", status_code=413)

        with pytest.raises(AIGatewayPayloadTooLargeError):
            execute_batch_with_split(single_item, failing_call)

    def test_malformed_output_raises_immediately_no_split(self) -> None:
        items = [BatchItem(id="i1", image=b"1"), BatchItem(id="i2", image=b"2")]
        call_count = 0

        def malformed_call(_items: Sequence[BatchItem]) -> list[BatchResult]:
            nonlocal call_count
            call_count += 1
            raise OCRValidationError("Malformed JSON output from model")

        with pytest.raises(OCRValidationError):
            execute_batch_with_split(items, malformed_call)

        assert call_count == 1  # No recursive split or retry on malformed output

    def test_ocr_batch_helper_function(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 60.0):
            reply = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "results": [{"id": "h1", "text": "Helper result"}]
                            }),
                        }
                    }
                ]
            }
            return FakeHTTPResponse(reply)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        items = [BatchItem(id="h1", image=b"test_img")]
        res = ocr_batch(
            items,
            endpoint="http://localhost:8000/v1",
            api_key="key",
            model="sub",
        )
        assert len(res) == 1
        assert res[0].text == "Helper result"


# ========================================================== 5. Test Vision Semantics
class TestVisionSemantics:
    def test_test_vision_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_requests: list[urllib.request.Request] = []

        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            captured_requests.append(req)
            raw_data = cast(bytes, req.data or b"")
            body = json.loads(raw_data.decode("utf-8"))
            # Ensure it is vision: user message has image_url block
            user_content = body["messages"][0]["content"]
            assert any(b.get("type") == "image_url" for b in user_content)

            reply = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "results": [
                                    {"id": "vision_test", "text": "TEST", "confidence": 0.99}
                                ]
                            }),
                        }
                    }
                ]
            }
            return FakeHTTPResponse(reply)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = test_vision(
            "http://localhost:8000/v1", "key", model="sub", expected_marker="TEST"
        )
        assert ok is True
        assert "hoạt động tốt" in msg
        assert len(captured_requests) == 1

    def test_test_vision_marker_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            reply = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "results": [
                                    {
                                        "id": "vision_test",
                                        "text": "UNEXPECTED_CONTENT",
                                        "confidence": 0.99,
                                    }
                                ]
                            }),
                        }
                    }
                ]
            }
            return FakeHTTPResponse(reply)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = test_vision(
            "http://localhost:8000/v1", "key", model="sub", expected_marker="TEST"
        )
        assert ok is False
        assert "không khớp ký tự mẫu" in msg

    def test_test_vision_error_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            err_body = io.BytesIO(b'{"error": "Bearer sk-super-secret-12345 failed"}')
            raise urllib.error.HTTPError(
                req.full_url, 401, "Unauthorized", cast(Any, {}), err_body
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        ok, msg = test_vision("http://localhost:8000/v1", "key", model="sub")
        assert ok is False
        assert "sk-super-secret" not in msg
        assert "sk-***" in msg or "Bearer ***" in msg

    def test_generate_marker_image(self) -> None:
        img_bytes = generate_marker_image("MARKER")
        assert len(img_bytes) > 0
        assert img_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes


# ========================================================== 6. HTTP Taxonomy & Sanitization
class TestHTTPTaxonomyAndSanitization:
    def test_taxonomy_status_and_classes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cases = [
            (401, b'{"error": "invalid_api_key"}', AIGatewayAuthError, False),
            (403, b'{"error": "forbidden"}', AIGatewayAuthError, False),
            (400, b'{"error": "bad_request"}', AIGatewayInvalidRequestError, False),
            (404, b'{"error": "model_not_found"}', AIGatewayInvalidRequestError, False),
            (413, b'{"error": "request_too_large"}', AIGatewayPayloadTooLargeError, True),
            (429, b'{"error": "rate_limited"}', AIGatewayRateLimitError, False),
            (500, b'{"error": "server_error"}', AIGatewayServerError, False),
            (503, b'{"error": "unavailable"}', AIGatewayServerError, False),
        ]

        for status, body, expected_cls, expected_split in cases:
            headers = {"Retry-After": "3.5"} if status == 429 else {}

            def mock_urlopen(
                req: urllib.request.Request,
                timeout: float = 60.0,
                s=status,
                b=body,
                h=headers,
            ):
                raise urllib.error.HTTPError(
                    req.full_url, s, "Error", cast(Any, h), io.BytesIO(b)
                )

            monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

            with pytest.raises(expected_cls) as excinfo:
                chat_completion(
                    "http://localhost:8000/v1",
                    "key",
                    model="sub",
                    messages=[{"role": "user", "content": "hi"}],
                )

            exc = excinfo.value
            assert exc.status_code == status
            assert exc.is_split_required == expected_split
            if status == 429:
                assert exc.retry_after == 3.5

    def test_400_context_length_classified_as_split_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 60.0):
            body = io.BytesIO(
                b'{"error": {"message": "context_length_exceeded: maximum context length is 8192"}}'
            )
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", cast(Any, {}), body
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        with pytest.raises(AIGatewayPayloadTooLargeError) as excinfo:
            chat_completion(
                "http://localhost:8000/v1",
                "key",
                model="sub",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert excinfo.value.is_split_required is True

    def test_timeout_and_connection_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_urlopen_timeout(req: urllib.request.Request, timeout: float = 60.0):
            raise TimeoutError("Socket timed out")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_timeout)
        with pytest.raises(AIGatewayTimeoutError) as exc_to:
            chat_completion("http://localhost:8000/v1", "key", model="sub", messages=[])
        assert exc_to.value.is_transient is True

        def mock_urlopen_conn(req: urllib.request.Request, timeout: float = 60.0):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_conn)
        with pytest.raises(AIGatewayConnectionError) as exc_conn:
            chat_completion("http://localhost:8000/v1", "key", model="sub", messages=[])
        assert exc_conn.value.is_transient is True

    def test_error_sanitizes_authorization_headers_and_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mock_urlopen(req: urllib.request.Request, timeout: float = 60.0):
            body = io.BytesIO(b'{"error": "Authorization: Bearer sk-1234567890abcdef is invalid"}')
            raise urllib.error.HTTPError(
                req.full_url, 401, "Unauthorized", cast(Any, {}), body
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        with pytest.raises(AIGatewayAuthError) as excinfo:
            chat_completion("http://localhost:8000/v1", "key", model="sub", messages=[])

        err_msg = str(excinfo.value)
        assert "sk-1234567890abcdef" not in err_msg
        assert "sk-***" in err_msg or "Bearer ***" in err_msg
