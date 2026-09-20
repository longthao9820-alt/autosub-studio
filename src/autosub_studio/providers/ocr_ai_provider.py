"""Provider doc chu tren hinh bang AI Vision (OpenAI-compatible batch vision request).

Invariants:
- Ho tro yeu cau vision theo me (batch) chuan OpenAI.
- AI_OCR_PROMPT_VERSION va prompt chat che: cropped subtitle strips, transcribe only visible
  original language, no translation/description/invention, punctuation/line content,
  uncertain => empty/uncertain, structured JSON only, every stable ID.
- Nhieu anh trong 1 user message voi ID text ngay truoc image_url.
- Phan hoi mong doi: {"results": [{"id", "text", "confidence", "uncertain?"}]} hoac top array.
- Bo kiem tra nghiem ngat (strict validator): ID khop chinh xac tap hop mong doi,
  tu choi ID thieu/thua/trung lap, non-string, malformed, commentary, truncated.
  Chi normalize harmless surrounding ```json fences. Cho phep text rong.
- Dataclasses BatchItem va BatchResult.
- test_vision gui anh thu nho tao san va xac minh marker nhan dien duoc, khong ping.
- Taxonomy loi HTTP tu ai_gateway: giu status_code/code/retry_after, phan biet auth,
  model/format, 413 split-required, 429 rate, 5xx, timeout/connection.
- Chia doi dong (dynamic split pure method) khi gap 413/context/request-too-large;
  single failure raises; malformed output khong chia va khong retry o day (scheduler quan ly).
- Uoc tinh kich thuoc payload request.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..services import ai_gateway

AI_OCR_PROMPT_VERSION: str = "v1"

AI_OCR_PROMPT: str = (
    "You are an expert video subtitle OCR engine. "
    "The input images are cropped subtitle strips. "
    "Transcribe only visible text in its original language. "
    "No translation, no description, no invention. "
    "Preserve visible punctuation and line content. "
    "If text is uncertain, blurry, or missing: mark uncertain as true or return empty text "
    "(uncertain => empty/uncertain). "
    "Respond with structured JSON only: "
    '{"results": [{"id": "...", "text": "...", "confidence": 1.0, "uncertain": false}]}. '
    "You must include every stable ID from the input images."
)


@dataclass
class BatchItem:
    """Mot khung hinh kem dinh danh de nhan dang trong me (batch)."""

    id: str
    image: bytes | Path | str | Any
    mime_type: str = "image/jpeg"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_image_bytes(self) -> bytes:
        """Lay noi dung bytes cua hinh anh tu nhieu kieu nguon khac nhau."""
        if hasattr(self.image, "payload"):
            return bytes(self.image.payload)
        if hasattr(self.image, "data"):
            return bytes(self.image.data)
        if isinstance(self.image, bytes):
            return self.image
        if isinstance(self.image, (str, Path)):
            return Path(self.image).read_bytes()
        raise TypeError(f"Không thể trích xuất bytes từ kiểu {type(self.image)}")


@dataclass
class BatchResult:
    """Ket qua nhan dang chu cho mot khung hinh trong me."""

    id: str
    text: str
    confidence: float = 1.0
    uncertain: bool = False
    raw: dict[str, Any] | None = None


class OCRValidationError(ai_gateway.AIGatewayError, ValueError):
    """Loi khi phan hoi OCR tu AI khong dung dinh dang hoac thieu/thua ID."""


def _clean_json_fences(raw_text: str) -> str:
    """Normalize harmless surrounding ```json fences only."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        first_line = lines[0].strip()
        if first_line.startswith("```"):
            fence_tag = first_line[3:].strip().lower()
            if fence_tag in ("", "json"):
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                    text = "\n".join(lines).strip()
    return text


def validate_ocr_response(
    raw_text: str,
    expected_ids: Sequence[str],
) -> list[BatchResult]:
    """Kiem tra nghiem ngat phan hoi OCR tu AI Gateway.

    Quy tac:
    - Bo qua cac fence ```json hoac ``` bao quanh (harmless surrounding fences only).
    - Tu choi commentary, chuoi khong phai JSON, du lieu bi cat cup (truncated).
    - Chap nhan format {"results": [...]} hoac top array [...].
    - IDs: phai khop chinh xac tap hop expected_ids (khong thieu, khong thua, khong trung lap).
    - text: bat buoc la chuoi, cho phep chuoi rong ("").
    - confidence: khong bat buoc, gioi han trong khoang [0.0, 1.0].
    - uncertain: khong bat buoc, la boolean.
    """
    if not raw_text or not raw_text.strip():
        raise OCRValidationError("Phản hồi OCR từ AI Gateway rỗng.")

    cleaned = _clean_json_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OCRValidationError(
            f"Phản hồi OCR không đúng định dạng JSON hoặc bị cắt cụt: {exc}"
        ) from exc

    items_list: list[Any]
    if isinstance(data, dict):
        if "results" not in data or not isinstance(data["results"], list):
            raise OCRValidationError("Phản hồi JSON thiếu trường danh sách 'results'.")
        items_list = data["results"]
    elif isinstance(data, list):
        items_list = data
    else:
        raise OCRValidationError(
            "Phản hồi JSON phải là object chứa 'results' hoặc top-level array."
        )

    expected_id_set = set(expected_ids)
    if len(expected_id_set) != len(expected_ids):
        raise ValueError("Danh sách expected_ids đầu vào có ID trùng lặp.")

    seen_ids: set[str] = set()
    results_by_id: dict[str, BatchResult] = {}

    for idx, item in enumerate(items_list):
        if not isinstance(item, dict):
            raise OCRValidationError(f"Mục thứ {idx} trong kết quả không phải là JSON object.")

        if "id" not in item:
            raise OCRValidationError(f"Mục thứ {idx} thiếu trường 'id'.")

        raw_id = item["id"]
        if not isinstance(raw_id, str):
            raise OCRValidationError(
                f"Trường 'id' của mục thứ {idx} phải là chuỗi (nhận được {type(raw_id).__name__})."
            )

        item_id = raw_id.strip()
        if not item_id:
            raise OCRValidationError(f"Trường 'id' của mục thứ {idx} rỗng.")

        if item_id not in expected_id_set:
            raise OCRValidationError(
                f"Phát hiện ID không mong muốn trong kết quả: '{item_id}'."
            )

        if item_id in seen_ids:
            raise OCRValidationError(
                f"Phát hiện ID bị trùng lặp trong kết quả: '{item_id}'."
            )
        seen_ids.add(item_id)

        if "text" not in item:
            raise OCRValidationError(f"Mục có ID '{item_id}' thiếu trường 'text'.")

        text_val = item["text"]
        if not isinstance(text_val, str):
            val_type = type(text_val).__name__
            raise OCRValidationError(
                f"Trường 'text' của ID '{item_id}' phải là chuỗi (nhận được {val_type})."
            )

        conf_val = item.get("confidence")
        confidence = 1.0
        if conf_val is not None:
            if isinstance(conf_val, (int, float)):
                confidence = max(0.0, min(1.0, float(conf_val)))
            else:
                raise OCRValidationError(
                    f"Trường 'confidence' của ID '{item_id}' không hợp lệ: {conf_val}"
                )

        unc_val = item.get("uncertain")
        uncertain = False
        if unc_val is not None:
            if isinstance(unc_val, bool):
                uncertain = unc_val
            elif isinstance(unc_val, (int, float)):
                uncertain = bool(unc_val)
            elif isinstance(unc_val, str):
                uncertain = unc_val.lower() in ("true", "1", "yes")

        results_by_id[item_id] = BatchResult(
            id=item_id,
            text=text_val,
            confidence=confidence,
            uncertain=uncertain,
            raw=item,
        )

    missing_ids = expected_id_set - seen_ids
    if missing_ids:
        missing_sorted = sorted(missing_ids)
        raise OCRValidationError(f"Thiếu các ID trong phản hồi OCR: {missing_sorted}")

    return [results_by_id[eid] for eid in expected_ids]


def build_batch_messages(
    items: Sequence[BatchItem],
    prompt: str = "",
) -> list[dict[str, Any]]:
    """Xay dung messages cho mot batch vision request theo chuan OpenAI.

    Dac diem:
    - Multiple images in one user message
    - ID text immediately before image_url
    """
    effective_prompt = (prompt or "").strip() or AI_OCR_PROMPT
    user_content: list[dict[str, Any]] = [{"type": "text", "text": effective_prompt}]

    for item in items:
        user_content.append({"type": "text", "text": f"ID: {item.id}"})
        b64_url = ai_gateway.image_to_base64_url(item.get_image_bytes(), mime=item.mime_type)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": b64_url},
        })

    return [
        {"role": "user", "content": user_content},
    ]


def estimate_batch_payload_size(
    items: Sequence[BatchItem],
    prompt: str = "",
) -> int:
    """Uoc tinh kich thuoc payload JSON request tinh bang bytes."""
    effective_prompt = (prompt or "").strip() or AI_OCR_PROMPT
    size = len(effective_prompt.encode("utf-8")) + 300
    for item in items:
        img_len = len(item.get_image_bytes())
        b64_size = ((img_len + 2) // 3) * 4 + 30
        id_size = len(item.id.encode("utf-8")) + 120
        size += b64_size + id_size
    return size


def is_split_required_error(exc: Exception) -> bool:
    """Kiem tra xem loi co phai do request qua lon (413 hoac context length) can split khong."""
    if isinstance(exc, ai_gateway.AIGatewayPayloadTooLargeError):
        return True
    if getattr(exc, "is_split_required", False):
        return True
    if getattr(exc, "status_code", None) == 413:
        return True

    msg = str(exc).lower()
    code = str(getattr(exc, "code", "") or "").lower()
    combined = f"{msg} {code}"
    terms = (
        "413",
        "payload_too_large",
        "payload too large",
        "request too large",
        "request entity too large",
        "context_length_exceeded",
        "context length",
        "maximum context length",
        "prompt is too long",
        "too many tokens",
    )
    return any(t in combined for t in terms)


def execute_batch_with_split(
    items: Sequence[BatchItem],
    call_fn: Callable[[Sequence[BatchItem]], list[BatchResult]],
) -> list[BatchResult]:
    """Thuc thi me voi co che chia doi dong (dynamic split pure method).

    Quy tac:
    - Neu call_fn thanh cong: tra ve ket qua.
    - Neu loi 413 / context length / request too large:
      - Neu me chi co 1 item (single failure): raise loi ngay, khong the chia tiep.
      - Neu me > 1 items: chia doi de quy (left, right), gop ket qua.
    - Neu loi bat ky khac (auth, 400, 429, 5xx, timeout, malformed output/validation):
      - Raise ngay lap tuc, khong thu lai o day (scheduler se quan ly retry policy sau).
    """
    if not items:
        return []

    try:
        return call_fn(items)
    except Exception as exc:
        if is_split_required_error(exc):
            if len(items) <= 1:
                raise
            mid = len(items) // 2
            left_items = items[:mid]
            right_items = items[mid:]
            results_left = execute_batch_with_split(left_items, call_fn)
            results_right = execute_batch_with_split(right_items, call_fn)
            return results_left + results_right

        raise


def generate_marker_image(marker: str = "TEST", width: int = 120, height: int = 40) -> bytes:
    """Tao mot anh thu nho chua ky tu mau ro rang de kiem tra vision."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 12), marker, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_vision(
    endpoint: str,
    api_key: str,
    *,
    model: str,
    thinking: str = "",
    timeout: float = 15.0,
    expected_marker: str = "TEST",
) -> tuple[bool, str]:
    """Kiem tra kha nang doc anh vision cua model bang cach gui anh that va xac thuc marker.

    Khong dung text ping ma thuc su gui mot anh mau nho chua chuoi marker de kiem tra.
    """
    if not model.strip():
        return False, "Chưa chỉ định tên model."
    if not endpoint.strip():
        return False, "Chưa nhập Endpoint."

    marker = expected_marker.strip() or "TEST"
    try:
        img_bytes = generate_marker_image(marker)
        item = BatchItem(id="vision_test", image=img_bytes)
        messages = build_batch_messages([item])

        raw_reply = ai_gateway.chat_completion(
            endpoint,
            api_key,
            model=model,
            messages=messages,
            thinking=thinking,
            timeout=timeout,
        )
        results = validate_ocr_response(raw_reply, ["vision_test"])
        recognized_text = results[0].text.strip()
        if marker.upper() in recognized_text.upper():
            return True, f"Vision model '{model}' hoạt động tốt. Nhận diện: '{recognized_text}'."
        return (
            False,
            f"Vision model '{model}' phản hồi nhưng không khớp ký tự mẫu '{marker}' "
            f"(nhận diện: '{recognized_text}').",
        )
    except Exception as exc:
        sanitized = ai_gateway._sanitize_error_text(str(exc))
        return False, f"Lỗi kiểm tra Vision model '{model}': {sanitized}"


test_vision.__test__ = False  # type: ignore[attr-defined]


class AIOCRProvider:
    """Provider xu ly OCR AI su dung OpenAI Vision Chat Completions."""

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        *,
        model: str = "sub",
        thinking: str = "",
        timeout: float = ai_gateway.DEFAULT_TIMEOUT,
        prompt: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.thinking = thinking
        self.timeout = timeout
        self.prompt = prompt

    def execute_batch(
        self,
        items: Sequence[BatchItem],
        *,
        allow_split: bool = True,
        custom_prompt: str = "",
    ) -> list[BatchResult]:
        """Thuc thi OCR cho mot danh sach BatchItem."""
        if not items:
            return []

        prompt_to_use = custom_prompt or self.prompt

        def _call_single_batch(sub_items: Sequence[BatchItem]) -> list[BatchResult]:
            messages = build_batch_messages(sub_items, prompt=prompt_to_use)
            raw_text = ai_gateway.chat_completion(
                self.endpoint,
                self.api_key,
                model=self.model,
                messages=messages,
                thinking=self.thinking,
                timeout=self.timeout,
            )
            expected_ids = [item.id for item in sub_items]
            return validate_ocr_response(raw_text, expected_ids)

        if allow_split:
            return execute_batch_with_split(items, _call_single_batch)
        return _call_single_batch(items)

    def test_vision(
        self,
        expected_marker: str = "TEST",
        timeout: float = 15.0,
    ) -> tuple[bool, str]:
        """Kiem tra kha nang vision cua provider."""
        return test_vision(
            self.endpoint,
            self.api_key,
            model=self.model,
            thinking=self.thinking,
            timeout=timeout,
            expected_marker=expected_marker,
        )


def ocr_batch(
    items: Sequence[BatchItem],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    thinking: str = "",
    timeout: float = ai_gateway.DEFAULT_TIMEOUT,
    prompt: str = "",
    allow_split: bool = True,
) -> list[BatchResult]:
    """Ham tien ich goi OCR me bang AI Vision."""
    provider = AIOCRProvider(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        thinking=thinking,
        timeout=timeout,
        prompt=prompt,
    )
    return provider.execute_batch(items, allow_split=allow_split)
