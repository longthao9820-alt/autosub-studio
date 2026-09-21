"""Cong giao tiep AI Gateway duy nhat cua ung dung (chuan OpenAI HTTP bang stdlib)."""

from __future__ import annotations

import base64
import contextlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .settings import Settings

ROLE_SUBTITLE_EXTRACTION = "sub"
ROLE_SUBTITLE_TRANSLATION = "prime"


@dataclass(frozen=True)
class GatewayCapabilities:
    """Nhung kieu tac vu ma adapter OpenAI-compatible hien tai thuc su ho tro.

    Day la contract cua adapter, khong phai suy doan ve nha cung cap dung sau
    Gateway. Khi Gateway co endpoint media chinh thuc, adapter co the mo rong
    contract nay ma khong lam ro ri ten vendor vao pipeline nghiep vu.
    """

    text: bool = True
    structured_text: bool = True
    image: bool = True
    multi_image: bool = True
    video: bool = False
    media_upload: bool = False


def capabilities() -> GatewayCapabilities:
    """Tra ve kha nang cua adapter dang duoc cai dat, khong tu bia endpoint."""
    return GatewayCapabilities()


def role_for_task(task: str) -> str:
    """Anh xa tac vu cua AutoSub sang role logic do Gateway quan ly."""
    clean = (task or "").strip().casefold()
    if clean in {"translation", "subtitle_translation", "translate", "prime"}:
        return ROLE_SUBTITLE_TRANSLATION
    if clean in {"extraction", "subtitle_extraction", "ocr", "vision", "sub"}:
        return ROLE_SUBTITLE_EXTRACTION
    raise ValueError(f"Không nhận diện được loại tác vụ AI Gateway: {task}")


DEFAULT_TIMEOUT = 60.0
TEST_TIMEOUT = 15.0


class AIGatewayError(RuntimeError):
    """Loi khi goi hoac ket noi toi AI Gateway."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        retry_after: float | None = None,
        is_transient: bool = False,
        is_split_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after
        self.is_transient = is_transient
        self.is_split_required = is_split_required


class AIGatewayAuthError(AIGatewayError):
    """Loi xac thuc (401, 403) - khong thu lai."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 401,
        code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            retry_after=retry_after,
            is_transient=False,
            is_split_required=False,
        )


class AIGatewayInvalidRequestError(AIGatewayError):
    """Loi yeu cau / model / dinh dang khong hop le (400, 404) - khong thu lai."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            retry_after=retry_after,
            is_transient=False,
            is_split_required=False,
        )


class AIGatewayPayloadTooLargeError(AIGatewayError):
    """Loi yeu cau qua lon (413 hoac vuot context length) - can chia nho batch (split-required)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 413,
        code: str | None = "payload_too_large",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            retry_after=retry_after,
            is_transient=False,
            is_split_required=True,
        )


class AIGatewayRateLimitError(AIGatewayError):
    """Loi vuot qua gioi han toc do (429) - co the thu lai."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        code: str | None = "rate_limit_exceeded",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            retry_after=retry_after,
            is_transient=True,
            is_split_required=False,
        )


class AIGatewayServerError(AIGatewayError):
    """Loi may chu AI Gateway (5xx) - loi tam thoi (transient)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            retry_after=retry_after,
            is_transient=True,
            is_split_required=False,
        )


class AIGatewayTimeoutError(AIGatewayError):
    """Loi het thoi gian cho (timeout) - loi tam thoi (transient)."""

    def __init__(
        self,
        message: str = "Hết thời gian chờ phản hồi từ AI Gateway (timeout).",
        *,
        code: str | None = "timeout",
    ) -> None:
        super().__init__(
            message,
            status_code=None,
            code=code,
            retry_after=None,
            is_transient=True,
            is_split_required=False,
        )


class AIGatewayConnectionError(AIGatewayError):
    """Loi ket noi mang (URLError / connection) - loi tam thoi (transient)."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "connection_failed",
    ) -> None:
        super().__init__(
            message,
            status_code=None,
            code=code,
            retry_after=None,
            is_transient=True,
            is_split_required=False,
        )


def normalize_chat_endpoint(endpoint: str) -> str:
    """Chuan hoa endpoint ve URL day du /chat/completions.

    Xu ly cac dang:
    - https://api.openai.com
    - https://api.openai.com/
    - https://api.openai.com/v1
    - https://api.openai.com/v1/
    - https://api.openai.com/v1/chat/completions
    - http://localhost:8000/chat/completions
    - http://localhost:8000/models
    - http://localhost:8000/v1/models
    """
    ep = (endpoint or "").strip().rstrip("/")
    if not ep:
        return ""
    if ep.endswith("/models"):
        ep = ep[:-7].rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep
    if ep.endswith("/v1"):
        return f"{ep}/chat/completions"
    return f"{ep}/v1/chat/completions"


def normalize_models_endpoint(endpoint: str) -> str:
    """Chuan hoa endpoint ve URL /models de kiem tra danh sach hoac ket noi."""
    chat_ep = normalize_chat_endpoint(endpoint)
    if not chat_ep:
        return ""
    return chat_ep.replace("/chat/completions", "/models")


def resolve_model(alias: str, settings: Settings) -> tuple[str, str]:
    """Chuyen alias 'sub' hoac 'prime' thanh ten model that va muc do suy nghi (thinking)."""
    clean = (alias or "").strip().lower()
    if clean == "prime":
        model_name = settings.ai_model_prime.strip() or "prime"
        thinking = settings.ai_thinking_prime.strip() or "medium"
        return model_name, thinking
    model_name = settings.ai_model_sub.strip() or "sub"
    thinking = settings.ai_thinking_sub.strip() or "low"
    return model_name, thinking


def _sanitize_error_text(text: str, max_len: int = 120) -> str:
    """Khu cac thong tin nhay cam va gioi han do dai cua chuoi loi."""
    if not text:
        return ""
    sanitized = re.sub(
        r"(?i)(authorization|api-key|x-api-key)\s*[:=]\s*[^\s,;'\"]+", r"\1: ***", text
    )
    sanitized = re.sub(r"(?i)bearer\s+[^\s'\"]+", "Bearer ***", sanitized)
    sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{6,}", "sk-***", sanitized)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) > max_len:
        return sanitized[:max_len] + "..."
    return sanitized


def _sanitize_content_type(content_type: str) -> str:
    """Chuan hoa Content-Type header."""
    if not content_type:
        return ""
    ct = content_type.split(";")[0].strip().lower()
    if re.match(r"^[a-z0-9_.-]+/[a-z0-9_.+-]+$", ct):
        return ct
    return ct[:50]


def _extract_text_blocks(content: Any) -> str:
    """Trich xuat chu tu content (chuoi hoac danh sach khoi text / output_text)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if txt is None:
                    txt = item.get("output_text")
                if txt is not None:
                    parts.append(str(txt))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _raise_classified_error(
    status_code: int | None,
    msg: str,
    *,
    err_code: str | None = None,
    retry_after: float | None = None,
    cause: Exception | None = None,
) -> None:
    """Phan loai va nem loi AIGatewayError phu hop theo status code va ma loi."""
    sanitized_msg = _sanitize_error_text(str(msg))
    prefix = (
        f"AI Gateway báo lỗi {status_code}: "
        if status_code is not None
        else "AI Gateway báo lỗi: "
    )
    clean_err_str = f"{prefix}{sanitized_msg}"

    lower_combined = f"{sanitized_msg} {err_code or ''}".lower()
    is_context_too_large = any(
        term in lower_combined
        for term in (
            "context_length_exceeded",
            "context length",
            "maximum context length",
            "request too large",
            "request entity too large",
            "payload too large",
            "prompt is too long",
            "too many tokens",
        )
    )

    err: AIGatewayError
    if status_code == 413 or is_context_too_large:
        err = AIGatewayPayloadTooLargeError(
            clean_err_str,
            status_code=status_code or 413,
            code=err_code or "payload_too_large",
            retry_after=retry_after,
        )
    elif status_code in (401, 403) or (err_code and "auth" in err_code.lower()):
        err = AIGatewayAuthError(
            clean_err_str,
            status_code=status_code or 401,
            code=err_code,
            retry_after=retry_after,
        )
    elif status_code == 429 or (err_code and "rate_limit" in err_code.lower()):
        err = AIGatewayRateLimitError(
            clean_err_str,
            status_code=status_code or 429,
            code=err_code or "rate_limit_exceeded",
            retry_after=retry_after,
        )
    elif status_code in (400, 404):
        err = AIGatewayInvalidRequestError(
            clean_err_str,
            status_code=status_code,
            code=err_code,
            retry_after=retry_after,
        )
    elif status_code is not None and 500 <= status_code <= 599:
        err = AIGatewayServerError(
            clean_err_str,
            status_code=status_code,
            code=err_code,
            retry_after=retry_after,
        )
    else:
        err = AIGatewayError(
            clean_err_str,
            status_code=status_code,
            code=err_code,
            retry_after=retry_after,
        )

    if cause is not None:
        raise err from cause
    raise err


def _parse_sse_stream(text: str) -> tuple[bool, str]:
    """Parse luong Server-Sent Events (SSE) khi AI Gateway bo qua stream=False."""
    clean_text = text.lstrip("\ufeff")
    lines = clean_text.splitlines()
    accumulated_parts: list[str] = []
    found_choice = False

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        if clean_line.startswith(":"):
            # Comment SSE bo qua
            continue
        if clean_line.startswith(("event:", "id:", "retry:")):
            # Metadata event bo qua
            continue
        if not clean_line.startswith("data:"):
            continue

        payload = clean_line[5:].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if not isinstance(chunk, dict):
            continue

        if "error" in chunk:
            err_obj = chunk["error"]
            code_val = None
            if isinstance(err_obj, dict):
                msg = str(err_obj.get("message", err_obj))
                code_val = err_obj.get("code") or err_obj.get("type")
            else:
                msg = str(err_obj)
            _raise_classified_error(None, msg, err_code=str(code_val) if code_val else None)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        found_choice = True
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            target: dict[str, Any] | None = None
            if "delta" in choice and isinstance(choice["delta"], dict):
                target = choice["delta"]
            elif "message" in choice and isinstance(choice["message"], dict):
                target = choice["message"]

            if target is not None:
                if "content" in target:
                    extracted = _extract_text_blocks(target["content"])
                    if extracted:
                        accumulated_parts.append(extracted)
                elif "output_text" in target:
                    extracted = str(target["output_text"])
                    if extracted:
                        accumulated_parts.append(extracted)

    if found_choice:
        return True, "".join(accumulated_parts)
    return False, ""


def parse_chat_response(raw: str, content_type: str = "") -> str:
    """Parse phan hoi Chat Completions (ho tro ca JSON chuan va SSE fallback)."""
    clean_text = (raw or "").lstrip("\ufeff").strip()
    if not clean_text:
        raise AIGatewayError("Phản hồi từ AI Gateway rỗng.")

    sanitized_ct = _sanitize_content_type(content_type)
    is_sse_ct = "event-stream" in sanitized_ct
    is_sse_text = (
        clean_text.startswith("data:")
        or "\ndata:" in clean_text
        or "\rdata:" in clean_text
    )

    # Uu tien thu parse SSE neu header hoac text co dau hieu SSE
    if is_sse_ct or is_sse_text:
        ok, sse_result = _parse_sse_stream(clean_text)
        if ok:
            return sse_result
        if is_sse_ct:
            ct_desc = f" (Content-Type: {sanitized_ct})" if sanitized_ct else ""
            bounded = _sanitize_error_text(clean_text)
            raise AIGatewayError(f"Phản hồi SSE từ AI Gateway không hợp lệ{ct_desc}: {bounded}")

    # Thu parse JSON chuan
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        # Fallback SSE neu chua thu truoc do
        if not is_sse_ct and not is_sse_text:
            ok, sse_result = _parse_sse_stream(clean_text)
            if ok:
                return sse_result

        ct_desc = f" (Content-Type: {sanitized_ct})" if sanitized_ct else ""
        bounded = _sanitize_error_text(clean_text)
        err_msg = f"Phản hồi từ AI Gateway không đúng định dạng JSON{ct_desc}: {bounded}"
        raise AIGatewayError(err_msg) from exc

    if not isinstance(data, dict):
        ct_desc = f" (Content-Type: {sanitized_ct})" if sanitized_ct else ""
        bounded = _sanitize_error_text(clean_text)
        err_msg = f"Phản hồi từ AI Gateway không đúng định dạng JSON{ct_desc}: {bounded}"
        raise AIGatewayError(err_msg)

    if "error" in data:
        err_obj = data["error"]
        code_val = None
        if isinstance(err_obj, dict):
            msg = str(err_obj.get("message", str(err_obj)))
            code_val = err_obj.get("code") or err_obj.get("type")
        else:
            msg = str(err_obj)
        _raise_classified_error(None, msg, err_code=str(code_val) if code_val else None)

    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise AIGatewayError("Phản hồi từ AI Gateway thiếu trường 'choices'.")

    first = choices[0]
    if isinstance(first, dict):
        msg_obj = first.get("message")
        if not isinstance(msg_obj, dict) and "delta" in first and isinstance(first["delta"], dict):
            msg_obj = first["delta"]
        if isinstance(msg_obj, dict) and "content" in msg_obj:
            return _extract_text_blocks(msg_obj["content"])

    raise AIGatewayError("Không tìm thấy nội dung phản hồi từ AI Gateway.")


def chat_completion(
    endpoint: str,
    api_key: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    thinking: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> str:
    """Goi Chat Completions theo chuan OpenAI HTTP dung urllib cua Python."""
    url = normalize_chat_endpoint(endpoint)
    if not url:
        raise AIGatewayError("Chưa cấu hình Endpoint AI Gateway.")
    valid_timeout = max(1.0, float(timeout))

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    clean_thinking = (thinking or "").strip().lower()
    if clean_thinking and clean_thinking not in ("none", "off"):
        payload["reasoning_effort"] = clean_thinking
    if response_format is not None:
        payload["response_format"] = response_format
    if temperature is not None:
        payload["temperature"] = temperature

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "AutoSub-Studio/AI-Gateway",
    }
    clean_key = (api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=valid_timeout) as resp:
            content_type = ""
            resp_headers = getattr(resp, "headers", None)
            if resp_headers:
                if hasattr(resp_headers, "get"):
                    content_type = resp_headers.get("Content-Type", "") or ""
                elif hasattr(resp_headers, "get_content_type"):
                    content_type = resp_headers.get_content_type() or ""
            raw_bytes = resp.read()
            if isinstance(raw_bytes, str):
                raw = raw_bytes
            else:
                raw = raw_bytes.decode("utf-8-sig", errors="replace")
    except urllib.error.HTTPError as exc:
        err_body = ""
        with contextlib.suppress(Exception):
            err_body = exc.read().decode("utf-8", errors="replace")
        msg = exc.reason
        err_code: str | None = None
        with contextlib.suppress(Exception):
            parsed = json.loads(err_body)
            if isinstance(parsed, dict) and "error" in parsed:
                err_obj = parsed["error"]
                if isinstance(err_obj, dict):
                    msg = err_obj.get("message", msg)
                    err_code = err_obj.get("code")
                    if not err_code and "type" in err_obj:
                        err_code = str(err_obj.get("type"))
                else:
                    msg = str(err_obj)

        retry_after: float | None = None
        if exc.headers:
            ra_header = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
            if ra_header:
                with contextlib.suppress(Exception):
                    retry_after = float(ra_header)

        _raise_classified_error(
            exc.code,
            str(msg),
            err_code=err_code,
            retry_after=retry_after,
            cause=exc,
        )
    except urllib.error.URLError as exc:
        sanitized_reason = _sanitize_error_text(str(exc.reason))
        if isinstance(exc.reason, TimeoutError):
            raise AIGatewayTimeoutError() from exc
        raise AIGatewayConnectionError(
            f"Không thể kết nối tới AI Gateway: {sanitized_reason}",
            code="connection_failed",
        ) from exc
    except TimeoutError as exc:
        raise AIGatewayTimeoutError() from exc
    except AIGatewayError:
        raise
    except Exception as exc:
        raise AIGatewayError(f"Lỗi gọi AI Gateway: {exc}") from exc

    return parse_chat_response(raw, content_type=content_type)


def test_connection(
    endpoint: str, api_key: str, timeout: float = TEST_TIMEOUT
) -> tuple[bool, str]:
    """Kiem tra ket noi toi endpoint va khoa API."""
    url = normalize_models_endpoint(endpoint)
    if not url:
        return False, "Chưa nhập Endpoint."
    valid_timeout = max(1.0, float(timeout))

    headers: dict[str, str] = {"User-Agent": "AutoSub-Studio/AI-Gateway"}
    clean_key = (api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=valid_timeout) as resp:
            if resp.status < 400:
                return True, "Kết nối AI Gateway thành công."
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405):
            # Endpoint khong ho tro GET /models, thu chat completion ping
            try:
                chat_completion(
                    endpoint,
                    api_key,
                    model="sub",
                    messages=[{"role": "user", "content": "ping"}],
                    timeout=valid_timeout,
                )
                return True, "Kết nối AI Gateway thành công."
            except Exception as sub_exc:
                return False, f"Lỗi kết nối: {sub_exc}"
        return False, f"Lỗi kết nối ({exc.code}): {exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"Không thể kết nối: {exc.reason}"
    except Exception as exc:
        return False, f"Lỗi kết nối: {exc}"
    return True, "Kết nối AI Gateway thành công."


def test_model(
    endpoint: str,
    api_key: str,
    *,
    model: str,
    thinking: str = "",
    timeout: float = TEST_TIMEOUT,
) -> tuple[bool, str]:
    """Kiem tra mot model cu the voi muc do suy nghi tuong ung."""
    if not model.strip():
        return False, "Chưa chỉ định tên model."
    try:
        reply = chat_completion(
            endpoint,
            api_key,
            model=model,
            messages=[{"role": "user", "content": "Trả lời đúng một từ: OK"}],
            thinking=thinking,
            timeout=timeout,
        )
        return True, f"Model '{model}' hoạt động tốt. Phản hồi: {reply.strip()[:60]}"
    except Exception as exc:
        return False, f"Lỗi kiểm tra model '{model}': {exc}"


def test_translation_role(
    endpoint: str,
    api_key: str,
    *,
    model: str,
    thinking: str = "",
    timeout: float = TEST_TIMEOUT,
) -> tuple[bool, str]:
    """Kiem tra role prime bang mot structured translation request nho."""
    try:
        reply = chat_completion(
            endpoint,
            api_key,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate subtitle items and return JSON only: "
                        '{"translations":[{"id":1,"text":"..."}]}'
                    ),
                },
                {
                    "role": "user",
                    "content": '{"target":"English","items":[{"id":1,"text":"等一下"}]}',
                },
            ],
            thinking=thinking,
            timeout=timeout,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(reply.strip().removeprefix("```json").removesuffix("```").strip())
        items = parsed.get("translations") if isinstance(parsed, dict) else None
        if not isinstance(items, list) or not any(
            isinstance(item, dict) and item.get("id") == 1 and str(item.get("text", "")).strip()
            for item in items
        ):
            return False, f"Role prime '{model}' trả structured translation không hợp lệ."
        return True, f"Role prime '{model}': structured translation OK."
    except Exception as exc:
        return False, f"Lỗi kiểm tra role prime '{model}': {exc}"


def image_to_base64_url(image_data: bytes | Path | str, mime: str = "image/jpeg") -> str:
    """Chuyen bytes hoac tep anh thanh data URL base64 chuan vision."""
    if isinstance(image_data, (str, Path)):
        raw_bytes = Path(image_data).read_bytes()
    else:
        raw_bytes = bytes(image_data)
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def ocr_image_with_ai(
    endpoint: str,
    api_key: str,
    *,
    model: str,
    image_data: bytes | Path | str,
    thinking: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Dung AI vision de doc chu tren mot khung hinh da crop."""
    b64_url = image_to_base64_url(image_data)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Đọc toàn bộ văn bản/chữ có trong hình ảnh này. "
                        "Chỉ trả về nội dung chữ đọc được, không thêm bất kỳ giải thích, "
                        "ghi chú hay định dạng markdown nào. "
                        "Nếu trong ảnh không có chữ, chỉ trả về chuỗi rỗng."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": b64_url},
                },
            ],
        }
    ]
    return chat_completion(
        endpoint,
        api_key,
        model=model,
        messages=messages,
        thinking=thinking,
        timeout=timeout,
    )
