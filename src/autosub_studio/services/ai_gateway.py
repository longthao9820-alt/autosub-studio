"""Cong giao tiep AI Gateway duy nhat cua ung dung (chuan OpenAI HTTP bang stdlib)."""

from __future__ import annotations

import base64
import contextlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .settings import Settings

DEFAULT_TIMEOUT = 60.0
TEST_TIMEOUT = 15.0


class AIGatewayError(RuntimeError):
    """Loi khi goi hoac ket noi toi AI Gateway."""


def normalize_chat_endpoint(endpoint: str) -> str:
    """Chuan hoa endpoint ve URL day du /chat/completions.

    Xu ly cac dang:
    - https://api.openai.com
    - https://api.openai.com/
    - https://api.openai.com/v1
    - https://api.openai.com/v1/
    - https://api.openai.com/v1/chat/completions
    - http://localhost:8000/chat/completions
    """
    ep = (endpoint or "").strip().rstrip("/")
    if not ep:
        return ""
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
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = ""
        with contextlib.suppress(Exception):
            err_body = exc.read().decode("utf-8")
        msg = exc.reason
        with contextlib.suppress(Exception):
            parsed = json.loads(err_body)
            if isinstance(parsed, dict) and "error" in parsed:
                err_obj = parsed["error"]
                msg = err_obj.get("message", msg) if isinstance(err_obj, dict) else str(err_obj)
        raise AIGatewayError(f"AI Gateway báo lỗi {exc.code}: {msg}") from exc
    except urllib.error.URLError as exc:
        raise AIGatewayError(f"Không thể kết nối tới AI Gateway: {exc.reason}") from exc
    except TimeoutError as exc:
        raise AIGatewayError("Hết thời gian chờ phản hồi từ AI Gateway (timeout).") from exc
    except Exception as exc:
        raise AIGatewayError(f"Lỗi gọi AI Gateway: {exc}") from exc

    try:
        res_json = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIGatewayError("Phản hồi từ AI Gateway không đúng định dạng JSON.") from exc

    choices = res_json.get("choices") if isinstance(res_json, dict) else None
    if not choices or not isinstance(choices, list):
        raise AIGatewayError("Phản hồi từ AI Gateway thiếu trường 'choices'.")
    first = choices[0]
    if isinstance(first, dict):
        msg_obj = first.get("message")
        if isinstance(msg_obj, dict):
            content = msg_obj.get("content")
            if content is not None:
                return str(content)
    raise AIGatewayError("Không tìm thấy nội dung phản hồi từ AI Gateway.")


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
