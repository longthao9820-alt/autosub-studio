"""Dich phu de: Google mien phi, AI Gateway (chuan OpenAI HTTP), hoac giu nguyen."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..services import ai_gateway
from ..services.settings import Settings

PROVIDER_NONE = "Khong dich (giu nguyen)"
PROVIDER_GOOGLE = "Google (mien phi)"
PROVIDER_SERVER_AI = "Server AI API"
PROVIDER_AI = "AI Gateway"

PROVIDERS = (PROVIDER_GOOGLE, PROVIDER_SERVER_AI, PROVIDER_NONE)

AI_MODELS = ("sub", "prime")

# Ma ngon ngu -> ten hien thi tren giao dien.
LANGUAGES: dict[str, str] = {
    "auto": "Tu nhan dang",
    "vi": "Tieng Viet",
    "en": "Tieng Anh",
    "zh-CN": "Tieng Trung (gian the)",
    "ja": "Tieng Nhat",
    "ko": "Tieng Han",
    "th": "Tieng Thai",
    "fr": "Tieng Phap",
    "de": "Tieng Duc",
    "es": "Tieng Tay Ban Nha",
    "ru": "Tieng Nga",
    "id": "Tieng Indonesia",
    "ms": "Tieng Malaysia",
    "hi": "Tieng An Do",
    "ar": "Tieng A Rap",
    "pt": "Tieng Bo Dao Nha",
}

_TAG_RE = re.compile(r"</?[a-zA-Z_][^>]*>")


class TranslationError(RuntimeError):
    """Loi khi goi dich vu dich."""


@dataclass
class TranslationRequest:
    """Mot lo cau can dich."""

    texts: list[str]
    source: str = "auto"
    target: str = "vi"
    context_before: str = ""
    context_after: str = ""
    glossary: dict[str, str] | None = None
    extra_prompt: str = ""


def cache_key(text: str, source: str, target: str, provider: str) -> str:
    raw = f"{provider}|{source}|{target}|{text}".encode()
    return hashlib.sha256(raw).hexdigest()


def apply_glossary(text: str, glossary: dict[str, str] | None) -> str:
    """Thay cac tu trong bang thuat ngu sau khi dich."""
    if not glossary:
        return text
    out = text
    for src, dst in glossary.items():
        if not src.strip():
            continue
        out = re.sub(re.escape(src), dst or src, out, flags=re.IGNORECASE)
    return out


def available_providers() -> list[str]:
    return list(PROVIDERS)


def is_ai_provider(provider: str) -> bool:
    return provider in (
        PROVIDER_SERVER_AI,
        PROVIDER_AI,
        "Server AI API",
        "AI Gateway",
    )


def provider_ready(provider: str, api_key: str = "") -> tuple[bool, str]:
    """Kiem tra nha cung cap co dung duoc khong. Tra ve (san sang, ly do)."""
    if provider == PROVIDER_NONE:
        return True, ""
    if provider == PROVIDER_GOOGLE:
        try:
            import deep_translator  # noqa: F401
        except ImportError:
            return False, "Chua cai thu vien deep-translator. Chay: pip install deep-translator"
        return True, "Can ket noi Internet."
    if is_ai_provider(provider):
        settings = Settings.load()
        if not settings.ai_endpoint.strip():
            return False, "Chưa cấu hình Endpoint AI Gateway (mở nút AI Gateway)."
        key = api_key.strip() or Settings.get_secret("ai_gateway_key")
        if not key.strip():
            return False, "Chưa nhập khóa API cho AI Gateway."
        return True, "Sẵn sàng (AI Gateway)."
    return False, f"Khong ho tro nha cung cap: {provider}"


def translate_batch(
    provider: str,
    request: TranslationRequest,
    *,
    api_key: str = "",
    model: str = "sub",
    endpoint: str = "",
    thinking: str = "",
    on_log: Callable[[str], None] | None = None,
) -> list[str]:
    """Dich mot lo cau, tra ve danh sach ban dich cung do dai voi dau vao."""
    if not request.texts:
        return []
    if provider == PROVIDER_NONE:
        return list(request.texts)
    if provider == PROVIDER_GOOGLE:
        out = _translate_google(request, on_log=on_log)
    elif is_ai_provider(provider):
        out = _translate_ai(
            request,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            thinking=thinking,
            on_log=on_log,
        )
    else:
        raise TranslationError(f"Khong ho tro nha cung cap: {provider}")
    return [apply_glossary(t, request.glossary) for t in out]


# --------------------------------------------------------------------------- Google


def _translate_google(
    request: TranslationRequest, *, on_log: Callable[[str], None] | None = None
) -> list[str]:
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise TranslationError(
            "Chua cai thu vien deep-translator. Chay: pip install deep-translator"
        ) from exc
    source = "auto" if request.source in ("", "auto") else request.source
    try:
        translator = GoogleTranslator(source=source, target=request.target)
    except Exception as exc:
        raise TranslationError(f"Khong tao duoc bo dich Google: {exc}") from exc

    results: list[str] = []
    for text in request.texts:
        clean = text.strip()
        if not clean:
            results.append("")
            continue
        try:
            out = translator.translate(clean)
        except Exception as exc:
            raise TranslationError(
                f"Goi Google Dich that bai. Kiem tra ket noi mang roi thu lai. Chi tiet: {exc}"
            ) from exc
        results.append((out or "").strip())
        if on_log:
            on_log(f"Google: {clean[:40]} -> {results[-1][:40]}")
    return results


# --------------------------------------------------------------------------- AI Gateway


def _translate_ai(
    request: TranslationRequest,
    *,
    api_key: str = "",
    model: str = "sub",
    endpoint: str = "",
    thinking: str = "",
    on_log: Callable[[str], None] | None = None,
) -> list[str]:
    settings = Settings.load()
    ep = endpoint.strip() or settings.ai_endpoint.strip()
    if not ep:
        raise TranslationError("Chưa cấu hình Endpoint AI Gateway.")
    key = api_key.strip() or Settings.get_secret("ai_gateway_key")
    if not key:
        raise TranslationError("Chưa nhập khóa API AI Gateway.")

    actual_model, default_thinking = ai_gateway.resolve_model(model or settings.llm_model, settings)
    actual_thinking = thinking if thinking else default_thinking

    src_name = LANGUAGES.get(request.source, request.source or "tu nhan dang")
    dst_name = LANGUAGES.get(request.target, request.target)
    lines = [{"id": i, "text": t} for i, t in enumerate(request.texts)]

    glossary_note = ""
    if request.glossary:
        pairs = ", ".join(f"{k} = {v}" for k, v in list(request.glossary.items())[:50])
        glossary_note = f"\nBảng thuật ngữ bắt buộc giữ nguyên hoặc dịch cố định: {pairs}"

    context_note = ""
    if request.context_before or request.context_after:
        context_note = (
            f"\nNgữ cảnh trước: {request.context_before}\nNgữ cảnh sau: {request.context_after}"
        )

    system_prompt = (
        "Bạn là biên dịch viên phụ đề chuyên nghiệp. Dịch chính xác từng câu sang ngôn ngữ đích.\n"
        "Quy tắc bắt buộc:\n"
        f"1. Số lượng câu dịch ra PHẢI ĐÚNG bằng số lượng câu đầu vào ({len(request.texts)} câu).\n"
        "2. Giữ nguyên thứ tự 1:1 theo id từ 0 đến N-1.\n"
        "3. Giữ nguyên tên riêng, thương hiệu và con số.\n"
        "4. Không thêm giải thích, không gộp câu, không tách câu, không bỏ sót câu.\n"
        "5. Trả về đúng định dạng JSON: {\"translations\": [{\"id\": 0, \"text\": \"...\"}, ...]}"
    )
    if request.extra_prompt.strip():
        system_prompt += "\n\nYêu cầu riêng của người dùng:\n" + request.extra_prompt.strip()

    prompt = (
        f"Dịch các câu phụ đề sau từ {src_name} sang {dst_name}."
        f"{glossary_note}{context_note}\n\n"
        f"Danh sách câu (JSON):\n{json.dumps(lines, ensure_ascii=False)}"
    )

    try:
        raw_reply = ai_gateway.chat_completion(
            ep,
            key,
            model=actual_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            thinking=actual_thinking,
            response_format={"type": "json_object"},
        )
    except ai_gateway.AIGatewayError as exc:
        raise TranslationError(str(exc)) from exc

    if on_log:
        on_log(f"AI Gateway ({actual_model}): đã dịch {len(request.texts)} câu.")

    return _parse_ai_json(raw_reply, len(request.texts), strict=True)


def _parse_ai_json(text: str, expected: int, *, strict: bool = True) -> list[str]:
    """Doc ket qua JSON tra ve tu AI Gateway.

    Neu strict=True: kiem tra bat buoc so cau tra ve dung bang expected,
    neu lech thi bao loi translation count mismatch.
    """
    cleaned = _TAG_RE.sub("", text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TranslationError("Kết quả trả về từ AI Gateway không đúng định dạng JSON.") from exc

    items = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise TranslationError("Kết quả trả về từ AI Gateway thiếu trường 'translations'.")

    if strict and len(items) != expected:
        raise TranslationError(
            f"Số lượng câu dịch ({len(items)}) không khớp với số lượng câu gốc ({expected})."
        )

    out = [""] * expected
    filled: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < expected:
            out[idx] = str(item.get("text", "")).strip()
            filled.add(idx)

    if strict and len(filled) != expected:
        raise TranslationError(
            f"Số lượng câu dịch ({len(filled)}) không khớp với số lượng câu gốc ({expected})."
        )
    return out


def chunk(items: Sequence[str], size: int = 25) -> list[list[str]]:
    """Chia danh sach thanh cac lo nho de goi dich vu."""
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
