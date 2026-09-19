"""Dich phu de: Google mien phi, mo hinh ngon ngu Claude, hoac giu nguyen."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

PROVIDER_NONE = "Khong dich (giu nguyen)"
PROVIDER_GOOGLE = "Google (mien phi)"
PROVIDER_CLAUDE = "Claude (can khoa API)"

PROVIDERS = (PROVIDER_GOOGLE, PROVIDER_CLAUDE, PROVIDER_NONE)

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

CLAUDE_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
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
    if provider == PROVIDER_CLAUDE:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "Chua cai thu vien anthropic. Chay: pip install anthropic"
        if not api_key:
            return False, "Chua nhap khoa API Claude trong tab Cai dat chung."
        return True, "Can ket noi Internet."
    return False, f"Khong ho tro nha cung cap: {provider}"


def translate_batch(
    provider: str,
    request: TranslationRequest,
    *,
    api_key: str = "",
    model: str = "claude-opus-5",
    on_log: Callable[[str], None] | None = None,
) -> list[str]:
    """Dich mot lo cau, tra ve danh sach ban dich cung do dai voi dau vao."""
    if not request.texts:
        return []
    if provider == PROVIDER_NONE:
        return list(request.texts)
    if provider == PROVIDER_GOOGLE:
        out = _translate_google(request, on_log=on_log)
    elif provider == PROVIDER_CLAUDE:
        out = _translate_claude(request, api_key=api_key, model=model, on_log=on_log)
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


# --------------------------------------------------------------------------- Claude

_SYSTEM_PROMPT = (
    "Ban la bien dich vien phu de chuyen nghiep. Dich tung cau sang ngon ngu dich, "
    "giu dung so luong cau va dung thu tu. Giu nguyen ten rieng, thuong hieu va con so. "
    "Cau dich phai ngan gon de vua thoi luong hien thi tren man hinh. "
    "Khong them giai thich, khong them dau ngoac, khong gop hay tach cau."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def _translate_claude(
    request: TranslationRequest,
    *,
    api_key: str,
    model: str = "claude-opus-5",
    on_log: Callable[[str], None] | None = None,
) -> list[str]:
    try:
        import anthropic
    except ImportError as exc:
        raise TranslationError("Chua cai thu vien anthropic. Chay: pip install anthropic") from exc
    if not api_key:
        raise TranslationError("Chua nhap khoa API Claude trong tab Cai dat chung.")

    client = anthropic.Anthropic(api_key=api_key)
    src_name = LANGUAGES.get(request.source, request.source or "tu nhan dang")
    dst_name = LANGUAGES.get(request.target, request.target)
    lines = [{"id": i, "text": t} for i, t in enumerate(request.texts)]
    glossary_note = ""
    if request.glossary:
        pairs = ", ".join(f"{k} = {v}" for k, v in list(request.glossary.items())[:50])
        glossary_note = f"\nBang thuat ngu bat buoc giu nguyen hoac dich co dinh: {pairs}"
    context_note = ""
    if request.context_before or request.context_after:
        context_note = (
            f"\nNgu canh truoc: {request.context_before}\nNgu canh sau: {request.context_after}"
        )
    prompt = (
        f"Dich cac cau phu de sau tu {src_name} sang {dst_name}."
        f"{glossary_note}{context_note}\n\n"
        f"Danh sach cau (JSON):\n{json.dumps(lines, ensure_ascii=False)}"
    )

    system_prompt = _SYSTEM_PROMPT
    if request.extra_prompt.strip():
        system_prompt += "\n\nYeu cau rieng cua nguoi dung:\n" + request.extra_prompt.strip()

    try:
        response = client.messages.create(
            model=model or "claude-opus-5",
            max_tokens=16000,
            system=system_prompt,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise TranslationError("Khoa API Claude khong dung hoac da bi thu hoi.") from exc
    except anthropic.RateLimitError as exc:
        raise TranslationError(
            "Dich vu Claude bao qua gioi han goi. Cho vai phut roi dich lai."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise TranslationError("Khong ket noi duoc toi Claude. Kiem tra mang.") from exc
    except anthropic.APIStatusError as exc:
        raise TranslationError(f"Claude bao loi {exc.status_code}: {exc.message}") from exc

    if response.stop_reason == "refusal":
        raise TranslationError(
            "Claude tu choi dich noi dung nay. Hay dung nha cung cap khac cho doan nay."
        )

    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
    if on_log:
        usage = getattr(response, "usage", None)
        if usage is not None:
            on_log(f"Claude: {usage.input_tokens} token vao, {usage.output_tokens} token ra")
    return _parse_claude_json(text, len(request.texts))


def _parse_claude_json(text: str, expected: int) -> list[str]:
    """Doc ket qua JSON tra ve, chap nhan ca truong hop thieu hoac thua cau."""
    cleaned = _TAG_RE.sub("", text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TranslationError("Ket qua tra ve tu Claude khong dung dinh dang JSON.") from exc
    items = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise TranslationError("Ket qua tra ve tu Claude thieu truong 'translations'.")
    out = [""] * expected
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < expected:
            out[idx] = str(item.get("text", "")).strip()
    return out


def chunk(items: Sequence[str], size: int = 25) -> list[list[str]]:
    """Chia danh sach thanh cac lo nho de goi dich vu."""
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
