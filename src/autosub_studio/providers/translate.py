"""Dich phu de: Google mien phi, AI Gateway (chuan OpenAI HTTP), hoac giu nguyen."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..services import ai_gateway
from ..services.ai_gateway_scheduler import get_global_gateway_scheduler
from ..services.ffmpeg import CancelledError
from ..services.settings import Settings

PROVIDER_NONE = "Khong dich (giu nguyen)"
PROVIDER_GOOGLE = "Google (mien phi)"
PROVIDER_SERVER_AI = "Server AI API"
PROVIDER_AI = "AI Gateway"

PROVIDERS = (PROVIDER_GOOGLE, PROVIDER_SERVER_AI, PROVIDER_NONE)

AI_MODELS = ("sub", "prime")
TRANSLATION_PROMPT_VERSION = "v2-document"
DEFAULT_SAFE_TRANSLATION_CHARS = 60_000

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


class TranslationValidationError(TranslationError):
    """Phan hoi co the giu lai mot phan va chi sua cac ID bi loi."""

    def __init__(
        self,
        message: str,
        *,
        partial: dict[int, str] | None = None,
        missing_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial = dict(partial or {})
        self.missing_ids = list(missing_ids or [])


@dataclass
class TranslationRequest:
    """Mot lo cau can dich."""

    texts: list[str]
    ids: list[int] | None = None
    source: str = "auto"
    target: str = "vi"
    context_before: str = ""
    context_after: str = ""
    glossary: dict[str, str] | None = None
    extra_prompt: str = ""


def cache_key(
    text: str,
    source: str,
    target: str,
    provider: str,
    *,
    role: str = "",
    prompt_version: str = "",
    settings_fingerprint: str = "",
) -> str:
    raw = (
        f"{provider}|{role}|{source}|{target}|{prompt_version}|"
        f"{settings_fingerprint}|{text}"
    ).encode()
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
    model: str = "prime",
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
            model=ai_gateway.role_for_task("subtitle_translation"),
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
    model: str = "prime",
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

    model_role = ai_gateway.role_for_task("subtitle_translation")
    actual_model, default_thinking = ai_gateway.resolve_model(model_role, settings)
    actual_thinking = thinking if thinking else default_thinking

    src_name = LANGUAGES.get(request.source, request.source or "tu nhan dang")
    dst_name = LANGUAGES.get(request.target, request.target)
    expected_ids = (
        list(request.ids)
        if request.ids is not None
        else list(range(len(request.texts)))
    )
    if len(expected_ids) != len(request.texts) or len(set(expected_ids)) != len(expected_ids):
        raise TranslationError("Danh sách ID dịch không hợp lệ hoặc bị trùng.")
    lines = [
        {"id": item_id, "text": text}
        for item_id, text in zip(expected_ids, request.texts, strict=True)
    ]

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
        "Bạn là biên dịch viên phụ đề chuyên nghiệp. Dịch tự nhiên, đúng nghĩa và phù hợp "
        "văn nói của phụ đề; duy trì nhất quán tên nhân vật, quan hệ, đại từ, giọng điệu "
        "và thuật ngữ trong toàn bộ payload.\n"
        "Quy tắc bắt buộc:\n"
        f"1. Số lượng câu dịch ra PHẢI ĐÚNG bằng số lượng câu đầu vào ({len(request.texts)} câu).\n"
        "2. Giữ nguyên chính xác id của từng mục đầu vào.\n"
        "3. Giữ đúng tên riêng, thương hiệu và con số theo ngữ cảnh.\n"
        "4. Không tóm tắt, kiểm duyệt, thêm ý, giải thích, gộp, tách hoặc bỏ sót câu.\n"
        "5. Chỉ trả JSON: {\"translations\": [{\"id\": <id gốc>, \"text\": \"...\"}, ...]}"
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

    return _parse_ai_json(raw_reply, expected_ids, strict=True, source_texts=request.texts)


def _parse_ai_json(
    text: str,
    expected: int | Sequence[int],
    *,
    strict: bool = True,
    source_texts: Sequence[str] | None = None,
) -> list[str]:
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
        raise TranslationValidationError(
            "Kết quả trả về từ AI Gateway không đúng định dạng JSON."
        ) from exc

    items = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise TranslationValidationError(
            "Kết quả trả về từ AI Gateway thiếu trường 'translations'."
        )

    expected_ids = list(range(expected)) if isinstance(expected, int) else list(expected)
    expected_set = set(expected_ids)
    source_position = {item_id: pos for pos, item_id in enumerate(expected_ids)}
    if len(expected_set) != len(expected_ids):
        raise TranslationError("Danh sách ID mong đợi bị trùng.")

    out_by_id: dict[int, str] = {}
    filled: set[int] = set()
    invalid_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id", -1))
        except (TypeError, ValueError):
            continue
        if idx not in expected_set or idx in filled:
            invalid_ids.add(idx)
            continue
        value = item.get("text", "")
        translated = value.strip() if isinstance(value, str) else ""
        source_index = source_position[idx]
        source = source_texts[source_index] if source_texts is not None else ""
        suspicious_limit = max(2_000, len(source) * 6 + 200)
        if not translated or "\ufffd" in translated or len(translated) > suspicious_limit:
            continue
        out_by_id[idx] = translated
        filled.add(idx)

    missing = [idx for idx in expected_ids if idx not in filled]
    if strict and (missing or invalid_ids or len(items) != len(expected_ids)):
        details: list[str] = []
        if missing:
            details.append(f"thiếu ID {missing[:20]}")
        if invalid_ids:
            details.append(f"ID lạ/trùng {sorted(invalid_ids)[:20]}")
        raise TranslationValidationError(
            "Số lượng câu dịch không khớp hoặc phản hồi không hợp lệ: "
            + ", ".join(details or [f"nhận {len(items)}/{len(expected_ids)}"]),
            partial=out_by_id,
            missing_ids=missing or expected_ids,
        )
    return [out_by_id.get(idx, "") for idx in expected_ids]


@dataclass(frozen=True)
class _TranslationUnit:
    id: int
    text: str


@dataclass(frozen=True)
class _TranslationResult:
    id: int
    text: str


def _chunk_units_by_size(
    units: Sequence[_TranslationUnit], max_characters: int
) -> list[list[_TranslationUnit]]:
    """Chia theo kich thuoc payload, khong theo batch 5-8 cau co dinh."""
    limit = max(2_000, int(max_characters))
    chunks: list[list[_TranslationUnit]] = []
    current: list[_TranslationUnit] = []
    current_size = 0
    for unit in units:
        item_size = len(unit.text) + 40
        if current and current_size + item_size > limit:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(unit)
        current_size += item_size
    if current:
        chunks.append(current)
    return chunks


def translate_document(
    provider: str,
    request: TranslationRequest,
    *,
    api_key: str = "",
    endpoint: str = "",
    thinking: str = "",
    max_characters: int = DEFAULT_SAFE_TRANSLATION_CHARS,
    max_retries: int = 3,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_chunk_complete: Callable[[list[int], list[str]], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> list[str]:
    """Dich mot tai lieu phu de bang don vi lon nhat an toan.

    AI Gateway dung role ``prime`` va di qua scheduler toan cuc. Tai lieu vua
    gioi han se la mot request. Tai lieu lon duoc chia theo kich thuoc; chi
    chunk/ID loi moi bi chia hoac gui sua lai.
    """
    if not request.texts:
        return []
    if not is_ai_provider(provider):
        return translate_batch(provider, request, api_key=api_key, on_log=on_log)
    if not ai_gateway.capabilities().structured_text:
        raise TranslationError("AI Gateway không hỗ trợ structured text cho role 'prime'.")

    ids = list(request.ids) if request.ids is not None else list(range(len(request.texts)))
    if len(ids) != len(request.texts) or len(set(ids)) != len(ids):
        raise TranslationError("Danh sách ID dịch không hợp lệ hoặc bị trùng.")

    units = [
        _TranslationUnit(item_id, text)
        for item_id, text in zip(ids, request.texts, strict=True)
    ]
    position_by_id = {unit.id: pos for pos, unit in enumerate(units)}
    translation_settings = Settings.load()
    _prime_model, default_prime_thinking = ai_gateway.resolve_model(
        ai_gateway.role_for_task("subtitle_translation"), translation_settings
    )
    effective_thinking = thinking if thinking else default_prime_thinking
    completed: dict[int, str] = {}
    scheduler = get_global_gateway_scheduler()
    baseline = scheduler.get_metrics()
    started = time.monotonic()
    request_count = 0
    repair_request_count = 0
    payload_characters = 0

    if on_log:
        on_log(
            f"AI Task: Subtitle Translation | Model Role: prime | "
            f"thinking={effective_thinking or 'none'} | cues={len(units)} | "
            f"chars={sum(len(u.text) for u in units)}"
        )

    def checkpoint(results: Sequence[_TranslationResult]) -> None:
        fresh = [r for r in results if r.id not in completed and r.text.strip()]
        for result in fresh:
            completed[result.id] = apply_glossary(result.text.strip(), request.glossary)
        if fresh and on_chunk_complete:
            on_chunk_complete(
                [result.id for result in fresh],
                [completed[result.id] for result in fresh],
            )
        if on_progress:
            on_progress(int(len(completed) / max(1, len(units)) * 100))

    def neighbor_context(batch_units: Sequence[_TranslationUnit]) -> tuple[str, str]:
        positions = [position_by_id[u.id] for u in batch_units if u.id in position_by_id]
        if not positions:
            return "", ""
        first = min(positions)
        last = max(positions)
        before = "\n".join(u.text for u in units[max(0, first - 10) : first])
        after = "\n".join(u.text for u in units[last + 1 : last + 11])
        return before, after

    def send(batch_units: list[_TranslationUnit], *, repair: bool = False) -> None:
        nonlocal request_count, repair_request_count, payload_characters
        if not batch_units:
            return
        if should_cancel and should_cancel():
            raise CancelledError("Dịch phụ đề đã bị hủy.")

        before, after = neighbor_context(batch_units)

        def call_provider(items: Sequence[_TranslationUnit]) -> list[_TranslationResult]:
            nonlocal request_count, repair_request_count, payload_characters
            request_count += 1
            if repair:
                repair_request_count += 1
            payload_characters += sum(len(item.text) for item in items)
            sub_request = TranslationRequest(
                texts=[item.text for item in items],
                ids=[item.id for item in items],
                source=request.source,
                target=request.target,
                context_before=before,
                context_after=after,
                glossary=request.glossary,
                extra_prompt=request.extra_prompt,
            )
            translated = _translate_ai(
                sub_request,
                api_key=api_key,
                model=ai_gateway.role_for_task("subtitle_translation"),
                endpoint=endpoint,
                thinking=effective_thinking,
                on_log=None,
            )
            return [
                _TranslationResult(item.id, text)
                for item, text in zip(items, translated, strict=True)
            ]

        try:
            result_map = scheduler.schedule_job(
                job_id=f"prime_{uuid.uuid4().hex}",
                batches=[batch_units],
                provider=call_provider,
                should_cancel=should_cancel,
                max_retries=max(0, int(max_retries)),
                fail_fast=True,
                raise_on_cancel=True,
            )
            checkpoint(
                [result_map[unit.id] for unit in batch_units if unit.id in result_map]
            )
        except TranslationValidationError as exc:
            partial_results = [
                _TranslationResult(item_id, value)
                for item_id, value in exc.partial.items()
                if item_id in {unit.id for unit in batch_units}
            ]
            checkpoint(partial_results)
            remaining = [unit for unit in batch_units if unit.id not in completed]
            if not remaining:
                return
            if len(remaining) < len(batch_units):
                send(remaining, repair=True)
                return
            if len(batch_units) <= 1:
                raise
            middle = len(batch_units) // 2
            send(batch_units[:middle], repair=True)
            send(batch_units[middle:], repair=True)
        except TranslationError as exc:
            cause: BaseException | None = exc.__cause__
            if isinstance(cause, ai_gateway.AIGatewayError) and not getattr(
                cause, "is_split_required", False
            ):
                raise
            if len(batch_units) <= 1:
                raise
            middle = len(batch_units) // 2
            send(batch_units[:middle], repair=True)
            send(batch_units[middle:], repair=True)

    initial_chunks = _chunk_units_by_size(units, max_characters)
    for initial in initial_chunks:
        send(initial)

    missing = [unit.id for unit in units if unit.id not in completed]
    if missing:
        raise TranslationError(f"Dịch còn thiếu các ID: {missing[:20]}")

    elapsed = time.monotonic() - started
    current_metrics = scheduler.get_metrics()
    retries = max(0, current_metrics.retries - baseline.retries)
    avg_payload = payload_characters / max(1, request_count)
    if on_log:
        on_log(
            f"Translation Metrics: role=prime, cues={len(units)}, "
            f"source_chars={sum(len(u.text) for u in units)}, requests={request_count}, "
            f"avg_payload_chars={avg_payload:.0f}, retries={retries}, "
            f"repair_requests={repair_request_count}, elapsed={elapsed:.2f}s"
        )
    if on_progress:
        on_progress(100)
    return [completed[item_id] for item_id in ids]


def chunk(items: Sequence[str], size: int = 25) -> list[list[str]]:
    """Chia danh sach thanh cac lo nho de goi dich vu."""
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
