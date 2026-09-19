"""Nhan dang giong noi bang faster-whisper (chay tren may, khong gui du lieu di dau)."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from ..core.models import Cue
from ..services.gpu import cuda_ready
from ..services.paths import bundled_dir

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEVICES = ("Tu chon", "CPU", "GPU (CUDA)")


class ASRUnavailable(RuntimeError):
    """Chua cai faster-whisper hoac thieu model."""


class ASRError(RuntimeError):
    """Nhan dang that bai."""


def is_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def install_hint() -> str:
    return (
        "Chua cai faster-whisper. Mo Cai dat chung de xem huong dan, hoac chay: "
        "pip install faster-whisper"
    )


def bundled_model_dirs() -> list[Path]:
    """Cac thu muc model kem theo ban dong goi, sap theo kich thuoc model."""
    root = bundled_dir("models")
    if root is None:
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "model.bin").is_file():
            found.append(child)
    return found


def bundled_model_for(size: str) -> Path | None:
    """Thu muc model kem theo ung dung hop voi kich thuoc yeu cau, neu co."""
    dirs = bundled_model_dirs()
    if not dirs:
        return None
    needle = size.lower().replace("-", "")
    for path in dirs:
        if needle in path.name.lower().replace("-", ""):
            return path
    return dirs[0]


def resolve_model_source(size: str, model_dir: str = "") -> tuple[str, bool]:
    """Chon nguon model. Tra ve (duong dan hoac ten model, co san tren may)."""
    if model_dir:
        p = Path(model_dir)
        if (p / "model.bin").is_file():
            return str(p), True
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "model.bin").is_file():
                    return str(child), True
    bundled = bundled_model_for(size)
    if bundled is not None:
        return str(bundled), True
    return size, model_is_local(size, model_dir)


def model_cache_dir(model_dir: str = "") -> Path:
    """Thu muc chua model da tai."""
    if model_dir:
        return Path(model_dir)
    hf = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf:
        return Path(hf)
    return Path.home() / ".cache" / "huggingface"


def model_is_local(size: str, model_dir: str = "") -> bool:
    """Kiem tra model da co san tren may chua (de khong tu y tai ve)."""
    p = Path(model_dir) if model_dir else None
    if p and p.is_dir():
        if (p / "model.bin").is_file():
            return True
        if any(child.is_dir() and (child / "model.bin").is_file() for child in p.iterdir()):
            return True
    root = model_cache_dir(model_dir)
    if not root.is_dir():
        return False
    needle = f"faster-whisper-{size}".lower()
    try:
        for child in root.rglob("*"):
            if child.is_dir() and needle in child.name.lower():
                return True
    except OSError:
        return False
    return False


def resolve_device(choice: str, use_gpu: bool) -> tuple[str, str]:
    """Chon thiet bi va kieu tinh toan phu hop.

    Neu nguoi dung muon dung card do hoa nhung may khong dap ung thi tu lui
    ve CPU, khong bao loi.
    """
    want_gpu = choice.startswith("GPU") or (choice == "Tu chon" and use_gpu)
    if want_gpu and cuda_ready()[0]:
        return "cuda", "float16"
    return "cpu", "int8"


def transcribe(
    audio_path: str | Path,
    *,
    model_size: str = "small",
    model_dir: str = "",
    language: str = "auto",
    device: str = "Tu chon",
    use_gpu: bool = False,
    vad: bool = True,
    duration: float = 0.0,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[Cue], str]:
    """Nhan dang mot tep am thanh. Tra ve (danh sach cau, ma ngon ngu nhan duoc)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ASRUnavailable(install_hint()) from exc

    src = Path(audio_path)
    if not src.is_file():
        raise ASRError(f"Khong tim thay tep am thanh: {src}")

    dev, compute = resolve_device(device, use_gpu)
    target, _local = resolve_model_source(model_size, model_dir)
    if on_log:
        label = Path(target).name if Path(target).is_dir() else target
        where = "card do hoa (GPU)" if dev == "cuda" else "bo xu ly (CPU)"
        on_log(f"Nap model '{label}' tren {where}, kieu tinh toan {compute}...")
        if dev == "cpu" and (device.startswith("GPU") or use_gpu):
            on_log(f"Khong dung duoc GPU: {cuda_ready()[1]}")
    try:
        model = WhisperModel(target, device=dev, compute_type=compute)
    except Exception as exc:
        if dev == "cuda":
            if on_log:
                on_log(f"Khong dung duoc GPU ({exc}). Chuyen sang CPU.")
            try:
                model = WhisperModel(target, device="cpu", compute_type="int8")
            except Exception as exc2:
                raise ASRError(_friendly(exc2)) from exc2
        else:
            raise ASRError(_friendly(exc)) from exc

    lang = None if language in ("", "auto") else language
    try:
        segments, info = model.transcribe(
            str(src),
            language=lang,
            vad_filter=bool(vad),
            vad_parameters={"min_silence_duration_ms": 500} if vad else None,
            beam_size=5,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
    except Exception as exc:
        raise ASRError(_friendly(exc)) from exc

    total = duration or float(getattr(info, "duration", 0.0) or 0.0)
    detected = str(getattr(info, "language", "") or "")
    cues: list[Cue] = []
    for seg in segments:
        if should_cancel is not None and should_cancel():
            break
        text = (seg.text or "").strip()
        if text:
            start, end = _segment_bounds(seg)
            cues.append(Cue(start=start, end=end, text=text))
        if on_progress and total > 0:
            on_progress(max(0, min(99, int(float(seg.end) / total * 100))))
    if on_progress:
        on_progress(100)
    return cues, detected


def _segment_bounds(segment) -> tuple[float, float]:
    """Cat khoang lang dau/cuoi bang timestamp tu cua faster-whisper."""
    fallback_start = float(getattr(segment, "start", 0.0) or 0.0)
    fallback_end = float(getattr(segment, "end", fallback_start) or fallback_start)
    words = [
        word
        for word in (getattr(segment, "words", None) or ())
        if str(getattr(word, "word", "") or "").strip()
        and getattr(word, "start", None) is not None
        and getattr(word, "end", None) is not None
    ]
    if not words:
        return fallback_start, fallback_end
    start = max(0.0, float(words[0].start))
    end = max(start + 0.05, float(words[-1].end))
    return start, end


def resegment(
    cues: list[Cue],
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    min_duration: float = 0.6,
    max_duration: float = 7.0,
) -> list[Cue]:
    """Chia lai cau cho vua man hinh va gop cac cau qua ngan."""
    limit = max(10, int(max_chars) * max(1, int(max_lines)))
    out: list[Cue] = []
    for cue in cues:
        text = " ".join((cue.text or "").split())
        if not text:
            continue
        if len(text) <= limit:
            out.append(
                Cue(
                    cue.start,
                    cue.end,
                    _wrap(text, max_chars, max_lines),
                    cue.translation,
                    cue.speaker,
                )
            )
            continue
        parts = _split_by_length(text, limit)
        span = max(cue.duration, 0.1)
        cursor = cue.start
        total_chars = sum(len(p) for p in parts) or 1
        for part in parts:
            share = span * (len(part) / total_chars)
            end = min(cue.end, cursor + share)
            out.append(
                Cue(
                    cursor,
                    max(cursor + 0.05, end),
                    _wrap(part, max_chars, max_lines),
                    "",
                    cue.speaker,
                )
            )
            cursor = end
    return _merge_short(out, min_duration=min_duration, max_duration=max_duration, limit=limit)


def _wrap(text: str, max_chars: int, max_lines: int) -> str:
    """Xuong dong sao cho moi dong khong qua `max_chars` ky tu."""
    if max_lines <= 1 or len(text) <= max_chars:
        return text
    if " " not in text:
        return "\n".join(
            text[index : index + max_chars]
            for index in range(0, min(len(text), max_chars * max_lines), max_chars)
        )
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
            if len(lines) == max_lines - 1:
                continue
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines[:max_lines]) if lines else text


def _split_by_length(text: str, limit: int) -> list[str]:
    if " " not in text and len(text) > limit:
        clauses = [part for part in re.split(r"(?<=[。！？!?；;，,、])", text) if part]
        chinese_parts: list[str] = []
        current = ""
        for clause in clauses:
            while len(clause) > limit:
                if current:
                    chinese_parts.append(current)
                    current = ""
                chinese_parts.append(clause[:limit])
                clause = clause[limit:]
            if current and len(current) + len(clause) > limit:
                chinese_parts.append(current)
                current = clause
            else:
                current += clause
        if current:
            chinese_parts.append(current)
        return chinese_parts or [text]
    words = text.split()
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [text]


def _merge_short(
    cues: list[Cue], *, min_duration: float, max_duration: float, limit: int
) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        if (
            out
            and cue.duration < min_duration
            and (out[-1].duration + cue.duration) <= max_duration
            and len(out[-1].text) + len(cue.text) + 1 <= limit
            and cue.start - out[-1].end < 0.4
        ):
            prev = out[-1]
            prev.end = cue.end
            prev.text = f"{prev.text} {cue.text}".strip()
            continue
        out.append(cue)
    return out


def _friendly(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return "Card do hoa het bo nho. Hay chon model nho hon hoac chuyen sang CPU."
    if "connect" in text or "resolve" in text or "network" in text:
        return (
            "Khong tai duoc model vi khong co mang. Hay noi may voi Internet lan dau, "
            "hoac chon thu muc model da tai san trong Cai dat chung."
        )
    if "no such file" in text:
        return "Khong tim thay tep model."
    return f"Nhan dang that bai: {exc}"
