"""Doc van ban thanh giong noi: V2 su dung duy nhat backend offline Piper Local."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..services.paths import config_dir
from .local_voice import (
    TTSError,
    get_catalog,
    list_catalog,
    piper_runtime_ready,
    synthesize_piper,
)

PROVIDER_LOCAL = "Local Voice"
PROVIDER_PIPER = PROVIDER_LOCAL
PROVIDERS = (PROVIDER_LOCAL,)

# Ho tro __getattr__ de cac vi tri cu kiem tra `if tts.PROVIDER_... in available`
# khong bi loi AttributeError ma danh gia ve False an toan truoc khi UI contract cap nhat.
_DEPRECATED_PROVIDERS = {
    "PROVIDER_VOICESTUDIO": "_deprecated_voicestudio",
    "PROVIDER_EDGE": "_deprecated_edge",
    "PROVIDER_SAPI": "_deprecated_sapi",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_PROVIDERS:
        return _DEPRECATED_PROVIDERS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def available_providers() -> list[str]:
    """Danh sach nha cung cap giong doc kha dung trong V2.

    V2 chi su dung duy nhat Piper Local, khong co provider du phong (no fallback).
    """
    return [PROVIDER_LOCAL]


def provider_ready(provider: str = PROVIDER_LOCAL) -> tuple[bool, str]:
    """Kiem tra nha cung cap da san sang hay chua."""
    if provider in (PROVIDER_LOCAL, "Piper Local (Offline)", ""):
        return piper_runtime_ready()
    return False, f"Khong ho tro: {provider}"


def list_voices(provider: str = PROVIDER_LOCAL, language: str = "") -> list[str]:
    """Danh sach giong doc kha dung tu catalog cua Piper Local."""
    if provider in (PROVIDER_LOCAL, "Piper Local (Offline)", ""):
        catalog_voices = list_catalog(language)
        return [v.id for v in catalog_voices]
    return []


def synthesize(
    provider: str,
    text: str,
    out_path: str | Path,
    *,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
    speed: float = 1.0,
    length_scale: float | None = None,
    speaker_id: int | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Tao tep am thanh cho mot cau bang Piper Local.

    Tra ve duong dan tep WAV vua tao.
    """
    if provider not in (PROVIDER_LOCAL, "Piper Local (Offline)", ""):
        raise TTSError(f"Khong ho tro nha cung cap giong doc: {provider}")

    # Neu goi voi rate cu va chua truyen speed tuy bien thi anh xa sang speed
    effective_speed = speed
    if rate != 0 and speed == 1.0 and length_scale is None:
        effective_speed = max(0.5, min(2.0, 1.0 + rate / 100.0))

    # Chon voice mac dinh neu de trong
    target_voice = voice
    if not target_voice:
        catalog = get_catalog()
        target_voice = next(iter(catalog.keys()), "en_US-bryce-medium")

    return synthesize_piper(
        text,
        out_path,
        voice_id=target_voice,
        speed=effective_speed,
        length_scale=length_scale,
        speaker_id=speaker_id,
        on_log=on_log,
    )


def voice_cache_key(
    provider: str,
    text: str,
    *,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
    speed: float = 1.0,
    speaker_id: int | None = None,
) -> str:
    """Khoa cache on dinh, doc lap voi provider (provider-free).

    Thay doi noi dung, giong doc hoac toc do thi tao ma hash moi.
    """
    payload = json.dumps(
        {
            "text": " ".join((text or "").split()),
            "voice": voice,
            "rate": int(rate),
            "speed": round(float(speed), 3),
            "volume": int(volume),
            "speaker_id": speaker_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_dictionary(text: str, specification: str = "") -> str:
    """Thay cach doc theo cac dong ``tu goc=cach doc`` nhu tu dien NTS."""
    result = str(text or "")
    pairs = []
    for line in str(specification or "").splitlines():
        source, separator, replacement = line.partition("=")
        if separator and source.strip():
            pairs.append((source.strip(), replacement.strip() or source.strip()))
    for source, replacement in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        result = re.sub(re.escape(source), replacement, result, flags=re.IGNORECASE)
    return result


def normalize_punctuation(text: str) -> str:
    """Chuan hoa dau cau de TTS khong doc ten ky hieu hoac ngat sai."""
    value = str(text or "").replace("…", "...").replace("；", ";").replace("，", ",")
    value = value.replace("。", ".").replace("！", "!").replace("？", "?")
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])(?=[^\s,.;:!?])", r"\1 ", value)
    return " ".join(value.split())


def default_voice_cache_dir() -> Path:
    """Cache voice dung chung giua cac project."""
    return config_dir().parent / "cache" / "voice"


def synthesize_cached(
    provider: str,
    text: str,
    out_path: str | Path,
    *,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
    speed: float = 1.0,
    length_scale: float | None = None,
    speaker_id: int | None = None,
    cache_dir: str | Path | None = None,
    enabled: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Tao voice mot lan va dung lai theo hash cho nhung lan sau (luon la .wav)."""
    if not enabled:
        return synthesize(
            provider,
            text,
            out_path,
            voice=voice,
            rate=rate,
            volume=volume,
            speed=speed,
            length_scale=length_scale,
            speaker_id=speaker_id,
            on_log=on_log,
        )
    folder = Path(cache_dir) if cache_dir is not None else default_voice_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    key = voice_cache_key(
        provider,
        text,
        voice=voice,
        rate=rate,
        volume=volume,
        speed=speed,
        speaker_id=speaker_id,
    )
    cached = folder / f"{key}.wav"
    if cached.is_file() and cached.stat().st_size >= 64:
        if on_log:
            on_log(f"Dung lai voice cache: {key[:10]}")
        return cached
    produced = synthesize(
        provider,
        text,
        out_path,
        voice=voice,
        rate=rate,
        volume=volume,
        speed=speed,
        length_scale=length_scale,
        speaker_id=speaker_id,
        on_log=on_log,
    )
    tmp = folder / f".{key}.{os.getpid()}.{threading.get_ident()}.wav"
    try:
        shutil.copyfile(produced, tmp)
        tmp.replace(cached)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return cached


def save_voice_cache(path: str | Path, data: dict[str, list[str]]) -> None:
    """Luu danh sach giong doc de lan sau mo nhanh."""
    with contextlib.suppress(OSError):
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_voice_cache(path: str | Path) -> dict[str, list[str]]:
    """Doc danh sach giong doc da cache."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}
