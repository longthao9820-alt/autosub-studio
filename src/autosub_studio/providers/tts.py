"""Doc van ban thanh giong noi: Windows SAPI (offline) va Edge TTS (can mang)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ..services.paths import config_dir

PROVIDER_SAPI = "Windows SAPI (offline)"
PROVIDER_EDGE = "Edge TTS (can Internet)"
PROVIDER_VOICESTUDIO = "VoiceStudio Local (English US)"
PROVIDERS = (PROVIDER_VOICESTUDIO, PROVIDER_EDGE, PROVIDER_SAPI)
VOICESTUDIO_API = "http://127.0.0.1:3900"
KITTEN_VOICES = (
    "Kitten English Male 2|kittentts|expr-voice-2-m",
    "Kitten English Female 2|kittentts|expr-voice-2-f",
    "Kitten English Male 3|kittentts|expr-voice-3-m",
    "Kitten English Female 3|kittentts|expr-voice-3-f",
    "Kitten English Male 4|kittentts|expr-voice-4-m",
    "Kitten English Female 4|kittentts|expr-voice-4-f",
    "Kitten English Male 5|kittentts|expr-voice-5-m",
    "Kitten English Female 5|kittentts|expr-voice-5-f",
)

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_PS_SAFE = re.compile(r"[`$\x00-\x1f]")


class TTSError(RuntimeError):
    """Loi khi tao giong doc."""


def available_providers() -> list[str]:
    # Luon hien VoiceStudio de nguoi dung co the chon roi bam Ket Noi, ke ca
    # khi app chua mo. provider_ready se vo hieu hoa START cho den khi san sang.
    out = [PROVIDER_VOICESTUDIO]
    if os.name == "nt":
        out.append(PROVIDER_SAPI)
    try:
        import edge_tts  # noqa: F401

        out.append(PROVIDER_EDGE)
    except ImportError:
        pass
    return out or [PROVIDER_SAPI]


def provider_ready(provider: str) -> tuple[bool, str]:
    if provider == PROVIDER_VOICESTUDIO:
        return voicestudio_ready()
    if provider == PROVIDER_SAPI:
        if os.name != "nt":
            return False, "Giong SAPI chi co tren Windows."
        return True, ""
    if provider == PROVIDER_EDGE:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False, "Chua cai edge-tts. Chay: pip install edge-tts"
        return True, "Can ket noi Internet."
    return False, f"Khong ho tro: {provider}"


def list_voices(provider: str, language: str = "") -> list[str]:
    """Danh sach giong doc kha dung."""
    if provider == PROVIDER_SAPI:
        return _sapi_voices()
    if provider == PROVIDER_EDGE:
        return _edge_voices(language)
    if provider == PROVIDER_VOICESTUDIO:
        return _voicestudio_voices()
    return []


def synthesize(
    provider: str,
    text: str,
    out_path: str | Path,
    *,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Tao tep am thanh cho mot cau. Tra ve duong dan tep vua tao."""
    clean = " ".join((text or "").split())
    if not clean:
        raise TTSError("Cau khong co noi dung de doc.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if provider == PROVIDER_SAPI:
        return _sapi_speak(clean, out, voice=voice, rate=rate, volume=volume)
    if provider == PROVIDER_EDGE:
        return _edge_speak(clean, out, voice=voice, rate=rate, volume=volume, on_log=on_log)
    if provider == PROVIDER_VOICESTUDIO:
        return _voicestudio_speak(clean, out, voice=voice, rate=rate, on_log=on_log)
    raise TTSError(f"Khong ho tro nha cung cap giong doc: {provider}")


def voice_cache_key(
    provider: str,
    text: str,
    *,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
) -> str:
    """Khoa cache on dinh; doi noi dung/cau hinh giong thi tao file moi."""
    payload = json.dumps(
        {
            "provider": provider,
            "text": " ".join((text or "").split()),
            "voice": voice,
            "rate": int(rate),
            "volume": int(volume),
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
    """Cache voice dung chung giua cac project, giong thu muc audio cua NTS."""
    return config_dir().parent / "cache" / "voice"


def synthesize_cached(
    provider: str,
    text: str,
    out_path: str | Path,
    *,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
    cache_dir: str | Path | None = None,
    enabled: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Tai/tao voice mot lan va dung lai theo hash cho nhung lan sau."""
    if not enabled:
        return synthesize(
            provider,
            text,
            out_path,
            voice=voice,
            rate=rate,
            volume=volume,
            on_log=on_log,
        )
    folder = Path(cache_dir) if cache_dir is not None else default_voice_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".mp3" if provider == PROVIDER_EDGE else ".wav"
    key = voice_cache_key(provider, text, voice=voice, rate=rate, volume=volume)
    cached = folder / f"{key}{suffix}"
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
        on_log=on_log,
    )
    tmp = folder / f".{key}.{os.getpid()}.{threading.get_ident()}{suffix}"
    try:
        shutil.copyfile(produced, tmp)
        tmp.replace(cached)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return cached


# --------------------------------------------------------------------------- VoiceStudio local API


def _json_request(path: str, *, timeout: float = 10.0) -> object:
    request = urllib.request.Request(VOICESTUDIO_API + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def voicestudio_ready() -> tuple[bool, str]:
    try:
        data = _json_request("/health", timeout=1.5)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False, "Hãy mở VoiceStudio trước; Toolsub sẽ kết nối local port 3900."
    if not isinstance(data, dict) or data.get("status") != "ok":
        return False, "VoiceStudio chưa sẵn sàng."
    device = str(data.get("device") or "local")
    return True, f"VoiceStudio local: {device}."


def start_voicestudio(timeout: float = 20.0) -> tuple[bool, str]:
    """Mo VoiceStudio da cai san va doi API local san sang."""
    ready = voicestudio_ready()
    if ready[0]:
        return ready
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "VoiceStudio"
        / "omnivoice-studio.exe",
        Path.home()
        / "AppData"
        / "Local"
        / "Programs"
        / "VoiceStudio"
        / "omnivoice-studio.exe",
    ]
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        return False, "Chưa cài VoiceStudio trên máy này."
    try:
        subprocess.Popen(
            [str(executable)],
            cwd=str(executable.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        return False, f"Không mở được VoiceStudio: {exc}"
    deadline = time.monotonic() + max(1.0, min(30.0, float(timeout)))
    while time.monotonic() < deadline:
        ready = voicestudio_ready()
        if ready[0]:
            return ready
        time.sleep(0.5)
    return False, "VoiceStudio đã mở nhưng API local chưa sẵn sàng; thử lại sau vài giây."


def _parse_vs_voice(value: str) -> tuple[str, str]:
    parts = str(value or "").split("|")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    return "omnivoice", str(value or "demo0001")


def _voicestudio_voices() -> list[str]:
    voices = list(KITTEN_VOICES)
    try:
        data = _json_request("/v1/audio/voices", timeout=5)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return voices
    if not isinstance(data, dict):
        return voices
    for item in data.get("voices") or []:
        if not isinstance(item, dict) or item.get("type") != "profile":
            continue
        language = str(item.get("language") or "").casefold()
        if language and not language.startswith("en") and "english" not in language:
            continue
        voice_id = str(item.get("voice_id") or "")
        name = str(item.get("name") or voice_id)
        if voice_id:
            voices.append(f"{name}|omnivoice|{voice_id}")
    return voices


def _voicestudio_speak(
    text: str,
    out: Path,
    *,
    voice: str,
    rate: int,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    ready, reason = voicestudio_ready()
    if not ready:
        raise TTSError(reason)
    engine, voice_id = _parse_vs_voice(voice or KITTEN_VOICES[0])
    speed = max(0.25, min(4.0, 1.0 + int(rate) / 100.0))
    payload = json.dumps(
        {
            "model": engine,
            "input": text,
            "voice": voice_id,
            "response_format": "wav",
            "speed": speed,
            "language": "en",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        VOICESTUDIO_API + "/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise TTSError(f"VoiceStudio báo lỗi {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise TTSError(f"Không gọi được VoiceStudio local: {exc}") from exc
    if len(audio) < 64:
        raise TTSError("VoiceStudio không trả về audio hợp lệ.")
    target = out.with_suffix(".wav")
    target.write_bytes(audio)
    if on_log:
        on_log(f"VoiceStudio: {engine} / {voice_id}")
    return target


# --------------------------------------------------------------------------- SAPI


def _run_powershell(script: str, timeout: float = 120) -> str:
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise TTSError("Khong tim thay PowerShell de goi giong doc Windows.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TTSError("Tao giong doc qua lau, da dung lai.") from exc
    if res.returncode != 0:
        raise TTSError(f"Giong doc Windows bao loi: {(res.stderr or '').strip()[:300]}")
    return res.stdout or ""


def _sapi_voices() -> list[str]:
    if os.name != "nt":
        return []
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
    )
    try:
        out = _run_powershell(script, timeout=30)
    except TTSError:
        return []
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip()
        and any(tag in line.casefold() for tag in ("david", "zira", "mark", "english", "us"))
    ]


def _sapi_speak(text: str, out: Path, *, voice: str, rate: int, volume: int) -> Path:
    safe_voice = _PS_SAFE.sub("", voice or "").replace("'", "''")
    tmp = Path(tempfile.gettempdir()) / f"autosub_tts_{os.getpid()}_{abs(hash(text)) % 99999}.txt"
    tmp.write_text(text, encoding="utf-8")
    target = str(out.resolve()).replace("'", "''")
    select = f"$s.SelectVoice('{safe_voice}'); " if safe_voice else ""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"{select}"
        f"$s.Rate = {max(-10, min(10, int(rate)))}; "
        f"$s.Volume = {max(0, min(100, int(volume)))}; "
        f"$s.SetOutputToWaveFile('{target}'); "
        f"$t = Get-Content -LiteralPath '{str(tmp).replace(chr(39), chr(39) * 2)}' "
        "-Raw -Encoding UTF8; "
        "$s.Speak($t); $s.Dispose()"
    )
    try:
        _run_powershell(script)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    if not out.is_file() or out.stat().st_size < 64:
        raise TTSError("Giong doc Windows khong tao duoc tep am thanh.")
    return out


# --------------------------------------------------------------------------- Edge TTS


def _edge_voices(language: str = "") -> list[str]:
    try:
        import edge_tts
    except ImportError:
        return []
    try:
        voices = asyncio.run(edge_tts.list_voices())
    except Exception:
        return []
    names = []
    for v in voices:
        name = str(v.get("ShortName", ""))
        locale = str(v.get("Locale", ""))
        if not name:
            continue
        # Toolsub nay duoc cau hinh rieng cho English (US), khong hien cac
        # giong dia phuong/da ngon ngu khac lam roi danh sach.
        if locale.casefold() != "en-us":
            continue
        names.append(name)
    return sorted(names)


def _edge_speak(
    text: str,
    out: Path,
    *,
    voice: str,
    rate: int,
    volume: int,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    try:
        import edge_tts
    except ImportError as exc:
        raise TTSError("Chua cai edge-tts. Chay: pip install edge-tts") from exc
    voice = voice or "vi-VN-NamMinhNeural"
    rate_str = f"{'+' if rate >= 0 else '-'}{abs(int(rate))}%"
    vol = max(0, min(100, int(volume))) - 100
    volume_str = f"{'+' if vol >= 0 else '-'}{abs(vol)}%"
    mp3 = out.with_suffix(".mp3")

    async def _run() -> None:
        last: Exception | None = None
        for attempt in range(3):
            with contextlib.suppress(OSError):
                mp3.unlink(missing_ok=True)
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate_str, volume=volume_str)
                await comm.save(str(mp3))
                if mp3.is_file() and mp3.stat().st_size >= 64:
                    return
            except Exception as exc:
                last = exc
            if attempt < 2:
                if on_log:
                    on_log(f"Edge TTS thu lai cau doc ({attempt + 2}/3)...")
                await asyncio.sleep(0.4 * (attempt + 1))
        if last is not None:
            raise last
        raise TTSError("Edge TTS khong tra ve du lieu am thanh sau 3 lan thu.")

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise TTSError(
            f"Goi Edge TTS that bai. Kiem tra ket noi mang hoac doi giong doc khac. Chi tiet: {exc}"
        ) from exc
    if not mp3.is_file() or mp3.stat().st_size < 64:
        raise TTSError("Edge TTS khong tra ve am thanh.")
    if on_log:
        on_log(f"Edge TTS: {voice}")
    return mp3


def save_voice_cache(path: str | Path, data: dict[str, list[str]]) -> None:
    """Luu danh sach giong doc de lan sau mo nhanh."""
    with contextlib.suppress(OSError):
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_voice_cache(path: str | Path) -> dict[str, list[str]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}
