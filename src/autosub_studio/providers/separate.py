"""Tach nhac nen va loi thoai. Mac dinh dung FFmpeg, co the dung Demucs neu da cai."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..services.ffmpeg import CREATE_NO_WINDOW, CancelToken, FFmpeg
from ..services.media import isolate_voice, remove_vocals

MODE_FFMPEG = "Co ban (FFmpeg, nhanh)"
MODE_DEMUCS = "Chat luong cao (Demucs, cham)"
MODES = (MODE_FFMPEG, MODE_DEMUCS)


class SeparationError(RuntimeError):
    """Tach am thanh that bai."""


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
    except ImportError:
        return False
    return True


def install_hint() -> str:
    return (
        "Chua cai Demucs. Chay: pip install demucs (tai ve khoang 2 GB, "
        "can card do hoa de chay nhanh)."
    )


def separate(
    ff: FFmpeg,
    src: str | Path,
    out_dir: str | Path,
    *,
    mode: str = MODE_FFMPEG,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    """Tach thanh (duong dan giong noi, duong dan nhac nen)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    voice = out / "voice.wav"
    music = out / "music.wav"

    if mode == MODE_DEMUCS and demucs_available():
        return _separate_demucs(ff, src, out, token=token, on_log=on_log, on_progress=on_progress)

    if on_log:
        on_log("Tach bang FFmpeg (khu kenh giua va loc dai tan giong noi).")
    isolate_voice(
        ff,
        src,
        voice,
        duration=duration,
        token=token,
        on_progress=lambda p: on_progress(int(p * 0.5)) if on_progress else None,
    )
    remove_vocals(
        ff,
        src,
        music,
        duration=duration,
        token=token,
        on_progress=lambda p: on_progress(50 + int(p * 0.5)) if on_progress else None,
    )
    return voice, music


def _separate_demucs(
    ff: FFmpeg,
    src: str | Path,
    out: Path,
    *,
    token: CancelToken | None,
    on_log: Callable[[str], None] | None,
    on_progress: Callable[[int], None] | None,
) -> tuple[Path, Path]:
    work = out / "demucs"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", str(work), str(src)]
    if on_log:
        on_log("Chay Demucs, buoc nay co the mat vai phut...")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise SeparationError(f"Khong chay duoc Demucs: {exc}") from exc
    assert proc.stdout is not None
    for line in proc.stdout:
        if token is not None and token.cancelled:
            proc.kill()
            raise SeparationError("Nguoi dung da huy.")
        line = line.strip()
        if line and on_log:
            on_log(line)
    proc.wait()
    if proc.returncode != 0:
        raise SeparationError("Demucs ket thuc voi loi. Xem log de biet chi tiet.")

    vocals = next(work.rglob("vocals.*"), None)
    other = next(work.rglob("no_vocals.*"), None)
    if vocals is None or other is None:
        raise SeparationError("Demucs khong tao ra tep ket qua nhu mong doi.")
    voice = out / "voice.wav"
    music = out / "music.wav"
    ff.run(
        ["-i", str(vocals), "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(voice)],
        token=token,
    )
    ff.run(
        ["-i", str(other), "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", str(music)],
        token=token,
    )
    if on_progress:
        on_progress(100)
    return voice, music


def gpu_available() -> bool:
    """Doan xem may co card NVIDIA dung duoc khong (chi de goi y cho nguoi dung)."""
    if os.name != "nt":
        return False
    try:
        res = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            creationflags=CREATE_NO_WINDOW,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0 and bool((res.stdout or "").strip())
