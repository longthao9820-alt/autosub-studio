"""Phan tach nguoi noi theo cao do giong (nam / nu), chay hoan toan tren may."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

from ..core.models import Cue

# Nguong tan so co ban de phan biet giong nam va giong nu.
PITCH_SPLIT_HZ = 165.0
MIN_PITCH_HZ = 70.0
MAX_PITCH_HZ = 350.0

LABEL_MALE = "Nam"
LABEL_FEMALE = "Nu"
LABEL_UNKNOWN = "?"


class DiarizeError(RuntimeError):
    """Phan tach nguoi noi that bai."""


def estimate_pitch(wav_path: str | Path, max_seconds: float = 4.0) -> float:
    """Uoc luong tan so co ban (Hz) cua mot doan WAV mono 16 bit. 0 = khong ro."""
    try:
        import numpy as np
    except ImportError as exc:
        raise DiarizeError("Thieu thu vien numpy de phan tich giong noi.") from exc
    p = Path(wav_path)
    if not p.is_file():
        return 0.0
    try:
        with wave.open(str(p), "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            channels = wf.getnchannels()
            frames = wf.readframes(int(min(max_seconds, 30.0) * rate))
    except (wave.Error, OSError):
        return 0.0
    if width != 2 or not frames or rate <= 0:
        return 0.0
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if data.size < rate // 10:
        return 0.0
    data = data - float(data.mean())
    peak = float(np.abs(data).max())
    if peak < 200:  # gan nhu im lang
        return 0.0
    data = data / peak

    # Tu tuong quan de tim chu ky lap lai cua song am.
    corr = np.correlate(data, data, mode="full")[data.size - 1 :]
    lo = int(rate / MAX_PITCH_HZ)
    hi = int(rate / MIN_PITCH_HZ)
    if hi <= lo or hi >= corr.size:
        return 0.0
    window = corr[lo:hi]
    if window.size == 0 or float(window.max()) <= 0:
        return 0.0
    lag = int(window.argmax()) + lo
    if lag <= 0:
        return 0.0
    pitch = rate / lag
    if not (MIN_PITCH_HZ <= pitch <= MAX_PITCH_HZ):
        return 0.0
    return float(pitch)


def label_from_pitch(pitch: float) -> str:
    if pitch <= 0:
        return LABEL_UNKNOWN
    return LABEL_MALE if pitch < PITCH_SPLIT_HZ else LABEL_FEMALE


def assign_speakers(
    cues: list[Cue],
    segment_paths: dict[int, Path],
    *,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Gan nhan nguoi noi cho tung cau dua tren cao do. Tra ve so cau da gan."""
    total = len(cues)
    done = 0
    for index, cue in enumerate(cues):
        if should_cancel is not None and should_cancel():
            break
        path = segment_paths.get(index)
        if path is None:
            continue
        pitch = estimate_pitch(path)
        label = label_from_pitch(pitch)
        if label != LABEL_UNKNOWN:
            cue.speaker = label
            done += 1
        if on_log and pitch > 0:
            on_log(f"Cau {index + 1}: {pitch:.0f} Hz -> {label}")
        if on_progress and total:
            on_progress(max(0, min(99, int((index + 1) / total * 100))))
    if on_progress:
        on_progress(100)
    return done
