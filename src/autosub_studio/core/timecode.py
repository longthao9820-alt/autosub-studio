"""Doc va ghi moc thoi gian cho cac dinh dang phu de."""

from __future__ import annotations

import re

_SRT_RE = re.compile(
    r"^\s*(?:(\d+):)?(\d{1,3}):(\d{1,2})(?:[.,](\d{1,3}))?\s*$",
)


class TimecodeError(ValueError):
    """Chuoi thoi gian khong hop le."""


def parse_timecode(text: str) -> float:
    """Doc chuoi `hh:mm:ss,mmm` / `hh:mm:ss.mmm` / `mm:ss` / `h:mm:ss.cc` ra giay.

    Chap nhan dau phay hoac dau cham lam dau thap phan, cho phep thieu phan gio.
    """
    if text is None:
        raise TimecodeError("Chuoi thoi gian rong")
    raw = str(text).strip()
    if not raw:
        raise TimecodeError("Chuoi thoi gian rong")
    m = _SRT_RE.match(raw)
    if not m:
        # Cho phep dang chi co so giay, vi du "12.5"
        try:
            value = float(raw.replace(",", "."))
        except ValueError as exc:
            raise TimecodeError(f"Khong doc duoc moc thoi gian: {text!r}") from exc
        if value < 0:
            raise TimecodeError(f"Moc thoi gian am: {text!r}")
        return value
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    frac_raw = m.group(4) or "0"
    # ASS dung 2 chu so (centisecond), SRT dung 3 chu so (millisecond).
    frac = int(frac_raw) / (10 ** len(frac_raw))
    if minutes > 59 or seconds > 59:
        raise TimecodeError(f"Phut/giay vuot qua 59: {text!r}")
    return hours * 3600 + minutes * 60 + seconds + frac


def format_srt(seconds: float) -> str:
    """Ghi giay ra `hh:mm:ss,mmm` dung chuan SRT."""
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt(seconds: float) -> str:
    """Ghi giay ra `hh:mm:ss.mmm` dung chuan WebVTT."""
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_ass(seconds: float) -> str:
    """Ghi giay ra `h:mm:ss.cc` dung chuan ASS."""
    h, m, s, ms = _split(seconds)
    return f"{h:d}:{m:02d}:{s:02d}.{ms // 10:02d}"


def format_display(seconds: float) -> str:
    """Chuoi hien thi tren giao dien, giong SRT."""
    return format_srt(seconds)


def _split(seconds: float) -> tuple[int, int, int, int]:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return h, m, s, ms
