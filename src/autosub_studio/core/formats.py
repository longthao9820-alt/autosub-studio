"""Doc va ghi cac dinh dang phu de SRT, WebVTT va ASS."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Cue, SubtitleDoc
from .timecode import TimecodeError, format_ass, format_srt, format_vtt, parse_timecode

SUPPORTED_SUFFIXES = (".srt", ".vtt", ".ass", ".ssa", ".txt")

_ARROW = re.compile(r"-->")
_TAG = re.compile(r"\{[^{}]*\}")
_HTML_TAG = re.compile(r"</?(?:b|i|u|font|ruby|rt|c|v)(?:\s[^>]*)?>", re.IGNORECASE)
_VTT_CUE_SETTING = re.compile(r"\s+(?:line|position|size|align|region|vertical):\S+")
_CJK = re.compile(r"[\u3400-\u9fff]")


@dataclass
class ParseResult:
    """Ket qua doc mot tep phu de, kem canh bao de hien thi cho nguoi dung."""

    doc: SubtitleDoc
    warnings: list[str] = field(default_factory=list)


class SubtitleFormatError(ValueError):
    """Tep phu de khong doc duoc."""


def read_text(path: str | Path) -> str:
    """Doc noi dung tep voi nhieu bang ma pho bien, uu tien UTF-8."""
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # SRT Trung Quoc tren Windows rat thuong la GBK/GB18030 hoac Big5. Cac
    # bang ma mot byte nhu cp1258/latin-1 giai ma duoc moi day byte nen neu
    # thu chung truoc se tao chu rac ma khong bao loi. Charset Normalizer
    # danh gia do "sach" va tinh nhat quan ngon ngu de chon bang ma phu hop.
    try:
        from charset_normalizer import from_bytes

        match = from_bytes(data).best()
        if match is not None:
            encoding = str(getattr(match, "encoding", "") or "").lower()
            if encoding in {"big5", "big5hkscs", "cp950", "gb18030", "gbk", "gb2312"}:
                chinese = _decode_legacy_chinese(data)
                if chinese is not None:
                    return chinese
            decoded = str(match)
            if decoded:
                return decoded
    except (ImportError, UnicodeError, ValueError):
        pass
    for encoding in ("gb18030", "big5", "shift_jis", "cp949", "cp1258", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _decode_legacy_chinese(data: bytes) -> str | None:
    """Phan biet GB18030 va Big5, uu tien gian the khi hai bang ma deu doc duoc."""
    candidates: list[tuple[int, int, str, str]] = []
    for priority, encoding in enumerate(("gb18030", "big5")):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        cjk_count = len(_CJK.findall(text))
        if not cjk_count:
            continue
        suspicious = sum(
            1
            for char in text
            if "\ue000" <= char <= "\uf8ff"
            or "\u3040" <= char <= "\u30ff"
            or "\u3100" <= char <= "\u312f"
        )
        candidates.append((suspicious, priority, encoding, text))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[3]


def load_subtitle(path: str | Path) -> ParseResult:
    """Doc tep phu de, tu nhan dang dinh dang theo duoi tep va noi dung."""
    p = Path(path)
    if not p.is_file():
        raise SubtitleFormatError(f"Khong tim thay tep: {p}")
    content = read_text(p)
    suffix = p.suffix.lower()
    if suffix in (".ass", ".ssa") or "[Events]" in content:
        return parse_ass(content)
    if suffix == ".vtt" or content.lstrip().upper().startswith("WEBVTT"):
        return parse_vtt(content)
    if suffix == ".txt" and "-->" not in content:
        return parse_plain_text(content)
    return parse_srt(content)


def save_subtitle(path: str | Path, doc: SubtitleDoc, text_mode: str = "original", **kw) -> Path:
    """Ghi tai lieu phu de ra tep, chon dinh dang theo duoi tep."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".ass", ".ssa"):
        content = write_ass(doc, text_mode=text_mode, **kw)
    elif suffix == ".vtt":
        content = write_vtt(doc, text_mode=text_mode)
    else:
        content = write_srt(doc, text_mode=text_mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)
    return p


# --------------------------------------------------------------------------- doc


def _clean(text: str) -> str:
    text = _TAG.sub("", text)
    text = _HTML_TAG.sub("", text)
    return text.replace("\\N", "\n").replace("\\n", "\n").strip()


def parse_srt(content: str) -> ParseResult:
    """Doc SRT. Bo qua khoi hong nhung van giu cac khoi con lai."""
    warnings: list[str] = []
    cues: list[Cue] = []
    blocks = re.split(r"\r?\n\s*\r?\n", content.replace("﻿", ""))
    for block_no, block in enumerate(blocks, start=1):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        time_idx = next((i for i, ln in enumerate(lines) if _ARROW.search(ln)), -1)
        if time_idx < 0:
            warnings.append(f"Khoi {block_no}: khong co dong thoi gian, da bo qua.")
            continue
        try:
            start, end = _parse_time_line(lines[time_idx])
        except TimecodeError as exc:
            warnings.append(f"Khoi {block_no}: {exc}")
            continue
        text = _clean("\n".join(lines[time_idx + 1 :]))
        cues.append(Cue(start=start, end=end, text=text))
    doc = SubtitleDoc(cues=cues)
    doc.sort()
    return ParseResult(doc=doc, warnings=warnings)


def parse_vtt(content: str) -> ParseResult:
    """Doc WebVTT, bo qua khoi NOTE/STYLE/REGION."""
    warnings: list[str] = []
    cues: list[Cue] = []
    body = re.sub(r"^\s*WEBVTT[^\n]*\n", "", content.replace("﻿", ""), count=1)
    blocks = re.split(r"\r?\n\s*\r?\n", body)
    for block_no, block in enumerate(blocks, start=1):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        head = lines[0].strip().upper()
        if head.startswith(("NOTE", "STYLE", "REGION")):
            continue
        time_idx = next((i for i, ln in enumerate(lines) if _ARROW.search(ln)), -1)
        if time_idx < 0:
            continue
        clean_line = _VTT_CUE_SETTING.sub("", lines[time_idx])
        try:
            start, end = _parse_time_line(clean_line)
        except TimecodeError as exc:
            warnings.append(f"Khoi {block_no}: {exc}")
            continue
        text = _clean("\n".join(lines[time_idx + 1 :]))
        cues.append(Cue(start=start, end=end, text=text))
    doc = SubtitleDoc(cues=cues)
    doc.sort()
    return ParseResult(doc=doc, warnings=warnings)


def parse_ass(content: str) -> ParseResult:
    """Doc ASS/SSA theo dong Format: trong muc [Events]."""
    warnings: list[str] = []
    cues: list[Cue] = []
    fields = [
        "Layer",
        "Start",
        "End",
        "Style",
        "Name",
        "MarginL",
        "MarginR",
        "MarginV",
        "Effect",
        "Text",
    ]
    in_events = False
    for line_no, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if line.startswith("["):
            in_events = line.lower().startswith("[events")
            continue
        if not in_events or not line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        if key == "format":
            fields = [f.strip() for f in rest.split(",")]
            continue
        if key not in ("dialogue", "comment"):
            continue
        if key == "comment":
            continue
        parts = rest.split(",", len(fields) - 1)
        if len(parts) < len(fields):
            warnings.append(f"Dong {line_no}: thieu cot, da bo qua.")
            continue
        row = dict(zip(fields, parts, strict=False))
        try:
            start = parse_timecode(row.get("Start", ""))
            end = parse_timecode(row.get("End", ""))
        except TimecodeError as exc:
            warnings.append(f"Dong {line_no}: {exc}")
            continue
        speaker = (row.get("Name") or "").strip()
        cues.append(Cue(start=start, end=end, text=_clean(row.get("Text", "")), speaker=speaker))
    doc = SubtitleDoc(cues=cues)
    doc.sort()
    return ParseResult(doc=doc, warnings=warnings)


def parse_plain_text(content: str, seconds_per_line: float = 3.0) -> ParseResult:
    """Doc van ban thuan: moi dong thanh mot cau, chia deu thoi gian."""
    cues: list[Cue] = []
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    cursor = 0.0
    for line in lines:
        cues.append(Cue(start=cursor, end=cursor + seconds_per_line, text=line))
        cursor += seconds_per_line
    warn = ["Tep van ban khong co moc thoi gian, da tam chia deu 3 giay moi dong."] if cues else []
    return ParseResult(doc=SubtitleDoc(cues=cues), warnings=warn)


def _parse_time_line(line: str) -> tuple[float, float]:
    left, _, right = line.partition("-->")
    if not right:
        raise TimecodeError("Dong thoi gian thieu dau -->")
    start = parse_timecode(left.strip())
    end = parse_timecode(right.strip().split()[0] if right.strip() else "")
    return start, end


# --------------------------------------------------------------------------- ghi


def write_srt(doc: SubtitleDoc, text_mode: str = "original") -> str:
    out: list[str] = []
    n = 0
    for cue in doc.cues:
        text = cue.display_text(text_mode)
        if not text:
            continue
        n += 1
        out.append(str(n))
        out.append(f"{format_srt(cue.start)} --> {format_srt(cue.end)}")
        out.append(text)
        out.append("")
    return "\n".join(out) + ("\n" if out else "")


def write_vtt(doc: SubtitleDoc, text_mode: str = "original") -> str:
    out: list[str] = ["WEBVTT", ""]
    for cue in doc.cues:
        text = cue.display_text(text_mode)
        if not text:
            continue
        out.append(f"{format_vtt(cue.start)} --> {format_vtt(cue.end)}")
        out.append(text)
        out.append("")
    return "\n".join(out) + "\n"


def write_ass(
    doc: SubtitleDoc,
    text_mode: str = "original",
    *,
    font: str = "Arial",
    font_size: int = 48,
    primary_color: str = "&H00FFFFFF",
    outline_color: str = "&H00000000",
    back_color: str = "&H80000000",
    bold: bool = True,
    outline: float = 2.0,
    shadow: float = 1.0,
    alignment: int = 2,
    margin_v: int = 40,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
) -> str:
    """Ghi ASS kem mot Style duy nhat dung cho render."""
    header = [
        "[Script Info]",
        "; Tao boi AutoSub Studio",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{font_size},{primary_color},&H000000FF,{outline_color},"
        f"{back_color},{-1 if bold else 0},0,0,0,100,100,0,0,1,{outline},{shadow},"
        f"{alignment},40,40,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in doc.cues:
        text = cue.display_text(text_mode)
        if not text:
            continue
        body = text.replace("\n", "\\N").replace(",", "‚") if False else text.replace("\n", "\\N")
        name = cue.speaker.replace(",", " ")
        header.append(
            f"Dialogue: 0,{format_ass(cue.start)},{format_ass(cue.end)},Default,{name},"
            f"0,0,0,,{body}"
        )
    return "\n".join(header) + "\n"
