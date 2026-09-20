"""Cac kieu du lieu va ham tien ich dung chung cho ca OCR noi bo va OCR AI.

Chua Row, Rect, timeline merge, dedupe, static text detection va cac ham xu ly
chuoi / toa do doc lap voi engine nhan dang. Khong phu thuoc RapidOCR hay GPU.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

from .models import Cue

_rapid_ratio: Any = None
try:
    from rapidfuzz.fuzz import ratio as _rapid_ratio_impl

    _rapid_ratio = _rapid_ratio_impl
except ImportError:
    pass

Rect = tuple[float, float, float, float]
SPECK_CHARS = 12


class Row(NamedTuple):
    """Mot vung chu doc duoc tren khung hinh."""

    order: tuple[float, float]
    text: str
    score: float
    rect: Rect | None


def _text_ratio(a: str, b: str) -> float:
    """Do gan nhau cua chu; uu tien RapidFuzz C++ va co duong lui an toan."""
    if _rapid_ratio is not None:
        return float(_rapid_ratio(a, b)) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _similar(a: str, b: str, threshold: float) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return _text_ratio(a, b) >= threshold


def _same_caption_version(a: str, b: str, threshold: float) -> bool:
    """Hai ban doc lien tiep co phai ban ngan/day du cua cung mot cau khong."""
    if _similar(a, b, threshold):
        return True
    left = "".join(a.split())
    right = "".join(b.split())
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return len(short) >= 3 and len(short) * 2 >= len(long) and short in long


def _better_read(text: str, score: float, current: str, current_score: float) -> bool:
    """Ban doc nao dang tin hon: uu tien do tin cay, ngang nhau thi lay ban day du hon."""
    if not current:
        return True
    if abs(score - current_score) > 0.015:
        return score > current_score
    return len(text) > len(current)


def is_ignored(text: str, ignore: Iterable[str]) -> bool:
    """Dong chu nay co nam trong danh sach can bo khong."""
    if not text:
        return False
    key = text.casefold()
    for item in ignore:
        if key == item:
            return True
        if abs(len(key) - len(item)) > max(4, len(item) // 2):
            continue
        if _similar(key, item, 0.85):
            return True
    return False


def _is_cjk(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _separator(left: str, right: str) -> str:
    """Tieng Trung khong chen dau cach giua cac cum OCR cung mot dong."""
    if not left or not right:
        return ""
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return ""
    if right[0] in "，。！？；：、,.!?;:)]}》」』”’":
        return ""
    return " "


def _trim_repeat(left: str, right: str, limit: int = 12) -> str:
    """Bo phan chu bi doc hai lan o cho hai vung chong len nhau."""
    top = min(len(left), len(right), limit)
    for k in range(top, 0, -1):
        if left[-k:].casefold() == right[:k].casefold():
            return right[k:]
    return right


def _area(rect: Rect | None) -> float:
    if rect is None:
        return 0.0
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _covered(small: Rect | None, big: Rect | None) -> float:
    """Phan tram dien tich cua vung nho nam long trong vung lon."""
    if small is None or big is None:
        return 0.0
    sx1, sy1, sx2, sy2 = small
    bx1, by1, bx2, by2 = big
    area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
    if area <= 0:
        return 0.0
    over_w = max(0.0, min(sx2, bx2) - max(sx1, bx1))
    over_h = max(0.0, min(sy2, by2) - max(sy1, by1))
    return (over_w * over_h) / area


def _box_order(row: Any) -> tuple[float, float]:
    """Sap cac dong chu doc duoc theo tren xuong duoi, trai sang phai."""
    try:
        points = row[0]
        ys = [float(p[1]) for p in points]
        xs = [float(p[0]) for p in points]
    except (IndexError, TypeError, ValueError):
        return (0.0, 0.0)
    return (min(ys), min(xs))


def _box_rect(row: Any) -> Rect | None:
    """Hinh chu nhat bao quanh mot vung chu doc duoc: (trai, tren, phai, duoi)."""
    try:
        points = row[0]
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (IndexError, TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_row(row: Any, min_confidence: float) -> Row | None:
    """Doi mot dong ket qua tho cua bo doc chu thanh Row, bo dong kem tin cay."""
    try:
        text = str(row[1]).strip()
        score = float(row[2]) if len(row) > 2 else 1.0
    except (IndexError, TypeError, ValueError):
        return None
    if not text or score < min_confidence:
        return None
    return Row(_box_order(row), text, score, _box_rect(row))


def _same_line(a: Rect | None, b: Rect | None) -> bool:
    """Hai vung chu co nam tren cung mot dong khong."""
    if a is None or b is None:
        return False
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return shorter > 0 and (bottom - top) / shorter > 0.5


def _reading_order(rows: list[Row]) -> list[Row]:
    """Sap dung trai-sang-phai trong cung dong, roi moi tren-xuong-duoi."""
    placed: list[list[Row]] = []
    loose: list[Row] = []
    for row in sorted(rows, key=lambda item: item.order):
        if row.rect is None:
            loose.append(row)
            continue
        for line in placed:
            if any(_same_line(row.rect, item.rect) for item in line):
                line.append(row)
                break
        else:
            placed.append([row])
    placed.sort(key=lambda line: min(item.rect[1] for item in line if item.rect is not None))
    ordered: list[Row] = []
    for line in placed:
        ordered.extend(
            sorted(line, key=lambda item: item.rect[0] if item.rect is not None else item.order[1])
        )
    ordered.extend(sorted(loose, key=lambda item: item.order))
    return ordered


def _overlap_x(a: Rect | None, b: Rect | None) -> float:
    if a is None or b is None:
        return 0.0
    return min(a[2], b[2]) - max(a[0], b[0])


def _join_parts(parts: list[tuple[object, str]]) -> str:
    """Ghep cac vung chu thanh cau, xu ly cho bo doc chu cat dong lam doi."""
    text = ""
    previous = None
    for rect_obj, piece in parts:
        rect = rect_obj if (isinstance(rect_obj, tuple) and len(rect_obj) == 4) else None
        if not piece:
            continue
        if not text:
            text, previous = piece, rect
            continue
        same_line = _same_line(previous, rect)
        if same_line and text.endswith(("-", "—", "_")) and piece and _is_cjk(piece[0]):
            text = text[:-1]
        if same_line and _overlap_x(previous, rect) > 0:
            text += _trim_repeat(text, piece)
        else:
            text += _separator(text, piece) + piece
        previous = rect
    return text.strip()


def join_rows(rows: list[Row]) -> tuple[str, float]:
    """Ghep cac vung chu tren mot khung hinh thanh mot cau kem do tin cay."""
    kept = [row for row in rows if row.text]
    if not kept:
        return "", 0.0
    kept = _reading_order(kept)
    text = _join_parts([(row.rect, row.text) for row in kept])
    mean = sum(row.score for row in kept) / len(kept)
    return text, mean


def _drop_overlaps(rows: list[Row], overlap: float = 0.7) -> list[Row]:
    """Bo cac vung chu nam long trong vung khac de khong doc mot chu hai lan."""
    order = sorted(
        range(len(rows)),
        key=lambda i: -_area(rows[i].rect),
    )
    kept: list[Row] = []
    for i in order:
        rect = rows[i].rect
        if any(_covered(rect, other.rect) >= overlap for other in kept):
            continue
        kept.append(rows[i])
    return kept


def _drop_flicker(reads: list[tuple[str, float]], similarity: float) -> list[tuple[str, float]]:
    """Bo cai nhay mot khung: chu la chi lo ra dung mot khung roi mat ngay."""
    if len(reads) < 3:
        return reads
    out = list(reads)
    for i in range(1, len(reads) - 1):
        before, middle, after = reads[i - 1], out[i], reads[i + 1]
        if not middle[0]:
            continue
        if _similar(middle[0], before[0], similarity) or _similar(middle[0], after[0], similarity):
            continue
        if before[0] and after[0] and _similar(before[0], after[0], similarity):
            out[i] = before if _better_read(before[0], before[1], after[0], after[1]) else after
            continue
        longest = max(len(before[0]), len(after[0]))
        if middle[0] and len(middle[0]) < SPECK_CHARS and len(middle[0]) * 2 < longest:
            out[i] = ("", 0.0)
    return out


def _stabilize_reads(
    reads: list[tuple[str, float]], similarity: float, votes: int = 3
) -> list[tuple[str, float]]:
    """Chon ban chu duoc nhieu khung gan nhau ung ho de giam loi tung hinh."""
    need = max(1, min(5, int(votes)))
    if need <= 1 or len(reads) < need:
        return reads
    radius = max(1, need - 1)
    out = list(reads)
    for i, current in enumerate(reads):
        if not current[0]:
            continue
        nearby = [
            item
            for item in reads[max(0, i - radius) : min(len(reads), i + radius + 1)]
            if item[0]
        ]
        best_group: list[tuple[str, float]] = []
        for candidate in nearby:
            group = [item for item in nearby if _similar(item[0], candidate[0], similarity)]
            if len(group) > len(best_group) or (
                len(group) == len(best_group)
                and group
                and sum(s for _t, s in group) > sum(s for _t, s in best_group)
            ):
                best_group = group
        if len(best_group) < need:
            continue
        chosen = best_group[0]
        for candidate in best_group[1:]:
            if _better_read(candidate[0], candidate[1], chosen[0], chosen[1]):
                chosen = candidate
        out[i] = chosen
    return out


def _best_caption(
    reads: Sequence[tuple[str, float]], similarity: float
) -> tuple[str, float]:
    """Chon ban chu duoc nhieu khung hinh ung ho nhat."""
    items = [(text, float(score)) for text, score in reads if text]
    if not items:
        return "", 0.0
    exact: dict[str, list[float]] = {}
    for text, score in items:
        exact.setdefault(text, []).append(score)

    best_text = items[0][0]
    best_key: tuple[float, ...] | None = None
    for text, scores in exact.items():
        ratios = [_text_ratio(text, other) for other, _s in items]
        fuzzy = sum(1 for ratio in ratios if ratio >= similarity)
        key = (
            float(len(scores)),
            float(fuzzy),
            sum(ratios),
            sum(scores) / len(scores),
            float(len(text)),
        )
        if best_key is None or key > best_key:
            best_text, best_key = text, key
    return best_text, sum(exact[best_text]) / len(exact[best_text])


def static_texts(
    per_frame: list[list[Row]],
    *,
    ratio: float = 0.6,
    min_frames: int = 8,
    spread_x: float = 14.0,
    spread_y: float = 10.0,
) -> frozenset[str]:
    """Cac dong chu dung im mot cho suot video: logo, watermark, chu quang cao."""
    frames = len(per_frame)
    if frames < min_frames:
        return frozenset()
    spots: list[dict[str, Any]] = []
    for index, rows in enumerate(per_frame):
        for row in rows:
            if row.rect is None or len(row.text) < 2:
                continue
            x, y = row.rect[0], row.rect[1]
            for spot in spots:
                if abs(x - spot["x"]) <= spread_x and abs(y - spot["y"]) <= spread_y:
                    spot["frames"].add(index)
                    spot["texts"].append(row.text)
                    break
            else:
                spots.append({"x": x, "y": y, "frames": {index}, "texts": [row.text]})
    need = max(min_frames, int(frames * ratio))
    out: set[str] = set()
    for spot in spots:
        if len(spot["frames"]) < need:
            continue
        texts: list[str] = spot["texts"]
        common = max(set(texts), key=texts.count)
        if texts.count(common) < len(texts) * 0.55:
            continue
        alike = sum(1 for t in texts if _similar(t, common, 0.7))
        if alike < len(texts) * 0.75:
            continue
        out.update(t.casefold() for t in texts)
    return frozenset(out)


def _merge_rows(
    per_frame: list[list[Row]],
    *,
    fps: float,
    start_offset: float,
    stamps: Sequence[float] | None,
    total: int,
    similarity: float,
    min_duration: float,
    ignore: frozenset[str],
    consensus: int = 1,
    on_log: Callable[[str], None] | None = None,
) -> list[Cue]:
    """Gop cac khung hinh doc ra chu giong nhau thanh tung cau co dau va cuoi."""
    step = 1.0 / max(0.2, float(fps))
    exact = list(stamps) if stamps is not None and len(stamps) == total else []
    reads = [join_rows([r for r in rows if not is_ignored(r.text, ignore)]) for rows in per_frame]
    reads = _stabilize_reads(reads, similarity, consensus)
    reads = _drop_flicker(reads, similarity)
    cues: list[Cue] = []
    current_text = ""
    variants: list[tuple[str, float]] = []
    current_start = 0.0

    def flush(end: float) -> None:
        if not current_text or end - current_start < min_duration:
            return
        best_text, _best_score = _best_caption(variants, similarity)
        cues.append(Cue(start=current_start, end=end, text=best_text or current_text))

    for i, (text, score) in enumerate(reads):
        stamp = exact[i] if exact else start_offset + i * step
        if _same_caption_version(text, current_text, similarity):
            if text:
                variants.append((text, score))
                current_text = _best_caption(variants, similarity)[0]
            continue
        flush(stamp)
        current_text = text
        variants = [(text, score)] if text else []
        current_start = stamp
        if on_log and text:
            on_log(f"{stamp:7.2f}s  {text[:60]}")

    if current_text:
        end = (exact[-1] + step) if exact else (start_offset + total * step)
        flush(end)
    return cues


def _merge_exact_repeats(cues: list[Cue], *, max_gap: float = 0.2) -> list[Cue]:
    """Noi lai mot caption bi OCR rot vai frame o giua."""
    if len(cues) < 2:
        return cues
    out = [cues[0]]
    for cue in cues[1:]:
        previous = out[-1]
        gap = cue.start - previous.end
        if cue.text == previous.text and 0 <= gap <= max(0.0, float(max_gap)):
            previous.end = max(previous.end, cue.end)
            continue
        out.append(cue)
    return out


def clean_cues(cues: list[Cue], *, drop_chars: str = "", drop_words: str = "") -> list[Cue]:
    """Loc bo ky tu rac va cac tu khong muon giu trong ket qua doc chu."""
    chars = [c.strip() for c in drop_chars.split(",") if c.strip()]
    words = [w.strip() for w in drop_words.split(",") if w.strip()]
    out: list[Cue] = []
    for cue in cues:
        text = cue.text
        for item in chars:
            text = text.replace(item, "")
        for item in words:
            text = re.sub(re.escape(item), "", text, flags=re.IGNORECASE)
        text = " ".join(text.split())
        if text:
            out.append(Cue(cue.start, cue.end, text, cue.translation, cue.speaker))
    return out


def make_continuous(cues: list[Cue], max_gap: float = 1.5) -> list[Cue]:
    """Keo dai moi cau den sat cau sau de phu de khong bi nhay ngat quang."""
    for current, following in zip(cues, cues[1:], strict=False):
        gap = following.start - current.end
        if 0 < gap <= max_gap:
            current.end = following.start
    return cues
