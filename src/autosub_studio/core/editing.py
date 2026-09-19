"""Thao tac bien tap phu de: tach, gop, doi thoi gian va kiem tra lech time."""

from __future__ import annotations

from dataclasses import dataclass

from .models import MIN_DURATION, Cue, SubtitleDoc

# Nguong mac dinh khi kiem tra chat luong phu de.
DEFAULT_MAX_CPS = 21.0
DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_DURATION = 8.0
DEFAULT_MAX_GAP = 15.0
DEFAULT_MAX_RATIO = 1.6


@dataclass
class Issue:
    """Mot loi hoac canh bao ve thoi gian/noi dung cua cau phu de."""

    index: int
    kind: str
    message: str
    severity: str = "warning"  # "error" hoac "warning"


def split_cue(doc: SubtitleDoc, index: int, at_seconds: float) -> int:
    """Tach cau `index` tai moc `at_seconds`. Tra ve vi tri cau moi."""
    if not 0 <= index < len(doc.cues):
        raise IndexError("Vi tri cau khong hop le")
    cue = doc.cues[index]
    if not (cue.start + MIN_DURATION <= at_seconds <= cue.end - MIN_DURATION):
        raise ValueError("Moc tach phai nam ben trong cau, cach hai dau it nhat 0,05 giay")
    ratio = (at_seconds - cue.start) / cue.duration if cue.duration else 0.5
    first_text, second_text = _split_text(cue.text, ratio)
    first_trans, second_trans = _split_text(cue.translation, ratio)
    new_cue = Cue(
        start=at_seconds,
        end=cue.end,
        text=second_text,
        translation=second_trans,
        speaker=cue.speaker,
    )
    cue.end = at_seconds
    cue.text = first_text
    cue.translation = first_trans
    doc.cues.insert(index + 1, new_cue)
    return index + 1


def merge_cues(doc: SubtitleDoc, indexes: list[int]) -> int:
    """Gop cac cau lien tiep thanh mot. Tra ve vi tri cau da gop."""
    if len(indexes) < 2:
        raise ValueError("Can chon it nhat 2 cau de gop")
    order = sorted(set(indexes))
    if order[-1] - order[0] != len(order) - 1:
        raise ValueError("Chi gop duoc cac cau lien tiep nhau")
    if order[-1] >= len(doc.cues):
        raise IndexError("Vi tri cau khong hop le")
    group = [doc.cues[i] for i in order]
    merged = Cue(
        start=min(c.start for c in group),
        end=max(c.end for c in group),
        text=" ".join(c.text.strip() for c in group if c.text.strip()),
        translation=" ".join(c.translation.strip() for c in group if c.translation.strip()),
        speaker=next((c.speaker for c in group if c.speaker), ""),
    )
    del doc.cues[order[0] : order[-1] + 1]
    doc.cues.insert(order[0], merged)
    return order[0]


def shift_cues(doc: SubtitleDoc, offset: float, indexes: list[int] | None = None) -> None:
    """Doi thoi gian cac cau mot khoang `offset` giay (co the am)."""
    targets = doc.cues if indexes is None else [doc.cues[i] for i in indexes]
    for cue in targets:
        cue.start = max(0.0, cue.start + offset)
        cue.end = max(cue.start + MIN_DURATION, cue.end + offset)


def scale_cues(doc: SubtitleDoc, factor: float) -> None:
    """Nhan thoi gian voi he so, dung khi video sai toc do khung hinh."""
    if factor <= 0:
        raise ValueError("He so phai lon hon 0")
    for cue in doc.cues:
        cue.start *= factor
        cue.end *= factor


def remove_overlaps(doc: SubtitleDoc, min_gap: float = 0.04) -> int:
    """Cat bot phan chong lan giua cac cau. Tra ve so cau da sua."""
    doc.sort()
    fixed = 0
    for i in range(len(doc.cues) - 1):
        cur, nxt = doc.cues[i], doc.cues[i + 1]
        if cur.end > nxt.start - min_gap:
            new_end = max(cur.start + MIN_DURATION, nxt.start - min_gap)
            if abs(new_end - cur.end) > 1e-6:
                cur.end = new_end
                fixed += 1
    return fixed


def check_timing(
    doc: SubtitleDoc,
    *,
    max_cps: float = DEFAULT_MAX_CPS,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_gap: float = DEFAULT_MAX_GAP,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> list[Issue]:
    """Do cac loi thoi gian pho bien, tra ve danh sach da sap theo vi tri cau."""
    issues: list[Issue] = []
    for i, cue in enumerate(doc.cues):
        if cue.end <= cue.start:
            issues.append(
                Issue(i, "negative", "Thoi gian ket thuc khong lon hon bat dau.", "error")
            )
        elif cue.duration < min_duration:
            issues.append(Issue(i, "short", f"Cau qua ngan ({cue.duration:.2f} giay).", "warning"))
        elif cue.duration > max_duration:
            issues.append(Issue(i, "long", f"Cau qua dai ({cue.duration:.1f} giay).", "warning"))
        if cue.duration > 0 and cue.cps > max_cps:
            issues.append(
                Issue(i, "fast", f"Chu chay qua nhanh ({cue.cps:.0f} ky tu/giay).", "warning")
            )
        if not cue.text.strip() and not cue.translation.strip():
            issues.append(Issue(i, "empty", "Cau khong co noi dung.", "warning"))
        if cue.translation.strip() and cue.length_ratio > max_ratio:
            issues.append(
                Issue(
                    i,
                    "ratio",
                    f"Ban dich dai gap {cue.length_ratio:.2f} lan ban goc.",
                    "warning",
                )
            )
        if i + 1 < len(doc.cues):
            nxt = doc.cues[i + 1]
            if cue.end > nxt.start + 1e-6:
                issues.append(
                    Issue(
                        i,
                        "overlap",
                        f"Chong lan cau {i + 2} ({cue.end - nxt.start:.2f}s).",
                        "error",
                    )
                )
            elif nxt.start - cue.end > max_gap:
                issues.append(
                    Issue(
                        i,
                        "gap",
                        f"Khoang trong {nxt.start - cue.end:.0f} giay truoc cau {i + 2}.",
                        "warning",
                    )
                )
    issues.sort(key=lambda it: (it.index, it.kind))
    return issues


def _split_text(text: str, ratio: float) -> tuple[str, str]:
    """Chia van ban theo ti le, uu tien cat tai khoang trang gan nhat."""
    text = (text or "").strip()
    if not text:
        return "", ""
    if "\n" in text:
        lines = text.split("\n")
        cut = max(1, min(len(lines) - 1, round(len(lines) * ratio)))
        return "\n".join(lines[:cut]).strip(), "\n".join(lines[cut:]).strip()
    target = max(1, min(len(text) - 1, int(round(len(text) * ratio))))
    left = text.rfind(" ", 0, target)
    right = text.find(" ", target)
    candidates = [c for c in (left, right) if c > 0]
    if not candidates:
        return text, ""
    cut = min(candidates, key=lambda c: abs(c - target))
    return text[:cut].strip(), text[cut:].strip()
