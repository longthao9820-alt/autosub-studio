"""Kieu du lieu cot loi: cau phu de va tai lieu phu de."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MIN_DURATION = 0.05


@dataclass
class Cue:
    """Mot cau phu de."""

    start: float
    end: float
    text: str = ""
    translation: str = ""
    speaker: str = ""

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.end = max(0.0, float(self.end))
        self.text = (self.text or "").strip()
        self.translation = (self.translation or "").strip()
        self.speaker = (self.speaker or "").strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def chars(self) -> int:
        return len(self.text.replace("\n", " "))

    @property
    def cps(self) -> float:
        """So ky tu tren mot giay cua cau goc."""
        if self.duration <= 0:
            return 0.0
        return self.chars / self.duration

    @property
    def length_ratio(self) -> float:
        """Ti le do dai ban dich so voi ban goc."""
        src = len(self.text.strip())
        if src == 0:
            return 0.0
        return len(self.translation.strip()) / src

    def display_text(self, mode: str = "original") -> str:
        """Van ban dung de hien thi hoac render theo che do chon."""
        original = self.text.strip()
        translated = self.translation.strip()
        if mode == "translation":
            return translated or original
        if mode == "both":
            if translated and original:
                return f"{original}\n{translated}"
            return translated or original
        return original or translated

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "translation": self.translation,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cue:
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            translation=str(data.get("translation", "")),
            speaker=str(data.get("speaker", "")),
        )

    def copy(self) -> Cue:
        return Cue(self.start, self.end, self.text, self.translation, self.speaker)


@dataclass
class SubtitleDoc:
    """Tap hop cac cau phu de cua mot du an."""

    cues: list[Cue] = field(default_factory=list)
    language: str = ""
    target_language: str = ""

    def __len__(self) -> int:
        return len(self.cues)

    def __iter__(self):
        return iter(self.cues)

    def __getitem__(self, index: int) -> Cue:
        return self.cues[index]

    @property
    def total_chars(self) -> int:
        return sum(c.chars for c in self.cues)

    @property
    def translated_count(self) -> int:
        return sum(1 for c in self.cues if c.translation.strip())

    @property
    def duration(self) -> float:
        return max((c.end for c in self.cues), default=0.0)

    def sort(self) -> None:
        self.cues.sort(key=lambda c: (c.start, c.end))

    def index_at(self, seconds: float) -> int:
        """Vi tri cau dang phat tai moc `seconds`, tra ve -1 neu khong co."""
        for i, cue in enumerate(self.cues):
            if cue.start <= seconds < cue.end:
                return i
        return -1

    def copy(self) -> SubtitleDoc:
        return SubtitleDoc(
            cues=[c.copy() for c in self.cues],
            language=self.language,
            target_language=self.target_language,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "target_language": self.target_language,
            "cues": [c.to_dict() for c in self.cues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleDoc:
        cues = [Cue.from_dict(item) for item in data.get("cues", []) if isinstance(item, dict)]
        return cls(
            cues=cues,
            language=str(data.get("language", "")),
            target_language=str(data.get("target_language", "")),
        )
