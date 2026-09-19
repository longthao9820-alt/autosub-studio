"""Quan ly thu muc du an va tep project.json co danh so phien ban."""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.models import SubtitleDoc
from ..services.paths import safe_name, unique_path, write_text_atomic

PROJECT_SCHEMA_VERSION = 4
PROJECT_FILE = "project.json"
RECOVERY_FILE = "project.recovery.json"
SUBDIRS = ("video", "audio", "subtitles", "temp", "exports")
OCR_TEMP_DIRS = ("ocr_frames", "ocr_window", "ocr_probe")


@dataclass
class ProjectData:
    """Noi dung mot du an, la nguon du lieu chinh khi lam viec."""

    schema_version: int = PROJECT_SCHEMA_VERSION
    project_id: int = 0
    name: str = ""
    folder: str = ""
    video_path: str = ""
    original_video: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    doc: SubtitleDoc = field(default_factory=SubtitleDoc)
    audio_path: str = ""
    music_path: str = ""
    voice_path: str = ""
    dub_path: str = ""
    dub_video_path: str = ""
    dub_timing: list[list[float]] = field(default_factory=list)
    render_path: str = ""
    auto_srt_path: str = ""  # tep .srt tu dong xuat canh video, ghi de moi lan tach lai
    ocr_region: list[int] = field(default_factory=list)  # x, y, w, h
    blur_region: list[int] = field(default_factory=list)
    lut_path: str = ""
    render_preset: str = ""
    output_video_path: str = ""
    notes: str = ""
    updated_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.folder)

    def sub_dir(self, name: str) -> Path:
        d = self.path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update({
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": self.project_id,
            "name": self.name,
            "folder": self.folder,
            "video_path": self.video_path,
            "original_video": self.original_video,
            "duration": round(self.duration, 3),
            "width": int(self.width),
            "height": int(self.height),
            "audio_path": self.audio_path,
            "music_path": self.music_path,
            "voice_path": self.voice_path,
            "dub_path": self.dub_path,
            "dub_video_path": self.dub_video_path,
            "dub_timing": [list(item[:2]) for item in self.dub_timing],
            "render_path": self.render_path,
            "auto_srt_path": self.auto_srt_path,
            "ocr_region": list(self.ocr_region),
            "blur_region": list(self.blur_region),
            "lut_path": self.lut_path,
            "render_preset": self.render_preset,
            "output_video_path": self.output_video_path,
            "notes": self.notes,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "subtitle": self.doc.to_dict(),
        })
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectData:
        version = int(data.get("schema_version", 1) or 1)
        if version > PROJECT_SCHEMA_VERSION:
            raise ProjectFormatError(
                f"Du an duoc tao boi phien ban moi hon (schema {version}). "
                "Hay cap nhat AutoSub Studio de mo."
            )
        known = {
            "schema_version",
            "project_id",
            "name",
            "folder",
            "video_path",
            "original_video",
            "duration",
            "width",
            "height",
            "audio_path",
            "music_path",
            "voice_path",
            "dub_path",
            "dub_video_path",
            "dub_timing",
            "render_path",
            "auto_srt_path",
            "ocr_region",
            "blur_region",
            "lut_path",
            "render_preset",
            "output_video_path",
            "notes",
            "updated_at",
            "subtitle",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        item = cls(
            schema_version=PROJECT_SCHEMA_VERSION,
            project_id=int(data.get("project_id", 0) or 0),
            name=str(data.get("name", "")),
            folder=str(data.get("folder", "")),
            video_path=str(data.get("video_path", "")),
            original_video=str(data.get("original_video", "")),
            duration=float(data.get("duration", 0.0) or 0.0),
            width=int(data.get("width", 0) or 0),
            height=int(data.get("height", 0) or 0),
            audio_path=str(data.get("audio_path", "")),
            music_path=str(data.get("music_path", "")),
            voice_path=str(data.get("voice_path", "")),
            dub_path=str(data.get("dub_path", "")),
            dub_video_path=str(data.get("dub_video_path", "")),
            render_path=str(data.get("render_path", "")),
            auto_srt_path=str(data.get("auto_srt_path", "")),
            lut_path=str(data.get("lut_path", "")),
            render_preset=str(data.get("render_preset", "")),
            output_video_path=str(data.get("output_video_path", "")),
            notes=str(data.get("notes", "")),
            updated_at=str(data.get("updated_at", "")),
            extra=extra,
        )
        item.ocr_region = [int(v) for v in (data.get("ocr_region") or [])][:4]
        item.blur_region = [int(v) for v in (data.get("blur_region") or [])][:4]
        item.dub_timing = [
            [float(value) for value in pair[:2]]
            for pair in (data.get("dub_timing") or [])
            if isinstance(pair, list) and len(pair) >= 2
        ]
        sub = data.get("subtitle")
        if isinstance(sub, dict):
            item.doc = SubtitleDoc.from_dict(sub)
        return item


class ProjectFormatError(ValueError):
    """Tep du an khong doc duoc."""


class ProjectStore:
    """Tao, doc, ghi va xoa thu muc du an trong khong gian lam viec."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.projects_dir = self.workspace / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create(self, project_id: int, name: str) -> ProjectData:
        folder = unique_path(self.projects_dir / f"{project_id}-{safe_name(name)}")
        folder.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (folder / sub).mkdir(exist_ok=True)
        data = ProjectData(project_id=project_id, name=name, folder=str(folder))
        self.save(data)
        return data

    def resolve_project_folder(self, folder: str | Path) -> Path:
        """Tim lai thu muc project theo ten sau khi ca ban portable bi di chuyen."""
        requested = Path(folder)
        try:
            if requested.is_dir():
                return requested
        except OSError:
            pass
        if not requested.name:
            return requested
        candidate = self.projects_dir / requested.name
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(self.projects_dir.resolve()) and resolved.is_dir():
                return resolved
        except OSError:
            pass
        return requested

    @staticmethod
    def relocate_project_path(value: str, old_folder: str | Path, new_folder: str | Path) -> str:
        """Doi duong dan con ben trong project sang vi tri portable moi."""
        if not value:
            return ""
        try:
            relative = Path(value).relative_to(Path(old_folder))
        except (OSError, ValueError):
            return value
        return str(Path(new_folder) / relative)

    def load(self, folder: str | Path) -> ProjectData:
        current_folder = self.resolve_project_folder(folder)
        p = current_folder / PROJECT_FILE
        if not p.is_file():
            raise ProjectFormatError(f"Khong tim thay {PROJECT_FILE} trong {folder}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectFormatError(f"Tep du an hong: {p}") from exc
        if not isinstance(raw, dict):
            raise ProjectFormatError(f"Tep du an hong: {p}")
        data = ProjectData.from_dict(raw)
        old_folder = data.folder or str(folder)
        for field_name in (
            "video_path",
            "original_video",
            "audio_path",
            "music_path",
            "voice_path",
            "dub_path",
            "dub_video_path",
            "render_path",
            "auto_srt_path",
            "lut_path",
            "output_video_path",
        ):
            setattr(
                data,
                field_name,
                self.relocate_project_path(
                    str(getattr(data, field_name) or ""), old_folder, current_folder
                ),
            )
        data.folder = str(current_folder)
        for sub in SUBDIRS:
            (current_folder / sub).mkdir(exist_ok=True)
        return data

    def save(self, data: ProjectData) -> Path:
        content = json.dumps(data.to_dict(), ensure_ascii=False, indent=2)
        path = write_text_atomic(Path(data.folder) / PROJECT_FILE, content)
        self.clear_recovery(data)
        return path

    def save_recovery(self, data: ProjectData) -> Path:
        content = json.dumps(data.to_dict(), ensure_ascii=False, indent=2)
        return write_text_atomic(Path(data.folder) / RECOVERY_FILE, content)

    def clear_recovery(self, data: ProjectData) -> None:
        p = Path(data.folder) / RECOVERY_FILE
        with contextlib.suppress(OSError):
            p.unlink(missing_ok=True)

    def recovery_path(self, folder: str | Path) -> Path | None:
        p = Path(folder) / RECOVERY_FILE
        return p if p.is_file() else None

    def delete(self, folder: str | Path, *, keep_files: bool = False) -> None:
        if keep_files:
            return
        p = Path(folder)
        if p.is_dir() and p.resolve().is_relative_to(self.projects_dir.resolve()):
            shutil.rmtree(p, ignore_errors=True)

    def clean_temp(self, data: ProjectData) -> int:
        """Xoa tep trung gian, tra ve so byte da giai phong."""
        temp = Path(data.folder) / "temp"
        freed = 0
        if not temp.is_dir():
            return 0
        for item in temp.iterdir():
            try:
                if item.is_file():
                    freed += item.stat().st_size
                    item.unlink()
                elif item.is_dir():
                    freed += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    shutil.rmtree(item, ignore_errors=True)
            except OSError:
                continue
        return freed

    def clean_ocr_temp(
        self, data: ProjectData, names: tuple[str, ...] = OCR_TEMP_DIRS
    ) -> int:
        """Xoa dung cac thu muc anh tam OCR, khong cham vao ket qua du an."""
        temp = Path(data.folder) / "temp"
        if not temp.is_dir():
            return 0
        temp_root = temp.resolve()
        projects_root = self.projects_dir.resolve()
        if not temp_root.is_relative_to(projects_root):
            return 0
        freed = 0
        for name in names:
            if name not in OCR_TEMP_DIRS:
                continue
            target = temp / name
            if not target.exists():
                continue
            try:
                resolved = target.resolve()
                if not resolved.is_relative_to(temp_root):
                    continue
                if target.is_file():
                    freed += target.stat().st_size
                    target.unlink()
                    continue
                freed += sum(
                    item.stat().st_size for item in target.rglob("*") if item.is_file()
                )
                shutil.rmtree(target, ignore_errors=True)
            except OSError:
                continue
        return freed

    def clean_stale_ocr_temp(self) -> int:
        """Don anh OCR con sot lai sau lan tat dot ngot hoac crash truoc do."""
        freed = 0
        for folder in self.projects_dir.iterdir():
            if not folder.is_dir():
                continue
            data = ProjectData(folder=str(folder))
            freed += self.clean_ocr_temp(data)
        return freed
