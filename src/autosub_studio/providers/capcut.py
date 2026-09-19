"""Tao CapCut draft moi tu mot draft mau co san cua nguoi dung."""

from __future__ import annotations

import copy
import json
import shutil
import time
import uuid
from pathlib import Path

from ..core.models import SubtitleDoc
from ..services.paths import safe_name, unique_path, write_text_atomic


class CapCutError(RuntimeError):
    """Khong doc/tao duoc CapCut draft."""


def drafts_root(capcut_path: str | Path) -> Path:
    root = Path(capcut_path)
    if root.name.casefold() == "com.lveditor.draft":
        return root
    return root / "User Data" / "Projects" / "com.lveditor.draft"


def list_drafts(capcut_path: str | Path) -> list[Path]:
    root = drafts_root(capcut_path)
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if (path / "draft_content.json").is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _new_id() -> str:
    return str(uuid.uuid4()).upper()


def _set_text(material: dict, text: str) -> None:
    material["recognize_text"] = text
    material["base_content"] = text
    try:
        content = json.loads(material.get("content") or "{}")
    except json.JSONDecodeError:
        content = {}
    content["text"] = text
    styles = content.get("styles")
    if not isinstance(styles, list) or not styles:
        styles = [{"range": [0, len(text)]}]
        content["styles"] = styles
    for style in styles:
        if isinstance(style, dict):
            style["range"] = [0, len(text)]
    material["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def export_draft(
    template_dir: str | Path,
    output_root: str | Path,
    name: str,
    video_path: str | Path,
    document: SubtitleDoc,
    *,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    timing: list[list[float]] | None = None,
    text_mode: str = "translation",
) -> Path:
    """Nhan ban draft mau, thay video chinh va tao lai track subtitle."""
    template = Path(template_dir)
    content_path = template / "draft_content.json"
    meta_path = template / "draft_meta_info.json"
    if not content_path.is_file() or not meta_path.is_file():
        raise CapCutError("Thu muc mau khong co draft_content.json/draft_meta_info.json.")
    try:
        content = json.loads(content_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapCutError(f"Khong doc duoc draft mau: {exc}") from exc

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    out = unique_path(root / safe_name(name or "AutoSub Studio"))
    shutil.copytree(template, out)

    duration_us = max(1, int(round(max(0.001, duration) * 1_000_000)))
    old_duration = max(1, int(content.get("duration") or duration_us))
    content["duration"] = duration_us
    content["name"] = name
    canvas = content.setdefault("canvas_config", {})
    if not canvas.get("width") or not canvas.get("height"):
        canvas["width"], canvas["height"] = int(width), int(height)

    materials = content.setdefault("materials", {})
    videos = materials.setdefault("videos", [])
    if not videos:
        raise CapCutError("Draft mau khong co video material de thay the.")
    main_video = videos[0]
    main_video_id = str(main_video.get("id") or _new_id())
    main_video["id"] = main_video_id
    video = Path(video_path).resolve()
    main_video.update(
        {
            "path": video.as_posix(),
            "media_path": "",
            "material_name": video.name,
            "duration": duration_us,
            "width": int(width),
            "height": int(height),
            "has_audio": True,
        }
    )

    tracks = content.setdefault("tracks", [])
    for track in tracks:
        if not isinstance(track, dict):
            continue
        for segment in track.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            target = segment.get("target_timerange")
            if str(segment.get("material_id")) == main_video_id:
                segment["source_timerange"] = {"start": 0, "duration": duration_us}
                segment["target_timerange"] = {"start": 0, "duration": duration_us}
                segment["speed"] = 1.0
            elif (
                isinstance(target, dict)
                and int(target.get("start") or 0) == 0
                and int(target.get("duration") or 0) >= int(old_duration * 0.8)
            ):
                target["duration"] = duration_us

    text_tracks = [
        track
        for track in tracks
        if isinstance(track, dict) and track.get("type") == "text"
    ]
    subtitle_track = max(
        text_tracks,
        key=lambda track: len(track.get("segments") or []),
        default=None,
    )
    if subtitle_track is None or not subtitle_track.get("segments"):
        raise CapCutError("Draft mau can co it nhat mot text/subtitle segment.")
    text_materials = materials.setdefault("texts", [])
    by_id = {str(item.get("id")): item for item in text_materials if isinstance(item, dict)}
    base_segment = subtitle_track["segments"][0]
    base_material = by_id.get(str(base_segment.get("material_id")))
    if base_material is None:
        raise CapCutError("Text material cua subtitle mau khong ton tai.")
    old_ids = {str(segment.get("material_id")) for segment in subtitle_track["segments"]}
    materials["texts"] = [
        item for item in text_materials if str(item.get("id")) not in old_ids
    ]
    new_materials = []
    new_segments = []
    pairs = timing if timing and len(timing) == len(document.cues) else None
    for index, cue in enumerate(document.cues):
        text = cue.display_text(text_mode).replace("\n", " ").strip()
        if not text:
            continue
        start, end = (pairs[index][:2] if pairs is not None else (cue.start, cue.end))
        if end <= start:
            continue
        material = copy.deepcopy(base_material)
        material_id = _new_id()
        material["id"] = material_id
        _set_text(material, text)
        segment = copy.deepcopy(base_segment)
        segment["id"] = _new_id()
        segment["material_id"] = material_id
        segment["target_timerange"] = {
            "start": int(round(start * 1_000_000)),
            "duration": max(1, int(round((end - start) * 1_000_000))),
        }
        segment["source_timerange"] = None
        segment["extra_material_refs"] = []
        segment["render_index"] = int(base_segment.get("render_index") or 0) + index
        new_materials.append(material)
        new_segments.append(segment)
    materials["texts"].extend(new_materials)
    subtitle_track["segments"] = new_segments

    now_us = int(time.time() * 1_000_000)
    draft_id = _new_id()
    meta.update(
        {
            "draft_name": name,
            "draft_fold_path": out.resolve().as_posix(),
            "draft_id": draft_id,
            "tm_draft_create": now_us,
            "tm_draft_modified": now_us,
            "tm_duration": duration_us,
        }
    )
    content["id"] = draft_id
    content["path"] = out.resolve().as_posix()
    content["create_time"] = now_us
    content["update_time"] = now_us
    write_text_atomic(
        out / "draft_content.json",
        json.dumps(content, ensure_ascii=False, separators=(",", ":")),
    )
    write_text_atomic(
        out / "draft_meta_info.json",
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
    )
    return out
