"""Kiem thu tao CapCut draft tu mot draft mau."""

from __future__ import annotations

import json
from pathlib import Path

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.providers import capcut


def _template(root: Path) -> Path:
    folder = root / "template"
    folder.mkdir()
    base_text = {
        "id": "SUB-TEXT-1",
        "type": "text",
        "content": json.dumps(
            {"text": "old", "styles": [{"range": [0, 3], "size": 10}]},
            ensure_ascii=False,
        ),
    }
    segment = {
        "id": "SUB-SEG-1",
        "material_id": "SUB-TEXT-1",
        "source_timerange": None,
        "target_timerange": {"start": 0, "duration": 1_000_000},
        "extra_material_refs": ["OLD-ANIMATION"],
        "render_index": 100,
    }
    content = {
        "id": "OLD-DRAFT",
        "duration": 10_000_000,
        "canvas_config": {"width": 1920, "height": 1080},
        "materials": {
            "videos": [
                {
                    "id": "VIDEO-1",
                    "path": "old.mp4",
                    "duration": 10_000_000,
                    "width": 1920,
                    "height": 1080,
                }
            ],
            "texts": [
                {"id": "TITLE", "type": "text", "content": '{"text":"title"}'},
                base_text,
            ],
        },
        "tracks": [
            {
                "type": "video",
                "segments": [
                    {
                        "material_id": "VIDEO-1",
                        "source_timerange": {"start": 0, "duration": 10_000_000},
                        "target_timerange": {"start": 0, "duration": 10_000_000},
                        "speed": 1.0,
                    }
                ],
            },
            {
                "type": "text",
                "segments": [
                    {
                        "material_id": "TITLE",
                        "target_timerange": {"start": 0, "duration": 10_000_000},
                    }
                ],
            },
            {"type": "text", "segments": [segment, dict(segment, id="SUB-SEG-2")]},
        ],
    }
    (folder / "draft_content.json").write_text(json.dumps(content), encoding="utf-8")
    (folder / "draft_meta_info.json").write_text(
        json.dumps({"draft_name": "old", "tm_duration": 10_000_000}), encoding="utf-8"
    )
    return folder


def test_export_draft_replaces_video_and_subtitle_track(tmp_path):
    template = _template(tmp_path)
    video = tmp_path / "new.mp4"
    video.write_bytes(b"video")
    doc = SubtitleDoc(
        cues=[
            Cue(0.5, 1.5, "goc 1", "dịch 1"),
            Cue(2.0, 3.25, "goc 2", "dịch 2"),
        ]
    )

    out = capcut.export_draft(
        template,
        tmp_path / "drafts",
        "Du an moi",
        video,
        doc,
        duration=4.0,
        timing=[[0.75, 1.75], [2.5, 3.9]],
    )

    data = json.loads((out / "draft_content.json").read_text(encoding="utf-8"))
    assert data["duration"] == 4_000_000
    assert data["materials"]["videos"][0]["path"] == video.resolve().as_posix()
    subtitle_track = max(
        (track for track in data["tracks"] if track["type"] == "text"),
        key=lambda track: len(track["segments"]),
    )
    assert [segment["target_timerange"] for segment in subtitle_track["segments"]] == [
        {"start": 750_000, "duration": 1_000_000},
        {"start": 2_500_000, "duration": 1_400_000},
    ]
    texts = [
        json.loads(item["content"])["text"]
        for item in data["materials"]["texts"]
        if item["id"] not in {"TITLE"}
    ]
    assert texts == ["dịch 1", "dịch 2"]
    assert json.loads((template / "draft_content.json").read_text(encoding="utf-8"))[
        "duration"
    ] == 10_000_000
