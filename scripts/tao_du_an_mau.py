"""Tao mot du an mau trong thu muc lam viec de nguoi dung thu ngay khi mo app.

Cach dung:
    python scripts/tao_du_an_mau.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosub_studio.core.models import Cue, SubtitleDoc  # noqa: E402
from autosub_studio.data.db import Database, Project  # noqa: E402
from autosub_studio.data.project import ProjectStore  # noqa: E402
from autosub_studio.services.ffmpeg import FFmpeg  # noqa: E402
from autosub_studio.services.paths import ensure_workspace  # noqa: E402
from autosub_studio.services.settings import Settings  # noqa: E402

NAME = "Video mau de thu"
LINES = [
    (0.3, 2.6, "This is the first sample sentence.", "Day la cau mau thu nhat."),
    (
        2.9,
        5.4,
        "The second line shows the translation column.",
        "Cau thu hai cho thay cot ban dich.",
    ),
    (
        5.7,
        8.0,
        "Press play to see subtitles on the video.",
        "Bam phat de thay phu de hien tren video.",
    ),
]


def main() -> int:
    settings = Settings.load()
    ensure_workspace(settings.workspace)
    ff = FFmpeg(settings.ffmpeg_path, settings.ffprobe_path)
    if not ff.available:
        print("Chua co FFmpeg nen khong tao duoc video mau.")
        return 2

    db = Database(Path(settings.workspace) / "db" / "app.db")
    db.ensure_default_scripts()
    store = ProjectStore(settings.workspace)

    with db.session() as s:
        if s.query(Project).filter(Project.name == NAME).first() is not None:
            print("Du an mau da ton tai, khong tao lai.")
            return 0
        record = Project(name=NAME, status="Moi tao")
        s.add(record)
        s.flush()
        project_id = record.id

    data = store.create(project_id, NAME)
    video = data.sub_dir("video") / "video_mau.mp4"
    ff.run(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=25:duration=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=180:duration=8",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
    )
    info = ff.probe(video)

    data.video_path = str(video)
    data.original_video = str(video)
    data.duration = info.duration
    data.doc = SubtitleDoc(
        cues=[Cue(a, b, c, d) for a, b, c, d in LINES],
        language="en",
        target_language="vi",
    )
    store.save(data)

    with db.session() as s:
        record = s.get(Project, project_id)
        if record is not None:
            record.folder = data.folder
            record.video_path = str(video)
            record.duration = info.duration
            record.cue_count = len(data.doc.cues)
            record.char_count = data.doc.total_chars
            record.has_subtitle = True
            record.has_translation = True
            record.language = "en"
            record.target_language = "vi"
            record.status = "San sang"

    print(f"Da tao du an mau tai: {data.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
