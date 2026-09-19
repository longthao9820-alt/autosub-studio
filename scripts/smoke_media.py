"""Chay thu toan bo luong xu ly media tren mot video tu tao.

Cach dung:
    python scripts/smoke_media.py [--ffmpeg C:\\duong-dan\\ffmpeg.exe] [--keep]

Script tu tao mot video mau bang FFmpeg roi lan luot chay: tach am thanh,
chuan hoa, che mo vung, ghep phu de cung, tao track long tieng, tron nhac nen
va xuat goi du an. Bao loi ro rang neu buoc nao that bai.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosub_studio.core.models import Cue, SubtitleDoc  # noqa: E402
from autosub_studio.data.project import ProjectStore  # noqa: E402
from autosub_studio.pipeline import steps as P  # noqa: E402
from autosub_studio.services import media  # noqa: E402
from autosub_studio.services.ffmpeg import CancelToken, FFmpeg  # noqa: E402
from autosub_studio.services.settings import Settings  # noqa: E402
from autosub_studio.services.tasks import TaskContext  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, func) -> object:
    try:
        result = func()
    except Exception as exc:  # bao cao thay vi dung han
        FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"  [HONG] {name}: {exc}")
        return None
    PASSED.append(name)
    print(f"  [OK]   {name}")
    return result


def make_test_video(ff: FFmpeg, out: Path, seconds: int = 6) -> Path:
    ff.run(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=25:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]
    )
    return out


def make_tone_wav(out: Path, seconds: float = 1.0) -> Path:
    import math

    rate = media.VOICE_RATE
    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = int(9000 * math.sin(2 * math.pi * 150 * i / rate))
        frames += value.to_bytes(2, "little", signed=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Chay thu luong media")
    parser.add_argument("--ffmpeg", default="", help="Duong dan toi ffmpeg.exe")
    parser.add_argument("--keep", action="store_true", help="Giu lai thu muc thu nghiem")
    args = parser.parse_args()

    ff = FFmpeg(args.ffmpeg)
    if not ff.available:
        print("KHONG TIM THAY FFMPEG. Dung --ffmpeg de chi duong dan.")
        return 2
    print(f"FFmpeg: {ff.version()}")

    work = Path(tempfile.mkdtemp(prefix="autosub_smoke_"))
    print(f"Thu muc thu nghiem: {work}\n")

    try:
        source = check("Tao video mau", lambda: make_test_video(ff, work / "source.mp4"))
        if source is None:
            return 1
        info = check("Doc thong tin video", lambda: ff.probe(source))
        if info is not None:
            print(f"         -> {info.resolution}, {info.duration:.2f}s, audio={info.has_audio}")

        settings = Settings()
        settings.workspace = str(work / "workspace")
        settings.render_preset = "ultrafast"
        settings.render_crf = 30
        settings.render_scale = "640x360"
        settings.render_fps = "25"
        store = ProjectStore(settings.workspace)
        project = store.create(1, "Du an thu nghiem")
        project.video_path = str(source)
        project.original_video = str(source)
        project.duration = info.duration if info else 6.0
        project.doc = SubtitleDoc(
            cues=[
                Cue(0.2, 2.0, "Cau thu nghiem mot", "Ban dich mot"),
                Cue(2.2, 4.0, "Cau thu nghiem hai", "Ban dich hai"),
                Cue(4.2, 5.8, "Cau thu nghiem ba", "Ban dich ba"),
            ]
        )
        project.blur_region = [40, 260, 200, 60]
        store.save(project)

        task = TaskContext(token=CancelToken(), _progress=lambda _p: None, _log=lambda _m: None)
        pc = P.PipelineContext(ff=ff, settings=settings, store=store, project=project, task=task)

        check("Tach am thanh 16 kHz", lambda: P.ensure_audio(pc))
        check("Chuan hoa video", lambda: P.step_normalize(pc))
        check("Xoa thoai giu nhac nen", lambda: P.step_keep_music(pc))
        check("Tach giong noi", lambda: P.step_keep_voice(pc))
        ass = check("Tao tep ASS", lambda: P.build_ass(pc))
        if ass is not None:
            content = Path(ass).read_text(encoding="utf-8")
            check(
                "ASS co du 3 cau",
                lambda: (
                    (_ for _ in ()).throw(AssertionError("thieu cau"))
                    if content.count("Dialogue:") != 3
                    else True
                ),
            )
        check("Che mo vung phu de goc", lambda: P.step_blur(pc))

        seg = make_tone_wav(work / "tone.wav", 1.2)
        voice = work / "voice_track.wav"
        check(
            "Ghep track long tieng",
            lambda: media.build_voice_track([(0.5, seg), (2.5, seg)], voice, 6.0),
        )
        music = Path(project.music_path)
        if music.is_file():
            check(
                "Tron giong doc voi nhac nen",
                lambda: media.mix_voice_and_music(
                    ff, voice, music, work / "mixed.wav", music_volume=30, ducking=True
                ),
            )
        project.dub_path = str(work / "mixed.wav") if (work / "mixed.wav").is_file() else ""
        store.save(project)

        rendered = check("Render video kem phu de", lambda: P.step_render(pc))
        if rendered is not None and project.render_path:
            out = Path(project.render_path)
            check(
                "Tep render ton tai va co du lieu",
                lambda: (
                    (_ for _ in ()).throw(AssertionError("tep rong"))
                    if (not out.is_file() or out.stat().st_size < 5000)
                    else True
                ),
            )
            check("Tep render doc duoc bang ffprobe", lambda: ff.probe(out))
        check("Xuat goi du an", lambda: P.step_export(pc))
        check(
            "Xuat SRT",
            lambda: __import__(
                "autosub_studio.core.formats", fromlist=["save_subtitle"]
            ).save_subtitle(work / "out.srt", project.doc, text_mode="translation"),
        )
    finally:
        if args.keep:
            print(f"\nGiu lai thu muc: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print(f"\nDat: {len(PASSED)} | Hong: {len(FAILED)}")
    for name, reason in FAILED:
        print(f"  - {name}: {reason}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
