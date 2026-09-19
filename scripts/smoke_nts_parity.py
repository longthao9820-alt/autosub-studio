"""Smoke test thu cong cho timeline NTS: TTS -> retime video -> mix -> mux."""

from __future__ import annotations

import tempfile
from pathlib import Path

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.data.project import ProjectStore
from autosub_studio.pipeline.steps import PipelineContext, step_dub
from autosub_studio.providers import tts
from autosub_studio.services.ffmpeg import CancelToken, FFmpeg
from autosub_studio.services.settings import Settings
from autosub_studio.services.tasks import TaskContext


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="autosub_nts_smoke_"))
    ff = FFmpeg()
    source = root / "source.mp4"
    ff.run(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=640x360:r=25:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    store = ProjectStore(root / "workspace")
    project = store.create(1, "NTS parity smoke")
    project.video_path = project.original_video = str(source)
    project.duration = 4.0
    project.doc = SubtitleDoc(
        cues=[
            Cue(0.0, 1.6, "This is the first American English voice test."),
            Cue(1.7, 3.6, "The second line is longer so the video follows the narration."),
        ]
    )
    settings = Settings()
    settings.tts_provider = tts.PROVIDER_VOICESTUDIO
    settings.tts_language = "en-US"
    settings.tts_voice = "Kitten English Male 2|kittentts|expr-voice-2-m"
    settings.dub_source = "original"
    settings.dub_output_mode = "video"
    settings.dub_timing_mode = "voice"
    settings.keep_original_audio = True
    settings.ducking = True
    settings.tts_cache_enabled = False
    logs = []
    context = PipelineContext(
        ff=ff,
        settings=settings,
        store=store,
        project=project,
        task=TaskContext(token=CancelToken(), _progress=lambda _p: None, _log=logs.append),
    )
    print(step_dub(context))
    info = ff.probe(project.dub_video_path)
    print(f"duration={info.duration:.3f} video={info.has_video} audio={info.has_audio}")
    print("\n".join(logs))
    return 0 if info.has_video and info.has_audio and info.duration >= 4.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
