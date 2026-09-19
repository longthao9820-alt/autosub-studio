"""Kiem thu viec tu dong xuat tep .srt canh video sau khi tach sub."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.data.project import ProjectStore
from autosub_studio.pipeline import steps
from autosub_studio.providers import tts
from autosub_studio.services import media
from autosub_studio.services.ffmpeg import CancelToken, FFmpeg
from autosub_studio.services.settings import Settings
from autosub_studio.services.tasks import TaskContext


def make_context(tmp_path: Path, *, video: Path | None, auto: bool = True):
    """Dung mot ngu canh toi thieu de chay rieng phan xuat tep."""
    store = ProjectStore(tmp_path / "workspace")
    project = store.create(1, "Du an thu")
    if video is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"khong phai video that, chi can co tep")
        project.original_video = str(video)
        project.video_path = str(video)
    project.doc = SubtitleDoc(cues=[Cue(0.0, 1.5, "Cau mot"), Cue(1.5, 3.0, "Cau hai")])
    logs: list[str] = []
    settings = Settings()
    settings.auto_export_srt = auto
    context = steps.PipelineContext(
        ff=FFmpeg(),
        settings=settings,
        store=store,
        project=project,
        task=TaskContext(token=CancelToken(), _progress=lambda _p: None, _log=logs.append),
    )
    return context, logs


class TestAutoExport:
    def test_voice_timeline_matches_nts_max_of_source_slot_and_voice(self, tmp_path):
        cues = [
            Cue(0.0, 2.56, "one"),
            Cue(2.64, 5.04, "two"),
            Cue(5.04, 6.80, "three"),
        ]
        voices = {
            0: (tmp_path / "one.wav", 2.986),
            1: (tmp_path / "two.wav", 2.786),
            2: (tmp_path / "three.wav", 2.560),
        }

        spans, voice_segments, timings, total = steps._voice_timeline(cues, voices, 7.0)

        assert spans[:3] == [
            (0.0, 2.64, 2.986),
            (2.64, 5.04, 2.786),
            (5.04, 6.8, 2.56),
        ]
        assert [round(start, 3) for start, _path in voice_segments] == [0.0, 2.986, 5.772]
        assert timings == [[0.0, 2.986], [2.986, 5.772], [5.772, 8.332]]
        assert total == pytest.approx(8.532)

    def test_writes_srt_next_to_video(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, logs = make_context(tmp_path, video=video)
        out = Path(steps.auto_export_srt(pc))
        assert out == video.with_suffix(".srt")
        assert out.is_file()
        assert "Cau mot" in out.read_text(encoding="utf-8")
        assert any("tu dong luu" in line for line in logs)

    def test_remembers_path_and_overwrites_next_time(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _ = make_context(tmp_path, video=video)
        first = steps.auto_export_srt(pc)
        assert pc.project.auto_srt_path == first
        pc.project.doc.cues = [Cue(0.0, 2.0, "Ban moi")]
        second = steps.auto_export_srt(pc)
        assert second == first
        assert "Ban moi" in Path(first).read_text(encoding="utf-8")
        assert len(list(Path(video).parent.glob("*.srt"))) == 1

    def test_does_not_clobber_a_file_already_there(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _ = make_context(tmp_path, video=video)
        cu = video.with_suffix(".srt")
        cu.write_text("phu de cua nguoi dung", encoding="utf-8")
        out = Path(steps.auto_export_srt(pc))
        assert out != cu
        assert cu.read_text(encoding="utf-8") == "phu de cua nguoi dung"
        assert out.is_file()

    def test_uses_translation_when_there_is_one(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _ = make_context(tmp_path, video=video)
        for cue in pc.project.doc.cues:
            cue.translation = "ban dich " + cue.text
        out = Path(steps.auto_export_srt(pc))
        text = out.read_text(encoding="utf-8")
        assert "ban dich Cau mot" in text

    def test_can_be_turned_off(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _ = make_context(tmp_path, video=video, auto=False)
        assert steps.auto_export_srt(pc) == ""
        assert not video.with_suffix(".srt").exists()

    def test_video_mode_mixes_original_audio_and_outputs_mp4(self, tmp_path, monkeypatch):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _logs = make_context(tmp_path, video=video)
        pc.settings.tts_provider = tts.PROVIDER_LOCAL
        pc.settings.dub_output_mode = "video"
        pc.settings.keep_original_audio = True
        pc.settings.original_audio_volume = 20
        pc.settings.dub_timing_mode = "voice"
        pc.settings.tts_fit_timing = False
        mixed_with = []

        def fake_synthesize(_provider, _text, out_path, **_kwargs):
            out = Path(out_path).with_suffix(".wav")
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(media.VOICE_RATE)
                wf.writeframes((1000).to_bytes(2, "little", signed=True) * media.VOICE_RATE)
            return out

        def fake_to_wav(_ff, source, out, **_kwargs):
            shutil.copyfile(source, out)
            return Path(out)

        def fake_mix(_ff, voice, bed, out, **kwargs):
            mixed_with.append((Path(bed), kwargs["music_volume"]))
            shutil.copyfile(voice, out)
            return Path(out)

        def fake_replace(_ff, _video, _audio, out, **_kwargs):
            Path(out).write_bytes(b"video co long tieng")
            return Path(out)

        def fake_retime(_ff, _video, _spans, _work_dir, out_path, **_kwargs):
            Path(out_path).write_bytes(b"retimed video")
            return Path(out_path)

        monkeypatch.setattr(tts, "provider_ready", lambda _provider: (True, ""))
        monkeypatch.setattr(tts, "synthesize_cached", fake_synthesize)
        monkeypatch.setattr(media, "to_wav", fake_to_wav)
        monkeypatch.setattr(media, "retime_video_segments", fake_retime)
        monkeypatch.setattr(media, "mix_voice_and_music", fake_mix)
        monkeypatch.setattr(media, "replace_audio", fake_replace)

        result = steps.step_dub(pc)

        assert Path(pc.project.dub_path).is_file()
        assert Path(pc.project.dub_video_path).is_file()
        assert mixed_with[0][1] == 20 and mixed_with[0][0].name in (
            "tap-01.mp4",
            "timeline_video.mp4",
        )
        assert "xuat video" in result
        again = pc.store.load(pc.project.folder)
        assert again.dub_video_path == pc.project.dub_video_path

    def test_no_cues_means_no_file(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _ = make_context(tmp_path, video=video)
        pc.project.doc.cues = []
        assert steps.auto_export_srt(pc) == ""
        assert not video.with_suffix(".srt").exists()

    def test_falls_back_to_project_folder_without_video(self, tmp_path):
        pc, _ = make_context(tmp_path, video=None)
        out = Path(steps.auto_export_srt(pc))
        assert out.is_file()
        assert out.parent == Path(pc.project.folder) / "exports"

    def test_falls_back_when_video_folder_is_gone(self, tmp_path):
        video = tmp_path / "o-usb-da-rut" / "tap-01.mp4"
        pc, logs = make_context(tmp_path, video=video)
        for item in video.parent.iterdir():
            item.unlink()
        video.parent.rmdir()
        out = Path(steps.auto_export_srt(pc))
        assert out.is_file()
        assert out.parent == Path(pc.project.folder) / "exports"
        assert logs

    def test_path_survives_saving_and_reloading_project(self, tmp_path):
        video = tmp_path / "phim" / "tap-01.mp4"
        pc, _ = make_context(tmp_path, video=video)
        steps.auto_export_srt(pc)
        pc.save()
        again = pc.store.load(pc.project.folder)
        assert again.auto_srt_path == pc.project.auto_srt_path


class TestOcrTemporaryCleanup:
    @pytest.mark.parametrize(
        ("step_name", "temp_names"),
        [
            (steps.STEP_OCR, ("ocr_frames", "ocr_window")),
            (steps.STEP_OCR_MEASURE, ("ocr_probe",)),
        ],
    )
    def test_cleans_images_when_ocr_step_is_stopped(
        self, tmp_path, monkeypatch, step_name, temp_names
    ):
        pc, _logs = make_context(tmp_path, video=tmp_path / "video.mp4")
        pc.settings.keep_temp = True
        for name in temp_names:
            folder = pc.project.sub_dir("temp") / name
            folder.mkdir()
            (folder / "frame.png").write_bytes(b"anh tam")

        def stopped(_context):
            raise steps.CancelledError("STOP")

        monkeypatch.setitem(steps.STEP_FUNCTIONS, step_name, stopped)

        with pytest.raises(steps.CancelledError):
            steps.run_step(step_name, pc)

        for name in temp_names:
            assert not (Path(pc.project.folder) / "temp" / name).exists()
