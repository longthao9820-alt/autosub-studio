"""Kiem thu he thong timeline va dubbing V2:
- Voice synthesis su dung duy nhat mot toc do global tts_speed_percent cho tat ca cac cau.
- Khong co per-cue atempo/tempo/speed adaptation trong media.to_wav hay TTS.
- Canh theo giong doc: target duration bang dung measured voice duration (ca expand va shrink).
- Deterministic cases: source 3.0 voice 4.5 -> 4.5; source 4.0 voice 2.8 -> 2.8.
- Explicit gap handling, cumulative timeline no drift.
- Subtitle mode giu source timeline ma khong thay doi toc do voice per-cue.
- Subtitle cues remapped va dong bo voi timeline moi.
- project.dub_timing migration compatible voi list/tuple.
"""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.data import project
from autosub_studio.pipeline import steps
from autosub_studio.providers import tts
from autosub_studio.services import media
from autosub_studio.services.ffmpeg import CancelToken, FFmpeg, MediaInfo
from autosub_studio.services.settings import Settings
from autosub_studio.services.tasks import TaskContext


def _create_wav(path: Path, duration: float, rate: int = media.VOICE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(media.VOICE_WIDTH)
        wf.setframerate(rate)
        total_frames = int(round(duration * rate))
        wf.writeframes((1000).to_bytes(2, "little", signed=True) * total_frames)
    return path


def _make_pipeline_context(tmp_path: Path, cues: list[Cue], video_duration: float = 10.0):
    store = project.ProjectStore(tmp_path / "workspace")
    proj = store.create(1, "Timeline Test")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"dummy video")
    proj.video_path = str(source_video)
    proj.original_video = str(source_video)
    proj.duration = video_duration
    proj.doc = SubtitleDoc(cues=cues)
    settings = Settings()
    logs: list[str] = []
    pc = steps.PipelineContext(
        ff=FFmpeg(),
        settings=settings,
        store=store,
        project=proj,
        task=TaskContext(token=CancelToken(), _progress=lambda _p: None, _log=logs.append),
    )
    return pc, logs


def _stub_common_media(monkeypatch):
    def fake_to_wav(_ff, s, d, **_k):
        shutil.copyfile(s, d)
        return Path(d)

    def fake_retime(*a, **kw):
        out = Path(kw.get("out_path", a[4]))
        out.write_bytes(b"vid")
        return out

    def fake_mix(_ff, v, _b, o, **_k):
        shutil.copyfile(v, o)
        return Path(o)

    def fake_replace(_ff, _v, _a, o, **_k):
        Path(o).write_bytes(b"final")
        return Path(o)

    monkeypatch.setattr(tts, "provider_ready", lambda _p: (True, ""))
    monkeypatch.setattr(media, "to_wav", fake_to_wav)
    monkeypatch.setattr(media, "retime_video_segments", fake_retime)
    monkeypatch.setattr(media, "mix_voice_and_music", fake_mix)
    monkeypatch.setattr(media, "replace_audio", fake_replace)
    return fake_retime


class TestVoiceTimeline:
    def test_deterministic_case_expand_3_0_to_4_5(self, tmp_path):
        """Case 1: source 3.0, voice 4.5 -> target ~4.5."""
        cues = [Cue(0.0, 3.0, "expand line")]
        v0 = _create_wav(tmp_path / "v0.wav", 4.5)
        voices = {0: (v0, 4.5)}

        spans, segs, timings, total = steps._voice_timeline(cues, voices, 3.0)

        assert len(spans) == 1
        assert spans[0] == (0.0, 3.0, 4.5)
        assert timings == [[0.0, 4.5]]
        assert segs == [(0.0, v0)]
        assert total == pytest.approx(4.5)

    def test_deterministic_case_shrink_4_0_to_2_8(self, tmp_path):
        """Case 2: source 4.0, voice 2.8 -> target ~2.8 (loai bo max(source, voice))."""
        cues = [Cue(0.0, 4.0, "shrink line")]
        v0 = _create_wav(tmp_path / "v0.wav", 2.8)
        voices = {0: (v0, 2.8)}

        spans, segs, timings, total = steps._voice_timeline(cues, voices, 4.0)

        assert len(spans) == 1
        assert spans[0] == (0.0, 4.0, 2.8)
        assert timings == [[0.0, 2.8]]
        assert segs == [(0.0, v0)]
        assert total == pytest.approx(2.8)

    def test_both_deterministic_cases_with_explicit_gap_no_drift(self, tmp_path):
        """Kiem tra ca hai case (3.0->4.5 va 4.0->2.8) voi gap ro rang giua cac cau."""
        cues = [
            Cue(1.0, 4.0, "cau mot source 3.0"),
            Cue(6.0, 10.0, "cau hai source 4.0"),
        ]
        v0 = _create_wav(tmp_path / "v0.wav", 4.5)
        v1 = _create_wav(tmp_path / "v1.wav", 2.8)
        voices = {0: (v0, 4.5), 1: (v1, 2.8)}

        spans, segs, timings, total = steps._voice_timeline(cues, voices, 12.0)

        assert spans == [
            (0.0, 1.0, 1.0),
            (1.0, 4.0, 4.5),
            (4.0, 6.0, 2.0),
            (6.0, 10.0, 2.8),
            (10.0, 12.0, 2.0),
        ]
        assert timings == [
            [1.0, 5.5],
            [7.5, 10.3],
        ]
        assert segs == [
            (1.0, v0),
            (7.5, v1),
        ]
        assert total == pytest.approx(12.3)

    def test_zero_and_invalid_duration_robustness(self, tmp_path):
        """Khong bi loi khi cue co thoi luong bang 0, am, hoac voice bi thieu/loi."""
        cues = [
            Cue(0.0, 0.0, "zero duration"),
            Cue(3.0, 2.0, "inverted duration"),
            Cue(4.0, 5.0, "normal line but missing voice"),
        ]
        v1 = _create_wav(tmp_path / "v1.wav", 1.5)
        voices = {
            0: (tmp_path / "zero.wav", 0.0),
            1: (v1, 1.5),
        }

        spans, segs, timings, total = steps._voice_timeline(cues, voices, 6.0)

        assert len(spans) >= 3
        for start, end, target in spans:
            assert end >= start
            assert target >= 0.04
        assert len(timings) == 3
        for start, end in timings:
            assert end > start
        assert total > 0.0


class TestDubbingInvariants:
    def test_synth_called_with_same_global_speed_for_all_cues(self, tmp_path, monkeypatch):
        """Tts synthesize duoc goi voi dung 1 toc do global tts_speed_percent duy nhat."""
        cues = [
            Cue(0.0, 2.0, "First line"),
            Cue(2.0, 4.0, "Second line"),
            Cue(4.0, 6.0, "Third line"),
        ]
        pc, _logs = _make_pipeline_context(tmp_path, cues)
        pc.settings.tts_speed_percent = 125
        pc.settings.tts_voice_profiles = [
            {"gender": "nam", "speed": 80, "voice": "v_nam"},
            {"gender": "nu", "speed": 175, "voice": "v_nu"},
        ]
        cues[0].speaker = "nam"
        cues[1].speaker = "nu"

        synthesize_speeds: list[float] = []

        def fake_synthesize(_provider, _text, out_path, **kwargs):
            synthesize_speeds.append(kwargs.get("speed", 1.0))
            return _create_wav(Path(out_path), 2.0)

        _stub_common_media(monkeypatch)
        monkeypatch.setattr(tts, "provider_ready", lambda _p: (True, ""))
        monkeypatch.setattr(tts, "synthesize_cached", fake_synthesize)

        steps.step_dub(pc)

        assert len(synthesize_speeds) == 3
        assert synthesize_speeds == [1.25, 1.25, 1.25]

    def test_no_media_to_wav_tempo_adaptation_for_fit_in_voice_mode(self, tmp_path, monkeypatch):
        """Khong goi media.to_wav voi tempo khac 1.0 de ep vua timestamp."""
        cues = [
            Cue(0.0, 1.0, "Short slot for longer voice"),
            Cue(2.0, 5.0, "Long slot for shorter voice"),
        ]
        pc, _logs = _make_pipeline_context(tmp_path, cues)
        pc.settings.tts_speed_percent = 100
        pc.settings.dub_timing_mode = "voice"

        to_wav_tempos: list[float] = []

        def fake_to_wav(_ff, src, dst, **kwargs):
            to_wav_tempos.append(kwargs.get("tempo", 1.0))
            shutil.copyfile(src, dst)
            return Path(dst)

        def fake_synthesize(_provider, text, out_path, **_kwargs):
            duration = 3.0 if "Short slot" in text else 1.5
            return _create_wav(Path(out_path), duration)

        _stub_common_media(monkeypatch)
        monkeypatch.setattr(tts, "provider_ready", lambda _p: (True, ""))
        monkeypatch.setattr(tts, "synthesize_cached", fake_synthesize)
        monkeypatch.setattr(media, "to_wav", fake_to_wav)

        steps.step_dub(pc)

        assert len(to_wav_tempos) == 2
        assert to_wav_tempos == [1.0, 1.0]

    def test_subtitle_mode_preserves_source_timeline_without_per_cue_tempo(
        self, tmp_path, monkeypatch
    ):
        """Che do subtitle giu timeline goc, khong thay doi toc do voice per-cue."""
        cues = [
            Cue(0.0, 1.0, "Cau ngan nhung doc dai"),
        ]
        pc, _logs = _make_pipeline_context(tmp_path, cues)
        pc.settings.dub_timing_mode = "subtitle"
        pc.settings.tts_allow_overlap = True

        to_wav_tempos: list[float] = []

        def fake_to_wav(_ff, src, dst, **kwargs):
            to_wav_tempos.append(kwargs.get("tempo", 1.0))
            shutil.copyfile(src, dst)
            return Path(dst)

        def fake_synthesize(_provider, _text, out_path, **_kwargs):
            return _create_wav(Path(out_path), 2.5)

        _stub_common_media(monkeypatch)
        retime_called = False

        def fake_retime(*_args, **_kwargs):
            nonlocal retime_called
            retime_called = True

        monkeypatch.setattr(tts, "provider_ready", lambda _p: (True, ""))
        monkeypatch.setattr(tts, "synthesize_cached", fake_synthesize)
        monkeypatch.setattr(media, "to_wav", fake_to_wav)
        monkeypatch.setattr(media, "retime_video_segments", fake_retime)

        steps.step_dub(pc)

        assert pc.project.dub_timing == []
        assert not retime_called
        assert to_wav_tempos == [1.0]

    def test_subtitle_cues_remapped_and_synced_with_video_and_voice(self, tmp_path, monkeypatch):
        """Phu de duoc remapped theo timeline moi va dong bo voi voice track."""
        cues = [
            Cue(0.0, 3.0, "Cau 1 nguon 3.0 voice 4.5"),
            Cue(3.0, 7.0, "Cau 2 nguon 4.0 voice 2.8"),
        ]
        pc, _logs = _make_pipeline_context(tmp_path, cues)
        pc.settings.dub_timing_mode = "voice"
        pc.settings.tts_end_pause_ms = 0

        def fake_synthesize(_provider, text, out_path, **_kwargs):
            dur = 4.5 if "Cau 1" in text else 2.8
            return _create_wav(Path(out_path), dur)

        _stub_common_media(monkeypatch)
        monkeypatch.setattr(tts, "provider_ready", lambda _p: (True, ""))
        monkeypatch.setattr(tts, "synthesize_cached", fake_synthesize)

        steps.step_dub(pc)

        assert len(pc.project.dub_timing) == 2
        cue0_start, cue0_end = pc.project.dub_timing[0]
        cue1_start, cue1_end = pc.project.dub_timing[1]

        assert cue0_start == pytest.approx(0.0)
        assert (cue0_end - cue0_start) == pytest.approx(4.5, abs=0.01)
        assert cue1_start == pytest.approx(cue0_end, abs=0.01)
        assert (cue1_end - cue1_start) == pytest.approx(2.8, abs=0.01)

        ass_path = steps.build_ass(pc, text_mode="original", timing=pc.project.dub_timing)
        assert ass_path.is_file()
        ass_content = ass_path.read_text(encoding="utf-8")
        assert "Dialogue:" in ass_content

        steps.step_dub(pc)

        assert len(pc.project.dub_timing) == 2
        cue0_start, cue0_end = pc.project.dub_timing[0]
        cue1_start, cue1_end = pc.project.dub_timing[1]

        assert cue0_start == pytest.approx(0.0)
        assert (cue0_end - cue0_start) == pytest.approx(4.5, abs=0.01)
        assert cue1_start == pytest.approx(cue0_end, abs=0.01)
        assert (cue1_end - cue1_start) == pytest.approx(2.8, abs=0.01)

        ass_path = steps.build_ass(pc, text_mode="original", timing=pc.project.dub_timing)
        assert ass_path.is_file()
        ass_content = ass_path.read_text(encoding="utf-8")
        assert "Dialogue:" in ass_content


class TestRetimeVideoAndProjectSchema:
    def test_retime_video_segments_stretches_and_shrinks(self, tmp_path):
        """retime_video_segments co the stretch (setpts > 1) hoac shrink (setpts < 1)."""
        class FakeFF:
            ffmpeg = "ffmpeg"

            def __init__(self):
                self.calls = []

            def probe(self, _path):
                return MediaInfo(path="source.mp4", fps=30.0, has_video=True, has_audio=True)

            def run(self, args, **_kwargs):
                self.calls.append(list(args))
                Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(args[-1]).write_bytes(b"part")

        ff = FakeFF()
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        out = tmp_path / "retimed.mp4"

        media.retime_video_segments(
            ff,
            source,
            [(0.0, 3.0, 4.5), (3.0, 7.0, 2.8)],
            tmp_path / "parts",
            out,
            workers=1,
        )

        filter_graphs = [
            call[call.index("-filter_complex") + 1]
            for call in ff.calls
            if "-filter_complex" in call
        ]
        assert len(filter_graphs) == 2
        assert "setpts=1.500000000*PTS" in filter_graphs[0]
        assert "setpts=0.700000000*PTS" in filter_graphs[1]
        assert out.is_file()

    def test_project_dub_timing_list_and_tuple_compatibility(self, tmp_path):
        """ProjectStore doc va ghi dung dub_timing o ca dang list va tuple."""
        store = project.ProjectStore(tmp_path / "workspace")
        p = store.create(1, "timing compatibility")
        p.dub_timing = [[0.0, 4.5], [4.5, 7.3]]
        store.save(p)

        loaded = store.load(p.folder)
        assert loaded.dub_timing == [[0.0, 4.5], [4.5, 7.3]]

        # Gia lap file project json chua tuple
        project_json = Path(p.folder) / "project.json"
        raw_dict = p.to_dict()
        raw_dict["dub_timing"] = [(0.0, 4.5), (4.5, 7.3)]
        import json
        project_json.write_text(json.dumps(raw_dict), encoding="utf-8")

        loaded2 = store.load(p.folder)
        assert loaded2.dub_timing == [[0.0, 4.5], [4.5, 7.3]]
