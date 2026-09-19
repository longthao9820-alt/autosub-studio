"""Kiem thu cho lop dich vu: media, du an, cau hinh, hang doi tac vu."""

from __future__ import annotations

import json
import threading
import time
import wave
from pathlib import Path

import pytest

from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.data.db import AutoScript, Database, Project
from autosub_studio.data.project import ProjectFormatError, ProjectStore
from autosub_studio.pipeline.steps import STEP_DUB
from autosub_studio.services import gpu, media, paths
from autosub_studio.services import settings as settings_service
from autosub_studio.services.ffmpeg import CancelToken, MediaInfo
from autosub_studio.services.settings import Settings, SubtitleStyle
from autosub_studio.services.tasks import CANCELLED, DONE, FAILED, PENDING, RUNNING, TaskManager


def _write_wav(path, seconds: float, rate: int = media.VOICE_RATE, value: int = 1000):
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(value.to_bytes(2, "little", signed=True) * frames)


class TestMediaHelpers:
    def test_atempo_within_range(self):
        assert media.atempo_chain(1.5) == "atempo=1.500000"

    def test_atempo_chains_for_large_values(self):
        chain = media.atempo_chain(4.0)
        assert chain.count("atempo") >= 2

    def test_atempo_chains_for_small_values(self):
        chain = media.atempo_chain(0.25)
        assert chain.count("atempo") >= 2

    def test_wav_duration(self, tmp_path):
        target = tmp_path / "a.wav"
        _write_wav(target, 1.5)
        assert media.wav_duration(target) == pytest.approx(1.5, abs=0.01)

    def test_wav_duration_missing_file(self, tmp_path):
        assert media.wav_duration(tmp_path / "khong-co.wav") == 0.0

    def test_append_silence_keeps_voice_and_adds_exact_pause(self, tmp_path):
        source = tmp_path / "source.wav"
        out = tmp_path / "padded.wav"
        _write_wav(source, 0.5)

        media.append_silence(source, out, 0.3)

        assert media.wav_duration(out) == pytest.approx(0.8, abs=0.01)
        with wave.open(str(out), "rb") as stream:
            stream.setpos(int(0.7 * media.VOICE_RATE))
            assert set(stream.readframes(10)) == {0}

    def test_build_voice_track_places_segments(self, tmp_path):
        seg = tmp_path / "seg.wav"
        _write_wav(seg, 0.5)
        out = tmp_path / "track.wav"
        media.build_voice_track([(1.0, seg)], out, total_duration=3.0)
        assert media.wav_duration(out) == pytest.approx(3.0, abs=0.02)
        with wave.open(str(out), "rb") as wf:
            wf.setpos(int(0.2 * media.VOICE_RATE))
            silent = wf.readframes(10)
            wf.setpos(int(1.2 * media.VOICE_RATE))
            loud = wf.readframes(10)
        assert set(silent) == {0}
        assert set(loud) != {0}

    def test_build_voice_track_ignores_bad_segment(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"khong phai wav")
        out = tmp_path / "track.wav"
        media.build_voice_track([(0.0, bad)], out, total_duration=1.0)
        assert out.is_file()

    def test_build_voice_track_clips_overflow(self, tmp_path):
        seg = tmp_path / "seg.wav"
        _write_wav(seg, 2.0)
        out = tmp_path / "track.wav"
        media.build_voice_track([(0.9, seg)], out, total_duration=1.0)
        assert media.wav_duration(out) == pytest.approx(1.0, abs=0.02)

    def test_build_voice_track_can_mix_overlapping_lines(self, tmp_path):
        first = tmp_path / "first.wav"
        second = tmp_path / "second.wav"
        _write_wav(first, 1.0, value=20_000)
        _write_wav(second, 1.0, value=20_000)
        out = tmp_path / "mixed.wav"

        media.build_voice_track(
            [(0.0, first), (0.5, second)],
            out,
            total_duration=2.0,
            mix_overlaps=True,
        )

        with wave.open(str(out), "rb") as wf:
            wf.setpos(int(0.75 * media.VOICE_RATE))
            sample = int.from_bytes(wf.readframes(1), "little", signed=True)
        assert sample == 32767

    def test_to_wav_applies_speed_and_pitch_independently(self, tmp_path):
        class FakeFF:
            def __init__(self):
                self.args = []

            def run(self, args, **_kwargs):
                self.args = list(args)
                _write_wav(Path(args[-1]), 0.1)

        ff = FakeFF()
        source = tmp_path / "source.wav"
        _write_wav(source, 0.1)

        media.to_wav(ff, source, tmp_path / "out.wav", tempo=1.2, pitch=1.1)

        graph = ff.args[ff.args.index("-filter:a") + 1]
        assert "asetrate=" in graph
        assert "atempo=" in graph

    def test_retime_video_segments_stretches_video_and_audio(self, tmp_path):
        class FakeFF:
            ffmpeg = "ffmpeg"

            def __init__(self):
                self.calls = []

            def probe(self, _path):
                return MediaInfo(path="source.mp4", fps=25.0, has_video=True, has_audio=True)

            def run(self, args, **_kwargs):
                self.calls.append(list(args))
                Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(args[-1]).write_bytes(b"media")

        ff = FakeFF()
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        out = tmp_path / "retimed.mp4"

        media.retime_video_segments(
            ff,
            source,
            [(0.0, 2.0, 3.0), (2.0, 4.0, 2.0)],
            tmp_path / "parts",
            out,
            workers=1,
        )

        first_graph = ff.calls[0][ff.calls[0].index("-filter_complex") + 1]
        assert "setpts=1.500000000*PTS" in first_graph
        assert "atempo=" in first_graph
        assert ff.calls[-1][:4] == ["-f", "concat", "-safe", "0"]
        assert out.is_file()


class TestPaths:
    def test_safe_name_strips_invalid(self):
        assert paths.safe_name('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"

    def test_safe_name_keeps_vietnamese(self):
        assert paths.safe_name("Tap 1 - Phim hay") == "Tap 1 - Phim hay"

    def test_safe_name_fallback(self):
        assert paths.safe_name("   ") == "du-an"

    def test_safe_name_reserved(self):
        assert paths.safe_name("CON").startswith("_")

    def test_unique_path(self, tmp_path):
        first = tmp_path / "a.txt"
        first.write_text("x", encoding="utf-8")
        assert paths.unique_path(first).name == "a_1.txt"

    def test_write_text_atomic(self, tmp_path):
        target = tmp_path / "sub" / "f.txt"
        paths.write_text_atomic(target, "noi dung")
        assert target.read_text(encoding="utf-8") == "noi dung"
        assert not (tmp_path / "sub" / "f.txt.tmp").exists()

    def test_human_size(self):
        assert paths.human_size(512) == "512 B"
        assert paths.human_size(2 * 1024 * 1024).endswith("MB")

    def test_not_portable_when_running_from_source(self):
        assert paths.is_portable() is False

    def test_config_dir_follows_appdata_when_not_portable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        paths.is_portable.cache_clear()
        assert paths.config_dir() == tmp_path / paths.APP_DIR_NAME

    def test_bundled_dir_missing(self):
        assert paths.bundled_dir("khong-ton-tai-dau") is None

    def test_bundled_dir_found(self, monkeypatch, tmp_path):
        (tmp_path / "ffmpeg").mkdir()
        monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
        assert paths.bundled_dir("ffmpeg") == tmp_path / "ffmpeg"

    def test_portable_data_dir_next_to_app(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
        assert paths.portable_data_dir() == tmp_path / "Data"


class TestProjectStore:
    def test_create_and_load(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(1, "Phim thu nghiem")
        data.doc = SubtitleDoc(cues=[Cue(0, 1, "xin chao")])
        store.save(data)
        again = store.load(data.folder)
        assert again.name == "Phim thu nghiem"
        assert again.doc.cues[0].text == "xin chao"

    def test_load_rejects_newer_schema(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(2, "x")
        raw = json.loads((store.projects_dir / "2-x" / "project.json").read_text("utf-8"))
        raw["schema_version"] = 99
        (store.projects_dir / "2-x" / "project.json").write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ProjectFormatError):
            store.load(data.folder)

    def test_load_broken_json(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(3, "y")
        (store.projects_dir / "3-y" / "project.json").write_text("{hong", encoding="utf-8")
        with pytest.raises(ProjectFormatError):
            store.load(data.folder)

    def test_project_round_trip_keeps_dub_timing(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(44, "dub")
        data.dub_timing = [[0.0, 2.5], [2.5, 4.75]]
        store.save(data)

        loaded = store.load(data.folder)

        assert loaded.dub_timing == [[0.0, 2.5], [2.5, 4.75]]

    def test_recovery_cycle(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(4, "z")
        store.save_recovery(data)
        assert store.recovery_path(data.folder) is not None
        store.save(data)
        assert store.recovery_path(data.folder) is None

    def test_clean_temp(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(5, "t")
        junk = data.sub_dir("temp") / "junk.bin"
        junk.write_bytes(b"0" * 1024)
        freed = store.clean_temp(data)
        assert freed >= 1024
        assert not junk.exists()

    def test_clean_ocr_temp_keeps_non_ocr_project_files(self, tmp_path):
        store = ProjectStore(tmp_path)
        data = store.create(6, "ocr")
        temp = data.sub_dir("temp")
        frames = temp / "ocr_frames"
        window = temp / "ocr_window"
        frames.mkdir()
        window.mkdir()
        (frames / "frame_000001.png").write_bytes(b"f" * 512)
        (window / "win_0001.png").write_bytes(b"w" * 256)
        keep = temp / "du-lieu-khac.bin"
        keep.write_bytes(b"giu lai")

        freed = store.clean_ocr_temp(data)

        assert freed == 768
        assert not frames.exists()
        assert not window.exists()
        assert keep.read_bytes() == b"giu lai"

    def test_clean_stale_ocr_temp_scans_all_projects(self, tmp_path):
        store = ProjectStore(tmp_path)
        first = store.create(7, "mot")
        second = store.create(8, "hai")
        for data in (first, second):
            probe = data.sub_dir("temp") / "ocr_probe"
            probe.mkdir()
            (probe / "probe_001.png").write_bytes(b"p" * 100)

        freed = store.clean_stale_ocr_temp()

        assert freed == 200
        assert not (Path(first.folder) / "temp" / "ocr_probe").exists()
        assert not (Path(second.folder) / "temp" / "ocr_probe").exists()

    def test_load_relocates_a_project_copied_with_portable_tool(self, tmp_path):
        workspace = tmp_path / "new-machine" / "Data" / "workspace"
        store = ProjectStore(workspace)
        data = store.create(9, "portable")
        new_folder = Path(data.folder)
        normalized = new_folder / "video" / "normalized.mp4"
        normalized.write_bytes(b"video")
        project_file = new_folder / "project.json"
        raw = json.loads(project_file.read_text(encoding="utf-8"))
        old_folder = Path("C:/Users/example/OldTool/Data/workspace/projects") / new_folder.name
        raw["folder"] = str(old_folder)
        raw["video_path"] = str(old_folder / "video" / "normalized.mp4")
        project_file.write_text(json.dumps(raw), encoding="utf-8")

        moved = store.load(old_folder)

        assert Path(moved.folder) == new_folder
        assert Path(moved.video_path) == normalized

    def test_delete_only_inside_workspace(self, tmp_path):
        store = ProjectStore(tmp_path)
        outside = tmp_path.parent / "khong-xoa"
        outside.mkdir(exist_ok=True)
        store.delete(outside)
        assert outside.exists()
        outside.rmdir()


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.workspace
        assert s.max_workers >= 1
        assert s.ocr_mode == "Nhanh Như NTS"
        assert s.ocr_language == "Simplified Chinese"
        assert s.ocr_fps == 15.0
        assert s.ocr_confidence == 70.0
        assert s.ocr_color_filter is False
        assert s.ocr_drop_static is False
        assert s.tts_provider == "Local Voice"
        assert s.tts_voice == ""
        assert s.local_voice == ""
        assert s.dub_output_mode == "video"
        assert s.original_audio_volume == 20

    def test_named_config_profiles_round_trip(self):
        settings = Settings()
        settings.tts_voice = "voice-default"
        settings.remember_active_profile()
        settings.active_config_profile = "review"
        settings.tts_voice = "voice-review"
        settings.original_audio_volume = 12
        settings.remember_active_profile()

        assert settings.apply_config_profile("default") is True
        assert settings.tts_voice == "voice-default"
        assert settings.apply_config_profile("review") is True
        assert settings.tts_voice == "voice-review"
        assert settings.original_audio_volume == 12

    @pytest.mark.parametrize("old_voice", ["Microsoft Zira Desktop", "Microsoft David Desktop"])
    @pytest.mark.parametrize("old_schema", [7, 9, 10])
    def test_old_english_sapi_voice_is_migrated_to_free_vietnamese(
        self, tmp_path, monkeypatch, old_voice, old_schema
    ):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps(
                {
                    "schema_version": old_schema,
                    "tts_provider": "Windows SAPI (offline)",
                    "tts_voice": old_voice,
                }
            ),
            encoding="utf-8",
        )

        upgraded = Settings.load()

        assert upgraded.tts_provider == "Local Voice"
        assert upgraded.tts_voice == ""
        assert upgraded.local_voice == ""
        assert upgraded.config_profiles["default"]["dub_output_mode"] == "video"
        assert upgraded.config_profiles["default"]["tts_provider"] == "Local Voice"

    def test_legacy_ocr_config_is_upgraded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps({"ocr_fps": 2.0, "ocr_confidence": 6.0, "ocr_similarity": 0.7}),
            encoding="utf-8",
        )
        upgraded = Settings.load()
        assert upgraded.ocr_mode == "Nhanh Như NTS"
        assert upgraded.ocr_fps == 15.0
        assert upgraded.ocr_confidence == 70.0
        assert upgraded.ocr_consensus == 1
        assert upgraded.ocr_color_filter is False
        assert upgraded.ocr_drop_static is False
        saved = json.loads(Settings.config_path().read_text(encoding="utf-8"))
        assert saved["schema_version"] == 12

    def test_schema_one_extreme_default_contrast_is_migrated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps({"schema_version": 1, "ocr_mode": "Chính Xác", "ocr_contrast": 100}),
            encoding="utf-8",
        )

        upgraded = Settings.load()

        assert upgraded.schema_version == 12
        assert upgraded.ocr_contrast == 0
        saved = json.loads(Settings.config_path().read_text(encoding="utf-8"))
        assert saved["schema_version"] == 12
        assert saved["ocr_contrast"] == 0

    def test_schema_four_nts_auto_filters_are_removed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "ocr_mode": "Nhanh Như NTS",
                    "ocr_server": "PP-OCRv4 Mobile (Nhanh Như NTS)",
                    "ocr_fps": 5.0,
                    "ocr_color_filter": True,
                    "ocr_text_color": "#FEFEFD",
                    "ocr_drop_static": True,
                }
            ),
            encoding="utf-8",
        )

        upgraded = Settings.load()

        assert upgraded.schema_version == 12
        assert upgraded.ocr_fps == 15.0
        assert upgraded.ocr_color_filter is False
        assert upgraded.ocr_text_color == ""
        assert upgraded.ocr_drop_static is False

    def test_schema_five_nts_rate_is_kept_at_high_quality(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "ocr_mode": "Nhanh Như NTS",
                    "ocr_server": "PP-OCRv4 Mobile (Nhanh Như NTS)",
                    "ocr_fps": 15.0,
                }
            ),
            encoding="utf-8",
        )

        upgraded = Settings.load()

        assert upgraded.schema_version == 12
        assert upgraded.ocr_fps == 15.0

    def test_schema_six_low_nts_rate_is_restored_for_quality(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps(
                {
                    "schema_version": 6,
                    "ocr_mode": "Nhanh Như NTS",
                    "ocr_server": "PP-OCRv4 Mobile (Nhanh Như NTS)",
                    "ocr_fps": 3.0,
                }
            ),
            encoding="utf-8",
        )

        upgraded = Settings.load()

        assert upgraded.schema_version == 12
        assert upgraded.ocr_fps == 15.0

    def test_style_round_trip(self):
        style = SubtitleStyle(font="Tahoma", font_size=60)
        assert SubtitleStyle.from_dict(style.to_dict()).font == "Tahoma"

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        s = Settings()
        s.asr_model = "medium"
        s.style.font_size = 72
        s.save()
        again = Settings.load()
        assert again.asr_model == "medium"
        assert again.style.font_size == 72

    def test_load_broken_config_uses_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text("{hong", encoding="utf-8")
        assert Settings.load().asr_model == "small"

    def test_portable_config_rebases_paths_from_the_old_computer(
        self, tmp_path, monkeypatch
    ):
        config = tmp_path / "new-tool" / "Data" / "config"
        config.mkdir(parents=True)
        workspace = tmp_path / "new-tool" / "Data" / "workspace"
        old_workspace = "C:/Users/example/OldTool/Data/workspace"
        (config / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "workspace": old_workspace,
                    "ffmpeg_path": "C:/Users/example/OldTool/ffmpeg.exe",
                    "model_dir": "C:/Users/example/OldTool/model",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_service, "config_dir", lambda: config)
        monkeypatch.setattr(settings_service, "default_workspace", lambda: workspace)
        monkeypatch.setattr(settings_service, "is_portable", lambda: True)

        loaded = Settings.load()

        assert Path(loaded.workspace) == workspace
        assert loaded.ffmpeg_path == ""
        assert loaded.model_dir == ""
        saved = json.loads((config / "config.json").read_text(encoding="utf-8"))
        assert Path(saved["workspace"]) == workspace

    def test_portable_config_keeps_an_explicit_custom_workspace(
        self, tmp_path, monkeypatch
    ):
        config = tmp_path / "Data" / "config"
        config.mkdir(parents=True)
        custom = tmp_path / "my-projects"
        (config / "config.json").write_text(
            json.dumps({"schema_version": 5, "workspace": str(custom)}),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_service, "config_dir", lambda: config)
        monkeypatch.setattr(
            settings_service, "default_workspace", lambda: tmp_path / "Data" / "workspace"
        )
        monkeypatch.setattr(settings_service, "is_portable", lambda: True)

        loaded = Settings.load()

        assert Path(loaded.workspace) == custom

    def test_hardware_result_is_cleared_after_copying_to_another_machine(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "hardware_machine_id": "old-machine",
                    "hardware_signature": "old-gpu-result",
                    "use_gpu": True,
                    "use_gpu_encoder": True,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(gpu, "machine_id", lambda: "new-machine")

        loaded = Settings.load()

        assert loaded.hardware_machine_id == "new-machine"
        assert loaded.hardware_signature == ""
        assert loaded.use_gpu is False
        assert loaded.use_gpu_encoder is False

    def test_hardware_result_is_kept_on_the_machine_that_checked_it(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.config_path().write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "hardware_machine_id": "this-machine",
                    "hardware_signature": "checked-gpu",
                    "use_gpu": True,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(gpu, "machine_id", lambda: "this-machine")

        loaded = Settings.load()

        assert loaded.hardware_signature == "checked-gpu"
        assert loaded.use_gpu is True

    def test_secret_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.set_secret("ai_gateway_key", "sk-test-123")
        assert Settings.get_secret("ai_gateway_key") == "sk-test-123"
        Settings.set_secret("ai_gateway_key", "")
        assert Settings.get_secret("ai_gateway_key") == ""

    def test_secret_not_stored_in_plain_text(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        Settings.set_secret("ai_gateway_key", "sk-bi-mat")
        blob = (tmp_path / "AutoSubStudio" / "secrets.dat").read_bytes()
        assert b"sk-bi-mat" not in blob


class TestDatabase:
    def test_create_and_query(self, tmp_path):
        db = Database(tmp_path / "app.db")
        with db.session() as s:
            s.add(Project(name="Du an 1"))
        with db.session() as s:
            assert s.query(Project).count() == 1

    def test_default_script(self, tmp_path):
        db = Database(tmp_path / "app.db")
        db.ensure_default_scripts()
        with db.session() as s:
            script = s.query(AutoScript).first()
            assert script is not None
            assert script.steps
            assert STEP_DUB in script.steps

    def test_old_default_script_gets_the_dubbing_step(self, tmp_path):
        from autosub_studio.pipeline.steps import STEP_ASR, STEP_RENDER, STEP_TRANSLATE

        db = Database(tmp_path / "app.db")
        with db.session() as session:
            script = AutoScript(name="Mac dinh")
            script.steps = [STEP_ASR, STEP_TRANSLATE, STEP_RENDER]
            session.add(script)

        db.ensure_default_scripts()

        with db.session() as session:
            script = session.query(AutoScript).filter(AutoScript.name == "Mac dinh").one()
            assert script.steps == [STEP_ASR, STEP_TRANSLATE, STEP_DUB, STEP_RENDER]

    def test_old_database_script_containing_deprecated_capcut_step_handled_safely(self, tmp_path):
        from autosub_studio.pipeline.steps import STEP_ASR, STEP_RENDER

        db = Database(tmp_path / "app.db")
        with db.session() as session:
            script = AutoScript(name="Custom Script")
            script.steps_json = json.dumps(["Tao CapCut draft", STEP_ASR, STEP_RENDER])
            session.add(script)

        # Loading script skips deprecated step without crash
        with db.session() as session:
            sc = session.query(AutoScript).filter(AutoScript.name == "Custom Script").one()
            assert sc.steps == [STEP_ASR, STEP_RENDER]

        # Normalization removes deprecated step from database
        db.ensure_default_scripts()
        with db.session() as session:
            sc = session.query(AutoScript).filter(AutoScript.name == "Custom Script").one()
            assert json.loads(sc.steps_json) == [STEP_ASR, STEP_RENDER]

    def test_recovers_projects_left_running_after_forced_exit(self, tmp_path):
        db = Database(tmp_path / "app.db")
        with db.session() as s:
            s.add(
                Project(
                    name="Dang OCR",
                    current_task="Tách Sub Bằng Chữ",
                    status="Đang Trích Xuất...",
                    progress=37,
                )
            )
            s.add(Project(name="Da xong", current_task="", status="Xong", progress=100))

        assert db.recover_interrupted_projects() == 1
        with db.session() as s:
            running = s.query(Project).filter(Project.name == "Dang OCR").one()
            finished = s.query(Project).filter(Project.name == "Da xong").one()
            assert running.current_task == ""
            assert running.progress == 0
            assert "đóng trước đó" in running.status
            assert finished.status == "Xong"

    def test_reopen_existing_database(self, tmp_path):
        path = tmp_path / "app.db"
        Database(path)
        db = Database(path)
        with db.session() as s:
            assert s.query(Project).count() == 0

    def test_rollback_on_error(self, tmp_path):
        db = Database(tmp_path / "app.db")
        with pytest.raises(RuntimeError), db.session() as s:
            s.add(Project(name="hong"))
            raise RuntimeError("that bai")
        with db.session() as s:
            assert s.query(Project).count() == 0


class TestTaskManager:
    def test_never_runs_more_than_two_jobs_at_once(self, qapp):
        manager = TaskManager(2)
        lock = threading.Lock()
        release = threading.Event()
        state = {"active": 0, "maximum": 0}
        done: list[tuple] = []
        manager.task_finished.connect(lambda *args: done.append(args))

        def held_job(ctx):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            release.wait(3.0)
            with lock:
                state["active"] -= 1
            return "ok"

        for number in range(5):
            manager.submit(f"job-{number}", held_job)
        _wait(qapp, lambda: state["active"] == 2)
        statuses = [record.status for record in manager.records()]
        assert statuses.count(RUNNING) == 2
        assert statuses.count(PENDING) == 3
        release.set()
        _wait(qapp, lambda: len(done) == 5)
        assert state["maximum"] == 2

    def test_runs_and_reports_result(self, qapp):
        manager = TaskManager(2)
        results: list[tuple] = []
        manager.task_finished.connect(lambda *args: results.append(args))
        manager.submit("cong viec", lambda ctx: "xong roi")
        _wait(qapp, lambda: bool(results))
        assert results[0][1] == DONE
        assert results[0][2] == "xong roi"

    def test_reports_failure_message(self, qapp):
        manager = TaskManager(1)
        results: list[tuple] = []
        manager.task_finished.connect(lambda *args: results.append(args))

        def boom(ctx):
            raise ValueError("loi nghiep vu")

        manager.submit("hong", boom)
        _wait(qapp, lambda: bool(results))
        assert results[0][1] == FAILED
        assert "loi nghiep vu" in results[0][3]

    def test_cancel_stops_task(self, qapp):
        manager = TaskManager(1)
        results: list[tuple] = []
        manager.task_finished.connect(lambda *args: results.append(args))

        def slow(ctx):
            for _ in range(200):
                ctx.check_cancel()
                time.sleep(0.01)
            return "khong nen toi day"

        task_id = manager.submit("cham", slow)
        _wait(qapp, lambda: manager.record(task_id).status == "dang chay")
        manager.cancel(task_id)
        _wait(qapp, lambda: bool(results), timeout=8.0)
        assert results[0][1] == CANCELLED

    def test_progress_signal(self, qapp):
        manager = TaskManager(1)
        seen: list[int] = []
        manager.task_progress.connect(lambda _tid, p: seen.append(p))
        manager.submit("tien trinh", lambda ctx: [ctx.progress(50), ctx.progress(100)])
        _wait(qapp, lambda: 100 in seen)
        assert 50 in seen

    def test_clear_finished(self, qapp):
        manager = TaskManager(1)
        done: list[tuple] = []
        manager.task_finished.connect(lambda *args: done.append(args))
        manager.submit("nhanh", lambda ctx: "ok")
        _wait(qapp, lambda: bool(done))
        manager.clear_finished()
        assert manager.records() == []


def _wait(qapp, predicate, timeout: float = 5.0) -> None:
    """Cho den khi dieu kien dung, van xu ly su kien Qt."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    qapp.processEvents()
    assert predicate(), "Cho qua lau ma dieu kien van chua dung"


def test_cancel_token():
    token = CancelToken()
    assert not token.cancelled
    token.cancel()
    assert token.cancelled
