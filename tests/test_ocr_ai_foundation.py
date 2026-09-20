"""Kiem thu nen tang OCR AI: cach ly khoi RapidOCR, di cu cai dat,
cache key, WAL checkpoint va tuong thich cache cu.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from autosub_studio.core.models import Cue
from autosub_studio.core.ocr_common import Row
from autosub_studio.data import ocr_cache
from autosub_studio.data.project import ProjectData, ProjectStore
from autosub_studio.pipeline import steps
from autosub_studio.providers import ocr, ocr_ai
from autosub_studio.services import ai_gateway
from autosub_studio.services.ffmpeg import CancelToken, FFmpeg
from autosub_studio.services.settings import Settings
from autosub_studio.services.tasks import TaskContext


def _dummy_task_context() -> TaskContext:
    return TaskContext(
        token=CancelToken(),
        _progress=lambda _p: None,
        _log=lambda _m: None,
    )


# ========================================================== 1. Zero RapidOCR in AI branch
class TestZeroLocalRapidOCRInAI:
    def test_prepare_filter_never_calls_local_probe_in_ai_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden_probe(*_args, **_kwargs):
            raise AssertionError("ocr.probe_frames was called in AI mode!")

        monkeypatch.setattr(ocr, "probe_frames", forbidden_probe)

        settings = Settings(ocr_mode="OCR AI", ocr_server="Server AI API")
        pc = steps.PipelineContext(
            ff=MagicMock(spec=FFmpeg),
            settings=settings,
            store=MagicMock(spec=ProjectStore),
            project=ProjectData(folder=str(tmp_path)),
            task=_dummy_task_context(),
        )

        dummy_frame = tmp_path / "frame.png"
        dummy_frame.write_bytes(b"dummy")

        flt = steps._prepare_filter(pc, [dummy_frame])
        assert flt is not None

        flt_explicit = steps._prepare_filter(pc, [dummy_frame], is_ai=True)
        assert flt_explicit is not None

    def test_step_ocr_measure_refuses_in_ai_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden_call(*_args, **_kwargs):
            raise AssertionError("Local OCR was called in step_ocr_measure in AI mode!")

        monkeypatch.setattr(ocr, "probe_frames", forbidden_call)
        monkeypatch.setattr(ocr, "is_available", forbidden_call)

        settings = Settings(ocr_mode="OCR AI", ocr_server="Server AI API")
        pc = steps.PipelineContext(
            ff=MagicMock(spec=FFmpeg),
            settings=settings,
            store=MagicMock(spec=ProjectStore),
            project=ProjectData(folder=str(tmp_path)),
            task=_dummy_task_context(),
        )

        with pytest.raises(steps.StepError) as excinfo:
            steps.step_ocr_measure(pc)
        assert "không hỗ trợ" in str(excinfo.value)

    def test_step_ocr_never_calls_local_read_or_probe_in_ai_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden_call(*_args, **_kwargs):
            raise AssertionError("Local OCR was called in step_ocr in AI mode!")

        monkeypatch.setattr(ocr, "probe_frames", forbidden_call)
        monkeypatch.setattr(ocr, "read_frames", forbidden_call)
        monkeypatch.setattr(ocr, "is_available", forbidden_call)

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake_video")

        settings = Settings(
            ocr_mode="OCR AI",
            ocr_server="Server AI API",
            ai_endpoint="http://localhost:8000/v1",
        )
        project = ProjectData(
            folder=str(tmp_path),
            video_path=str(video_path),
            ocr_region=[0, 100, 200, 50],
            duration=5.0,
        )
        pc = steps.PipelineContext(
            ff=MagicMock(spec=FFmpeg),
            settings=settings,
            store=MagicMock(spec=ProjectStore),
            project=project,
            task=_dummy_task_context(),
            api_key="test-key",
        )

        frame1 = tmp_path / "f1.png"
        frame1.write_bytes(b"frame1")
        monkeypatch.setattr(
            steps.media, "extract_frames", lambda *_args, **_kwargs: [(0.0, frame1)]
        )

        ai_called = []

        def mock_run_ai_ocr_pipeline(*_args, **_kwargs):
            ai_called.append(True)
            return [Cue(0.0, 1.0, "Test AI Cue")]

        monkeypatch.setattr(ocr_ai, "run_ai_ocr_pipeline", mock_run_ai_ocr_pipeline)
        monkeypatch.setattr(ocr_ai, "read_frames_ai", mock_run_ai_ocr_pipeline)

        msg = steps.step_ocr(pc)
        assert "1 cau" in msg
        assert len(ai_called) == 1
        assert len(pc.project.doc.cues) == 1
        assert pc.project.doc.cues[0].text == "Test AI Cue"


# ========================================================== 2. Settings v13 and Migration
class TestSettingsV13AndMigration:
    def test_default_settings_schema_and_values(self) -> None:
        s = Settings()
        assert s.schema_version == 13
        assert s.ocr_ai_batch_size == 8
        assert s.ocr_ai_max_concurrency == 4
        assert s.ocr_ai_timeout == 60
        assert s.ocr_ai_max_retries == 3
        assert s.ocr_ai_image_quality == 88
        assert s.ocr_ai_diff_threshold == 4.0
        assert s.ocr_ai_consensus_mode == "disabled"
        assert s.ocr_ai_consensus_frames == 1
        assert s.ocr_ai_prompt_version == "v1"
        assert s.ocr_ai_custom_prompt == ""
        # Local ocr_batch_size separate
        assert s.ocr_batch_size == 5

    def test_migration_from_v12_preserves_and_sets_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "config.json"
        monkeypatch.setattr(Settings, "config_path", staticmethod(lambda *_args: cfg_file))

        # Older schema 12 payload without v13 fields
        old_data = {
            "schema_version": 12,
            "ocr_batch_size": 10,  # user custom local batch
            "ocr_mode": "OCR AI",
            "ocr_server": "Server AI API",
            "ai_endpoint": "http://my-ai/v1",
        }
        import json

        cfg_file.write_text(json.dumps(old_data, ensure_ascii=False), encoding="utf-8")

        loaded = Settings.load()
        assert loaded.schema_version == 13
        assert loaded.ocr_batch_size == 10  # preserved local
        assert loaded.ocr_ai_batch_size == 8  # default
        assert loaded.ocr_ai_max_concurrency == 4
        assert loaded.ocr_ai_timeout == 60
        assert loaded.ocr_ai_max_retries == 3
        assert loaded.ocr_ai_image_quality == 88
        assert loaded.ocr_ai_diff_threshold == 4.0
        assert loaded.ocr_ai_consensus_mode == "disabled"
        assert loaded.ocr_ai_consensus_frames == 1
        assert loaded.ocr_ai_prompt_version == "v1"

    def test_migration_repeat_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "config.json"
        monkeypatch.setattr(Settings, "config_path", staticmethod(lambda *_args: cfg_file))

        s = Settings(
            schema_version=13,
            ocr_ai_batch_size=16,
            ocr_ai_max_concurrency=8,
            ocr_ai_timeout=90,
            ocr_ai_custom_prompt="Custom OCR prompt",
        )
        s.save()

        # Load twice, ensure custom values are never reset
        loaded1 = Settings.load()
        assert loaded1.ocr_ai_batch_size == 16
        assert loaded1.ocr_ai_max_concurrency == 8
        assert loaded1.ocr_ai_timeout == 90
        assert loaded1.ocr_ai_custom_prompt == "Custom OCR prompt"

        loaded2 = Settings.load()
        assert loaded2.ocr_ai_batch_size == 16
        assert loaded2.ocr_ai_max_concurrency == 8


# ========================================================== 3. Cache Key Invalidation
class TestAICacheKeyInvalidation:
    def test_cache_key_invalidation_on_parameters(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"content_1")

        base_params: dict[str, Any] = {
            "video": video,
            "region": [0, 100, 200, 50],
            "fps": 2.0,
            "endpoint": "http://localhost:8000/v1",
            "model_alias": "sub",
            "actual_model": "sub",
            "thinking": "low",
            "prompt_version": "v1",
            "custom_prompt": "",
            "preprocessing_version": "v1",
            "diff_threshold": 4.0,
            "image_quality": 88,
            "consensus_mode": "disabled",
            "consensus_frames": 1,
        }

        k_base = ocr_cache.make_ai_ocr_cache_key(**base_params)

        # Change model
        k_model = ocr_cache.make_ai_ocr_cache_key(
            **{**base_params, "model_alias": "prime", "actual_model": "prime"}
        )
        assert k_model != k_base

        # Change region
        k_region = ocr_cache.make_ai_ocr_cache_key(
            **{**base_params, "region": [0, 150, 200, 50]}
        )
        assert k_region != k_base

        # Change prompt version
        k_prompt = ocr_cache.make_ai_ocr_cache_key(
            **{**base_params, "prompt_version": "v2"}
        )
        assert k_prompt != k_base

        # Change custom prompt
        k_custom = ocr_cache.make_ai_ocr_cache_key(
            **{**base_params, "custom_prompt": "Read carefully"}
        )
        assert k_custom != k_base

        # Change endpoint
        k_ep = ocr_cache.make_ai_ocr_cache_key(
            **{**base_params, "endpoint": "http://remote-gateway/v1"}
        )
        assert k_ep != k_base

        # Change video content/mtime
        video.write_bytes(b"content_longer_version")
        k_vid = ocr_cache.make_ai_ocr_cache_key(**base_params)
        assert k_vid != k_base

    def test_api_key_never_stored_in_cache(self, tmp_path: Path) -> None:
        cache_db = tmp_path / "ocr_cache.sqlite3"
        video = tmp_path / "v.mp4"
        video.write_bytes(b"data")

        cache_key = ocr_cache.make_ai_ocr_cache_key(
            video=video,
            region=[0, 0, 10, 10],
            fps=1.0,
            endpoint="http://localhost:8000/v1",
            model_alias="sub",
            actual_model="sub",
        )

        ocr_cache.save_cache_meta(
            cache_db,
            cache_key,
            {
                "engine": "ai_gateway",
                "endpoint_hash": ocr_cache.hash_endpoint("http://localhost:8000/v1"),
                "model_alias": "sub",
                "actual_model": "sub",
            },
        )

        meta = ocr_cache.get_cache_meta(cache_db, cache_key)
        assert meta is not None
        assert "api_key" not in meta
        # Verify db file text does not contain secret
        content = cache_db.read_bytes()
        assert b"secret" not in content


# ========================================================== 4. WAL Resume & Immediate Checkpoint
class TestWALImmediateResumeAndConcurrency:
    def test_immediate_batch_checkpoint_and_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(
            ai_endpoint="http://localhost:8000/v1",
            ocr_ai_batch_size=2,
            ocr_ai_max_retries=1,
        )
        monkeypatch.setattr(Settings, "load", lambda: settings)
        monkeypatch.setattr(Settings, "get_secret", lambda _name: "test-key")

        # Create 4 distinct frames with high contrast (not blank)
        frames = []
        for i in range(4):
            f = tmp_path / f"frame_{i}.jpg"
            img = Image.new("RGB", (64, 64), color=(0, 0, 0))
            for x in range(10 + i * 10, 60):
                for y in range(10, 50):
                    img.putpixel((x, y), (255, 255, 255))
            img.save(f)
            frames.append(f)

        distinct_texts = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]
        total_calls = [0]
        ai_calls = []

        def mock_ai(*_args, **_kwargs):
            ai_calls.append(True)
            text = distinct_texts[total_calls[0] % len(distinct_texts)]
            total_calls[0] += 1
            return text

        monkeypatch.setattr(ai_gateway, "ocr_image_with_ai", mock_ai)

        cache_db = tmp_path / "ocr_cache.sqlite3"
        cache_key = "test_resume_key"

        def should_cancel():
            return len(ai_calls) >= 2

        cues1 = ocr_ai.read_frames_ai(
            frames,
            fps=1.0,
            cache_path=cache_db,
            cache_key=cache_key,
            batch_size=2,
            should_cancel=should_cancel,
        )
        assert len(cues1) >= 1

        assert len(ai_calls) == 2
        # Check SQLite WAL has the checkpointed batch
        assert cache_db.is_file()
        with sqlite3.connect(cache_db) as conn:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "wal"
            count = conn.execute(
                "SELECT count(*) FROM frames WHERE cache_key=?", (cache_key,)
            ).fetchone()[0]
            assert count == 2

        # Second run: resume with all 4 frames. Only frames 2 and 3 should call AI.
        ai_calls.clear()
        cues2 = ocr_ai.read_frames_ai(
            frames,
            fps=1.0,
            cache_path=cache_db,
            cache_key=cache_key,
            batch_size=2,
        )

        assert len(ai_calls) == 2  # Only the remaining 2 frames called AI
        assert len(cues2) >= 3

    def test_concurrent_writes_and_reads_thread_safety(self, tmp_path: Path) -> None:
        cache_db = tmp_path / "concurrent_cache.sqlite3"
        cache_key = "concurrent_key"

        def worker(w_id: int):
            batch = {
                w_id * 10 + j: [Row((0.0, 0.0), f"text_{w_id}_{j}", 0.9, None)]
                for j in range(10)
            }
            hashes = {w_id * 10 + j: f"hash_{w_id}_{j}" for j in range(10)}
            ocr_cache.save_frame_batch(cache_db, cache_key, batch, hashes)
            read_back = ocr_cache.load_frame_cache(cache_db, cache_key, 200)
            assert len(read_back) >= 10

        threads = []
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        total = ocr_cache.load_frame_cache(cache_db, cache_key, 200)
        assert len(total) == 80


# =========================================================================== 5. Legacy Cache Table
class TestLegacyCacheTable:
    def test_preserves_old_project_cache_opening(self, tmp_path: Path) -> None:
        cache_db = tmp_path / "legacy_cache.sqlite3"
        # Create legacy table without content_hash or ai_cache_meta
        with sqlite3.connect(cache_db) as conn:
            conn.execute(
                "CREATE TABLE frames ("
                "cache_key TEXT NOT NULL, "
                "frame_idx INTEGER NOT NULL, "
                "rows_json TEXT NOT NULL, "
                "PRIMARY KEY(cache_key, frame_idx))"
            )
            import json

            legacy_rows = json.dumps([[[0.0, 0.0], "Old Caption", 0.95, [10, 20, 100, 50]]])
            conn.execute(
                "INSERT INTO frames VALUES (?, ?, ?)",
                ("old_key", 0, legacy_rows),
            )
            conn.commit()

        # Load using load_frame_cache
        loaded = ocr_cache.load_frame_cache(cache_db, "old_key", 10)
        assert 0 in loaded
        assert loaded[0][0].text == "Old Caption"
        assert loaded[0][0].score == 0.95
        assert loaded[0][0].rect == (10.0, 20.0, 100.0, 50.0)

        # Save a new batch: table should migrate column seamlessly
        new_batch = {1: [Row((0.0, 0.0), "New Caption", 0.99, None)]}
        ocr_cache.save_frame_batch(cache_db, "old_key", new_batch, {1: "hash1"})

        updated = ocr_cache.load_frame_cache(cache_db, "old_key", 10)
        assert len(updated) == 2
        assert updated[1][0].text == "New Caption"
