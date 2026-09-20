"""Cac bai kiem tra toan dien cho AI OCR Pipeline (Contract ai-ocr-pipeline-r4).

14 bai kiem tra bat buoc:
1. crop region respected: mock extraction nhan dung region va gateway nhan dung kich thuoc crop.
2. no local symbols called: khong goi bat ky ham/bieu tuong nao cua local OCR.
3. redundant frame reduction: loc bo frame trong/tinh, giam so luong frame can gui len AI.
4. batch creation: tao batch dung kich thuoc ocr_ai_batch_size voi ID xac dinh, duy nhat.
5. split via provider: batch gap loi 413 / payload too large tu dong chia doi va thuc thi.
6. global scheduler use: su dung AIOCRScheduler toan cuc va ton trong max_concurrency.
7. out-of-order: batch hoan thanh nguoc thu tu van duoc anh xa dung theo ID va giu moc thoi gian.
8. checkpoint resume: dung giua chung luu lai WAL checkpoint, resume chi gui segment thieu.
9. invalidation model/region/prompt: thay doi model, region, prompt lam doi cache key.
10. cancellation: huy bo tra ve/raise CancelledError, don dep thu muc tam va giu checkpoint.
11. timeout/errors: loi batch giu lai doan da xong, bao loi, khong sinh sub rong, khong fallback.
12. multi-video shared concurrency: nhieu video chia se scheduler khong vuot max_concurrency.
13. timing/SRT order: moc thoi gian tang dan, gop doan trung o ranh gioi chunk, tu dong xuat SRT.
14. bounded chunk cleanup: toi da 1-2 thu muc chunk tai moi thoi diem, don dep sau moi chunk.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from autosub_studio.core import formats
from autosub_studio.core.models import SubtitleDoc
from autosub_studio.core.ocr_common import merge_segment_cues
from autosub_studio.data import ocr_cache
from autosub_studio.data.ocr_cache import SegmentRecord
from autosub_studio.data.project import ProjectData, ProjectStore
from autosub_studio.pipeline import steps
from autosub_studio.providers import ocr, ocr_ai
from autosub_studio.providers.ocr_ai_provider import (
    BatchItem,
    BatchResult,
    execute_batch_with_split,
)
from autosub_studio.services import ai_gateway, media
from autosub_studio.services.ai_ocr_scheduler import (
    AIOCRScheduler,
    CancelledError,
    CancelToken,
    get_global_scheduler,
    shutdown_global_scheduler,
)
from autosub_studio.services.ffmpeg import FFmpeg, MediaInfo
from autosub_studio.services.settings import Settings


def _make_dummy_task_context():
    tc = MagicMock()
    tc.token = CancelToken()
    tc.log = MagicMock()
    tc.progress = MagicMock()
    tc.check_cancel = MagicMock()
    return tc


def _create_test_image(
    width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ===========================================================================
# 1. Crop region respected
# ===========================================================================
def test_crop_region_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock extraction nhan dung region va gateway nhan dung kich thuoc crop."""
    crop_w, crop_h = 160, 48
    region = [20, 100, crop_w, crop_h]

    extracted_regions = []

    def mock_extract_chunks(ff, video, out_dir_parent, **kwargs):
        extracted_regions.append(kwargs.get("region"))
        d = Path(out_dir_parent) / "chunk_0"
        d.mkdir(parents=True, exist_ok=True)
        frame_path = d / "frame_000001.png"
        img = Image.new("RGB", (crop_w, crop_h), color=(0, 0, 0))
        for x in range(10, 50):
            for y in range(10, 30):
                img.putpixel((x, y), (255, 255, 255))
        img.save(frame_path)
        yield 0, 0.0, 5.0, [(1.0, frame_path)], d

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    received_image_dims = []

    def mock_chat_completion(endpoint, api_key, **kwargs):
        messages = kwargs.get("messages", [])
        for m in messages:
            for c in m.get("content", []):
                if c.get("type") == "image_url":
                    url = c["image_url"]["url"]
                    b64_data = url.split(",")[-1]
                    raw = base64.b64decode(b64_data)
                    with Image.open(io.BytesIO(raw)) as pil_img:
                        received_image_dims.append((pil_img.width, pil_img.height))
        # Return structured JSON with matching ID
        user_c = messages[0]["content"]
        ids = [
            c["text"].replace("ID: ", "")
            for c in user_c
            if c.get("text", "").startswith("ID: ")
        ]
        res = [{"id": i, "text": "CROP_TEXT", "confidence": 1.0} for i in ids]
        return json.dumps({"results": res})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"dummy_video")

    ff = MagicMock(spec=FFmpeg)
    temp_dir = tmp_path / "temp_ocr"

    cues = ocr_ai.run_ai_ocr_pipeline(
        ff,
        video_path,
        temp_dir,
        region=region,
        endpoint="http://mock.endpoint/v1",
        api_key="mock_key",
        model="sub",
        batch_size=4,
    )

    assert len(extracted_regions) == 1
    assert extracted_regions[0] == region
    assert len(received_image_dims) >= 1
    for w, h in received_image_dims:
        assert w == crop_w
        assert h == crop_h
    assert len(cues) >= 1
    assert cues[0].text == "CROP_TEXT"


# ===========================================================================
# 2. No local symbols called
# ===========================================================================
def test_no_local_symbols_called(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tuyen bo loi ngay neu bat ky ham hoac bieu tuong nao cua Local OCR bi goi."""
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Local OCR symbol was called in AI OCR mode!")

    monkeypatch.setattr(ocr, "probe_frames", forbidden_call)
    monkeypatch.setattr(ocr, "read_frames", forbidden_call)
    monkeypatch.setattr(ocr, "refine_boundaries", forbidden_call)
    monkeypatch.setattr(ocr, "is_available", forbidden_call)
    monkeypatch.setattr(ocr, "apply_probe", forbidden_call)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    def mock_extract_chunks(*args, **kwargs):
        d = tmp_path / "chunk_0"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "frame_000001.png"
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(10, 40):
            for y in range(8, 24):
                img.putpixel((x, y), (255, 255, 255))
        img.save(f)
        yield 0, 0.0, 5.0, [(0.5, f)], d

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    def mock_chat_completion(*args, **kwargs):
        messages = kwargs.get("messages", [])
        ids = [
            c["text"].replace("ID: ", "")
            for c in messages[0]["content"]
            if c.get("text", "").startswith("ID: ")
        ]
        return json.dumps({"results": [{"id": ids[0], "text": "NO_LOCAL", "confidence": 1.0}]})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    settings = Settings(
        ocr_mode="OCR AI",
        ocr_server="Server AI API",
        ai_endpoint="http://localhost:8000/v1",
    )
    project = ProjectData(
        folder=str(tmp_path),
        video_path=str(video_path),
        ocr_region=[0, 0, 64, 32],
        duration=5.0,
    )
    pc = steps.PipelineContext(
        ff=MagicMock(spec=FFmpeg),
        settings=settings,
        store=MagicMock(spec=ProjectStore),
        project=project,
        task=_make_dummy_task_context(),
        api_key="test-key",
    )

    msg = steps.step_ocr(pc)
    assert "1 cau" in msg
    assert pc.project.doc.cues[0].text == "NO_LOCAL"


# ===========================================================================
# 3. Redundant frame reduction
# ===========================================================================
def test_redundant_frame_reduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selector loc bo khung trong va khung trung lap, giam so luong goi len gateway."""
    frames = []
    # 5 blank frames
    for i in range(5):
        f = tmp_path / f"blank_{i}.png"
        Image.new("RGB", (64, 32), color=(0, 0, 0)).save(f)
        frames.append((float(i) * 0.5, f))
    # 10 identical static frames (pattern A)
    for i in range(5, 15):
        f = tmp_path / f"static_a_{i}.png"
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(10, 40):
            for y in range(8, 24):
                img.putpixel((x, y), (255, 255, 255))
        img.save(f)
        frames.append((float(i) * 0.5, f))
    # 5 identical static frames (pattern B)
    for i in range(15, 20):
        f = tmp_path / f"static_b_{i}.png"
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(25, 55):
            for y in range(12, 28):
                img.putpixel((x, y), (255, 255, 255))
        img.save(f)
        frames.append((float(i) * 0.5, f))

    def mock_extract_chunks(*args, **kwargs):
        d = tmp_path / "chunk_0"
        d.mkdir(parents=True, exist_ok=True)
        yield 0, 0.0, 10.0, frames, d

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    items_sent = []

    def mock_chat_completion(endpoint, api_key, **kwargs):
        messages = kwargs.get("messages", [])
        user_content = messages[0]["content"]
        ids = [
            c["text"].replace("ID: ", "")
            for c in user_content
            if c.get("text", "").startswith("ID: ")
        ]
        items_sent.extend(ids)
        res = [{"id": i, "text": f"TEXT_{i}", "confidence": 1.0} for i in ids]
        return json.dumps({"results": res})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    ff = MagicMock(spec=FFmpeg)
    temp_dir = tmp_path / "temp_ocr"

    cues = ocr_ai.run_ai_ocr_pipeline(
        ff,
        tmp_path / "v.mp4",
        temp_dir,
        region=[0, 0, 64, 32],
        endpoint="http://mock/v1",
        api_key="key",
        batch_size=8,
    )

    # 20 frames -> reduced to 2 visual segments
    assert len(items_sent) == 2
    assert len(cues) == 2


# ===========================================================================
# 4. Batch creation
# ===========================================================================
def test_batch_creation() -> None:
    """Tach cac segment thanh batch dung kich thuoc ocr_ai_batch_size voi ID xac dinh."""
    items = [
        BatchItem(id=f"seg_{i:04d}", image=_create_test_image(32, 16))
        for i in range(11)
    ]
    batches = media.chunk_sequence(items, chunk_size=4)
    assert len(batches) == 3
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 3

    # Check unique deterministic IDs
    all_ids = [item.id for b in batches for item in b]
    assert len(set(all_ids)) == 11
    assert all_ids == [f"seg_{i:04d}" for i in range(11)]


# ===========================================================================
# 5. Split via provider
# ===========================================================================
def test_split_via_provider() -> None:
    """Loi 413 / payload too large duoc provider tu dong chia doi de quy va hoan tat."""
    items = [
        BatchItem(id=f"id_{i}", image=_create_test_image(32, 16))
        for i in range(4)
    ]

    calls = []

    def mock_call(sub_items: Sequence[BatchItem]) -> list[BatchResult]:
        calls.append([it.id for it in sub_items])
        if len(sub_items) > 1:
            raise ai_gateway.AIGatewayPayloadTooLargeError(
                "Payload too large", status_code=413
            )
        return [BatchResult(id=sub_items[0].id, text=f"RESULT_{sub_items[0].id}")]

    results = execute_batch_with_split(items, mock_call)
    assert len(results) == 4
    assert [r.id for r in results] == ["id_0", "id_1", "id_2", "id_3"]
    assert len(calls) == 7


# ===========================================================================
# 6. Global scheduler use
# ===========================================================================
def test_global_scheduler_use(tmp_path: Path) -> None:
    """Kiem tra AIOCRScheduler toan cuc tien trinh ton trong max_concurrency."""
    shutdown_global_scheduler()
    scheduler = get_global_scheduler(max_concurrency=2)
    assert scheduler.max_concurrency == 2

    active_counts = []
    lock = threading.Lock()

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        with lock:
            active_counts.append(scheduler.metrics.active)
        time.sleep(0.05)
        return [BatchResult(id=item.id, text="OK") for item in items]

    batches = [
        [BatchItem(id=f"b_{i}_{j}", image=_create_test_image(16, 16)) for j in range(2)]
        for i in range(6)
    ]

    res = scheduler.schedule_job("test_job", batches, mock_provider, max_pending=4)
    assert len(res) == 12
    assert max(active_counts) <= 2
    shutdown_global_scheduler()


# ===========================================================================
# 7. Out-of-order execution
# ===========================================================================
def test_out_of_order_execution() -> None:
    """Batch hoan thanh nguoc thu tu van duoc ghep dung theo ID va giu moc thoi gian."""
    shutdown_global_scheduler()
    scheduler = AIOCRScheduler(max_concurrency=4)

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        item_id = items[0].id
        if "first" in item_id:
            time.sleep(0.08)
            return [BatchResult(id=item_id, text="FIRST")]
        else:
            time.sleep(0.01)
            return [BatchResult(id=item_id, text="SECOND")]

    b1 = [BatchItem(id="seg_0000_1.000_2.000_first", image=_create_test_image(16, 16))]
    b2 = [BatchItem(id="seg_0001_3.000_4.000_second", image=_create_test_image(16, 16))]

    results = scheduler.schedule_job("out_of_order_job", [b1, b2], mock_provider)

    assert "seg_0000_1.000_2.000_first" in results
    assert "seg_0001_3.000_4.000_second" in results
    assert results["seg_0000_1.000_2.000_first"].text == "FIRST"
    assert results["seg_0001_3.000_4.000_second"].text == "SECOND"

    records = [
        SegmentRecord(
            segment_id=k,
            start=1.0 if "first" in k else 3.0,
            end=2.0 if "first" in k else 4.0,
            content_hash="",
            text=v.text,
        )
        for k, v in results.items()
    ]
    cues = merge_segment_cues(records)
    assert len(cues) == 2
    assert cues[0].start == 1.0 and cues[0].text == "FIRST"
    assert cues[1].start == 3.0 and cues[1].text == "SECOND"
    scheduler.shutdown()


# ===========================================================================
# 8. Checkpoint resume after interruption
# ===========================================================================
def test_checkpoint_resume_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interruption luu lai checkpoint SQLite WAL; resume chi gui missing segments."""
    cache_db = tmp_path / "cache.sqlite3"
    cache_key = "test_chk_key"

    frames = []
    for i in range(4):
        f = tmp_path / f"f_{i}.png"
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(5 + i * 12, 15 + i * 12):
            for y in range(5, 25):
                img.putpixel((x, y), (255, 255, 255))
        img.save(f)
        frames.append((float(i) * 2.0, f))

    def mock_extract_chunks(*args, **kwargs):
        d = tmp_path / "chunk_0"
        d.mkdir(parents=True, exist_ok=True)
        yield 0, 0.0, 8.0, frames, d

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    called_ids = []
    distinct_words = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    id_to_text: dict[str, str] = {}

    def get_distinct_text(seg_id: str) -> str:
        if seg_id not in id_to_text:
            id_to_text[seg_id] = distinct_words[len(id_to_text) % len(distinct_words)]
        return id_to_text[seg_id]

    def mock_chat_completion(endpoint, api_key, **kwargs):
        messages = kwargs.get("messages", [])
        user_content = messages[0]["content"]
        ids = [
            c["text"].replace("ID: ", "")
            for c in user_content
            if c.get("text", "").startswith("ID: ")
        ]
        called_ids.extend(ids)
        res = [{"id": i, "text": get_distinct_text(i), "confidence": 1.0} for i in ids]
        return json.dumps({"results": res})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    ff = MagicMock(spec=FFmpeg)
    temp_dir = tmp_path / "temp_ocr"

    token = CancelToken()

    def should_cancel_first():
        if len(called_ids) >= 2:
            token.cancel()
            return True
        return False

    with pytest.raises(CancelledError):
        ocr_ai.run_ai_ocr_pipeline(
            ff,
            tmp_path / "v.mp4",
            temp_dir,
            region=[0, 0, 64, 32],
            endpoint="http://mock/v1",
            api_key="key",
            batch_size=1,
            cache_path=cache_db,
            cache_key=cache_key,
            token=token,
            should_cancel=should_cancel_first,
        )

    saved_records = ocr_cache.load_segment_checkpoint(cache_db, cache_key)
    assert len(saved_records) == 2

    called_ids.clear()
    cues = ocr_ai.run_ai_ocr_pipeline(
        ff,
        tmp_path / "v.mp4",
        temp_dir,
        region=[0, 0, 64, 32],
        endpoint="http://mock/v1",
        api_key="key",
        batch_size=1,
        cache_path=cache_db,
        cache_key=cache_key,
    )

    assert len(called_ids) == 2
    assert len(cues) == 4
    assert [c.text for c in cues] == ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    for k in range(len(cues)):
        assert cues[k].start < cues[k].end
        if k > 0:
            assert cues[k].start > cues[k - 1].start
            assert cues[k].start >= cues[k - 1].end


# ===========================================================================
# 9. Invalidation model/region/prompt
# ===========================================================================
def test_invalidation_model_region_prompt(tmp_path: Path) -> None:
    """Thay doi model, region, prompt, diff_threshold lam thay doi cache key."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    k_base = ocr_cache.make_ai_ocr_cache_key(
        video=video,
        region=[0, 0, 100, 50],
        fps=2.0,
        endpoint="http://host/v1",
        model_alias="sub",
        actual_model="model-v1",
        custom_prompt="Prompt A",
    )

    k_model = ocr_cache.make_ai_ocr_cache_key(
        video=video,
        region=[0, 0, 100, 50],
        fps=2.0,
        endpoint="http://host/v1",
        model_alias="sub",
        actual_model="model-v2",
        custom_prompt="Prompt A",
    )
    assert k_base != k_model

    k_region = ocr_cache.make_ai_ocr_cache_key(
        video=video,
        region=[0, 0, 100, 60],
        fps=2.0,
        endpoint="http://host/v1",
        model_alias="sub",
        actual_model="model-v1",
        custom_prompt="Prompt A",
    )
    assert k_base != k_region

    k_prompt = ocr_cache.make_ai_ocr_cache_key(
        video=video,
        region=[0, 0, 100, 50],
        fps=2.0,
        endpoint="http://host/v1",
        model_alias="sub",
        actual_model="model-v1",
        custom_prompt="Prompt B",
    )
    assert k_base != k_prompt

    k_diff = ocr_cache.make_ai_ocr_cache_key(
        video=video,
        region=[0, 0, 100, 50],
        fps=2.0,
        endpoint="http://host/v1",
        model_alias="sub",
        actual_model="model-v1",
        custom_prompt="Prompt A",
        diff_threshold=5.0,
    )
    assert k_base != k_diff


# ===========================================================================
# 10. Cancellation
# ===========================================================================
def test_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Huy bo sau mot so batch thanh cong: raise CancelledError va resume hoan tat."""
    cache_db = tmp_path / "cancel_cache.sqlite3"
    cache_key = "cancel_key"

    def make_frame(idx: int, folder: Path) -> Path:
        f = folder / f"frame_{idx}.png"
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(5 + idx * 10, 15 + idx * 10):
            for y in range(5, 25):
                img.putpixel((x, y), (255, 255, 255))
        img.save(f)
        return f

    token = CancelToken()

    def mock_extract_chunks(*args, **kwargs):
        d0 = tmp_path / "c0"
        d0.mkdir(parents=True, exist_ok=True)
        f0 = make_frame(0, d0)
        yield 0, 0.0, 30.0, [(1.0, f0)], d0

        token.cancel()
        d1 = tmp_path / "c1"
        d1.mkdir(parents=True, exist_ok=True)
        f1 = make_frame(1, d1)
        yield 1, 28.0, 60.0, [(35.0, f1)], d1

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    called_ids = []

    def mock_chat_completion(endpoint, api_key, **kwargs):
        messages = kwargs.get("messages", [])
        ids = [
            c["text"].replace("ID: ", "")
            for c in messages[0]["content"]
            if c.get("text", "").startswith("ID: ")
        ]
        called_ids.extend(ids)
        res = [{"id": i, "text": f"TEXT_{i}", "confidence": 1.0} for i in ids]
        return json.dumps({"results": res})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    ff = MagicMock(spec=FFmpeg)
    temp_dir = tmp_path / "temp_cancel"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(CancelledError):
        ocr_ai.run_ai_ocr_pipeline(
            ff,
            tmp_path / "v.mp4",
            temp_dir,
            region=[0, 0, 64, 32],
            endpoint="http://mock/v1",
            api_key="key",
            token=token,
            cache_path=cache_db,
            cache_key=cache_key,
        )

    # Chunk 0 succeeded and was saved to checkpoint
    saved = ocr_cache.load_segment_checkpoint(cache_db, cache_key)
    assert len(saved) == 1

    # Resume run without cancellation completes chunk 1
    def mock_extract_resume(*args, **kwargs):
        d0 = tmp_path / "c0_res"
        d0.mkdir(parents=True, exist_ok=True)
        f0 = make_frame(0, d0)
        yield 0, 0.0, 30.0, [(1.0, f0)], d0

        d1 = tmp_path / "c1_res"
        d1.mkdir(parents=True, exist_ok=True)
        f1 = make_frame(1, d1)
        yield 1, 28.0, 60.0, [(35.0, f1)], d1

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_resume)
    called_ids.clear()

    cues = ocr_ai.run_ai_ocr_pipeline(
        ff,
        tmp_path / "v.mp4",
        temp_dir,
        region=[0, 0, 64, 32],
        endpoint="http://mock/v1",
        api_key="key",
        cache_path=cache_db,
        cache_key=cache_key,
    )

    assert len(called_ids) == 1  # Only chunk 1 sent to AI
    assert len(cues) == 2


# ===========================================================================
# 11. Timeout/errors preserve completed
# ===========================================================================
def test_timeout_errors_preserve_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loi bat ky giu lai cac doan da hoan thanh, khong fallback va khong sinh sub rong."""
    cache_db = tmp_path / "error_cache.sqlite3"
    cache_key = "error_key"

    def make_frame(idx: int, folder: Path) -> Path:
        f = folder / f"frame_{idx}.png"
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(10 + idx * 20, 25 + idx * 20):
            for y in range(5, 25):
                img.putpixel((x, y), (255, 255, 255))
        img.save(f)
        return f

    def mock_extract_chunks(*args, **kwargs):
        d0 = tmp_path / "err_c0"
        d0.mkdir(parents=True, exist_ok=True)
        f0 = make_frame(0, d0)
        yield 0, 0.0, 30.0, [(1.0, f0)], d0

        d1 = tmp_path / "err_c1"
        d1.mkdir(parents=True, exist_ok=True)
        f1 = make_frame(1, d1)
        yield 1, 30.0, 60.0, [(35.0, f1)], d1

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    call_count = [0]

    def mock_chat_completion(endpoint, api_key, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            messages = kwargs.get("messages", [])
            ids = [
                c["text"].replace("ID: ", "")
                for c in messages[0]["content"]
                if c.get("text", "").startswith("ID: ")
            ]
            res = [{"id": ids[0], "text": "SAVED_BEFORE_ERROR", "confidence": 1.0}]
            return json.dumps({"results": res})
        raise ai_gateway.AIGatewayAuthError("Invalid API key 401", status_code=401)

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    ff = MagicMock(spec=FFmpeg)
    temp_dir = tmp_path / "temp_ocr"

    with pytest.raises(Exception) as excinfo:
        ocr_ai.run_ai_ocr_pipeline(
            ff,
            tmp_path / "v.mp4",
            temp_dir,
            region=[0, 0, 64, 32],
            endpoint="http://mock/v1",
            api_key="key",
            batch_size=1,
            cache_path=cache_db,
            cache_key=cache_key,
        )

    assert "401" in str(excinfo.value) or "xác thực" in str(excinfo.value)

    saved = ocr_cache.load_segment_checkpoint(cache_db, cache_key)
    assert len(saved) == 1
    rec = list(saved.values())[0]
    assert rec.text == "SAVED_BEFORE_ERROR"


# ===========================================================================
# 12. Multi-video shared concurrency integration
# ===========================================================================
def test_multi_video_shared_concurrency_integration() -> None:
    """Nhieu job chia se scheduler toan cuc khong bao gio vuot qua max_concurrency."""
    shutdown_global_scheduler()
    scheduler = get_global_scheduler(max_concurrency=2)

    active_records = []
    lock = threading.Lock()

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        with lock:
            active_records.append(scheduler.metrics.active)
        time.sleep(0.04)
        return [BatchResult(id=item.id, text="OK") for item in items]

    batches_v1 = [[BatchItem(id=f"v1_{i}", image=_create_test_image(16, 16))] for i in range(4)]
    batches_v2 = [[BatchItem(id=f"v2_{i}", image=_create_test_image(16, 16))] for i in range(4)]

    res1, res2 = {}, {}

    def run_v1():
        nonlocal res1
        res1 = scheduler.schedule_job("video_1", batches_v1, mock_provider)

    def run_v2():
        nonlocal res2
        res2 = scheduler.schedule_job("video_2", batches_v2, mock_provider)

    t1 = threading.Thread(target=run_v1)
    t2 = threading.Thread(target=run_v2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(res1) == 4
    assert len(res2) == 4
    assert max(active_records) <= 2
    shutdown_global_scheduler()


# ===========================================================================
# 13. Timing/SRT order
# ===========================================================================
def test_timing_srt_order(tmp_path: Path) -> None:
    """Moc thoi gian tang dan, gop doan trung o ranh gioi chunk va xuat SRT dung chuan."""
    segments = [
        SegmentRecord(
            segment_id="s1", start=1.0, end=3.0, content_hash="h1", text="Hello world!"
        ),
        SegmentRecord(
            segment_id="s2", start=28.5, end=31.0, content_hash="h2", text="Chunk boundary text"
        ),
        SegmentRecord(
            segment_id="s3", start=29.0, end=32.0, content_hash="h3", text="Chunk boundary text"
        ),
        SegmentRecord(
            segment_id="s4", start=35.0, end=38.0, content_hash="h4", text="End text."
        ),
    ]

    cues = merge_segment_cues(segments, max_gap=0.5, similarity=0.85)

    assert len(cues) == 3
    assert cues[0].start == 1.0 and cues[0].end == 3.0
    assert cues[0].text == "Hello world!"
    assert cues[1].start == 28.5 and cues[1].end == 32.0
    assert cues[1].text == "Chunk boundary text"
    assert cues[2].start == 35.0 and cues[2].end == 38.0

    doc = SubtitleDoc(cues=cues)
    srt_file = tmp_path / "output.srt"
    formats.save_subtitle(srt_file, doc)

    assert srt_file.is_file()
    content = srt_file.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:03,000" in content
    assert "00:00:28,500 --> 00:00:32,000" in content
    assert "Chunk boundary text" in content


# ===========================================================================
# 14. Bounded chunk cleanup
# ===========================================================================
def test_bounded_chunk_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Toi da 1-2 thu muc chunk tai moi thoi diem, don dep sau moi chunk."""
    temp_dir = tmp_path / "temp_bounded"
    temp_dir.mkdir(parents=True, exist_ok=True)

    active_dir_counts = []

    ff = MagicMock(spec=FFmpeg)
    ff.probe.return_value = MediaInfo(path=str(tmp_path / "v.mp4"), duration=70.0, fps=25.0)

    def mock_ff_run(args, **kwargs):
        out_pattern = args[-1]
        out_dir = Path(out_pattern).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = out_dir / "frame_000001.png"
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(frame)

    ff.run = mock_ff_run

    chunk_dirs_seen = []
    for _idx, _start, _end, _frames, c_dir in media.extract_video_chunks(
        ff,
        tmp_path / "v.mp4",
        temp_dir,
        chunk_duration=30.0,
        carryover=2.0,
        total_duration=70.0,
    ):
        chunk_dirs_seen.append(c_dir.name)
        existing = list(temp_dir.glob("chunk_*"))
        active_dir_counts.append(len(existing))
        shutil.rmtree(c_dir, ignore_errors=True)

    assert len(chunk_dirs_seen) == 3
    assert chunk_dirs_seen == ["chunk_0", "chunk_1", "chunk_0"]
    assert max(active_dir_counts) <= 2
    assert len(list(temp_dir.glob("chunk_*"))) == 0


# ===========================================================================
# 15. Content hash boundary carryover reuse
# ===========================================================================
def test_content_hash_boundary_carryover_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doan trung content hash o ranh gioi chunk khong phai goi AI lai, chi copy timing."""
    cache_db = tmp_path / "hash_cache.sqlite3"
    cache_key = "hash_key"

    # Create image file for chunk 0 and chunk 1 with identical content
    d0 = tmp_path / "c0"
    d0.mkdir(parents=True, exist_ok=True)
    f0 = d0 / "frame_0.png"
    img0 = Image.new("RGB", (64, 32), color=(0, 0, 0))
    for x in range(10, 40):
        for y in range(8, 24):
            img0.putpixel((x, y), (255, 255, 255))
    img0.save(f0)

    d1 = tmp_path / "c1"
    d1.mkdir(parents=True, exist_ok=True)
    f1 = d1 / "frame_1.png"
    img0.save(f1)  # Identical image content

    def mock_extract_chunks(*args, **kwargs):
        yield 0, 0.0, 30.0, [(28.5, f0)], d0
        yield 1, 28.0, 60.0, [(28.5, f1)], d1

    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    called_ids = []

    def mock_chat_completion(endpoint, api_key, **kwargs):
        messages = kwargs.get("messages", [])
        ids = [
            c["text"].replace("ID: ", "")
            for c in messages[0]["content"]
            if c.get("text", "").startswith("ID: ")
        ]
        called_ids.extend(ids)
        res = [{"id": i, "text": "REUSED_TEXT", "confidence": 0.95} for i in ids]
        return json.dumps({"results": res})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    ff = MagicMock(spec=FFmpeg)
    temp_dir = tmp_path / "temp_hash"

    cues = ocr_ai.run_ai_ocr_pipeline(
        ff,
        tmp_path / "v.mp4",
        temp_dir,
        region=[0, 0, 64, 32],
        endpoint="http://mock/v1",
        api_key="key",
        cache_path=cache_db,
        cache_key=cache_key,
    )

    # Only called AI once for chunk 0! Chunk 1 hit the content hash checkpoint.
    assert len(called_ids) == 1
    assert len(cues) == 1
    assert cues[0].text == "REUSED_TEXT"
