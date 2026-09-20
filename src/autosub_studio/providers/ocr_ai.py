"""Dieu phoi nhan dang chu tren video bang AI Vision (AI OCR Orchestrator).

Kien truc:
1. FFmpeg trich xuat khung hinh theo tung chunk (e.g. 30s) co carryover ranh gioi (e.g. 2s)
   va crop truc tiep theo vung da chon.
2. Visual Selector xu ly anh da crop, loc bo khung trong (blank), khung trung lap (static),
   chon khung dai dien ro net nhat va phan doan thanh VisualSegment on dinh.
3. Chia thanh cac lo (batches) theo cau hinh ocr_ai_batch_size.
4. Dieu phoi thong qua AIOCRScheduler toan cuc tien trinh voi gioi han max_concurrency
   duoc chia se giua tat ca cac du an.
5. AIOCRProvider goi OpenAI-compatible vision chat completion voi prompt chat che.
6. Checkpoint ngay lap tuc sau moi batch thanh cong vao SQLite WAL; resume chi gui
   cac doan chua co trong checkpoint.
7. Don dep thu muc tam theo tung chunk; khi huy bo don dep thu muc tam va giu nguyen checkpoint.
8. Tai tao phu de (reconstruction): chuan hoa khoang trang, gop doan trung lap o ranh gioi chunk,
   sap xep start/end va xuat SRT tu dong.
9. Ghi log chi tiet kien truc va cac chi so hieu nang (metrics).
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..core.models import Cue
from ..core.ocr_common import (
    Row,
    _merge_exact_repeats,
    _merge_rows,
    merge_segment_cues,
    normalize_subtitle_text,
    static_texts,
)
from ..data import ocr_cache
from ..data.ocr_cache import (
    SegmentRecord,
    load_frame_cache,
    save_frame_batch,
    save_segment_checkpoint,
)
from ..services import ai_gateway, media
from ..services.ai_ocr_scheduler import (
    CancelledError,
    CancelToken,
    get_global_scheduler,
    is_cancelled,
)
from ..services.ffmpeg import FFmpeg
from ..services.settings import Settings
from .ocr_ai_provider import AIOCRProvider, BatchItem, BatchResult
from .ocr_filter import TextFilter
from .ocr_selector import (
    VisualSegment,
    select_visual_segments,
)


def run_ai_ocr_pipeline(
    ff: FFmpeg,
    video: str | Path,
    temp_dir: str | Path,
    *,
    region: Sequence[int],
    fps: float = 2.0,
    chunk_duration: float = 30.0,
    carryover_duration: float = 2.0,
    endpoint: str = "",
    api_key: str = "",
    model: str = "sub",
    thinking: str = "",
    prompt: str = "",
    prompt_version: str = "v1",
    batch_size: int = 8,
    max_concurrency: int = 4,
    timeout: float = 60.0,
    max_retries: int = 3,
    diff_threshold: float = 4.0,
    blank_threshold: float = 3.0,
    image_quality: int = 88,
    consensus_frames: int = 1,
    accuracy_mode: bool = False,
    similarity: float = 0.85,
    min_duration: float = 0.3,
    total_duration: float = 0.0,
    cache_path: str | Path | None = None,
    cache_key: str = "",
    token: CancelToken | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> list[Cue]:
    """Dieu phoi toan bo luong nhan dang chu OCR AI tren video."""
    settings = Settings.load()
    ep = endpoint.strip() or settings.ai_endpoint.strip()
    if not ep:
        raise ai_gateway.AIGatewayError("Chưa cấu hình Endpoint AI Gateway cho OCR AI.")
    key = api_key.strip() or Settings.get_secret("ai_gateway_key")
    if not key:
        raise ai_gateway.AIGatewayError("Chưa có khóa API cho AI Gateway.")

    actual_model, default_thinking = ai_gateway.resolve_model(
        model or settings.ocr_ai_model, settings
    )
    actual_thinking = thinking if thinking else default_thinking

    effective_cache_key = cache_key
    if not effective_cache_key and cache_path:
        effective_cache_key = ocr_cache.make_ai_ocr_cache_key(
            video=video,
            region=region,
            fps=fps,
            endpoint=ep,
            model_alias=model,
            actual_model=actual_model,
            thinking=actual_thinking,
            prompt_version=prompt_version,
            custom_prompt=prompt,
            diff_threshold=diff_threshold,
            image_quality=image_quality,
            consensus_mode="accuracy" if accuracy_mode else "disabled",
            consensus_frames=consensus_frames,
        )

    if cache_path and effective_cache_key:
        ocr_cache.save_cache_meta(
            cache_path,
            effective_cache_key,
            {
                "engine": "ai_gateway",
                "endpoint_hash": ocr_cache.hash_endpoint(ep),
                "model_alias": model,
                "actual_model": actual_model,
                "thinking": actual_thinking,
                "prompt_version": prompt_version,
                "preprocessing_version": "v1",
                "video_identity": ocr_cache.file_identity(video),
                "region": list(region),
                "sampling_settings": {
                    "fps": fps,
                    "diff_threshold": diff_threshold,
                    "image_quality": image_quality,
                    "consensus_frames": consensus_frames,
                    "accuracy_mode": accuracy_mode,
                },
            },
        )

    completed_checkpoint: dict[str, SegmentRecord] = {}
    hash_checkpoint: dict[str, SegmentRecord] = {}
    if cache_path and effective_cache_key:
        completed_checkpoint = ocr_cache.load_segment_checkpoint(cache_path, effective_cache_key)
        hash_checkpoint = ocr_cache.load_segment_by_hash(cache_path, effective_cache_key)

    all_segment_records: dict[str, SegmentRecord] = dict(completed_checkpoint)

    if on_log:
        on_log(
            f"OCR AI Architecture: Chunk Extraction ({chunk_duration:g}s) -> "
            f"Visual Selector -> Batch Scheduler (max={max_concurrency}) -> "
            f"AI Gateway ({actual_model})"
        )
        if completed_checkpoint:
            on_log(f"Dùng lại checkpoint OCR AI: {len(completed_checkpoint)} đoạn đã hoàn thành.")

    t_pipeline_start = time.monotonic()
    extract_time = 0.0
    select_time = 0.0
    ocr_time = 0.0
    reconstruct_time = 0.0

    sampled_frames = 0
    rejected_frames = 0
    total_segments = 0
    total_representatives = 0
    total_images = 0
    total_requests = 0
    cache_hits = len(completed_checkpoint)
    cache_misses = 0

    provider = AIOCRProvider(
        endpoint=ep,
        api_key=key,
        model=actual_model,
        thinking=actual_thinking,
        timeout=float(timeout),
        prompt=prompt,
    )
    scheduler = get_global_scheduler(max_concurrency=max_concurrency)
    sched_baseline = scheduler.get_metrics()

    if is_cancelled(token, should_cancel):
        media.cleanup_chunk_dirs(temp_dir)
        raise CancelledError("OCR AI bị hủy.")

    effective_duration = float(total_duration)
    if effective_duration <= 0.0:
        try:
            effective_duration = float(ff.probe(video).duration)
        except Exception:
            effective_duration = 0.0

    last_progress = 0

    try:
        chunk_iter = media.extract_video_chunks(
            ff,
            video,
            temp_dir,
            chunk_duration=chunk_duration,
            carryover=carryover_duration,
            fps=fps,
            region=region,
            total_duration=effective_duration,
            token=token,
        )

        while True:
            t_ext_start = time.monotonic()
            try:
                chunk_data = next(chunk_iter)
            except StopIteration:
                break
            extract_time += time.monotonic() - t_ext_start

            chunk_idx, _c_start, actual_end, timed_frames, chunk_dir = chunk_data

            if is_cancelled(token, should_cancel):
                raise CancelledError("OCR AI bị hủy.")

            sampled_frames += len(timed_frames)

            t_sel_start = time.monotonic()
            chunk_segments = select_visual_segments(
                timed_frames,
                fps=fps,
                diff_threshold=diff_threshold,
                blank_threshold=blank_threshold,
                jpeg_quality=image_quality,
                consensus_frames=consensus_frames,
                accuracy_mode=accuracy_mode,
                token=token,
                should_cancel=should_cancel,
            )
            select_time += time.monotonic() - t_sel_start

            total_segments += len(chunk_segments)
            total_representatives += sum(len(s.representatives) for s in chunk_segments)
            frames_in_segs = sum(s.frame_count for s in chunk_segments)
            rejected_frames += max(0, len(timed_frames) - frames_in_segs)

            missing_segments: list[VisualSegment] = []
            immediate_saved: list[SegmentRecord] = []

            for s in chunk_segments:
                if s.id in all_segment_records:
                    continue
                if s.content_hash and s.content_hash in hash_checkpoint:
                    matched = hash_checkpoint[s.content_hash]
                    rec = SegmentRecord(
                        segment_id=s.id,
                        start=s.start,
                        end=s.end,
                        content_hash=s.content_hash,
                        text=matched.text,
                        confidence=matched.confidence,
                        uncertain=matched.uncertain,
                    )
                    all_segment_records[s.id] = rec
                    immediate_saved.append(rec)
                    cache_hits += 1
                else:
                    missing_segments.append(s)

            if immediate_saved and cache_path and effective_cache_key:
                save_segment_checkpoint(cache_path, effective_cache_key, immediate_saved)

            cache_misses += len(missing_segments)

            if missing_segments:
                seg_items_map: dict[str, list[str]] = {}
                items: list[BatchItem] = []

                for s in missing_segments:
                    if consensus_frames > 1 or accuracy_mode:
                        s_ids: list[str] = []
                        for k, rep in enumerate(s.representatives):
                            item_id = f"{s.id}_c{k}"
                            s_ids.append(item_id)
                            items.append(
                                BatchItem(
                                    id=item_id,
                                    image=rep.payload,
                                    mime_type="image/jpeg",
                                    metadata={"segment_id": s.id, "sample_index": k, "segment": s},
                                )
                            )
                        seg_items_map[s.id] = s_ids
                    else:
                        seg_items_map[s.id] = [s.id]
                        items.append(
                            BatchItem(
                                id=s.id,
                                image=s.representative.payload,
                                mime_type="image/jpeg",
                                metadata={"segment_id": s.id, "segment": s},
                            )
                        )

                total_images += len(items)
                batches = media.chunk_sequence(items, max(1, int(batch_size)))
                total_requests += len(batches)

                def make_on_batch_complete(
                    segs_list: list[VisualSegment],
                    item_mapping: dict[str, list[str]],
                ) -> Callable[[list[BatchResult]], None]:
                    seg_map = {s.id: s for s in segs_list}

                    def on_complete(results: list[BatchResult]) -> None:
                        res_by_id = {r.id: r for r in results}
                        to_save: list[SegmentRecord] = []
                        for s_id, item_ids in item_mapping.items():
                            if s_id in all_segment_records:
                                continue
                            if all(iid in res_by_id for iid in item_ids):
                                seg = seg_map.get(s_id)
                                if not seg:
                                    continue
                                if len(item_ids) == 1:
                                    r = res_by_id[item_ids[0]]
                                    rec = SegmentRecord(
                                        segment_id=s_id,
                                        start=seg.start,
                                        end=seg.end,
                                        content_hash=seg.content_hash,
                                        text=r.text,
                                        confidence=r.confidence,
                                        uncertain=r.uncertain,
                                    )
                                else:
                                    cands = [res_by_id[iid] for iid in item_ids]
                                    valid_texts = [c.text for c in cands if c.text.strip()]
                                    if valid_texts:
                                        norm_texts = [
                                            normalize_subtitle_text(t) for t in valid_texts
                                        ]
                                        best_norm = max(set(norm_texts), key=norm_texts.count)
                                        chosen_text = next(
                                            t
                                            for t in valid_texts
                                            if normalize_subtitle_text(t) == best_norm
                                        )
                                        avg_conf = sum(c.confidence for c in cands) / len(cands)
                                        unc = any(c.uncertain for c in cands)
                                    else:
                                        chosen_text = ""
                                        avg_conf = 1.0
                                        unc = True
                                    rec = SegmentRecord(
                                        segment_id=s_id,
                                        start=seg.start,
                                        end=seg.end,
                                        content_hash=seg.content_hash,
                                        text=chosen_text,
                                        confidence=avg_conf,
                                        uncertain=unc,
                                    )
                                to_save.append(rec)
                                all_segment_records[s_id] = rec
                                if rec.content_hash:
                                    hash_checkpoint[rec.content_hash] = rec

                        if to_save and cache_path and effective_cache_key:
                            save_segment_checkpoint(cache_path, effective_cache_key, to_save)

                    return on_complete

                t_ocr_start = time.monotonic()
                cb = make_on_batch_complete(missing_segments, seg_items_map)
                job_id = f"ai_ocr_{chunk_idx}_{int(time.monotonic() * 1000)}"

                scheduler.schedule_job(
                    job_id=job_id,
                    batches=batches,
                    provider=provider.execute_batch,
                    token=token,
                    should_cancel=should_cancel,
                    on_batch_complete=cb,
                    max_retries=max_retries,
                    total_timeout_per_batch=float(timeout) * 2.0,
                    endpoint_key=ocr_cache.hash_endpoint(ep),
                    fail_fast=True,
                    raise_on_cancel=True,
                )
                ocr_time += time.monotonic() - t_ocr_start

            shutil.rmtree(chunk_dir, ignore_errors=True)

            if on_progress and effective_duration > 0.0:
                prog_val = max(last_progress, min(95, int((actual_end / effective_duration) * 95)))
                last_progress = prog_val
                on_progress(prog_val)

    except CancelledError:
        media.cleanup_chunk_dirs(temp_dir)
        raise
    except Exception:
        media.cleanup_chunk_dirs(temp_dir)
        raise
    finally:
        media.cleanup_chunk_dirs(temp_dir)

    t_rec_start = time.monotonic()
    all_records = list(all_segment_records.values())
    cues = merge_segment_cues(
        all_records,
        similarity=similarity,
        min_duration=min_duration,
    )
    reconstruct_time += time.monotonic() - t_rec_start

    elapsed = time.monotonic() - t_pipeline_start
    sched_now = scheduler.get_metrics()
    retries_delta = max(0, sched_now.retries - sched_baseline.retries)
    avg_b = (total_images / total_requests) if total_requests > 0 else 0.0

    if on_log:
        on_log(
            f"OCR AI Metrics: sampled={sampled_frames}, rejected={rejected_frames}, "
            f"segments={total_segments}, representatives={total_representatives}, "
            f"images={total_images}, requests={total_requests}, "
            f"avg_batch={avg_b:.1f}, latency={sched_now.avg_latency:.2f}s, "
            f"retries={retries_delta}, cache_hits={cache_hits}, "
            f"cache_misses={cache_misses}, elapsed={elapsed:.2f}s, "
            f"stages=(extract={extract_time:.2f}s, select={select_time:.2f}s, "
            f"ocr={ocr_time:.2f}s, reconstruct={reconstruct_time:.2f}s)"
        )

    if on_progress:
        on_progress(100)

    return cues


# --------------------------------------------------------------------------- Compatibility API


def _image_to_small_gray(
    frame_path: Path, flt: TextFilter | None = None, quality: int = 88
) -> tuple[Any, bytes | None, str]:
    """Doc anh, ap dung bo loc neu co, tra ve anh thu nho xam, bytes va content hash."""
    img_bytes: bytes | None = None
    small: Any = None
    try:
        import cv2

        img = cv2.imread(str(frame_path))
        if img is not None:
            if flt is not None and flt.touches_image:
                from . import ocr_filter

                img = ocr_filter.adjust(img, flt.brightness, flt.contrast)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))]
            _, buf = cv2.imencode(".jpg", img, encode_param)
            img_bytes = buf.tobytes()
    except Exception:
        pass

    if img_bytes is None:
        try:
            import io

            from PIL import Image

            with Image.open(frame_path) as pil_img:
                rgb = pil_img.convert("RGB")
                small = rgb.convert("L").resize((64, 64))
                buf_io = io.BytesIO()
                rgb.save(buf_io, format="JPEG", quality=max(1, min(100, int(quality))))
                img_bytes = buf_io.getvalue()
        except Exception:
            pass

    content_hash = hashlib.sha256(img_bytes).hexdigest() if img_bytes else ""
    return small, img_bytes, content_hash


def _is_blank_frame(small_img: Any) -> bool:
    """Kiem tra khung hinh co phai don sac/khong co chi tiet nao khong."""
    if small_img is None:
        return True
    try:
        import numpy as np

        arr = np.asarray(small_img)
        if arr.std() < 2.0 or arr.mean() < 3.0:
            return True
    except Exception:
        pass
    return False


def _diff_metric(a: Any, b: Any) -> float:
    """Do sai khac trung binh giua hai khung hinh thu nho."""
    if a is None or b is None:
        return 999.0
    try:
        import numpy as np

        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        return float(np.mean(np.abs(arr_a - arr_b)))
    except Exception:
        return 999.0


def read_frames_ai(
    frames: Sequence[Path],
    *,
    fps: float = 2.0,
    start_offset: float = 0.0,
    stamps: Sequence[float] | None = None,
    similarity: float = 0.82,
    min_duration: float = 0.4,
    model: str = "sub",
    endpoint: str = "",
    api_key: str = "",
    thinking: str = "",
    consensus: int = 1,
    text_filter: TextFilter | None = None,
    batch_size: int = 8,
    diff_threshold: float = 4.0,
    image_quality: int = 88,
    timeout: float = 60.0,
    max_retries: int = 3,
    prompt: str = "",
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    cache_path: str | Path | None = None,
    cache_key: str = "",
) -> list[Cue]:
    """Doc chu tren cac khung hinh bang AI vision (API tuong thich cho cac bai kiem tra)."""
    if not frames:
        return []

    settings = Settings.load()
    ep = endpoint.strip() or settings.ai_endpoint.strip()
    if not ep:
        raise ai_gateway.AIGatewayError("Chưa cấu hình Endpoint AI Gateway cho OCR AI.")
    key = api_key.strip() or Settings.get_secret("ai_gateway_key")
    if not key:
        raise ai_gateway.AIGatewayError("Chưa có khóa API cho AI Gateway.")

    actual_model, default_thinking = ai_gateway.resolve_model(
        model or settings.ocr_ai_model, settings
    )
    actual_thinking = thinking if thinking else default_thinking

    total = len(frames)
    cached = load_frame_cache(cache_path, cache_key, total)
    rows_by_index: dict[int, list[Row]] = dict(cached)
    missing_indices = [i for i in range(total) if i not in rows_by_index]

    if on_log:
        if missing_indices:
            on_log(
                f"OCR AI ({actual_model}): xử lý {len(missing_indices)}/{total} khung hình "
                "(chỉ upload keyframe mới)..."
            )
        else:
            on_log("Tất cả khung hình đã có trong cache OCR AI.")
        if cached:
            on_log(f"Dùng lại cache OCR AI: {len(cached)}/{total} khung hình.")

    batch_rows: dict[int, list[Row]] = {}
    batch_hashes: dict[int, str] = {}
    last_keyframe_small: Any = None
    last_keyframe_rows: list[Row] = []

    for done, index in enumerate(missing_indices):
        if should_cancel is not None and should_cancel():
            break

        frame_path = frames[index]
        small, img_bytes, content_hash = _image_to_small_gray(
            frame_path, text_filter, quality=image_quality
        )

        if _is_blank_frame(small):
            current_rows: list[Row] = []
            last_keyframe_small = small
            last_keyframe_rows = []
        elif (
            last_keyframe_small is not None
            and _diff_metric(small, last_keyframe_small) < float(diff_threshold)
        ):
            current_rows = list(last_keyframe_rows)
        else:
            text = ""
            if img_bytes:
                retries = max(0, int(max_retries))
                for attempt in range(retries + 1):
                    try:
                        text = ai_gateway.ocr_image_with_ai(
                            ep,
                            key,
                            model=actual_model,
                            image_data=img_bytes,
                            thinking=actual_thinking,
                            timeout=float(timeout),
                        )
                        break
                    except Exception as exc:
                        if attempt == retries:
                            raise
                        if on_log:
                            on_log(
                                f"Lỗi gọi AI Gateway (lần {attempt + 1}/{retries + 1}): {exc}. "
                                "Đang thử lại..."
                            )

            clean_text = text.strip()
            if clean_text:
                current_rows = [Row((0.0, 0.0), clean_text, 1.0, None)]
            else:
                current_rows = []

            last_keyframe_small = small
            last_keyframe_rows = current_rows

        batch_rows[index] = current_rows
        batch_hashes[index] = content_hash
        rows_by_index[index] = current_rows

        if len(batch_rows) >= max(1, int(batch_size)):
            save_frame_batch(cache_path, cache_key, batch_rows, batch_hashes)
            batch_rows.clear()
            batch_hashes.clear()

        if on_progress:
            completed = len(cached) + done + 1
            on_progress(max(0, min(96, int(completed / total * 96))))

    if batch_rows:
        save_frame_batch(cache_path, cache_key, batch_rows, batch_hashes)
        batch_rows.clear()
        batch_hashes.clear()

    per_frame: list[list[Row]] = []
    for index in range(total):
        if index not in rows_by_index:
            break
        per_frame.append(rows_by_index[index])

    ignore: frozenset[str] = frozenset()
    if text_filter is not None and text_filter.drop_static:
        ignore = static_texts(per_frame)
        if ignore and on_log:
            shown = ", ".join(sorted(ignore)[:4])
            on_log(f"Bỏ {len(ignore)} dòng chữ cố định: {shown}")

    processed = len(per_frame)
    used_stamps = list(stamps[:processed]) if stamps is not None else None
    cues = _merge_rows(
        per_frame,
        fps=fps,
        start_offset=start_offset,
        stamps=used_stamps,
        total=processed,
        similarity=similarity,
        min_duration=min_duration,
        ignore=ignore,
        consensus=consensus,
        on_log=on_log,
    )
    cues = _merge_exact_repeats(cues, max_gap=0.2)
    if on_progress:
        on_progress(100)
    return cues
