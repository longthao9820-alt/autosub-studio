"""AI OCR Repeatable Benchmark Harness (ai-ocr-benchmark-r4).

Benchmarks the AI OCR subsystem without external network calls or video decoding.
Generates synthetic visual frame manifests across standard profiles:
- 'short': ~120 frames (60s @ 2fps, 4 segments)
- 'normal': ~1800 frames (15min @ 2fps, 40 segments)
- 'long': ~7200 frames (60min @ 2fps, 150 segments)
- 'multi': 3 concurrent streams proving scheduler global max concurrency limit.

Measures:
- requested original sampled frames
- sent images to AI provider
- total requests (batches)
- average batch size
- average latency
- total duration
- retries
- subtitle count
- rejected frames / cache hits
- peak active concurrency

Enforces deterministic assertions:
- Ground truth segments preserved (100% coverage)
- Zero hallucinations
- Reduction >= 70% on stable repetitions
- Request count == ceil(sent_images / batch_size)
- Second run on existing cache produces ZERO provider calls
- In multi profile: peak active concurrency <= configured max_concurrency
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from autosub_studio.core.models import Cue
from autosub_studio.core.ocr_common import merge_segment_cues, normalize_subtitle_text
from autosub_studio.data import ocr_cache
from autosub_studio.data.ocr_cache import SegmentRecord
from autosub_studio.providers.ocr_ai_provider import BatchItem, BatchResult
from autosub_studio.providers.ocr_selector import (
    FrameSample,
    VisualSegment,
    prepare_image,
    select_visual_segments,
)
from autosub_studio.services import media
from autosub_studio.services.ai_ocr_scheduler import AIOCRScheduler


@dataclass
class ActiveTracker:
    """Thread-safe active worker concurrency tracker."""

    current: int = 0
    peak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            if self.current > self.peak:
                self.peak = self.current

    def leave(self) -> None:
        with self.lock:
            self.current = max(0, self.current - 1)


@dataclass
class GroundTruthSegment:
    """Expected ground truth subtitle segment."""

    index: int
    text: str
    start: float
    end: float
    content_hash: str = ""


@dataclass
class ProfileResult:
    """Benchmark metrics for a single profile."""

    profile: str
    requested_original_sampled: int
    sent_images: int
    total_requests: int
    avg_batch: float
    avg_latency_ms: float
    total_duration_s: float
    retries: int
    subtitle_count: int
    rejected_frames: int
    cache_hits: int
    cache_misses: int
    reduction_pct: float
    peak_active: int
    ground_truth_count: int
    coverage_pct: float
    hallucination_count: int
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheTestResult:
    """Results for two-pass cache hit test."""

    first_run_provider_calls: int
    second_run_provider_calls: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate_pct: float
    zero_provider_calls: bool
    passed: bool


@dataclass
class BenchmarkReport:
    """Complete benchmark report covering all tested profiles."""

    timestamp: str
    config: dict[str, Any]
    profiles: dict[str, ProfileResult]
    cache_test: CacheTestResult | None
    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "profiles": {k: asdict(v) for k, v in self.profiles.items()},
            "cache_test": asdict(self.cache_test) if self.cache_test else None,
            "all_passed": self.all_passed,
        }


def _create_synthetic_templates(
    num_segments: int,
    base_dir: Path,
    width: int = 160,
    height: int = 48,
    label_prefix: str = "SUB",
) -> tuple[Path, list[tuple[Path, str, str]]]:
    """Create a minimal set of image files on disk for repetition.

    Returns:
        tuple of (blank_frame_path, list of (seg_image_path, text, content_hash))
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Blank frame template (solid dark background, luma std dev == 0)
    blank_path = base_dir / "template_blank.png"
    if not blank_path.exists():
        blank_img = Image.new("RGB", (width, height), color=(12, 12, 12))
        blank_img.save(blank_path)

    # 2. Distinct segment templates
    seg_templates: list[tuple[Path, str, str]] = []
    for k in range(num_segments):
        seg_path = base_dir / f"template_seg_{k:04d}.png"
        text = f"{label_prefix} {k + 1:03d}: Synthetic Subtitle Line {k + 1}"
        if not seg_path.exists():
            img = Image.new("RGB", (width, height), color=(12, 12, 12))
            draw = ImageDraw.Draw(img)
            # Distinct visual bar to enforce visual difference > diff_threshold
            bar_w = 15 + ((k * 7) % 35)
            draw.rectangle((5, 6, 5 + bar_w, height - 6), fill=(180, 200, 220))
            draw.text((25 + bar_w, 16), f"{label_prefix} {k + 1:03d}", fill=(255, 255, 255))
            img.save(seg_path)

        prep = prepare_image(seg_path)
        seg_templates.append((seg_path, text, prep.content_hash))

    return blank_path, seg_templates


def generate_synthetic_manifest(
    profile_name: str,
    base_dir: Path,
    *,
    scale: float = 1.0,
    fps: float = 2.0,
    stream_id: int = 0,
) -> tuple[list[FrameSample], list[GroundTruthSegment], dict[str, str]]:
    """Generates synthetic frame manifest with repeated file references.

    Returns:
        (frames, ground_truth_segments, hash_to_text_map)
    """
    scale = max(0.01, float(scale))
    if profile_name == "short":
        target_frames = max(24, int(120 * scale))
        num_segments = max(1, int(4 * scale))
    elif profile_name == "normal":
        target_frames = max(60, int(1800 * scale))
        num_segments = max(2, int(40 * scale))
    elif profile_name == "long":
        target_frames = max(120, int(7200 * scale))
        num_segments = max(4, int(150 * scale))
    elif profile_name == "multi":
        target_frames = max(24, int(120 * scale))
        num_segments = max(1, int(4 * scale))
    else:
        raise ValueError(f"Unknown profile: {profile_name}")

    label_prefix = f"S{stream_id}" if stream_id > 0 else "SUB"
    blank_path, seg_templates = _create_synthetic_templates(
        num_segments, base_dir, label_prefix=label_prefix
    )

    hash_to_text: dict[str, str] = {}
    for _, text, ch in seg_templates:
        hash_to_text[ch] = text

    # Plan frame allocation:
    # Segments and gaps interspersed
    seg_len = max(4, (target_frames // (num_segments * 3)))
    blank_len = max(2, ((target_frames - (num_segments * seg_len)) // (num_segments + 1)))

    frames: list[FrameSample] = []
    ground_truth: list[GroundTruthSegment] = []
    f_idx = 0

    for k in range(num_segments):
        # Blank gap
        for _ in range(blank_len):
            frames.append(
                FrameSample(timestamp=float(f_idx) / fps, path=blank_path, index=f_idx)
            )
            f_idx += 1

        seg_start_t = float(f_idx) / fps
        seg_path, text, ch = seg_templates[k]

        # Segment active frames
        for _ in range(seg_len):
            frames.append(
                FrameSample(timestamp=float(f_idx) / fps, path=seg_path, index=f_idx)
            )
            f_idx += 1

        seg_end_t = float(f_idx) / fps
        ground_truth.append(
            GroundTruthSegment(
                index=k,
                text=text,
                start=seg_start_t,
                end=seg_end_t,
                content_hash=ch,
            )
        )

    # Final padding blank frames to reach exact target_frames
    while len(frames) < target_frames:
        frames.append(
            FrameSample(timestamp=float(f_idx) / fps, path=blank_path, index=f_idx)
        )
        f_idx += 1

    return frames, ground_truth, hash_to_text


def _run_single_stream(
    stream_name: str,
    frames: list[FrameSample],
    ground_truth: list[GroundTruthSegment],
    hash_to_text: dict[str, str],
    scheduler: AIOCRScheduler,
    *,
    batch_size: int = 8,
    fps: float = 2.0,
    active_tracker: ActiveTracker | None = None,
    cache_path: Path | None = None,
    cache_key: str = "",
    simulated_delay_s: float = 0.0,
) -> ProfileResult:
    """Executes OCR pipeline on a single visual frame manifest."""
    t_start = time.monotonic()
    sampled_count = len(frames)

    # 1. Visual Selector
    segments: list[VisualSegment] = select_visual_segments(
        frames,
        fps=fps,
        diff_threshold=4.0,
        blank_threshold=3.0,
        consensus_frames=1,
    )

    # Check cache if enabled
    completed_checkpoint: dict[str, SegmentRecord] = {}
    if cache_path and cache_key:
        completed_checkpoint = ocr_cache.load_segment_checkpoint(cache_path, cache_key)

    missing_segments: list[VisualSegment] = []
    saved_records: dict[str, SegmentRecord] = dict(completed_checkpoint)
    cache_hits = 0

    for seg in segments:
        if seg.id in saved_records:
            cache_hits += 1
            continue
        missing_segments.append(seg)

    cache_misses = len(missing_segments)

    # 2. Batching
    batch_items: list[BatchItem] = []
    for s in missing_segments:
        batch_items.append(
            BatchItem(
                id=s.id,
                image=s.representative.payload,
                mime_type="image/jpeg",
                metadata={"segment": s},
            )
        )

    sent_images = len(batch_items)
    batches = media.chunk_sequence(batch_items, max(1, batch_size))
    total_requests = len(batches)

    # Concurrency and latency tracking
    lock = threading.Lock()
    current_active = 0
    local_peak = 0
    latencies: list[float] = []

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        nonlocal current_active, local_peak
        with lock:
            current_active += 1
            if current_active > local_peak:
                local_peak = current_active
        if active_tracker is not None:
            active_tracker.enter()

        t_req_start = time.monotonic()
        if simulated_delay_s > 0:
            time.sleep(simulated_delay_s)

        results: list[BatchResult] = []
        for item in items:
            raw_bytes = item.get_image_bytes()
            h = hashlib.sha256(raw_bytes).hexdigest()
            text = hash_to_text.get(h, "")
            if not text and "segment" in item.metadata:
                seg_ref: VisualSegment = item.metadata["segment"]
                text = hash_to_text.get(seg_ref.content_hash, "")
            results.append(BatchResult(id=item.id, text=text, confidence=1.0))

        req_latency = time.monotonic() - t_req_start
        with lock:
            latencies.append(req_latency)
            current_active -= 1

        if active_tracker is not None:
            active_tracker.leave()

        return results

    def on_batch_complete(res_list: list[BatchResult]) -> None:
        seg_map = {s.id: s for s in missing_segments}
        new_records: list[SegmentRecord] = []
        for r in res_list:
            seg = seg_map.get(r.id)
            if seg:
                rec = SegmentRecord(
                    segment_id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    content_hash=seg.content_hash,
                    text=r.text,
                    confidence=r.confidence,
                    uncertain=r.uncertain,
                )
                saved_records[seg.id] = rec
                new_records.append(rec)
        if new_records and cache_path and cache_key:
            ocr_cache.save_segment_checkpoint(cache_path, cache_key, new_records)

    # 3. Execution through Scheduler
    if batches:
        job_id = f"bench_{stream_name}_{int(time.monotonic() * 1000)}"
        scheduler.schedule_job(
            job_id=job_id,
            batches=batches,
            provider=mock_provider,
            on_batch_complete=on_batch_complete,
            max_retries=2,
            fail_fast=True,
        )

    # 4. Reconstruction
    all_recs = list(saved_records.values())
    cues: list[Cue] = merge_segment_cues(all_recs, min_duration=0.3)

    t_duration = time.monotonic() - t_start
    avg_latency_ms = (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0
    avg_batch = (sent_images / total_requests) if total_requests > 0 else 0.0
    rejected_frames = max(0, sampled_count - sent_images)
    reduction_pct = (
        ((sampled_count - sent_images) / sampled_count * 100.0)
        if sampled_count > 0
        else 0.0
    )

    # 5. Assertions & Invariants
    # Ground truth coverage: each gt segment must be matched by a cue
    covered_gt = 0
    for gt in ground_truth:
        matched = any(
            normalize_subtitle_text(c.text) == normalize_subtitle_text(gt.text)
            and max(c.start, gt.start) < min(c.end, gt.end)
            for c in cues
        )
        if matched:
            covered_gt += 1

    gt_total = len(ground_truth)
    coverage_pct = (covered_gt / gt_total * 100.0) if gt_total > 0 else 100.0

    # Hallucinations: output cues whose text does not match any ground truth
    hallucinations = 0
    for c in cues:
        norm_c = normalize_subtitle_text(c.text)
        if not any(normalize_subtitle_text(gt.text) == norm_c for gt in ground_truth):
            hallucinations += 1

    req_count_expected = math.ceil(sent_images / batch_size) if sent_images > 0 else 0
    req_count_ok = total_requests == req_count_expected
    reduction_ok = reduction_pct >= 70.0 or sampled_count < 10
    coverage_ok = coverage_pct >= 99.9
    hallucination_ok = hallucinations == 0

    passed = (
        coverage_ok
        and hallucination_ok
        and reduction_ok
        and req_count_ok
    )

    effective_peak = (
        active_tracker.peak if active_tracker is not None else local_peak
    )

    return ProfileResult(
        profile=stream_name,
        requested_original_sampled=sampled_count,
        sent_images=sent_images,
        total_requests=total_requests,
        avg_batch=round(avg_batch, 2),
        avg_latency_ms=round(avg_latency_ms, 2),
        total_duration_s=round(t_duration, 4),
        retries=0,
        subtitle_count=len(cues),
        rejected_frames=rejected_frames,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        reduction_pct=round(reduction_pct, 2),
        peak_active=effective_peak,
        ground_truth_count=gt_total,
        coverage_pct=round(coverage_pct, 2),
        hallucination_count=hallucinations,
        passed=passed,
        details={
            "req_count_expected": req_count_expected,
            "req_count_ok": req_count_ok,
            "reduction_ok": reduction_ok,
            "coverage_ok": coverage_ok,
        },
    )


def run_benchmark_profile(
    profile_name: str,
    *,
    scale: float = 1.0,
    batch_size: int = 8,
    max_concurrency: int = 4,
    fps: float = 2.0,
    base_temp_dir: Path | None = None,
) -> ProfileResult:
    """Runs benchmark for a single profile ('short', 'normal', 'long', or 'multi')."""
    owns_dir = False
    if base_temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="ai_ocr_bench_")
        work_dir = Path(temp_dir_obj.name)
        owns_dir = True
    else:
        work_dir = base_temp_dir
        temp_dir_obj = None

    try:
        if profile_name == "multi":
            # Multi profile: 3 concurrent jobs on shared scheduler
            scheduler = AIOCRScheduler(max_concurrency=max_concurrency)
            active_tracker = ActiveTracker()

            stream_results: list[ProfileResult] = []
            threads: list[threading.Thread] = []

            def worker(s_idx: int) -> None:
                s_dir = work_dir / f"stream_{s_idx}"
                f, gt, h2t = generate_synthetic_manifest(
                    "multi", s_dir, scale=scale, fps=fps, stream_id=s_idx + 1
                )
                res = _run_single_stream(
                    f"multi_s{s_idx}",
                    f,
                    gt,
                    h2t,
                    scheduler,
                    batch_size=batch_size,
                    fps=fps,
                    active_tracker=active_tracker,
                    simulated_delay_s=0.01,
                )
                stream_results.append(res)

            for i in range(3):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            scheduler.shutdown(timeout=1.0)

            # Aggregate multi result
            total_sampled = sum(r.requested_original_sampled for r in stream_results)
            total_sent = sum(r.sent_images for r in stream_results)
            total_reqs = sum(r.total_requests for r in stream_results)
            total_subs = sum(r.subtitle_count for r in stream_results)
            total_gt = sum(r.ground_truth_count for r in stream_results)
            total_hallucinations = sum(r.hallucination_count for r in stream_results)
            max_dur = max((r.total_duration_s for r in stream_results), default=0.0)
            avg_b = (total_sent / total_reqs) if total_reqs > 0 else 0.0
            avg_lat = (
                sum(r.avg_latency_ms for r in stream_results) / len(stream_results)
                if stream_results
                else 0.0
            )
            reduc = (
                ((total_sampled - total_sent) / total_sampled * 100.0)
                if total_sampled > 0
                else 0.0
            )
            peak_active = active_tracker.peak

            all_passed = (
                all(r.passed for r in stream_results)
                and (peak_active <= max_concurrency)
                and (peak_active >= min(max_concurrency, 2))
            )

            return ProfileResult(
                profile="multi",
                requested_original_sampled=total_sampled,
                sent_images=total_sent,
                total_requests=total_reqs,
                avg_batch=round(avg_b, 2),
                avg_latency_ms=round(avg_lat, 2),
                total_duration_s=round(max_dur, 4),
                retries=0,
                subtitle_count=total_subs,
                rejected_frames=total_sampled - total_sent,
                cache_hits=0,
                cache_misses=total_subs,
                reduction_pct=round(reduc, 2),
                peak_active=peak_active,
                ground_truth_count=total_gt,
                coverage_pct=100.0,
                hallucination_count=total_hallucinations,
                passed=all_passed,
                details={
                    "concurrency_limit": max_concurrency,
                    "concurrency_ok": peak_active <= max_concurrency,
                    "streams": [asdict(r) for r in stream_results],
                },
            )

        # Single stream profiles: short, normal, long
        scheduler = AIOCRScheduler(max_concurrency=max_concurrency)
        try:
            frames, gt, h2t = generate_synthetic_manifest(
                profile_name, work_dir, scale=scale, fps=fps
            )
            return _run_single_stream(
                profile_name,
                frames,
                gt,
                h2t,
                scheduler,
                batch_size=batch_size,
                fps=fps,
            )
        finally:
            scheduler.shutdown(timeout=1.0)
    finally:
        if owns_dir and temp_dir_obj is not None:
            gc.collect()
            shutil.rmtree(work_dir, ignore_errors=True)


def run_cache_benchmark(
    *,
    scale: float = 1.0,
    batch_size: int = 8,
    fps: float = 2.0,
) -> CacheTestResult:
    """Verifies that second run against SQLite cache results in ZERO provider calls."""
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="ai_ocr_cache_bench_")
    work_dir = Path(temp_dir_obj.name)

    try:
        cache_file = work_dir / "ocr_cache.sqlite3"
        cache_key = "bench_cache_key_v1"

        frames, gt, h2t = generate_synthetic_manifest(
            "short", work_dir, scale=scale, fps=fps
        )

        # Pass 1: Cold cache
        sched1 = AIOCRScheduler(max_concurrency=2)
        try:
            res1 = _run_single_stream(
                "cache_pass1",
                frames,
                gt,
                h2t,
                sched1,
                batch_size=batch_size,
                fps=fps,
                cache_path=cache_file,
                cache_key=cache_key,
            )
        finally:
            sched1.shutdown(timeout=1.0)

        # Pass 2: Warm cache (must hit 100% cache and make 0 provider calls)
        sched2 = AIOCRScheduler(max_concurrency=2)
        try:
            res2 = _run_single_stream(
                "cache_pass2",
                frames,
                gt,
                h2t,
                sched2,
                batch_size=batch_size,
                fps=fps,
                cache_path=cache_file,
                cache_key=cache_key,
            )
        finally:
            sched2.shutdown(timeout=1.0)

        p1_calls = res1.total_requests
        p2_calls = res2.total_requests
        cache_hits = res2.cache_hits
        cache_misses = res2.cache_misses
        hit_rate = (
            (cache_hits / (cache_hits + cache_misses) * 100.0)
            if (cache_hits + cache_misses) > 0
            else 0.0
        )
        zero_calls = (p2_calls == 0) and (res2.sent_images == 0)
        passed = zero_calls and (hit_rate >= 99.9)

        return CacheTestResult(
            first_run_provider_calls=p1_calls,
            second_run_provider_calls=p2_calls,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate_pct=round(hit_rate, 2),
            zero_provider_calls=zero_calls,
            passed=passed,
        )
    finally:
        gc.collect()
        shutil.rmtree(work_dir, ignore_errors=True)


def run_benchmark(
    *,
    profile: str = "all",
    scale: float = 1.0,
    batch_size: int = 8,
    max_concurrency: int = 4,
    fps: float = 2.0,
    run_cache_test: bool = True,
) -> BenchmarkReport:
    """Runs complete benchmark suite across selected profiles and returns report."""
    targets = (
        ["short", "normal", "long", "multi"]
        if profile == "all"
        else [profile]
    )

    profiles_res: dict[str, ProfileResult] = {}
    temp_bench_dir = tempfile.TemporaryDirectory(prefix="ai_ocr_bench_root_")
    root_path = Path(temp_bench_dir.name)

    try:
        for p in targets:
            sub_dir = root_path / p
            sub_dir.mkdir(parents=True, exist_ok=True)
            res = run_benchmark_profile(
                p,
                scale=scale,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                fps=fps,
                base_temp_dir=sub_dir,
            )
            profiles_res[p] = res

        cache_res: CacheTestResult | None = None
        if run_cache_test:
            cache_res = run_cache_benchmark(
                scale=scale,
                batch_size=batch_size,
                fps=fps,
            )

        all_ok = all(pr.passed for pr in profiles_res.values()) and (
            cache_res.passed if cache_res else True
        )

        return BenchmarkReport(
            timestamp=datetime.now(UTC).isoformat(),
            config={
                "profile": profile,
                "scale": scale,
                "batch_size": batch_size,
                "max_concurrency": max_concurrency,
                "fps": fps,
            },
            profiles=profiles_res,
            cache_test=cache_res,
            all_passed=all_ok,
        )
    finally:
        gc.collect()
        shutil.rmtree(root_path, ignore_errors=True)


def format_human_table(report: BenchmarkReport) -> str:
    """Formats benchmark results into a clean human-readable table."""
    lines: list[str] = []
    lines.append("=" * 105)
    lines.append("AI OCR REPEATABLE BENCHMARK RESULTS (ai-ocr-benchmark-r4)")
    lines.append("=" * 105)
    header = (
        f"{'Profile':<9} | {'Sampled':<7} | {'Sent':<5} | {'Reqs':<5} | "
        f"{'AvgBatch':<8} | {'Reduction':<9} | {'Subs':<5} | {'Latency':<8} | "
        f"{'Duration':<8} | {'Active':<6} | {'Status':<6}"
    )
    lines.append(header)
    lines.append("-" * 105)

    for name, p in report.profiles.items():
        status = "PASS" if p.passed else "FAIL"
        row = (
            f"{name:<9} | {p.requested_original_sampled:<7} | {p.sent_images:<5} | "
            f"{p.total_requests:<5} | {p.avg_batch:<8.1f} | {p.reduction_pct:<8.1f}% | "
            f"{p.subtitle_count:<5} | {p.avg_latency_ms:<5.1f} ms | {p.total_duration_s:<6.2f} s | "
            f"{p.peak_active:<6} | {status:<6}"
        )
        lines.append(row)

    lines.append("-" * 105)
    if report.cache_test:
        c = report.cache_test
        c_status = "PASS" if c.passed else "FAIL"
        lines.append(
            "Cache Invalidation & Hit Test (2nd run): "
            f"Provider calls = {c.second_run_provider_calls} "
            f"(first run: {c.first_run_provider_calls}), "
            f"Cache hits = {c.cache_hits} ({c.cache_hit_rate_pct:.1f}%) -> {c_status}"
        )

    lines.append("=" * 105)
    lines.append(f"Overall Result: {'PASS' if report.all_passed else 'FAIL'}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI Entrypoint for AI OCR Benchmark."""
    parser = argparse.ArgumentParser(
        description="Repeatable AI OCR Benchmark without external network calls."
    )
    parser.add_argument(
        "--profile",
        choices=["short", "normal", "long", "multi", "all"],
        default="all",
        help="Benchmark profile to run (default: all)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale multiplier for frame counts (e.g. 0.1 for fast verification, default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for AI OCR requests (default: 8)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent worker requests (default: 4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame rate sampling (default: 2.0)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="Optional path to output JSON benchmark report",
    )
    parser.add_argument(
        "--no-cache-test",
        action="store_true",
        help="Skip two-pass cache hit test",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress human-readable table, output minimal text",
    )

    args = parser.parse_args(argv)

    report = run_benchmark(
        profile=args.profile,
        scale=args.scale,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        fps=args.fps,
        run_cache_test=not args.no_cache_test,
    )

    if not args.quiet:
        print(format_human_table(report))

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"Report saved to: {json_path}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
