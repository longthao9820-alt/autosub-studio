"""Tests for AI OCR Benchmark harness (ai-ocr-benchmark-r4).

Verifies:
- Fast synthetic manifest generation and execution (scale=0.1)
- Preservation of ground truth segments (100% coverage)
- Zero hallucinations in subtitle cues
- Frame reduction >= 70%
- Batch request count == ceil(sent_images / batch_size)
- Global scheduler max concurrency limit respected in multi profile
- Second run cache hit rate == 100% with zero provider calls
- CLI JSON file generation and schema integrity
- Baseline fixture verification
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.benchmark_ai_ocr import (
    main,
    run_benchmark,
    run_benchmark_profile,
    run_cache_benchmark,
)


def test_benchmark_fast_profiles_all_pass() -> None:
    """Runs all benchmark profiles with scale=0.1 in < 1 second and checks invariants."""
    report = run_benchmark(profile="all", scale=0.1, batch_size=8, max_concurrency=4)

    assert report.all_passed is True
    assert set(report.profiles.keys()) == {"short", "normal", "long", "multi"}

    for name, p in report.profiles.items():
        assert p.passed is True, f"Profile {name} failed: {p.details}"
        assert p.coverage_pct >= 99.9, f"Profile {name} dropped ground truth segments"
        assert p.hallucination_count == 0, f"Profile {name} produced hallucinated subtitles"
        assert p.reduction_pct >= 70.0, f"Profile {name} reduction below 70%: {p.reduction_pct}%"

        if name == "multi":
            expected_reqs = sum(s["total_requests"] for s in p.details["streams"])
        else:
            expected_reqs = math.ceil(p.sent_images / 8) if p.sent_images > 0 else 0
        assert p.total_requests == expected_reqs, (
            f"Profile {name} requests mismatch: {p.total_requests} != {expected_reqs}"
        )
        assert p.subtitle_count == p.ground_truth_count


def test_benchmark_multi_concurrency_limit() -> None:
    """Proves that multi-stream benchmark strictly enforces scheduler max_concurrency."""
    max_limit = 2
    res = run_benchmark_profile("multi", scale=0.1, max_concurrency=max_limit)

    assert res.passed is True
    assert res.peak_active <= max_limit
    assert res.peak_active >= 1
    assert res.coverage_pct >= 99.9
    assert res.hallucination_count == 0


def test_benchmark_cache_second_run_zero_calls() -> None:
    """Proves that running second pass on warm SQLite cache produces zero provider calls."""
    res = run_cache_benchmark(scale=0.1)

    assert res.passed is True
    assert res.zero_provider_calls is True
    assert res.second_run_provider_calls == 0
    assert res.cache_hit_rate_pct >= 99.9
    assert res.cache_hits > 0


def test_benchmark_cli_json_export(tmp_path: Path) -> None:
    """Verifies that CLI generates expected JSON report without network calls."""
    out_json = tmp_path / "benchmark_result.json"
    rc = main(["--profile", "short", "--scale", "0.1", "--json", str(out_json), "--quiet"])

    assert rc == 0
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert "timestamp" in data
    assert "short" in data["profiles"]
    short_prof = data["profiles"]["short"]
    assert short_prof["passed"] is True
    assert short_prof["reduction_pct"] >= 70.0
    assert short_prof["coverage_pct"] >= 99.9
    assert short_prof["hallucination_count"] == 0
    assert data["cache_test"]["zero_provider_calls"] is True


def test_benchmark_baseline_fixture_matches() -> None:
    """Verifies baseline fixture exists and conforms to benchmark invariants."""
    baseline_path = (
        Path(__file__).resolve().parent / "fixtures" / "ai_ocr_benchmark_baseline.json"
    )
    assert baseline_path.is_file()

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["all_passed"] is True
    for _p_name, p in baseline["profiles"].items():
        assert p["passed"] is True
        assert p["coverage_pct"] >= 99.9
        assert p["hallucination_count"] == 0
        assert p["reduction_pct"] >= p["min_reduction_pct"]
        assert p["reduction_pct"] >= 70.0
    assert baseline["cache_test"]["zero_provider_calls"] is True
    assert baseline["cache_test"]["cache_hit_rate_pct"] == 100.0


def test_benchmark_deterministic_reproducibility() -> None:
    """Verifies repeated runs of short profile yield identical metrics."""
    res1 = run_benchmark_profile("short", scale=0.1)
    res2 = run_benchmark_profile("short", scale=0.1)

    assert res1.requested_original_sampled == res2.requested_original_sampled
    assert res1.sent_images == res2.sent_images
    assert res1.total_requests == res2.total_requests
    assert res1.subtitle_count == res2.subtitle_count
    assert res1.reduction_pct == res2.reduction_pct
    assert res1.ground_truth_count == res2.ground_truth_count
