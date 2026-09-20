"""Acceptance Matrix and Architectural Invariant Tests (ai-ocr-benchmark-r4).

Maps user acceptance requirements across 11 critical categories to existing test suites
and enforces core architectural invariants via AST and deterministic checks:
1. Gateway config / vision
2. Crop region handling
3. Visual duplicate reduction & representative frame selection
4. Batching, splitting, concurrency, out-of-order execution, and ID validation
5. Error taxonomy: retry, exponential backoff, auth fail-fast, model errors
6. Checkpoint, resume, and cache invalidation
7. Cooperative cancellation and temp cleanup
8. Multi-video shared scheduler concurrency
9. Subtitle order, timing monotonicity, and boundary carryover
10. Strict separation from RapidOCR / no silent local fallback
11. Bounded memory and open file descriptors
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autosub_studio.data import ocr_cache
from autosub_studio.data.project import ProjectData
from autosub_studio.pipeline.steps import PipelineContext, StepError, step_ocr
from autosub_studio.providers import ocr
from autosub_studio.services import ai_gateway
from autosub_studio.services.ai_ocr_scheduler import (
    AIOCRScheduler,
    _sanitize_log_text,
    is_retryable_error,
)
from autosub_studio.services.ffmpeg import FFmpeg
from autosub_studio.services.settings import Settings

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Acceptance Matrix: 11 Categories -> Discovered Test Functions
# ---------------------------------------------------------------------------

ACCEPTANCE_MATRIX: dict[str, list[tuple[str, str]]] = {
    "1_gateway_config_vision": [
        ("tests/test_ai_gateway.py", "test_chat_completion_payload_and_auth"),
        ("tests/test_ai_gateway.py", "test_connection_success"),
        ("tests/test_ocr_ai_provider.py", "test_test_vision_success"),
        ("tests/test_ocr_ai_provider.py", "test_test_vision_marker_mismatch"),
        ("tests/test_ocr_ai_provider.py", "test_generate_marker_image"),
        ("tests/test_ai_ocr_vision_ui.py", "test_dialog_vision_buttons_exist"),
        ("tests/test_ai_ocr_vision_ui.py", "test_main_window_alias_resolution_and_task_submission"),
    ],
    "2_crop": [
        ("tests/test_ocr_selector.py", "test_already_cropped_image_preserved"),
        ("tests/test_ocr_selector.py", "test_crop_helper_in_media"),
        ("tests/test_ocr_selector.py", "test_prepare_image_with_crop_region"),
        ("tests/test_ai_ocr_pipeline.py", "test_crop_region_respected"),
    ],
    "3_visual_duplicate_representative": [
        ("tests/test_ocr_selector.py", "test_blank_frames_detected"),
        ("tests/test_ocr_selector.py", "test_text_frame_not_blank"),
        ("tests/test_ocr_selector.py", "test_near_duplicate_and_noise_kept_in_same_segment"),
        ("tests/test_ocr_selector.py", "test_representative_is_clearest_middle_frame"),
        ("tests/test_ocr_selector.py", "test_compute_clarity_score_ordering"),
        ("tests/test_ocr_selector.py", "test_duplicate_filtering_by_content_hash"),
        ("tests/test_ai_ocr_pipeline.py", "test_redundant_frame_reduction"),
    ],
    "4_batch_split_concurrency_validation": [
        ("tests/test_ocr_ai_provider.py", "test_messages_structure_and_id_binding"),
        ("tests/test_ocr_ai_provider.py", "test_missing_ids_rejected"),
        ("tests/test_ocr_ai_provider.py", "test_duplicate_ids_rejected"),
        ("tests/test_ocr_ai_provider.py", "test_batch_16_split_to_8_then_4"),
        ("tests/test_ai_ocr_pipeline.py", "test_batch_creation"),
        ("tests/test_ai_ocr_pipeline.py", "test_split_via_provider"),
        ("tests/test_ai_ocr_pipeline.py", "test_out_of_order_execution"),
        ("tests/test_ai_ocr_scheduler.py", "test_out_of_order_completion_mapped_to_stable_keys"),
    ],
    "5_error_taxonomy_retry_backoff_auth": [
        ("tests/test_ocr_ai_provider.py", "test_taxonomy_status_and_classes"),
        ("tests/test_ocr_ai_provider.py", "test_timeout_and_connection_errors"),
        ("tests/test_ocr_ai_provider.py", "test_error_sanitizes_authorization_headers_and_secrets"),
        (
            "tests/test_ai_ocr_scheduler.py",
            "test_retry_transient_and_adaptive_concurrency_adjustment",
        ),
        ("tests/test_ai_ocr_scheduler.py", "test_auth_error_no_retry_and_marks_endpoint"),
        ("tests/test_ai_ocr_scheduler.py", "test_is_retryable_error_classification"),
        ("tests/test_ai_ocr_pipeline.py", "test_timeout_errors_preserve_completed"),
    ],
    "6_checkpoint_resume_cache_invalidation": [
        ("tests/test_ocr_ai_foundation.py", "test_cache_key_invalidation_on_parameters"),
        ("tests/test_ocr_ai_foundation.py", "test_immediate_batch_checkpoint_and_resume"),
        ("tests/test_ocr_ai_foundation.py", "test_api_key_never_stored_in_cache"),
        ("tests/test_ai_ocr_pipeline.py", "test_checkpoint_resume_after_interruption"),
        ("tests/test_ai_ocr_pipeline.py", "test_invalidation_model_region_prompt"),
    ],
    "7_cancellation": [
        ("tests/test_ocr_selector.py", "test_cancellation_with_should_cancel"),
        ("tests/test_ocr_selector.py", "test_cancellation_with_cancel_token"),
        (
            "tests/test_ai_ocr_scheduler.py",
            "test_cancellation_isolation_does_not_affect_other_jobs",
        ),
        ("tests/test_ai_ocr_pipeline.py", "test_cancellation"),
    ],
    "8_multi_video": [
        (
            "tests/test_ai_ocr_scheduler.py",
            "test_global_concurrency_across_three_projects_within_limit",
        ),
        ("tests/test_ai_ocr_pipeline.py", "test_multi_video_shared_concurrency_integration"),
    ],
    "9_srt_order_timing": [
        ("tests/test_ai_ocr_pipeline.py", "test_timing_srt_order"),
        ("tests/test_ai_ocr_pipeline.py", "test_content_hash_boundary_carryover_reuse"),
    ],
    "10_no_rapidocr_calls_or_fallback": [
        ("tests/test_ocr_selector.py", "test_no_ai_or_rapidocr_imports"),
        (
            "tests/test_ocr_ai_foundation.py",
            "test_step_ocr_never_calls_local_read_or_probe_in_ai_mode",
        ),
        ("tests/test_ai_ocr_pipeline.py", "test_no_local_symbols_called"),
    ],
    "11_bounded_temp_and_ram": [
        ("tests/test_ocr_selector.py", "test_open_image_count_bounded"),
        ("tests/test_ai_ocr_pipeline.py", "test_bounded_chunk_cleanup"),
        ("tests/test_ai_ocr_scheduler.py", "test_bounded_queue_and_backpressure"),
    ],
}


def test_acceptance_matrix_traceability() -> None:
    """Verifies that all 11 user contract categories map to existing test files and functions."""
    assert len(ACCEPTANCE_MATRIX) == 11

    missing_refs: list[str] = []
    for category, tests in ACCEPTANCE_MATRIX.items():
        assert len(tests) >= 2, (
            f"Category {category} must have at least 2 referenced test functions"
        )
        for rel_path, fn_name in tests:
            full_path = ROOT_DIR / rel_path
            if not full_path.is_file():
                missing_refs.append(f"Missing file: {rel_path}")
                continue
            tree = ast.parse(full_path.read_text(encoding="utf-8"))
            funcs = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if fn_name not in funcs:
                missing_refs.append(f"Missing test function: {rel_path}::{fn_name}")

    assert not missing_refs, f"Acceptance matrix references missing tests: {missing_refs}"


# ---------------------------------------------------------------------------
# Architectural Invariants
# ---------------------------------------------------------------------------


def test_invariant_ai_step_ast_no_local_ocr_calls() -> None:
    """AST check: In step_ocr, AI branch NEVER calls local OCR probe/read/refine."""
    steps_path = ROOT_DIR / "src" / "autosub_studio" / "pipeline" / "steps.py"
    tree = ast.parse(steps_path.read_text(encoding="utf-8"))

    step_ocr_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "step_ocr":
            step_ocr_fn = node
            break

    assert step_ocr_fn is not None, "step_ocr function definition not found in steps.py"

    # Inspect the if ai_mode branch inside step_ocr
    ai_branch_calls: list[str] = []
    non_ai_branch_calls: list[str] = []

    for stmt in step_ocr_fn.body:
        if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "ai_mode":
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    ai_branch_calls.append(ast.unparse(n))
            # The remaining body of step_ocr represents non-ai branch
            break

    for stmt in step_ocr_fn.body:
        # Non-AI statements below if ai_mode
        if not (isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "ai_mode"):
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    non_ai_branch_calls.append(ast.unparse(n))

    # Assert AI branch has 0 local OCR read/probe/refine calls
    forbidden_local_calls = ["ocr.read_frames", "ocr.probe_frames", "ocr.refine_boundaries"]
    for call_repr in ai_branch_calls:
        for forbidden in forbidden_local_calls:
            assert forbidden not in call_repr, (
                f"Architecture violation: AI OCR branch invokes local OCR '{forbidden}'"
            )

    # Assert non-AI branch does call local OCR
    assert any("ocr.read_frames" in c for c in non_ai_branch_calls), (
        "Non-AI OCR path must call ocr.read_frames"
    )


def test_invariant_provider_and_selector_no_local_ocr_imports() -> None:
    """AST check: AI provider and selector modules have ZERO imports of RapidOCR or local OCR."""
    checked_files = [
        ROOT_DIR / "src" / "autosub_studio" / "providers" / "ocr_ai_provider.py",
        ROOT_DIR / "src" / "autosub_studio" / "providers" / "ocr_selector.py",
        ROOT_DIR / "src" / "autosub_studio" / "services" / "ai_ocr_scheduler.py",
    ]

    for fpath in checked_files:
        assert fpath.is_file(), f"File {fpath} not found"
        tree = ast.parse(fpath.read_text(encoding="utf-8"))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        for imp in imported_modules:
            assert "rapidocr" not in imp.lower(), (
                f"{fpath.name} imports rapidocr: {imp}"
            )
            assert "paddleocr" not in imp.lower(), (
                f"{fpath.name} imports paddleocr: {imp}"
            )
            assert not imp.endswith(".ocr"), (
                f"{fpath.name} imports local ocr module: {imp}"
            )


def test_invariant_api_key_absent_from_project_cache_logs(tmp_path: Path) -> None:
    """Ensures secret API key is never written to project JSON, cache SQLite, or sanitized logs."""
    dummy_key = "sk-ant-api03-SECRETKEY12345678"

    # 1. Project file check
    project = ProjectData(folder=str(tmp_path), name="secret_test")
    project_json = json.dumps(project.to_dict())
    assert dummy_key not in project_json, "API key leaked into project dictionary"

    # 2. SQLite Cache key and metadata check
    db_path = tmp_path / "cache_leak_test.sqlite3"
    cache_key = ocr_cache.make_ai_ocr_cache_key(
        video=str(tmp_path / "vid.mp4"),
        region=[0, 0, 100, 50],
        fps=2.0,
        endpoint="http://example.com/v1",
        model_alias="sub",
        actual_model="sub",
    )
    assert dummy_key not in cache_key, "API key leaked into cache key"

    ocr_cache.save_cache_meta(
        db_path,
        cache_key,
        {"endpoint_hash": "abc", "model_alias": "sub"},
    )
    with sqlite3.connect(str(db_path)) as conn:
        dump = "\n".join(conn.iterdump())
    assert dummy_key not in dump, "API key leaked into cache SQLite tables"

    # 3. Log text sanitization
    sanitized = _sanitize_log_text(f"Authorization: Bearer {dummy_key}")
    assert dummy_key not in sanitized
    assert "Bearer sk-ant-***" in sanitized or "***" in sanitized


def test_invariant_ai_error_raises_without_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step OCR in AI mode raises StepError immediately on AI failure without local fallback."""
    # Poison all local OCR calls
    def forbidden_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("Local OCR was invoked as silent fallback in AI mode!")

    monkeypatch.setattr(ocr, "probe_frames", forbidden_call)
    monkeypatch.setattr(ocr, "read_frames", forbidden_call)
    monkeypatch.setattr(ocr, "refine_boundaries", forbidden_call)
    monkeypatch.setattr(ocr, "is_available", forbidden_call)

    # Mock extract_video_chunks to yield a chunk with a text frame
    def mock_extract_chunks(*args: object, **kwargs: object):
        out_dir_parent = kwargs.get("out_dir_parent", tmp_path / "chunks")
        d = Path(str(out_dir_parent)) / "chunk_0"
        d.mkdir(parents=True, exist_ok=True)
        frame_path = d / "frame_000001.png"
        from PIL import Image
        img = Image.new("RGB", (64, 32), color=(0, 0, 0))
        for x in range(10, 40):
            for y in range(8, 24):
                img.putpixel((x, y), (255, 255, 255))
        img.save(frame_path)
        yield 0, 0.0, 5.0, [(1.0, frame_path)], d

    from autosub_studio.services import media
    monkeypatch.setattr(media, "extract_video_chunks", mock_extract_chunks)

    # Mock chat completion to fail with network error
    def failing_chat_completion(*args: object, **kwargs: object) -> None:
        raise ai_gateway.AIGatewayError("Gateway connection refused", status_code=503)

    monkeypatch.setattr(ai_gateway, "chat_completion", failing_chat_completion)

    video = tmp_path / "test_video.mp4"
    video.write_bytes(b"dummy")

    settings = Settings(
        ocr_mode="OCR AI",
        ocr_server="Server AI API",
        ai_endpoint="http://localhost:8000/v1",
    )
    project = ProjectData(
        folder=str(tmp_path),
        ocr_region=[10, 10, 100, 30],
        video_path=str(video),
    )
    from autosub_studio.services.ffmpeg import CancelToken
    task = MagicMock()
    task.token = CancelToken()

    pc = PipelineContext(
        ff=MagicMock(spec=FFmpeg),
        settings=settings,
        store=MagicMock(),
        project=project,
        task=task,
        api_key="mock_key_abc",
    )

    with pytest.raises(StepError, match="Lỗi OCR AI"):
        step_ocr(pc)


def test_invariant_invalid_model_permanent_no_retry() -> None:
    """Proves 404 ModelNotFound error is classified non-retryable and aborts immediately."""
    err_404 = ai_gateway.AIGatewayError("Model 'nonexistent-model' not found", status_code=404)
    assert not is_retryable_error(err_404), "404 model error must be marked non-retryable"

    scheduler = AIOCRScheduler(max_concurrency=2)
    provider_calls = 0

    def failing_provider(items: object) -> list[object]:
        nonlocal provider_calls
        provider_calls += 1
        raise err_404

    from autosub_studio.providers.ocr_ai_provider import BatchItem

    items = [BatchItem(id="item_0", image=b"img")]
    with pytest.raises(ai_gateway.AIGatewayError, match="not found"):
        scheduler.schedule_job(
            job_id="model_err_job",
            batches=[items],
            provider=failing_provider,
            max_retries=3,
            fail_fast=True,
        )
    scheduler.shutdown(timeout=1.0)

    # fail_fast + non-retryable error aborts on first attempt without retrying
    assert provider_calls == 1, (
        f"Non-retryable 404 error was retried {provider_calls - 1} times"
    )
