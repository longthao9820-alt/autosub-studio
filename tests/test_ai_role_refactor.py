from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from autosub_studio.core.models import Cue
from autosub_studio.data.db import Database, TranslationCache
from autosub_studio.data.project import ProjectStore
from autosub_studio.pipeline import steps
from autosub_studio.providers import translate
from autosub_studio.services import ai_gateway, media
from autosub_studio.services.ai_ocr_scheduler import AIOCRScheduler, is_retryable_error
from autosub_studio.services.ffmpeg import CancelledError, CancelToken, FFmpeg
from autosub_studio.services.settings import Settings


def _scheduler(monkeypatch: pytest.MonkeyPatch) -> AIOCRScheduler:
    scheduler = AIOCRScheduler(max_concurrency=2, base_retry_delay=0.0)
    monkeypatch.setattr(translate, "get_global_gateway_scheduler", lambda: scheduler)
    return scheduler


def test_role_mapping_and_adapter_capabilities() -> None:
    assert ai_gateway.role_for_task("subtitle_extraction") == "sub"
    assert ai_gateway.role_for_task("subtitle_translation") == "prime"
    caps = ai_gateway.capabilities()
    assert caps.multi_image is True
    assert caps.structured_text is True
    assert caps.video is False
    assert caps.media_upload is False

    try:
        try:
            raise ai_gateway.AIGatewayTimeoutError()
        except ai_gateway.AIGatewayError as cause:
            raise translate.TranslationError("wrapped") from cause
    except translate.TranslationError as wrapped:
        assert is_retryable_error(wrapped) is True


def test_whole_document_uses_one_prime_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(monkeypatch)
    settings = Settings(
        ai_endpoint="http://gateway/v1",
        ai_model_sub="vision-role",
        ai_model_prime="translation-role",
        llm_model="sub",
    )
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(Settings, "get_secret", lambda _name: "secret")
    calls: list[dict] = []

    def fake_chat(*_args, **kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"].split("JSON):\n", 1)[1])
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "text": f"T:{item['text']}"}
                    for item in payload
                ]
            }
        )

    monkeypatch.setattr(ai_gateway, "chat_completion", fake_chat)
    try:
        out = translate.translate_document(
            translate.PROVIDER_AI,
            translate.TranslationRequest(
                texts=["one", "two", "three"],
                ids=[10, 11, 12],
                target="vi",
            ),
            api_key="secret",
        )
    finally:
        scheduler.shutdown(timeout=1.0)

    assert out == ["T:one", "T:two", "T:three"]
    assert len(calls) == 1
    assert calls[0]["model"] == "translation-role"


def test_missing_id_repairs_only_missing_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(monkeypatch)
    settings = Settings(ai_endpoint="http://gateway/v1")
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(Settings, "get_secret", lambda _name: "secret")
    seen_ids: list[list[int]] = []
    checkpoints: list[list[int]] = []

    def fake_chat(*_args, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"].split("JSON):\n", 1)[1])
        ids = [item["id"] for item in payload]
        seen_ids.append(ids)
        returned = ids if len(seen_ids) > 1 else [ids[0], ids[-1]]
        return json.dumps(
            {
                "translations": [
                    {"id": item_id, "text": f"translated-{item_id}"}
                    for item_id in returned
                ]
            }
        )

    monkeypatch.setattr(ai_gateway, "chat_completion", fake_chat)
    try:
        result = translate.translate_document(
            translate.PROVIDER_AI,
            translate.TranslationRequest(
                texts=["a", "b", "c"], ids=[100, 101, 102], target="en"
            ),
            api_key="secret",
            on_chunk_complete=lambda ids, _values: checkpoints.append(ids),
        )
    finally:
        scheduler.shutdown(timeout=1.0)

    assert result == ["translated-100", "translated-101", "translated-102"]
    assert seen_ids == [[100, 101, 102], [101]]
    assert checkpoints == [[100, 102], [101]]


def test_translation_chunks_by_payload_size_not_old_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(monkeypatch)
    settings = Settings(ai_endpoint="http://gateway/v1")
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(Settings, "get_secret", lambda _name: "secret")
    request_sizes: list[int] = []

    def fake_chat(*_args, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"].split("JSON):\n", 1)[1])
        request_sizes.append(len(payload))
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "text": "ok"} for item in payload
                ]
            }
        )

    monkeypatch.setattr(ai_gateway, "chat_completion", fake_chat)
    try:
        translate.translate_document(
            translate.PROVIDER_AI,
            translate.TranslationRequest(texts=["x" * 1_200] * 3),
            api_key="secret",
            max_characters=2_000,
        )
    finally:
        scheduler.shutdown(timeout=1.0)

    assert request_sizes == [1, 1, 1]


def test_translation_cancellation_is_cooperative(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    settings = Settings(ai_endpoint="http://gateway/v1")
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(Settings, "get_secret", lambda _name: "secret")
    try:
        with pytest.raises(CancelledError):
            translate.translate_document(
                translate.PROVIDER_AI,
                translate.TranslationRequest(texts=["cancel me"]),
                api_key="secret",
                should_cancel=lambda: True,
            )
    finally:
        scheduler.shutdown(timeout=1.0)


def test_frame_pts_keeps_real_timestamps_when_showinfo_has_boundary_extra() -> None:
    assert media._align_frame_pts([0.0, 0.08, 0.16, 0.24], 3, 1 / 15) == [
        0.0,
        0.08,
        0.16,
    ]


def test_translation_cache_reuses_validated_document_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _scheduler(monkeypatch)
    settings = Settings(
        workspace=str(tmp_path / "workspace"),
        translate_provider=translate.PROVIDER_AI,
        ai_endpoint="http://gateway/v1",
        source_language="zh-CN",
        target_language="vi",
    )
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(Settings, "get_secret", lambda _name: "secret")
    calls = 0

    def fake_chat(*_args, **kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(kwargs["messages"][1]["content"].split("JSON):\n", 1)[1])
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "text": f"VI:{item['text']}"}
                    for item in payload
                ]
            }
        )

    monkeypatch.setattr(ai_gateway, "chat_completion", fake_chat)
    store = ProjectStore(settings.workspace)
    database = Database(tmp_path / "workspace" / "db" / "app.db")

    def make_context(project_id: int):
        project = store.create(project_id, f"project-{project_id}")
        project.doc.cues = [Cue(0, 1, "甲"), Cue(1, 2, "乙")]
        task = MagicMock()
        task.token = CancelToken()
        return project, steps.PipelineContext(
            ff=FFmpeg(),
            settings=settings,
            store=store,
            project=project,
            task=task,
            api_key="secret",
            db=database,
        )

    try:
        first_project, first = make_context(1)
        steps.step_translate(first)
        assert calls == 1
        assert [cue.translation for cue in first_project.doc.cues] == ["VI:甲", "VI:乙"]

        with database.session() as session:
            assert session.query(TranslationCache).count() == 2

        second_project, second = make_context(2)
        result = steps.step_translate(second)
        assert "da co ban dich" in result.lower()
        assert calls == 1
        assert [cue.translation for cue in second_project.doc.cues] == ["VI:甲", "VI:乙"]
    finally:
        scheduler.shutdown(timeout=1.0)
        database.engine.dispose()


def test_changing_target_language_does_not_resume_old_translations(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _scheduler(monkeypatch)
    settings = Settings(
        workspace=str(tmp_path / "workspace"),
        translate_provider=translate.PROVIDER_AI,
        ai_endpoint="http://gateway/v1",
        source_language="zh-CN",
        target_language="vi",
    )
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(Settings, "get_secret", lambda _name: "secret")
    calls = 0

    def fake_chat(*_args, **kwargs):
        nonlocal calls
        calls += 1
        payload = json.loads(kwargs["messages"][1]["content"].split("JSON):\n", 1)[1])
        return json.dumps(
            {"translations": [{"id": item["id"], "text": "bản mới"} for item in payload]}
        )

    monkeypatch.setattr(ai_gateway, "chat_completion", fake_chat)
    store = ProjectStore(settings.workspace)
    project = store.create(1, "target-change")
    project.doc.target_language = "en"
    project.doc.cues = [Cue(0, 1, "甲", translation="old English")]
    task = MagicMock()
    task.token = CancelToken()
    context = steps.PipelineContext(
        ff=FFmpeg(),
        settings=settings,
        store=store,
        project=project,
        task=task,
        api_key="secret",
    )
    try:
        steps.step_translate(context)
    finally:
        scheduler.shutdown(timeout=1.0)

    assert calls == 1
    assert project.doc.target_language == "vi"
    assert project.doc.cues[0].translation == "bản mới"
