"""Kiem thu bo lap lich AI OCR Scheduler toan cuc.

Bao gom:
1. Concurrency toan cuc giua 3 projects luon <= gioi han da cau hinh.
2. Dieu phoi round-robin cong bang (fair order), khong de video dai gay doi kiet.
3. Ket qua duoc anh xa chinh xac theo ID bat ke thu tu hoan thanh thuc te (out-of-order mapping).
4. Thu lai voi loi transient (429, 5xx, timeout) va co che dieu tiet thich ung (adaptive pressure).
5. Loi auth khong thu lai (auth no retry) va danh dau endpoint that bai.
6. Malformed batch (OCRValidationError) chi thu lai tren batch bi loi.
7. Cach ly huy bo (cancellation isolation) giua cac jobs.
8. Hang doi bi chan (bounded queue) va co che ap luc nguoc (backpressure).
9. Shutdown dung han trong thoi gian huu han (finite shutdown).
10. Ban chup thong ke thread-safe (metrics snapshot).
11. Khong tiet lo thong tin nhay cam (no secret logs).
12. Tai cau hinh khong nhan ban worker pool va lien ket vong doi app (Qt lifecycle).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator, Sequence
from typing import Any

import pytest

from autosub_studio.app import setup_app_lifecycle
from autosub_studio.providers.ocr_ai_provider import (
    BatchItem,
    BatchResult,
    OCRValidationError,
)
from autosub_studio.services import ai_gateway
from autosub_studio.services.ai_ocr_scheduler import (
    AIOCRScheduler,
    _sanitize_log_text,
    get_global_scheduler,
    is_retryable_error,
    shutdown_global_scheduler,
)
from autosub_studio.services.ffmpeg import CancelToken


def _make_item(item_id: str) -> BatchItem:
    """Tao BatchItem gia lap voi payload nho de test."""
    return BatchItem(id=item_id, image=b"fake_jpeg_image_data")


def _make_result(item_id: str, text: str = "sample text") -> BatchResult:
    """Tao BatchResult tuong ung voi ID."""
    return BatchResult(id=item_id, text=text, confidence=1.0)


@pytest.fixture(autouse=True)
def _cleanup_global_scheduler() -> Generator[None, None, None]:
    """Don dep scheduler toan cuc truoc va sau moi test."""
    shutdown_global_scheduler(timeout=1.0)
    yield
    shutdown_global_scheduler(timeout=1.0)


class FakeClock:
    """Dong ho gia lap cho phep dieu khien thoi gian xac dinh trong test."""

    def __init__(self, initial: float = 100.0) -> None:
        self.current = initial

    def now(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


# 1. Global concurrency across 3 projects <= limit
def test_global_concurrency_across_three_projects_within_limit() -> None:
    """Kiem tra tong so HTTP requests dong thoi giua 3 du an khong bao gio vuot qua limit."""
    limit = 2
    peak_active = 0
    current_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(limit)
    thread_errors: list[Exception] = []
    concurrency_violations: list[int] = []

    scheduler = AIOCRScheduler(max_concurrency=limit)

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        nonlocal peak_active, current_active
        with lock:
            current_active += 1
            if current_active > peak_active:
                peak_active = current_active
            if current_active > limit:
                concurrency_violations.append(current_active)

        try:
            # Dong bo hoa de 2 requests cung chay song song
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=1.0)
            return [_make_result(item.id) for item in items]
        finally:
            with lock:
                current_active -= 1

    batches_p1 = [[_make_item("p1_1")], [_make_item("p1_2")]]
    batches_p2 = [[_make_item("p2_1")], [_make_item("p2_2")]]
    batches_p3 = [[_make_item("p3_1")], [_make_item("p3_2")]]

    res_p1: dict[str, BatchResult] = {}
    res_p2: dict[str, BatchResult] = {}
    res_p3: dict[str, BatchResult] = {}

    def run_p1() -> None:
        nonlocal res_p1
        try:
            res_p1 = scheduler.schedule_job("p1", batches_p1, mock_provider)
        except Exception as exc:
            thread_errors.append(exc)

    def run_p2() -> None:
        nonlocal res_p2
        try:
            res_p2 = scheduler.schedule_job("p2", batches_p2, mock_provider)
        except Exception as exc:
            thread_errors.append(exc)

    def run_p3() -> None:
        nonlocal res_p3
        try:
            res_p3 = scheduler.schedule_job("p3", batches_p3, mock_provider)
        except Exception as exc:
            thread_errors.append(exc)

    threads = [
        threading.Thread(target=run_p1),
        threading.Thread(target=run_p2),
        threading.Thread(target=run_p3),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    scheduler.shutdown(timeout=1.0)

    assert not thread_errors
    assert not concurrency_violations
    assert peak_active <= limit
    assert peak_active > 0
    assert len(res_p1) == 2
    assert len(res_p2) == 2
    assert len(res_p3) == 2


# 2. Fair order (round-robin across job IDs)
def test_fair_order_round_robin_interleaving() -> None:
    """Kiem tra round-robin luan phien cong bang giua cac jobs, tranh starvation."""
    limit = 1  # 1 slot de kiem tra thu tu tuan tu
    execution_order: list[str] = []
    order_lock = threading.Lock()
    all_registered = threading.Event()
    first_call_hit = threading.Event()

    clock = FakeClock()
    scheduler = AIOCRScheduler(
        max_concurrency=limit,
        clock_fn=clock.now,
        sleep_fn=lambda _d: None,
    )

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        first_call_hit.set()
        all_registered.wait(timeout=2.0)
        with order_lock:
            job_prefix = items[0].id.split("_")[0]
            execution_order.append(job_prefix)
        return [_make_result(item.id) for item in items]

    batches_j1 = [[_make_item("j1_1")], [_make_item("j1_2")], [_make_item("j1_3")]]
    batches_j2 = [[_make_item("j2_1")], [_make_item("j2_2")], [_make_item("j2_3")]]
    batches_j3 = [[_make_item("j3_1")], [_make_item("j3_2")], [_make_item("j3_3")]]

    t1 = threading.Thread(target=lambda: scheduler.schedule_job("j1", batches_j1, mock_provider))
    t2 = threading.Thread(target=lambda: scheduler.schedule_job("j2", batches_j2, mock_provider))
    t3 = threading.Thread(target=lambda: scheduler.schedule_job("j3", batches_j3, mock_provider))

    for t in (t1, t2, t3):
        t.start()

    first_call_hit.wait(timeout=2.0)
    # Doi cho ca 3 job deu co mat trong _job_order
    sync_event = threading.Event()
    for _ in range(100):
        with scheduler._lock:
            if len(scheduler._job_order) == 3:
                break
        sync_event.wait(0.01)
    all_registered.set()

    for t in (t1, t2, t3):
        t.join(timeout=3.0)

    scheduler.shutdown(timeout=1.0)

    # Khong the co truong hop j1 thuc thi ca 3 batch truoc khi j2 hoac j3 bat dau
    assert execution_order[0:3] != ["j1", "j1", "j1"]
    # Co su xen ke luan phien giua cac jobs
    assert "j2" in execution_order[:3] or "j3" in execution_order[:3]
    assert len(execution_order) == 9


# 3. Out-of-order ordered mapping
def test_out_of_order_completion_mapped_to_stable_keys() -> None:
    """Kiem tra batch 2 hoan thanh truoc batch 1 nhung ket qua tra ve dung ID."""
    scheduler = AIOCRScheduler(max_concurrency=2)
    b2_finished_event = threading.Event()
    completed_batches: list[list[BatchResult]] = []

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        first_id = items[0].id
        if first_id == "b1_item1":
            # Batch 1 cho Batch 2 hoan thanh truoc
            b2_finished_event.wait(timeout=2.0)
            return [_make_result(item.id, f"text_{item.id}") for item in items]
        if first_id == "b2_item1":
            res = [_make_result(item.id, f"text_{item.id}") for item in items]
            b2_finished_event.set()
            return res
        return [_make_result(item.id) for item in items]

    def on_complete_batch(res: list[BatchResult]) -> None:
        completed_batches.append(res)

    batches = [
        [_make_item("b1_item1"), _make_item("b1_item2")],
        [_make_item("b2_item1"), _make_item("b2_item2")],
    ]

    results = scheduler.schedule_job(
        "job_ooo",
        batches,
        mock_provider,
        on_batch_complete=on_complete_batch,
    )

    scheduler.shutdown(timeout=1.0)

    # Batch 2 da hoan thanh truoc batch 1 trong callback checkpoint
    assert completed_batches[0][0].id == "b2_item1"
    assert completed_batches[1][0].id == "b1_item1"

    # Ket qua tra ve van day du va map chinh xac theo ID
    assert set(results.keys()) == {"b1_item1", "b1_item2", "b2_item1", "b2_item2"}
    assert results["b1_item1"].text == "text_b1_item1"
    assert results["b2_item2"].text == "text_b2_item2"


# 4. 429/5xx/timeout retry and adaptive lower/recover
def test_retry_transient_and_adaptive_concurrency_adjustment() -> None:
    """Kiem tra thu lai loi 429/5xx/timeout va dieu tiet giam/phuc hoi concurrency."""
    call_attempts = 0
    recorded_sleeps: list[float] = []

    clock = FakeClock()
    scheduler = AIOCRScheduler(
        max_concurrency=4,
        clock_fn=clock.now,
        sleep_fn=recorded_sleeps.append,
    )

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        nonlocal call_attempts
        call_attempts += 1
        if call_attempts == 1:
            raise ai_gateway.AIGatewayRateLimitError("Rate limit", retry_after=2.5)
        if call_attempts == 2:
            raise ai_gateway.AIGatewayServerError("500 Internal Error")
        if call_attempts == 3:
            raise ai_gateway.AIGatewayTimeoutError("Timeout")
        return [_make_result(item.id) for item in items]

    batches = [[_make_item("item_retry")]]
    results = scheduler.schedule_job("job_retries", batches, mock_provider, max_retries=3)

    assert len(results) == 1
    assert call_attempts == 4
    # Co 3 lan sleep tuong ung 3 lan retry
    assert len(recorded_sleeps) == 3
    assert recorded_sleeps[0] == 2.5  # Ton trong Retry-After

    # Kiem tra adaptive concurrency manager
    mgr = scheduler._get_endpoint_manager("default")
    # Ban dau la 4, sau 3 loi giam xuong 1 (min)
    assert mgr.current_limit == 1

    # Cho them cac cu goi thanh cong de phuc hoi
    mgr.record_success()
    mgr.record_success()  # Dat recovery_threshold = 2 -> tang len 2
    assert mgr.current_limit == 2

    mgr.record_success()
    mgr.record_success()  # Tang len 3
    assert mgr.current_limit == 3

    mgr.record_success()
    mgr.record_success()  # Tang len 4 (max)
    assert mgr.current_limit == 4

    scheduler.shutdown(timeout=1.0)


# 5. Auth no retry
def test_auth_error_no_retry_and_marks_endpoint() -> None:
    """Loi 401/403 khong duoc phep thu lai va danh dau endpoint that bai."""
    call_count = 0
    recorded_sleeps: list[float] = []

    scheduler = AIOCRScheduler(
        max_concurrency=2,
        sleep_fn=recorded_sleeps.append,
    )

    def mock_auth_fail(items: Sequence[BatchItem]) -> list[BatchResult]:
        nonlocal call_count
        call_count += 1
        raise ai_gateway.AIGatewayAuthError("Unauthorized 401", status_code=401)

    batches = [[_make_item("auth_item")]]
    with pytest.raises(ai_gateway.AIGatewayAuthError):
        scheduler.schedule_job(
            "job_auth",
            batches,
            mock_auth_fail,
            max_retries=3,
            fail_fast=True,
            endpoint_key="ep_auth_test",
        )

    # Tuyet doi khong retry cu goi auth
    assert call_count == 1
    assert len(recorded_sleeps) == 0

    mgr = scheduler._get_endpoint_manager("ep_auth_test")
    assert mgr.is_auth_failed() is True

    scheduler.shutdown(timeout=1.0)


# 6. Malformed retry only batch
def test_malformed_ocr_validation_error_retries_only_failed_batch() -> None:
    """Loi OCRValidationError chi thu lai duy nhat batch bi loi."""
    b1_calls = 0
    b2_calls = 0
    recorded_sleeps: list[float] = []

    scheduler = AIOCRScheduler(
        max_concurrency=2,
        sleep_fn=recorded_sleeps.append,
    )

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        nonlocal b1_calls, b2_calls
        first_id = items[0].id
        if first_id == "b1":
            b1_calls += 1
            if b1_calls == 1:
                raise OCRValidationError("Missing expected ID: b1")
            return [_make_result("b1")]
        if first_id == "b2":
            b2_calls += 1
            return [_make_result("b2")]
        return []

    batches = [[_make_item("b1")], [_make_item("b2")]]
    results = scheduler.schedule_job("job_malformed", batches, mock_provider, max_retries=2)

    scheduler.shutdown(timeout=1.0)

    assert b1_calls == 2  # Batch 1 bi loi duoc thu lai
    assert b2_calls == 1  # Batch 2 khong bi anh huong
    assert "b1" in results
    assert "b2" in results


# 7. Cancellation isolation
def test_cancellation_isolation_does_not_affect_other_jobs() -> None:
    """Huy job 1 khong lam anh huong toi job 2 dang chay dong thoi."""
    scheduler = AIOCRScheduler(max_concurrency=1)
    token_j1 = CancelToken()
    token_j2 = CancelToken()

    j1_in_flight_event = threading.Event()
    j1_can_proceed = threading.Event()

    j1_completed_items: list[str] = []
    j2_completed_items: list[str] = []
    thread_errors: list[Exception] = []

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        first_id = items[0].id
        if first_id.startswith("j1"):
            j1_in_flight_event.set()
            j1_can_proceed.wait(timeout=2.0)
            return [_make_result(item.id) for item in items]
        if first_id.startswith("j2"):
            return [_make_result(item.id) for item in items]
        return []

    batches_j1 = [[_make_item("j1_1")], [_make_item("j1_2")], [_make_item("j1_3")]]
    batches_j2 = [[_make_item("j2_1")], [_make_item("j2_2")]]

    res_j1: dict[str, BatchResult] = {}
    res_j2: dict[str, BatchResult] = {}

    def run_j1() -> None:
        nonlocal res_j1
        try:
            res_j1 = scheduler.schedule_job(
                "job1",
                batches_j1,
                mock_provider,
                token=token_j1,
                on_batch_complete=lambda b: j1_completed_items.extend(r.id for r in b),
            )
        except Exception as exc:
            thread_errors.append(exc)

    def run_j2() -> None:
        nonlocal res_j2
        try:
            res_j2 = scheduler.schedule_job(
                "job2",
                batches_j2,
                mock_provider,
                token=token_j2,
                on_batch_complete=lambda b: j2_completed_items.extend(r.id for r in b),
            )
        except Exception as exc:
            thread_errors.append(exc)

    t1 = threading.Thread(target=run_j1)
    t2 = threading.Thread(target=run_j2)

    t1.start()
    t2.start()

    # Doi batch 1 cua Job 1 bat dau roi huy Job 1
    j1_in_flight_event.wait(timeout=2.0)
    token_j1.cancel()
    # Cho phep in-flight batch cua Job 1 tiep tuc hoan tat
    j1_can_proceed.set()

    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    scheduler.shutdown(timeout=1.0)

    assert not thread_errors
    assert "j1_1" in j1_completed_items
    assert "j1_2" not in j1_completed_items
    assert "j1_3" not in j1_completed_items
    assert "j1_1" in res_j1
    assert "j1_2" not in res_j1
    assert "j1_3" not in res_j1
    assert set(j2_completed_items) == {"j2_1", "j2_2"}
    assert set(res_j2.keys()) == {"j2_1", "j2_2"}
    assert set(j2_completed_items) == {"j2_1", "j2_2"}  # Job 2 khong he bi anh huong


# 8. Bounded queue & backpressure
def test_bounded_queue_and_backpressure() -> None:
    """Kiem tra ap luc nguoc (backpressure) giu hang doi khong vuot qua max_pending."""
    max_pending = 2
    scheduler = AIOCRScheduler(max_concurrency=1)

    batch_started_events = [threading.Event() for _ in range(5)]
    batch_finish_events = [threading.Event() for _ in range(5)]
    current_idx = 0
    lock = threading.Lock()

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        nonlocal current_idx
        with lock:
            idx = current_idx
            current_idx += 1
        batch_started_events[idx].set()
        batch_finish_events[idx].wait(timeout=2.0)
        return [_make_result(item.id) for item in items]

    yielded_count = 0

    def batch_generator() -> Generator[list[BatchItem], None, None]:
        nonlocal yielded_count
        for i in range(5):
            yielded_count += 1
            yield [_make_item(f"item_{i}")]

    def run_feeder() -> None:
        scheduler.schedule_job(
            "job_bounded",
            batch_generator(),
            mock_provider,
            max_pending=max_pending,
        )

    t = threading.Thread(target=run_feeder)
    t.start()

    # Cho batch 0 bat dau
    batch_started_events[0].wait(timeout=2.0)
    # Luc nay: batch 0 in-flight, max_pending (2) nam trong pending_queue
    # Tong yielded <= 1 (in-flight) + 2 (pending) = 3
    assert yielded_count <= 3

    # Giai phong batch 0
    batch_finish_events[0].set()
    batch_started_events[1].wait(timeout=2.0)
    # Giai phong tiep cac batch con lai
    for i in range(1, 5):
        batch_finish_events[i].set()

    t.join(timeout=3.0)
    scheduler.shutdown(timeout=1.0)
    assert yielded_count == 5


# 9. Scheduler shutdown
def test_scheduler_shutdown_terminates_workers() -> None:
    """Kiem tra scheduler shutdown ket thuc tat ca worker thread sach se."""
    scheduler = AIOCRScheduler(max_concurrency=3)
    workers = list(scheduler._workers)
    assert len(workers) == 3
    assert all(w.is_alive() for w in workers)

    scheduler.shutdown(timeout=2.0)

    assert scheduler.is_shutdown is True
    assert all(not w.is_alive() for w in workers)

    # Goi tiep se bao loi
    with pytest.raises(RuntimeError):
        scheduler.schedule_job("j", [[_make_item("x")]], lambda _items: [])


# 10. Thread-safe metrics snapshot
def test_thread_safe_metrics_snapshot() -> None:
    """Kiem tra ban chup thong ke day du cac chi so hoat dong."""
    scheduler = AIOCRScheduler(max_concurrency=2)
    scheduler.record_cache_hit(3)
    scheduler.record_cache_miss(1)
    scheduler.record_payload_bytes(1024)

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        return [_make_result(item.id) for item in items]

    batches = [
        [_make_item("m1"), _make_item("m2")],
        [_make_item("m3"), _make_item("m4")],
    ]
    scheduler.schedule_job("job_metrics", batches, mock_provider)

    snap = scheduler.get_metrics()
    assert snap.active == 0
    assert snap.queue == 0
    assert snap.completed == 2
    assert snap.failed == 0
    assert snap.avg_batch_size == 2.0
    assert snap.cache_hits == 3
    assert snap.cache_misses == 1
    assert snap.payload_bytes >= 1024
    assert snap.elapsed >= 0.0

    scheduler.reset_metrics()
    snap2 = scheduler.get_metrics()
    assert snap2.completed == 0
    assert snap2.cache_hits == 0

    scheduler.shutdown(timeout=1.0)


# 11. No secret logs
def test_no_secret_logs_sanitizes_tokens() -> None:
    """Kiem tra chuoi loi va log duoc khu toan bo token/mat khau truoc khi ghi."""
    dirty = "Error with Authorization: Bearer sk-secret-123456789 and x-api-key: my_pass_123"
    clean = _sanitize_log_text(dirty)
    assert "sk-secret-123456789" not in clean
    assert "my_pass_123" not in clean
    assert "***" in clean


# 12. Reconfiguration and lifecycle without multiplying pools
def test_reconfiguration_and_lifecycle_without_multiplying_pools() -> None:
    """Kiem tra lazy singleton, reconfigure an toan ma khong tao them thread pools."""
    s1 = get_global_scheduler(max_concurrency=2)
    assert s1.max_concurrency == 2
    initial_workers = len(s1._workers)
    assert initial_workers == 2

    # Reconfigure len 4
    s2 = get_global_scheduler(max_concurrency=4)
    assert s2 is s1  # Cung mot doi tuong singleton
    assert s2.max_concurrency == 4
    assert len(s2._workers) == 4

    # Reconfigure xuong 2 khong tao them thread pool
    s2.reconfigure(2)
    assert s2.max_concurrency == 2
    # So luong thread van duoc quan ly gon gang
    assert len(s2._workers) == 4

    # Kiem tra setup_app_lifecycle
    class FakeSignal:
        def __init__(self) -> None:
            self.slot: Any = None

        def connect(self, fn: Any) -> None:
            self.slot = fn

        def emit(self) -> None:
            if self.slot:
                self.slot()

    class FakeQtApp:
        def __init__(self) -> None:
            self.aboutToQuit = FakeSignal()

    fake_app = FakeQtApp()
    setup_app_lifecycle(fake_app)
    assert fake_app.aboutToQuit.slot is shutdown_global_scheduler

    fake_app.aboutToQuit.emit()
    assert s1.is_shutdown is True


def test_is_retryable_error_classification() -> None:
    """Kiem tra phan loai loi retryable theo dung quy tac."""
    assert is_retryable_error(ai_gateway.AIGatewayRateLimitError("429")) is True
    assert is_retryable_error(ai_gateway.AIGatewayServerError("500")) is True
    assert is_retryable_error(ai_gateway.AIGatewayTimeoutError()) is True
    assert is_retryable_error(OCRValidationError("Malformed JSON")) is True

    assert is_retryable_error(ai_gateway.AIGatewayAuthError("401")) is False
    assert is_retryable_error(ai_gateway.AIGatewayInvalidRequestError("400")) is False
    assert is_retryable_error(ai_gateway.AIGatewayPayloadTooLargeError("413")) is False


def test_endpoint_shared_across_different_projects_adaptive_limit() -> None:
    """Kiem tra endpoint dung chung giua cac du an chia se concurrency va ap luc."""
    shared_ep = "https://api.openai.com/v1"
    clock = FakeClock()
    recorded_sleeps: list[float] = []

    scheduler = AIOCRScheduler(
        max_concurrency=4,
        clock_fn=clock.now,
        sleep_fn=recorded_sleeps.append,
    )

    # Ca project A va project B dung chung endpoint
    mgr = scheduler._get_endpoint_manager(shared_ep)
    assert mgr.current_limit == 4

    # Project A gap loi 429
    mgr.record_failure(ai_gateway.AIGatewayRateLimitError("429 Too Many Requests"))
    assert mgr.current_limit == 3

    # Project B cung dung endpoint nay va nhin thay concurrency da bi giam xuong 3
    mgr_b = scheduler._get_endpoint_manager(shared_ep)
    assert mgr_b is mgr
    assert mgr_b.current_limit == 3

    # Project B gap tiep 500
    mgr_b.record_failure(ai_gateway.AIGatewayServerError("500 Server Error"))
    assert mgr.current_limit == 2

    # Hai cu goi thanh cong tu ca 2 project de phuc hoi
    mgr.record_success()
    mgr_b.record_success()
    assert mgr.current_limit == 3

    mgr.record_success()
    mgr_b.record_success()
    assert mgr.current_limit == 4

    scheduler.shutdown(timeout=1.0)


def test_cancellation_after_completion_retains_completed_results() -> None:
    """Kiem tra viec huy sau khi da hoan thanh khong lam mat ket qua da co."""
    scheduler = AIOCRScheduler(max_concurrency=2)
    token = CancelToken()

    def mock_provider(items: Sequence[BatchItem]) -> list[BatchResult]:
        return [_make_result(item.id, f"text_{item.id}") for item in items]

    batches = [[_make_item("item_done1")], [_make_item("item_done2")]]
    results = scheduler.schedule_job("job_done", batches, mock_provider, token=token)

    assert len(results) == 2
    assert "item_done1" in results
    assert "item_done2" in results

    # Huy token sau khi batch da hoan tat
    token.cancel()

    # Ket qua tra ve ban dau van nguyen ven, khong bi xoa boi cancel sau do
    assert len(results) == 2

    scheduler.shutdown(timeout=1.0)
