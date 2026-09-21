"""Bo lap lich xu ly OCR AI toan cuc (Process-global AI OCR Scheduler).

Invariants & Kien truc:
1. Singleton toan cuc tien trinh (process-global singleton) duoc chia se boi tat ca
   cong viec OCR cua cac du an (projects):
   - Tong so HTTP request dang hoat dong dong thoi (active requests) khong bao gio vuot qua
     gioi han toan cuc duoc cau hinh (max_concurrency).
   - Dieu phoi cong bang (fair round-robin) giua cac job_id / project de mot video dai khong
     the chiem dung hang doi va gay doi kiet (starvation) cho cac video ngan khac.
   - Hang doi co gioi han va ap luc nguoc (bounded pending queue & backpressure): viec submit
     batch se block/cho hop tac neu hang doi vuot qua nguong cho phep, tranh tao vo han future
     hay gay tran RAM boi payload hinh anh.

2. API:
   - Nhan vao: job_id, batches (co the la list hoac generator/iterator), provider callable,
     CancelToken/should_cancel, callbacks (checkpoint, error, progress).
   - Tra ve ket qua: dict[str, BatchResult] voi key la item ID, doc lap voi thu tu hoan thanh
     (out-of-order execution mapped to deterministic ID keys).

3. Chinh sach thu lai (Retry policy):
   - Chi thu lai loi tam thoi (is_transient == True: timeout, connection, 429, 5xx) va
     OCRValidationError (batch ket qua bi loi dinh dang / thieu ID tu LLM).
   - Loi xac thuc (auth 401/403), loi model/request/format (400, 404) dung job/global config
     ngay lap tuc, khong thu lai (khong hammer).
   - Loi split-required (413 / context length) duoc uy quyen cho provider xu ly theo co che
     chia doi dong, scheduler khong tu dong chia lai.
   - Gioi han so lan retry (bounded max retries), exponential backoff + jitter co the tiem
     (inject) clock va random de test xac dinh (deterministic tests), ton trong Retry-After,
     thoi gian cho retry co the bi huy interruptible boi CancelToken.

4. Ap luc thich ung (Adaptive pressure):
   - Khi gap loi 429 hoac 5xx lien tiep, tu dong giam muc concurrency hieu dung cho endpoint/key
     tuong ung; khi cac cu goi thanh cong lien tiep, dan dan phuc hoi ve muc gioi han da cau hinh;
     tuyet doi khong tiep tuc ban pha (no hammer).

5. Giai thich Timeout (Network Timeout Note):
   - urllib cua Python chi ho tro mot tham so timeout duy nhat o cap socket,
     khong tach biet connect/read/total timeout:
     + Provider timeout (vi du AIOCRProvider.timeout): chi phoi timeout socket HTTP request.
     + Scheduler total_timeout_per_batch (batch deadline): chi phoi thoi han tong cong cho
       toan bo batch tinh ca cac lan retry. Neu qua thoi han nay, scheduler se huy batch.

6. Cach ly loi (Error isolation):
   - Loi vinh vien cua mot batch chi anh huong toi batch/job do, khong lam hong cac job khac.
   - Loi xac thuc auth co the danh dau endpoint that bai de cac job sau khong hammer, khong co
     co che fallback cuc bo gia mao (no local fallback).

7. Huy bo (Cancellation) & Shutdown:
   - Huy bo se lap tuc dung nap batch moi, don sach batch dang cho (pending), cho phep ket qua
     in-flight hoan tat va kich hoat checkpoint callback de luu tien do da xong.
   - Scheduler shutdown hoan tat trong thoi gian huu han (finite shutdown).

8. Thong ke (Thread-safe Metrics):
   - Cung cap snapshot: active, queue, completed, failed, retries, avg latency, last latency,
     payload bytes, avg batch size, cache counters externally recordable, elapsed.

9. Vong doi (Lifecycle):
   - Singleton lazy qua get_global_scheduler() va shutdown_global_scheduler(), tich hop
     aboutToQuit cua Qt app, reconfigure an toan ma khong tao them worker pool moi.
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
import urllib.error
from collections import deque
from collections.abc import Callable, Iterable, Sequence, Sized
from dataclasses import dataclass, field
from typing import Any

from ..providers.ocr_ai_provider import (
    BatchItem,
    BatchResult,
    OCRValidationError,
    estimate_batch_payload_size,
)
from ..services import ai_gateway
from .ai_ocr_metrics import SchedulerMetricsCollector, SchedulerMetricsSnapshot
from .ffmpeg import CancelledError, CancelToken

__all__ = [
    "AIOCRScheduler",
    "BatchTask",
    "CancelToken",
    "CancelledError",
    "EndpointPressureManager",
    "JobState",
    "compute_backoff",
    "get_global_scheduler",
    "is_cancelled",
    "is_retryable_error",
    "schedule_ocr_job",
    "shutdown_global_scheduler",
]


def _sanitize_log_text(text: str) -> str:
    """Loai bo token, mat khau, authorization truoc khi ghi log."""
    return ai_gateway._sanitize_error_text(str(text))


def is_cancelled(
    token: Any = None,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """Kiem tra xem tac vu da bi huy bo boi token hoac callback hay chua."""
    if should_cancel is not None:
        try:
            if should_cancel():
                return True
        except Exception:
            pass
    if token is not None:
        val = getattr(token, "cancelled", None)
        if callable(val):
            try:
                if val():
                    return True
            except Exception:
                pass
        elif bool(val):
            return True

        val_is = getattr(token, "is_cancelled", None)
        if callable(val_is):
            try:
                if val_is():
                    return True
            except Exception:
                pass
        elif bool(val_is):
            return True

        event = getattr(token, "_event", None)
        if isinstance(event, threading.Event) and event.is_set():
            return True
    return False


def is_retryable_error(exc: Exception) -> bool:
    """Xac dinh xem loi co phai la loi co the thu lai (retryable) o cap scheduler khong.

    Thu lai:
    1. Transient errors (timeout, connection, 429 rate limit, 5xx server).
    2. OCRValidationError (malformed batch output, thieu/thua ID tu LLM).

    Khong thu lai:
    - Auth errors (401, 403).
    - Invalid request/model/format (400, 404).
    - Split-required errors (413 / context length) duoc xu ly dong bo boi provider.
    """
    # Khong thu lai loi xac thuc
    if isinstance(exc, ai_gateway.AIGatewayAuthError):
        return False

    # Khong thu lai loi cu phap / request / model khong hop le
    if isinstance(exc, ai_gateway.AIGatewayInvalidRequestError):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code in (400, 401, 403, 404):
        return False

    # Split-required can duoc chia nho boi provider, khong retry nguyen me
    if getattr(exc, "is_split_required", False) or isinstance(
        exc, ai_gateway.AIGatewayPayloadTooLargeError
    ):
        return False
    if status_code == 413:
        return False

    # OCRValidationError: LLM tra ve ket qua loi dinh dang hoac thieu ID -> retry me nay
    if isinstance(exc, OCRValidationError):
        return True

    # Transient: timeout, connection, 429, 5xx
    if getattr(exc, "is_transient", False):
        return True

    if isinstance(
        exc,
        (
            ai_gateway.AIGatewayRateLimitError,
            ai_gateway.AIGatewayServerError,
            ai_gateway.AIGatewayTimeoutError,
            ai_gateway.AIGatewayConnectionError,
            TimeoutError,
            urllib.error.URLError,
        ),
    ):
        return True

    # Provider nghiep vu co the boc loi Gateway (vi du TranslationError).
    # Van giu mot retry/backoff policy chung thay vi moi workload tu retry rieng.
    cause = exc.__cause__
    if isinstance(cause, Exception) and cause is not exc:
        return is_retryable_error(cause)

    return bool(status_code == 429 or (status_code is not None and 500 <= status_code <= 599))


def compute_backoff(
    attempt: int,
    *,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_after: float | None = None,
    jitter_range: tuple[float, float] = (0.0, 0.2),
    random_fn: Callable[[float, float], float] = random.uniform,
) -> float:
    """Tinh thoi gian cho exponential backoff voi jitter va Retry-After."""
    if retry_after is not None and retry_after > 0:
        return float(retry_after)

    delay = min(max_delay, base_delay * (2.0**attempt))
    jitter = random_fn(jitter_range[0], jitter_range[1])
    return max(0.0, delay + jitter)


class EndpointPressureManager:
    """Quan ly ap luc ket noi va concurrency thich ung cho mot endpoint / API key."""

    def __init__(
        self,
        max_concurrency: int = 4,
        min_concurrency: int = 1,
        recovery_threshold: int = 2,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.min_concurrency = min(min_concurrency, self.max_concurrency)
        self.current_limit = self.max_concurrency
        self.recovery_threshold = recovery_threshold
        self.clock_fn = clock_fn

        self._consecutive_successes = 0
        self._consecutive_errors = 0
        self._backoff_until: float = 0.0
        self._auth_failed = False
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Ghi nhan mot cu goi thanh cong de tung buoc phuc hoi concurrency."""
        with self._lock:
            self._consecutive_errors = 0
            if self.current_limit < self.max_concurrency:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.recovery_threshold:
                    self._consecutive_successes = 0
                    self.current_limit = min(self.max_concurrency, self.current_limit + 1)

    def record_failure(self, exc: Exception) -> None:
        """Ghi nhan loi de giam concurrency thich ung hoac danh dau auth failed."""
        root = exc
        while isinstance(root.__cause__, Exception) and root.__cause__ is not root:
            root = root.__cause__
        with self._lock:
            self._consecutive_successes = 0
            if isinstance(root, ai_gateway.AIGatewayAuthError) or getattr(
                root, "status_code", None
            ) in (401, 403):
                self._auth_failed = True
                return

            status_code = getattr(root, "status_code", None)
            is_transient = getattr(root, "is_transient", False) or (
                status_code in (429,) or (status_code is not None and 500 <= status_code <= 599)
            )

            if is_transient or isinstance(
                root, (ai_gateway.AIGatewayRateLimitError, ai_gateway.AIGatewayServerError)
            ):
                self._consecutive_errors += 1
                self.current_limit = max(self.min_concurrency, self.current_limit - 1)
                retry_after = getattr(root, "retry_after", None)
                if retry_after is not None and retry_after > 0:
                    self._backoff_until = max(self._backoff_until, self.clock_fn() + retry_after)
                elif status_code == 429:
                    self._backoff_until = max(self._backoff_until, self.clock_fn() + 0.5)

    def is_auth_failed(self) -> bool:
        """Kiem tra xem endpoint da bi khoa do loi xac thuc chua."""
        with self._lock:
            return self._auth_failed

    def get_backoff_wait(self) -> float:
        """Tra ve so giay con phai cho truoc khi co the gui request toi endpoint nay."""
        with self._lock:
            rem = self._backoff_until - self.clock_fn()
            return max(0.0, rem)

    def reconfigure(self, new_max: int) -> None:
        """Cap nhat gioi han toi da ma khong reset trang thai loi."""
        with self._lock:
            self.max_concurrency = max(1, new_max)
            self.min_concurrency = min(self.min_concurrency, self.max_concurrency)
            self.current_limit = min(self.current_limit, self.max_concurrency)


@dataclass(eq=False)
class BatchTask:
    """Doi tuong cong viec cua mot batch can xu ly."""

    task_id: str
    job_id: str
    items: Sequence[BatchItem]
    provider: Callable[[Sequence[BatchItem]], list[BatchResult]]
    on_complete: Callable[[list[BatchResult]], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    max_retries: int = 3
    total_deadline: float | None = None
    endpoint_key: str = ""


@dataclass
class JobState:
    """Trang thai dieu phoi cua mot cong viec OCR (job_id)."""

    job_id: str
    max_pending: int
    token: Any = None
    should_cancel: Callable[[], bool] | None = None
    fail_fast: bool = False
    endpoint_key: str = ""
    pending_queue: deque[BatchTask] = field(default_factory=deque)
    in_flight: set[BatchTask] = field(default_factory=set)
    completed_results: dict[str, BatchResult] = field(default_factory=dict)
    failed: bool = False
    last_error: Exception | None = None
    all_enqueued: bool = False
    failed_tasks: list[BatchTask] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)


class AIOCRScheduler:
    """Bo lap lich OCR AI toan cuc, quan ly luong request HTTP va dieu phoi round-robin."""

    def __init__(
        self,
        max_concurrency: int = 4,
        *,
        base_retry_delay: float = 0.5,
        max_retry_delay: float = 30.0,
        clock_fn: Callable[[], float] = time.monotonic,
        random_fn: Callable[[float, float], float] = random.uniform,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._max_concurrency = max(1, max_concurrency)
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._clock_fn = clock_fn
        self._random_fn = random_fn
        self._sleep_fn = sleep_fn

        self._lock = threading.Lock()
        self._worker_condition = threading.Condition(self._lock)
        self._active_requests: int = 0
        self._active_per_endpoint: dict[str, int] = {}
        self._shutdown: bool = False

        self._registered_jobs: dict[str, JobState] = {}
        self._job_order: list[str] = []
        self._rr_idx: int = 0

        self._endpoint_managers: dict[str, EndpointPressureManager] = {}
        self._endpoint_managers_lock = threading.Lock()
        self._metrics = SchedulerMetricsCollector(clock_fn=self._clock_fn)

        self._workers: list[threading.Thread] = []
        self._ensure_workers()

    @property
    def max_concurrency(self) -> int:
        with self._lock:
            return self._max_concurrency

    @property
    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    @property
    def metrics(self) -> SchedulerMetricsSnapshot:
        return self.get_metrics()

    def get_metrics(self) -> SchedulerMetricsSnapshot:
        """Tra ve ban chup thong ke tai thoi diem hien tai."""
        with self._lock:
            active = self._active_requests
            q_size = sum(len(j.pending_queue) for j in self._registered_jobs.values())
        return self._metrics.snapshot(queue_size=q_size, active_count=active)

    def record_cache_hit(self, count: int = 1) -> None:
        """Ghi nhan so lan trung cache OCR tu ben ngoai."""
        self._metrics.record_cache_hit(count)

    def record_cache_miss(self, count: int = 1) -> None:
        """Ghi nhan so lan hut cache OCR tu ben ngoai."""
        self._metrics.record_cache_miss(count)

    def record_payload_bytes(self, byte_count: int) -> None:
        """Ghi nhan dung luong payload tu ben ngoai."""
        self._metrics.record_payload_bytes(byte_count)

    def reset_metrics(self) -> None:
        """Dat lai bo dem thong ke."""
        self._metrics.reset()

    def _get_endpoint_manager(self, endpoint_key: str) -> EndpointPressureManager:
        """Lay hoac tao moi bo quan ly ap luc cho endpoint."""
        key = endpoint_key.strip() or "default"
        with self._endpoint_managers_lock:
            if key not in self._endpoint_managers:
                self._endpoint_managers[key] = EndpointPressureManager(
                    max_concurrency=self._max_concurrency,
                    clock_fn=self._clock_fn,
                )
            return self._endpoint_managers[key]

    def _ensure_workers(self) -> None:
        """Khoi chay cac worker thread cho toi khi du so luong max_concurrency."""
        with self._lock:
            if self._shutdown:
                return
            needed = self._max_concurrency - len(self._workers)
            for _ in range(needed):
                idx = len(self._workers) + 1
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"AIOCRWorker-{idx}",
                    daemon=True,
                )
                self._workers.append(t)
                t.start()

    def reconfigure(self, max_concurrency: int) -> None:
        """Cap nhat gioi han concurrency an toan ma khong nhan ban worker pool."""
        with self._lock:
            if self._shutdown:
                return
            new_max = max(1, int(max_concurrency))
            self._max_concurrency = new_max
            with self._endpoint_managers_lock:
                for mgr in self._endpoint_managers.values():
                    mgr.reconfigure(new_max)
            self._worker_condition.notify_all()

        self._ensure_workers()

    def _interruptible_wait(
        self,
        delay: float,
        *,
        token: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> bool:
        """Cho doi delay mot cach co the bi ngat boi CancelToken hoac scheduler shutdown."""
        if delay <= 0:
            return is_cancelled(token, should_cancel) or self._shutdown

        if self._sleep_fn is not None:
            self._sleep_fn(delay)
            return is_cancelled(token, should_cancel) or self._shutdown

        # Su dung event cua CancelToken neu co de khong ton CPU
        cancel_evt = getattr(token, "_event", None)
        if isinstance(cancel_evt, threading.Event):
            signaled = cancel_evt.wait(timeout=delay)
            return signaled or is_cancelled(token, should_cancel) or self._shutdown

        # Fallback polling ngan neu khong co Event
        end_time = self._clock_fn() + delay
        while self._clock_fn() < end_time:
            if is_cancelled(token, should_cancel) or self._shutdown:
                return True
            slice_wait = min(0.05, end_time - self._clock_fn())
            if slice_wait > 0:
                time.sleep(slice_wait)
        return is_cancelled(token, should_cancel) or self._shutdown

    def _pop_next_task_round_robin(self) -> BatchTask | None:
        """Chon batch tiep theo theo thuat toan round-robin cong bang giua cac job."""
        if self._active_requests >= self._max_concurrency:
            return None

        if not self._job_order:
            return None

        n = len(self._job_order)
        for step in range(n):
            idx = (self._rr_idx + step) % n
            cand_job_id = self._job_order[idx]
            job_state = self._registered_jobs.get(cand_job_id)
            if job_state is None:
                continue

            # Neu job da bi huy bo, giai phong toan bo pending queue
            if is_cancelled(job_state.token, job_state.should_cancel):
                if job_state.pending_queue:
                    cleared = len(job_state.pending_queue)
                    job_state.pending_queue.clear()
                    self._metrics.record_queue_change(-cleared)
                    job_state.condition.notify_all()
                continue

            if job_state.failed and job_state.fail_fast:
                continue

            if not job_state.pending_queue:
                continue

            task_cand = job_state.pending_queue[0]
            endpoint_mgr = self._get_endpoint_manager(task_cand.endpoint_key)

            # Neu endpoint da bi loi auth, tu choi ngay de khong hammer
            if endpoint_mgr.is_auth_failed():
                job_state.pending_queue.popleft()
                self._metrics.record_queue_change(-1)
                exc = ai_gateway.AIGatewayAuthError(
                    f"Endpoint '{task_cand.endpoint_key}' đã bị từ chối xác thực (401/403)."
                )
                job_state.failed = True
                job_state.last_error = exc
                if task_cand.on_error:
                    task_cand.on_error(exc)
                job_state.condition.notify_all()
                continue

            # Kiem tra gioi han ap luc thich ung tren endpoint
            active_on_endpoint = self._active_per_endpoint.get(task_cand.endpoint_key, 0)
            if active_on_endpoint >= endpoint_mgr.current_limit:
                continue

            # Kiem tra backoff cooldown tren endpoint
            wait_sec = endpoint_mgr.get_backoff_wait()
            if wait_sec > 0.0:
                continue

            # Lay task hop le
            task = job_state.pending_queue.popleft()
            self._metrics.record_queue_change(-1)
            job_state.in_flight.add(task)
            self._active_requests += 1
            self._active_per_endpoint[task.endpoint_key] = active_on_endpoint + 1
            self._metrics.record_active_change(1)

            # Chuyen con tro round-robin sang job tiep theo
            self._rr_idx = (idx + 1) % n
            job_state.condition.notify_all()
            return task

        return None

    def _worker_loop(self) -> None:
        """Vong lap chinh cua worker thread lay va xu ly batch."""
        while True:
            task: BatchTask | None = None
            with self._lock:
                while not self._shutdown:
                    task = self._pop_next_task_round_robin()
                    if task is not None:
                        break
                    self._worker_condition.wait(timeout=0.1)

                if self._shutdown and task is None:
                    break

            if task is None:
                continue

            self._execute_task(task)

    def _execute_task(self, task: BatchTask) -> None:
        """Thuc thi mot batch voi chinh sach retry, deadline va metric."""
        job_state: JobState | None = None
        with self._lock:
            job_state = self._registered_jobs.get(task.job_id)

        token = job_state.token if job_state else None
        should_cancel = job_state.should_cancel if job_state else None
        endpoint_mgr = self._get_endpoint_manager(task.endpoint_key)

        start_time = self._clock_fn()
        deadline = (
            start_time + task.total_deadline
            if task.total_deadline is not None and task.total_deadline > 0
            else None
        )

        results: list[BatchResult] | None = None
        last_exc: Exception | None = None
        success = False

        payload_size = 0
        with contextlib.suppress(Exception):
            payload_size = estimate_batch_payload_size(task.items)

        for attempt in range(task.max_retries + 1):
            if is_cancelled(token, should_cancel) or self._shutdown:
                last_exc = CancelledError(f"Task '{task.task_id}' bị hủy.")
                break

            if deadline is not None and self._clock_fn() >= deadline:
                last_exc = ai_gateway.AIGatewayTimeoutError(
                    f"Quá thời hạn deadline xử lý batch ({task.total_deadline}s)."
                )
                break

            call_start = self._clock_fn()
            try:
                results = task.provider(task.items)
                call_latency = self._clock_fn() - call_start
                success = True
                endpoint_mgr.record_success()
                self._metrics.record_batch_completion(
                    batch_size=len(task.items),
                    latency=call_latency,
                    payload_bytes=payload_size,
                )
                break
            except Exception as exc:
                last_exc = exc
                endpoint_mgr.record_failure(exc)

                if not is_retryable_error(exc):
                    break

                if attempt >= task.max_retries:
                    break

                if deadline is not None and self._clock_fn() >= deadline:
                    last_exc = ai_gateway.AIGatewayTimeoutError(
                        f"Quá thời hạn deadline xử lý batch ({task.total_deadline}s)."
                    )
                    break

                self._metrics.record_retry()

                retry_source = exc
                while (
                    isinstance(retry_source.__cause__, Exception)
                    and retry_source.__cause__ is not retry_source
                ):
                    retry_source = retry_source.__cause__
                retry_after = getattr(retry_source, "retry_after", None)
                delay = compute_backoff(
                    attempt,
                    base_delay=self._base_retry_delay,
                    max_delay=self._max_retry_delay,
                    retry_after=retry_after,
                    random_fn=self._random_fn,
                )

                interrupted = self._interruptible_wait(
                    delay, token=token, should_cancel=should_cancel
                )
                if interrupted or is_cancelled(token, should_cancel) or self._shutdown:
                    last_exc = CancelledError(
                        f"Task '{task.task_id}' bị hủy trong thời gian chờ retry."
                    )
                    break

        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            curr_ep_active = self._active_per_endpoint.get(task.endpoint_key, 1)
            self._active_per_endpoint[task.endpoint_key] = max(0, curr_ep_active - 1)
            self._metrics.record_active_change(-1)

            if job_state is not None:
                if success and results is not None:
                    for r in results:
                        job_state.completed_results[r.id] = r
                else:
                    self._metrics.record_batch_failure()
                    job_state.failed = True
                    job_state.last_error = last_exc
                    job_state.failed_tasks.append(task)

            self._worker_condition.notify_all()

        # Goi callbacks ngoai lock de tranh deadlock
        if success and results is not None:
            if task.on_complete is not None:
                with contextlib.suppress(Exception):
                    task.on_complete(results)
        elif task.on_error is not None and last_exc is not None:
            with contextlib.suppress(Exception):
                task.on_error(last_exc)

        with self._lock:
            if job_state is not None:
                job_state.in_flight.discard(task)
                job_state.condition.notify_all()

    def schedule_job(
        self,
        job_id: str,
        batches: Iterable[Sequence[BatchItem]] | Sequence[Sequence[BatchItem]],
        provider: Callable[[Sequence[BatchItem]], list[BatchResult]],
        *,
        token: Any = None,
        cancel_token: Any = None,
        should_cancel: Callable[[], bool] | None = None,
        on_batch_complete: Callable[[list[BatchResult]], None] | None = None,
        on_batch_error: Callable[[Exception], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        max_retries: int = 3,
        total_timeout_per_batch: float | None = None,
        max_pending: int = 4,
        endpoint_key: str = "",
        fail_fast: bool = False,
        raise_on_cancel: bool = False,
    ) -> dict[str, BatchResult]:
        """Dieu phoi va thuc thi toan bo cac batch cua mot job OCR.

        Ho tro ap luc nguoc (backpressure) giu hang doi khong vuot qua max_pending,
        cho ket qua theo ID doc lap voi thu tu hoan thanh thuc te.
        """
        effective_token = cancel_token if cancel_token is not None else token
        effective_max_pending = max(1, int(max_pending))

        with self._lock:
            if self._shutdown:
                raise RuntimeError("Scheduler đã shutdown.")

            job_condition = threading.Condition(self._lock)
            job_state = JobState(
                job_id=job_id,
                max_pending=effective_max_pending,
                token=effective_token,
                should_cancel=should_cancel,
                fail_fast=fail_fast,
                endpoint_key=endpoint_key,
                condition=job_condition,
            )
            self._registered_jobs[job_id] = job_state
            if job_id not in self._job_order:
                self._job_order.append(job_id)

        total_submitted = 0
        known_total = len(batches) if isinstance(batches, Sized) else 0
        batch_iter = iter(batches)

        # Nap cac batch voi co che ap luc nguoc (backpressure)
        while True:
            with self._lock:
                while len(job_state.pending_queue) >= effective_max_pending:
                    if (
                        is_cancelled(effective_token, should_cancel)
                        or (job_state.failed and fail_fast)
                        or self._shutdown
                    ):
                        break
                    job_condition.wait(timeout=0.05)

                if (
                    is_cancelled(effective_token, should_cancel)
                    or (job_state.failed and fail_fast)
                    or self._shutdown
                ):
                    break

            try:
                batch = next(batch_iter)
            except StopIteration:
                break

            with self._lock:
                task = BatchTask(
                    task_id=f"{job_id}_{total_submitted}",
                    job_id=job_id,
                    items=batch,
                    provider=provider,
                    on_complete=on_batch_complete,
                    on_error=on_batch_error,
                    max_retries=max_retries,
                    total_deadline=total_timeout_per_batch,
                    endpoint_key=endpoint_key,
                )
                job_state.pending_queue.append(task)
                total_submitted += 1
                self._metrics.record_queue_change(1)
                self._worker_condition.notify()

            if on_progress is not None and known_total > 0:
                completed_so_far = len(job_state.completed_results)
                on_progress(completed_so_far, known_total)

        with self._lock:
            job_state.all_enqueued = True

            # Cho doi tat ca pending va in-flight hoan tat
            while (job_state.pending_queue or job_state.in_flight) and not self._shutdown:
                if is_cancelled(effective_token, should_cancel):
                    if job_state.pending_queue:
                        cleared = len(job_state.pending_queue)
                        job_state.pending_queue.clear()
                        self._metrics.record_queue_change(-cleared)
                    # In-flight result may finish
                    while job_state.in_flight and not self._shutdown:
                        job_condition.wait(timeout=0.05)
                    break

                if job_state.failed and fail_fast:
                    if job_state.pending_queue:
                        cleared = len(job_state.pending_queue)
                        job_state.pending_queue.clear()
                        self._metrics.record_queue_change(-cleared)
                    break

                job_condition.wait(timeout=0.05)

                if on_progress is not None and known_total > 0:
                    completed_so_far = len(job_state.completed_results)
                    on_progress(completed_so_far, known_total)

            # Don dep dang ky job
            if job_id in self._registered_jobs:
                del self._registered_jobs[job_id]
            if job_id in self._job_order:
                self._job_order.remove(job_id)
            if self._job_order:
                self._rr_idx = self._rr_idx % len(self._job_order)
            else:
                self._rr_idx = 0
            self._worker_condition.notify_all()

        if on_progress is not None and known_total > 0:
            on_progress(len(job_state.completed_results), known_total)

        if is_cancelled(effective_token, should_cancel) and raise_on_cancel:
            raise CancelledError(f"Job '{job_id}' bị hủy.")

        if job_state.failed and fail_fast and job_state.last_error is not None:
            raise job_state.last_error

        return dict(job_state.completed_results)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Dung toan bo worker thread va tat scheduler trong finite timeout."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._worker_condition.notify_all()
            for job in self._registered_jobs.values():
                job.condition.notify_all()

        for worker in self._workers:
            worker.join(timeout=max(0.1, timeout))
        self._workers.clear()


_global_scheduler: AIOCRScheduler | None = None
_global_scheduler_lock = threading.Lock()


def get_global_scheduler(
    max_concurrency: int | None = None,
    **kwargs: Any,
) -> AIOCRScheduler:
    """Lay singleton scheduler toan cuc cua tien trinh (lazy singleton).

    Neu scheduler da ton tai va max_concurrency duoc truyen vao, tai cau hinh
    an toan ma khong tao them worker pool moi.
    """
    global _global_scheduler
    with _global_scheduler_lock:
        if _global_scheduler is None or _global_scheduler.is_shutdown:
            concurrency = max_concurrency
            if concurrency is None:
                try:
                    from .settings import Settings

                    settings = Settings.load()
                    concurrency = getattr(settings, "ai_ocr_concurrency", 4)
                except Exception:
                    concurrency = 4
            _global_scheduler = AIOCRScheduler(max_concurrency=concurrency, **kwargs)
        else:
            if max_concurrency is not None:
                _global_scheduler.reconfigure(max_concurrency=max_concurrency)
        return _global_scheduler


def shutdown_global_scheduler(timeout: float = 5.0) -> None:
    """Tat scheduler toan cuc va doi tat ca worker ket thuc trong finite timeout."""
    global _global_scheduler
    with _global_scheduler_lock:
        if _global_scheduler is not None:
            _global_scheduler.shutdown(timeout=timeout)
            _global_scheduler = None


def schedule_ocr_job(
    job_id: str,
    batches: Iterable[Sequence[BatchItem]] | Sequence[Sequence[BatchItem]],
    provider: Callable[[Sequence[BatchItem]], list[BatchResult]],
    *,
    token: Any = None,
    cancel_token: Any = None,
    should_cancel: Callable[[], bool] | None = None,
    on_batch_complete: Callable[[list[BatchResult]], None] | None = None,
    on_batch_error: Callable[[Exception], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    max_retries: int = 3,
    total_timeout_per_batch: float | None = None,
    max_pending: int = 4,
    endpoint_key: str = "",
    fail_fast: bool = False,
    raise_on_cancel: bool = False,
) -> dict[str, BatchResult]:
    """Tien ich goi scheduler toan cuc de thuc thi mot job OCR."""
    scheduler = get_global_scheduler()
    return scheduler.schedule_job(
        job_id,
        batches,
        provider,
        token=token,
        cancel_token=cancel_token,
        should_cancel=should_cancel,
        on_batch_complete=on_batch_complete,
        on_batch_error=on_batch_error,
        on_progress=on_progress,
        max_retries=max_retries,
        total_timeout_per_batch=total_timeout_per_batch,
        max_pending=max_pending,
        endpoint_key=endpoint_key,
        fail_fast=fail_fast,
        raise_on_cancel=raise_on_cancel,
    )
