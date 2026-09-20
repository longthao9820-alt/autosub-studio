"""Thread-safe metrics snapshot va collector cho AI OCR Scheduler."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerMetricsSnapshot:
    """Ban chup thong ke hieu nang luong xu ly OCR AI tai mot thoi diem."""

    active: int
    queue: int
    completed: int
    failed: int
    retries: int
    avg_latency: float
    last_latency: float
    payload_bytes: int
    avg_batch_size: float
    cache_hits: int
    cache_misses: int
    elapsed: float


class SchedulerMetricsCollector:
    """Thu thap va tinh toan chi so thong ke thread-safe cho AI OCR Scheduler."""

    def __init__(
        self,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock_fn = clock_fn
        self._lock = threading.Lock()
        self._start_time: float = self._clock_fn()

        self._active_requests: int = 0
        self._queue_size: int = 0
        self._completed_batches: int = 0
        self._failed_batches: int = 0
        self._retries_count: int = 0
        self._total_latency: float = 0.0
        self._last_latency: float = 0.0
        self._payload_bytes: int = 0
        self._total_items: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def record_active_change(self, delta: int) -> None:
        """Cap nhat so luong HTTP requests dang chay."""
        with self._lock:
            self._active_requests = max(0, self._active_requests + delta)

    def record_queue_change(self, delta: int) -> None:
        """Cap nhat so luong batches dang cho trong hang doi."""
        with self._lock:
            self._queue_size = max(0, self._queue_size + delta)

    def record_batch_completion(
        self,
        batch_size: int,
        latency: float,
        payload_bytes: int = 0,
    ) -> None:
        """Ghi nhan mot batch hoan thanh thanh cong."""
        with self._lock:
            self._completed_batches += 1
            self._total_items += max(0, batch_size)
            self._last_latency = max(0.0, float(latency))
            self._total_latency += self._last_latency
            self._payload_bytes += max(0, int(payload_bytes))

    def record_batch_failure(self) -> None:
        """Ghi nhan mot batch that bai vinh vien."""
        with self._lock:
            self._failed_batches += 1

    def record_retry(self) -> None:
        """Ghi nhan mot lan thu lai (retry)."""
        with self._lock:
            self._retries_count += 1

    def record_cache_hit(self, count: int = 1) -> None:
        """Ghi nhan so lan trung cache (cache hit)."""
        with self._lock:
            self._cache_hits += max(0, int(count))

    def record_cache_miss(self, count: int = 1) -> None:
        """Ghi nhan so lan khong trung cache (cache miss)."""
        with self._lock:
            self._cache_misses += max(0, int(count))

    def record_payload_bytes(self, byte_count: int) -> None:
        """Ghi nhan them dung luong payload (bytes)."""
        with self._lock:
            self._payload_bytes += max(0, int(byte_count))

    def snapshot(
        self,
        queue_size: int | None = None,
        active_count: int | None = None,
    ) -> SchedulerMetricsSnapshot:
        """Tra ve ban chup thong ke bat bien (immutable) tai thoi diem hien tai."""
        with self._lock:
            q = self._queue_size if queue_size is None else queue_size
            act = self._active_requests if active_count is None else active_count
            avg_lat = (
                self._total_latency / self._completed_batches
                if self._completed_batches > 0
                else 0.0
            )
            avg_b_size = (
                self._total_items / self._completed_batches
                if self._completed_batches > 0
                else 0.0
            )
            now = self._clock_fn()
            elapsed = max(0.0, now - self._start_time)

            return SchedulerMetricsSnapshot(
                active=max(0, act),
                queue=max(0, q),
                completed=self._completed_batches,
                failed=self._failed_batches,
                retries=self._retries_count,
                avg_latency=avg_lat,
                last_latency=self._last_latency,
                payload_bytes=self._payload_bytes,
                avg_batch_size=avg_b_size,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                elapsed=elapsed,
            )

    def reset(self) -> None:
        """Dat lai toan bo bo dem ve 0."""
        with self._lock:
            self._start_time = self._clock_fn()
            self._active_requests = 0
            self._queue_size = 0
            self._completed_batches = 0
            self._failed_batches = 0
            self._retries_count = 0
            self._total_latency = 0.0
            self._last_latency = 0.0
            self._payload_bytes = 0
            self._total_items = 0
            self._cache_hits = 0
            self._cache_misses = 0
