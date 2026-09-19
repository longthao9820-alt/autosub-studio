"""Hang doi tac vu nen: chay song song, bao tien trinh, huy va gioi han thoi gian."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from .ffmpeg import CancelledError, CancelToken

PENDING = "cho"
RUNNING = "dang chay"
DONE = "xong"
FAILED = "loi"
CANCELLED = "da huy"


@dataclass
class TaskContext:
    """Doi tuong truyen vao ham xu ly de bao tien trinh va kiem tra huy."""

    token: CancelToken
    _progress: Callable[[int], None]
    _log: Callable[[str], None]

    def progress(self, percent: int) -> None:
        self._progress(max(0, min(100, int(percent))))

    def log(self, message: str) -> None:
        self._log(str(message))

    def check_cancel(self) -> None:
        self.token.raise_if_cancelled()

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled


class TaskSignals(QObject):
    started = Signal(str)
    progress = Signal(str, int)
    log = Signal(str, str)
    finished = Signal(str, str, object, str)  # task_id, status, result, message


@dataclass
class TaskRecord:
    """Ban ghi mot tac vu trong hang doi."""

    task_id: str
    name: str
    project_id: int = 0
    status: str = PENDING
    percent: int = 0
    message: str = ""
    token: CancelToken = field(default_factory=CancelToken)


class _Runner(QRunnable):
    def __init__(
        self,
        record: TaskRecord,
        func: Callable[..., Any],
        signals: TaskSignals,
        timeout: float,
    ) -> None:
        super().__init__()
        self.record = record
        self.func = func
        self.signals = signals
        self.timeout = timeout
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        rec = self.record
        if rec.token.cancelled:
            self.signals.finished.emit(rec.task_id, CANCELLED, None, "Da huy truoc khi chay")
            return
        self.signals.started.emit(rec.task_id)
        ctx = TaskContext(
            token=rec.token,
            _progress=lambda p: self.signals.progress.emit(rec.task_id, p),
            _log=lambda m: self.signals.log.emit(rec.task_id, m),
        )
        try:
            result = self.func(ctx)
        except CancelledError:
            self.signals.finished.emit(rec.task_id, CANCELLED, None, "Nguoi dung da huy")
        except Exception as exc:  # loi nghiep vu duoc bao ve nguoi dung, khong lam sap app
            detail = f"{exc}".strip() or exc.__class__.__name__
            self.signals.log.emit(rec.task_id, traceback.format_exc())
            self.signals.finished.emit(rec.task_id, FAILED, None, detail)
        else:
            self.signals.finished.emit(rec.task_id, DONE, result, "Hoan tat")


class TaskManager(QObject):
    """Quan ly toan bo tac vu nen cua ung dung."""

    task_started = Signal(str)
    task_progress = Signal(str, int)
    task_log = Signal(str, str)
    task_finished = Signal(str, str, object, str)
    queue_changed = Signal()

    def __init__(self, max_workers: int = 2, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_workers)))
        self._records: dict[str, TaskRecord] = {}
        self._counter = 0
        self._signals = TaskSignals(self)
        self._signals.started.connect(self._on_started)
        self._signals.progress.connect(self._on_progress)
        self._signals.log.connect(self.task_log)
        self._signals.finished.connect(self._on_finished)

    def set_max_workers(self, count: int) -> None:
        self._pool.setMaxThreadCount(max(1, int(count)))

    @property
    def max_workers(self) -> int:
        return self._pool.maxThreadCount()

    def submit(
        self,
        name: str,
        func: Callable[[TaskContext], Any],
        *,
        project_id: int = 0,
        timeout: float = 0.0,
    ) -> str:
        """Dua mot ham vao hang doi. Tra ve ma tac vu."""
        self._counter += 1
        task_id = f"T{self._counter:05d}"
        record = TaskRecord(task_id=task_id, name=name, project_id=project_id)
        self._records[task_id] = record
        self._pool.start(_Runner(record, func, self._signals, timeout))
        self.queue_changed.emit()
        return task_id

    def cancel(self, task_id: str) -> bool:
        rec = self._records.get(task_id)
        if rec is None or rec.status in (DONE, FAILED, CANCELLED):
            return False
        rec.token.cancel()
        rec.message = "Dang dung..."
        self.queue_changed.emit()
        return True

    def cancel_project(self, project_id: int) -> int:
        count = 0
        for rec in list(self._records.values()):
            if rec.project_id == project_id and rec.status in (PENDING, RUNNING):
                rec.token.cancel()
                count += 1
        if count:
            self.queue_changed.emit()
        return count

    def cancel_all(self) -> int:
        count = 0
        for rec in list(self._records.values()):
            if rec.status in (PENDING, RUNNING):
                rec.token.cancel()
                count += 1
        if count:
            self.queue_changed.emit()
        return count

    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if r.status in (PENDING, RUNNING))

    def records(self) -> list[TaskRecord]:
        return list(self._records.values())

    def record(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    def running_for_project(self, project_id: int) -> TaskRecord | None:
        for rec in self._records.values():
            if rec.project_id == project_id and rec.status in (PENDING, RUNNING):
                return rec
        return None

    def wait_for_done(self, msecs: int = 30000) -> bool:
        """Cho tat ca tac vu ket thuc, dung khi dong ung dung."""
        return self._pool.waitForDone(msecs)

    def clear_finished(self) -> None:
        for tid in [t for t, r in self._records.items() if r.status in (DONE, FAILED, CANCELLED)]:
            del self._records[tid]
        self.queue_changed.emit()

    # ------------------------------------------------------------------ noi bo

    def _on_started(self, task_id: str) -> None:
        rec = self._records.get(task_id)
        if rec:
            rec.status = RUNNING
            rec.message = "Dang chay"
        self.task_started.emit(task_id)
        self.queue_changed.emit()

    def _on_progress(self, task_id: str, percent: int) -> None:
        rec = self._records.get(task_id)
        if rec:
            rec.percent = percent
        self.task_progress.emit(task_id, percent)

    def _on_finished(self, task_id: str, status: str, result: object, message: str) -> None:
        rec = self._records.get(task_id)
        if rec:
            rec.status = status
            rec.message = message
            rec.percent = 100 if status == DONE else rec.percent
        self.task_finished.emit(task_id, status, result, message)
        self.queue_changed.emit()
