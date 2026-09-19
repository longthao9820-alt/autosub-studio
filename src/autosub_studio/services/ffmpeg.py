"""Tim va goi FFmpeg/FFprobe duoi dang tien trinh con."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .paths import app_root

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_TIME_RE = re.compile(r"out_time_ms=(-?\d+)")


class FFmpegNotFound(RuntimeError):
    """Khong tim thay FFmpeg tren may."""


class FFmpegError(RuntimeError):
    """FFmpeg tra ve ma loi."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


class CancelledError(RuntimeError):
    """Nguoi dung da huy tac vu."""


class CancelToken:
    """Co hieu de huy mot tac vu dang chay."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("Tac vu da bi huy")


@dataclass
class MediaInfo:
    """Thong tin cua mot tep video/audio."""

    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    size_bytes: int = 0

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "-"


def _candidate_dirs() -> Iterable[Path]:
    """Thu muc co kha nang chua ffmpeg, uu tien ban kem theo ung dung."""
    seen: set[Path] = set()
    bases: list[Path] = [app_root(), app_root() / "_internal"]
    source_root = Path(__file__).resolve().parents[3]
    if source_root not in bases:
        bases.append(source_root)
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        bases.append(Path(meipass))

    for base in bases:
        for parent in (base, *list(base.parents)[:3]):
            for name in ("ffmpeg", "ffmpeg/bin", "bin", "tools", ""):
                d = (parent / name) if name else parent
                if d not in seen:
                    seen.add(d)
                    yield d


def find_binary(name: str, configured: str = "") -> str:
    """Tim duong dan tep thuc thi.

    Thu tu uu tien: duong dan nguoi dung da cau hinh, ban kem theo ung dung,
    roi moi den PATH cua he thong.
    """
    exe = f"{name}.exe" if os.name == "nt" else name
    if configured:
        p = Path(configured)
        if p.is_dir():
            cand = p / exe
            if cand.is_file():
                return str(cand)
        elif p.is_file():
            return str(p)
    for d in _candidate_dirs():
        cand = d / exe
        try:
            if cand.is_file():
                return str(cand)
        except OSError:
            continue
    return shutil.which(name) or ""


class FFmpeg:
    """Bo goi FFmpeg co ho tro tien trinh, huy va gioi han thoi gian."""

    def __init__(self, ffmpeg_path: str = "", ffprobe_path: str = "") -> None:
        self.ffmpeg = find_binary("ffmpeg", ffmpeg_path)
        self.ffprobe = find_binary("ffprobe", ffprobe_path) or _sibling(self.ffmpeg, "ffprobe")

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg) and Path(self.ffmpeg).is_file()

    def require(self) -> None:
        if not self.available:
            raise FFmpegNotFound(
                "Chua tim thay FFmpeg. Vao tab 'Cai dat chung' de chon duong dan ffmpeg.exe, "
                "hoac cai FFmpeg roi them vao PATH."
            )

    def version(self) -> str:
        if not self.available:
            return ""
        try:
            out = subprocess.run(
                [self.ffmpeg, "-version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                timeout=15,
            )
            return out.stdout.splitlines()[0] if out.stdout else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def probe(self, path: str | Path) -> MediaInfo:
        """Doc thong tin media. Nem FFmpegError neu tep hong."""
        self.require()
        p = Path(path)
        if not p.is_file():
            raise FFmpegError(f"Khong tim thay tep: {p}")
        if not self.ffprobe:
            raise FFmpegNotFound("Khong tim thay ffprobe.exe canh ffmpeg.exe")
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(p),
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError("Doc thong tin tep qua lau, da dung lai.") from exc
        if res.returncode != 0:
            raise FFmpegError(f"Tep media khong doc duoc: {p.name}", res.stderr)
        try:
            data = json.loads(res.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise FFmpegError(f"Ket qua ffprobe khong hop le cho {p.name}") from exc
        info = MediaInfo(path=str(p), size_bytes=p.stat().st_size)
        fmt = data.get("format") or {}
        try:
            info.duration = float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            info.duration = 0.0
        for stream in data.get("streams") or []:
            kind = stream.get("codec_type")
            if kind == "video" and not info.has_video:
                info.has_video = True
                info.width = int(stream.get("width") or 0)
                info.height = int(stream.get("height") or 0)
                info.video_codec = str(stream.get("codec_name") or "")
                info.fps = _parse_fraction(stream.get("avg_frame_rate") or "0/0")
            elif kind == "audio" and not info.has_audio:
                info.has_audio = True
                info.audio_codec = str(stream.get("codec_name") or "")
        return info

    def run(
        self,
        args: Sequence[str],
        *,
        duration: float = 0.0,
        on_progress: Callable[[int], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        token: CancelToken | None = None,
        timeout: float = 0.0,
        cwd: str | Path | None = None,
    ) -> None:
        """Chay FFmpeg. Nem FFmpegError khi that bai, CancelledError khi bi huy."""
        self.require()
        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-y", *args]
        if duration > 0:
            cmd += ["-progress", "pipe:1", "-nostats"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            cwd=str(cwd) if cwd else None,
        )
        tail: list[str] = []
        started = time.monotonic()
        stop = threading.Event()

        def pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if len(tail) > 60:
                    del tail[0]
                if on_log:
                    on_log(line)

        t = threading.Thread(target=pump_stderr, daemon=True)
        t.start()
        try:
            if duration > 0 and proc.stdout is not None:
                for line in proc.stdout:
                    if stop.is_set():
                        break
                    m = _TIME_RE.search(line)
                    if m and on_progress:
                        done = max(0, int(m.group(1))) / 1_000_000.0
                        on_progress(max(0, min(99, int(done / duration * 100))))
                    if _should_stop(token, started, timeout):
                        break
            while proc.poll() is None:
                if _should_stop(token, started, timeout):
                    break
                time.sleep(0.1)
        finally:
            stop.set()
        if proc.poll() is None:
            _kill(proc)
            t.join(timeout=2)
            if token is not None and token.cancelled:
                raise CancelledError("Tac vu da bi huy")
            raise FFmpegError("Tac vu vuot qua thoi gian cho phep va da bi dung.")
        t.join(timeout=2)
        if proc.returncode != 0:
            log = "\n".join(tail)
            raise FFmpegError(_friendly_error(log), log)
        if on_progress:
            on_progress(100)


def _should_stop(token: CancelToken | None, started: float, timeout: float) -> bool:
    if token is not None and token.cancelled:
        return True
    return timeout > 0 and (time.monotonic() - started) > timeout


def _kill(proc: subprocess.Popen) -> None:
    """Dung tien trinh con va toan bo tien trinh chau."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                timeout=15,
            )
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        with contextlib.suppress(OSError):
            proc.kill()


def _friendly_error(log: str) -> str:
    """Chuyen loi FFmpeg thanh cau tieng Viet de hieu."""
    low = log.lower()
    if "no space left" in low:
        return "O dia da day, khong ghi duoc tep ket qua."
    if "permission denied" in low or "access is denied" in low:
        return "Khong co quyen ghi tep, hoac tep dang bi phan mem khac mo."
    if "invalid data found" in low or "moov atom not found" in low:
        return "Tep video bi hong hoac khong doc duoc."
    if "does not contain any stream" in low:
        return "Tep khong co luong hinh/tieng nao."
    if "unable to find a suitable output format" in low:
        return "Duoi tep dau ra khong duoc ho tro."
    if "no such file or directory" in low:
        return "Khong tim thay tep dau vao."
    if "fontconfig" in low or "unable to open" in low and ".ass" in low:
        return "Khong doc duoc tep phu de hoac font khi ghep phu de."
    for line in reversed(log.splitlines()):
        if line.strip():
            return f"FFmpeg bao loi: {line.strip()}"
    return "FFmpeg that bai nhung khong tra ve thong tin loi."


def _sibling(binary: str, name: str) -> str:
    if not binary:
        return ""
    exe = f"{name}.exe" if os.name == "nt" else name
    cand = Path(binary).with_name(exe)
    return str(cand) if cand.is_file() else (shutil.which(name) or "")


def _parse_fraction(text: str) -> float:
    try:
        num, _, den = str(text).partition("/")
        d = float(den) if den else 1.0
        return float(num) / d if d else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
