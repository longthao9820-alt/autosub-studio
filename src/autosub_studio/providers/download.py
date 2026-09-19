"""Tai video/audio bang yt-dlp di kem, khong qua shell."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from ..services.ffmpeg import CancelledError, CancelToken
from ..services.paths import app_root, bundled_dir

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")


class DownloadError(RuntimeError):
    pass


def find_ytdlp() -> Path | None:
    bundled = bundled_dir("download")
    candidates = [
        bundled / "yt-dlp.exe" if bundled else None,
        app_root() / "assets" / "download" / "yt-dlp.exe",
        app_root() / "yt-dlp.exe",
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def is_available() -> bool:
    return find_ytdlp() is not None


def download_video(
    url: str,
    out_dir: str | Path,
    *,
    audio_only: bool = False,
    proxy: str = "",
    token: CancelToken | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Tai mot URL thanh MP4 hoac audio, tra ve duong dan that sau merge."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Link video khong hop le; chi chap nhan http/https.")
    executable = find_ytdlp()
    if executable is None:
        raise DownloadError("Khong tim thay yt-dlp di kem tool.")
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    template = str(folder / "%(title).180B [%(id)s].%(ext)s")
    args = [
        str(executable),
        "--no-playlist",
        "--newline",
        "--windows-filenames",
        "--progress-template",
        "download:%(progress._percent_str)s",
        "--print",
        "after_move:__AUTOSUB_FILE__%(filepath)s",
        "-o",
        template,
    ]
    if audio_only:
        args += ["-x", "--audio-format", "mp3"]
    else:
        args += ["-f", "bestvideo*+bestaudio/best", "--merge-output-format", "mp4"]
    if proxy.strip():
        args += ["--proxy", proxy.strip()]
    args.append(str(url).strip())
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )
    output: Path | None = None
    tail = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        tail.append(line)
        del tail[:-30]
        if token is not None and token.cancelled:
            process.terminate()
            process.wait(timeout=10)
            raise CancelledError("Da dung tai video.")
        if line.startswith("__AUTOSUB_FILE__"):
            output = Path(line.removeprefix("__AUTOSUB_FILE__"))
        found = _PERCENT.search(line)
        if found and on_progress:
            on_progress(max(0, min(99, int(float(found.group(1))))))
        if on_log and not line.startswith("download:"):
            on_log(line)
    code = process.wait()
    if code != 0:
        raise DownloadError("Tai video that bai: " + " | ".join(tail[-5:]))
    if output is None or not output.is_file():
        files = sorted(folder.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        output = next((path for path in files if path.is_file()), None)
    if output is None or not output.is_file():
        raise DownloadError("yt-dlp bao xong nhung khong tim thay tep dau ra.")
    if on_progress:
        on_progress(100)
    return output
