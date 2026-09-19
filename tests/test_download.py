"""Kiem thu wrapper yt-dlp an toan va co tien trinh."""

from __future__ import annotations

import pytest

from autosub_studio.providers import download


def test_rejects_non_http_links(tmp_path):
    with pytest.raises(download.DownloadError):
        download.download_video("file:///C:/secret.txt", tmp_path)


def test_download_passes_url_as_argument_and_returns_printed_file(tmp_path, monkeypatch):
    executable = tmp_path / "yt-dlp.exe"
    executable.write_bytes(b"exe")
    output = tmp_path / "video.mp4"
    output.write_bytes(b"video")
    seen = {}

    class FakeProcess:
        stdout = iter(["download: 42.5%\n", f"__AUTOSUB_FILE__{output}\n"])

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def popen(args, **kwargs):
        seen["args"] = list(args)
        seen["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(download, "find_ytdlp", lambda: executable)
    monkeypatch.setattr(download.subprocess, "Popen", popen)
    progress = []

    result = download.download_video(
        "https://example.com/watch?v=1",
        tmp_path,
        on_progress=progress.append,
    )

    assert result == output
    assert seen["args"][-1] == "https://example.com/watch?v=1"
    assert "--no-playlist" in seen["args"]
    assert "shell" not in seen["kwargs"]
    assert progress == [42, 100]
