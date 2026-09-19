"""Kiem thu backend Piper Local V2 (Contract v2-p3-voice-backend-r2)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autosub_studio.providers import local_voice
from autosub_studio.providers.local_voice import (
    STATUS_NOT_DOWNLOADED,
    STATUS_READY,
    ChecksumMismatchError,
    PiperVoiceManager,
    VoiceArtifact,
    VoiceInfo,
    _is_cancelled,
    build_synthesis_command,
    get_catalog,
    get_voice_info,
    get_voice_metadata,
    list_catalog,
    piper_runtime_ready,
    register_voice,
    synthesize_piper,
)
from autosub_studio.services import paths
from autosub_studio.services.ffmpeg import CancelToken


class TestCatalog:
    """Kiem tra catalog, tinh mo rong va metadata giay phep."""

    def test_initial_catalog_contains_required_voices(self):
        catalog = get_catalog()
        assert "en_US-bryce-medium" in catalog
        assert "vi_VN-vais1000-medium" in catalog
        assert "zh_CN-chaowen-medium" in catalog

    def test_en_us_bryce_metadata_and_hashes(self):
        info = get_voice_info("en_US-bryce-medium")
        assert info is not None
        assert info.language == "en-US"
        assert info.license == "Public Domain"
        assert info.commercial_use is True
        assert "bryce" in info.model_card_url
        assert info.num_speakers == 1
        assert len(info.artifacts) == 2

        onnx = next(a for a in info.artifacts if a.name.endswith(".onnx"))
        config = next(a for a in info.artifacts if a.name.endswith(".onnx.json"))
        assert onnx.sha256 == "dc9caa6c313199ffb5ac698b6e542fa6cba388aeaf2731e25262e33b9810aef1"
        assert config.sha256 == "7ceb1bc4af6d4e41b6d1edbb86c67e91e01eaa71f66db4cd0ae92ac704d415be"
        assert onnx.size_bytes == 63531379
        assert config.size_bytes == 4966

    def test_vi_vn_vais1000_metadata_and_hashes(self):
        info = get_voice_info("vi_VN-vais1000-medium")
        assert info is not None
        assert info.language == "vi-VN"
        assert info.license == "CC BY 4.0"
        assert info.commercial_use is True
        assert "vais1000" in info.model_card_url
        assert info.num_speakers == 1

        onnx = next(a for a in info.artifacts if a.name.endswith(".onnx"))
        config = next(a for a in info.artifacts if a.name.endswith(".onnx.json"))
        assert onnx.sha256 == "ec7c89e2c85f4d1edc24b6120c18aaf1bda614f06b511567eb9c7c0de15e2dab"
        assert config.sha256 == "fafb9da1354ed4b77c31af228ed41fb41cd825c14cffa105454b25e6ae751ee0"
        assert onnx.size_bytes == 63201294
        assert config.size_bytes == 4860

    def test_zh_cn_chaowen_metadata_and_hashes(self):
        info = get_voice_info("zh_CN-chaowen-medium")
        assert info is not None
        assert info.language == "zh-CN"
        assert info.license == "CC0"
        assert info.commercial_use is True
        assert "chaowen" in info.model_card_url
        assert info.num_speakers == 1

        onnx = next(a for a in info.artifacts if a.name.endswith(".onnx"))
        config = next(a for a in info.artifacts if a.name.endswith(".onnx.json"))
        assert onnx.sha256 == "820d64ac16048fbcf38dd0823d37fab5f5e0c2bd71b01ca5a50f553fac19e746"
        assert config.sha256 == "a6bb2caafa0645642f13cbf7e2f6fbbb16fded66e51109fc26d622f6472fa16f"
        assert onnx.size_bytes == 63221984
        assert config.size_bytes == 2927

    def test_list_catalog_filter(self):
        en_voices = list_catalog("en-US")
        assert any(v.id == "en_US-bryce-medium" for v in en_voices)
        assert not any(v.id == "vi_VN-vais1000-medium" for v in en_voices)

        vi_voices = list_catalog("vi-VN")
        assert any(v.id == "vi_VN-vais1000-medium" for v in vi_voices)

    def test_register_custom_voice_extensibility(self):
        custom = VoiceInfo(
            id="custom-voice-test",
            name="Custom Test Voice",
            language="de-DE",
            language_name="German",
            license="MIT",
            artifacts=(
                VoiceArtifact(
                    name="custom.onnx",
                    url="https://example.com/custom.onnx",
                    sha256="abc123",
                    size_bytes=100,
                ),
            ),
        )
        register_voice(custom)
        assert get_voice_info("custom-voice-test") is custom
        assert any(v.id == "custom-voice-test" for v in list_catalog("de-DE"))

    def test_get_voice_metadata_exposure(self):
        meta = get_voice_metadata("en_US-bryce-medium")
        assert meta["id"] == "en_US-bryce-medium"
        assert meta["license"] == "Public Domain"
        assert meta["commercial_use"] is True
        assert len(meta["artifacts"]) == 2
        assert "sha256" in meta["artifacts"][0]


class TestVoiceManagerIntegrityAndPersistence:
    """Kiem tra quan ly model, tinh toan ven SHA256 va persistence sau khoi tao lai."""

    def test_status_not_downloaded_when_empty(self, tmp_path):
        mgr = PiperVoiceManager(tmp_path)
        assert mgr.get_status("en_US-bryce-medium") == STATUS_NOT_DOWNLOADED

    def test_status_not_downloaded_when_partial_files(self, tmp_path):
        mgr = PiperVoiceManager(tmp_path)
        vdir = tmp_path / "en_US-bryce-medium"
        vdir.mkdir()
        # Chi co file onnx, thieu onnx.json
        (vdir / "en_US-bryce-medium.onnx").write_bytes(b"some bytes")
        assert mgr.get_status("en_US-bryce-medium") == STATUS_NOT_DOWNLOADED

    def test_status_not_downloaded_when_corrupt_hash(self, tmp_path):
        test_id = "test-corrupt-voice"
        art = VoiceArtifact("test.onnx", "https://x/t.onnx", "expectedhash123", 10)
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art,)))

        mgr = PiperVoiceManager(tmp_path)
        vdir = tmp_path / test_id
        vdir.mkdir()
        (vdir / "test.onnx").write_bytes(b"corrupted content")

        assert mgr.get_status(test_id, verify_checksum=True) == STATUS_NOT_DOWNLOADED

    def test_status_ready_and_persistence_across_manager_recreation(self, tmp_path):
        test_id = "test-persistence-voice"
        content = b"valid onnx content"
        content_hash = hashlib.sha256(content).hexdigest()
        art = VoiceArtifact("test.onnx", "https://x/t.onnx", content_hash, len(content))
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art,)))

        vdir = tmp_path / test_id
        vdir.mkdir()
        (vdir / "test.onnx").write_bytes(content)

        # Instance 1
        mgr1 = PiperVoiceManager(tmp_path)
        assert mgr1.get_status(test_id) == STATUS_READY

        # Instance 2 (tai tao manager tren cung thu muc du lieu)
        mgr2 = PiperVoiceManager(tmp_path)
        assert mgr2.get_status(test_id) == STATUS_READY

    def test_safe_delete_voice(self, tmp_path):
        test_id = "test-delete-voice"
        art = VoiceArtifact("test.onnx", "https://x/t.onnx", "abc", 10)
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art,)))

        mgr = PiperVoiceManager(tmp_path)
        vdir = tmp_path / test_id
        vdir.mkdir()
        (vdir / "test.onnx").write_bytes(b"test")

        assert mgr.delete_voice(test_id) is True
        assert not vdir.exists()
        assert mgr.get_status(test_id) == STATUS_NOT_DOWNLOADED


class TestDownloadManager:
    """Kiem tra tai ve: .part, atomic rename, checksum failure, cancellation, retry."""

    def test_download_success_atomic_rename(self, tmp_path, monkeypatch):
        test_id = "test-dl-success"
        c1 = b"onnx-model-binary-data"
        c2 = b'{"sample_rate": 22050}'
        h1 = hashlib.sha256(c1).hexdigest()
        h2 = hashlib.sha256(c2).hexdigest()

        art1 = VoiceArtifact("m.onnx", "https://mock/m.onnx", h1, len(c1))
        art2 = VoiceArtifact("m.onnx.json", "https://mock/m.onnx.json", h2, len(c2))
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art1, art2)))

        responses = {
            "https://mock/m.onnx": c1,
            "https://mock/m.onnx.json": c2,
        }

        class MockResponse:
            def __init__(self, data: bytes):
                self._io = io.BytesIO(data)

            def read(self, amt: int = -1):
                return self._io.read(amt)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout=60.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url in responses:
                return MockResponse(responses[url])
            raise RuntimeError(f"Unexpected URL: {url}")

        monkeypatch.setattr(local_voice.urllib.request, "urlopen", mock_urlopen)

        mgr = PiperVoiceManager(tmp_path)
        progress_events = []
        out_dir = mgr.download_voice(
            test_id,
            on_progress=lambda cur, total: progress_events.append((cur, total)),
        )

        assert out_dir == tmp_path / test_id
        assert (out_dir / "m.onnx").read_bytes() == c1
        assert (out_dir / "m.onnx.json").read_bytes() == c2
        # Khong con file .part
        assert not (out_dir / "m.onnx.part").exists()
        assert not (out_dir / "m.onnx.json.part").exists()
        assert mgr.get_status(test_id) == STATUS_READY
        assert len(progress_events) > 0

    def test_download_checksum_failure_cleans_up(self, tmp_path, monkeypatch):
        test_id = "test-dl-checksum-fail"
        correct_content = b"correct data"
        wrong_content = b"corrupted data from server"
        correct_hash = hashlib.sha256(correct_content).hexdigest()

        art = VoiceArtifact(
            "model.onnx", "https://mock/model.onnx", correct_hash, len(correct_content)
        )
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art,)))

        class MockResponse:
            def __init__(self, data: bytes):
                self._io = io.BytesIO(data)

            def read(self, amt: int = -1):
                return self._io.read(amt)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            local_voice.urllib.request,
            "urlopen",
            lambda req, **kw: MockResponse(wrong_content),
        )

        mgr = PiperVoiceManager(tmp_path)
        with pytest.raises(ChecksumMismatchError):
            mgr.download_voice(test_id)

        vdir = tmp_path / test_id
        # File target khong duoc tao, file .part phai bi xoa sach
        assert not (vdir / "model.onnx").exists()
        assert not (vdir / "model.onnx.part").exists()
        assert mgr.get_status(test_id) == STATUS_NOT_DOWNLOADED

    def test_is_cancelled_supports_various_token_types(self):
        token = CancelToken()
        assert _is_cancelled(token) is False
        token.cancel()
        assert _is_cancelled(token) is True
        assert _is_cancelled(None) is False
        assert _is_cancelled(lambda: True) is True
        assert _is_cancelled(lambda: False) is False

        class CustomToken:
            def __init__(self, c: bool):
                self.is_cancelled = c

        assert _is_cancelled(CustomToken(True)) is True
        assert _is_cancelled(CustomToken(False)) is False

    def test_download_cancellation_with_ffmpeg_cancel_token(self, tmp_path, monkeypatch):
        test_id = "test-dl-ffmpeg-cancel"
        content = b"sample model data for cancellation"
        content_hash = hashlib.sha256(content).hexdigest()
        art = VoiceArtifact("cancel.onnx", "https://mock/cancel.onnx", content_hash, len(content))
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art,)))

        token = CancelToken()

        class StreamResponse:
            def __init__(self):
                self._chunks = [b"chunk1", b"chunk2", b"chunk3"]
                self._idx = 0

            def read(self, amt: int = -1):
                if self._idx < len(self._chunks):
                    data = self._chunks[self._idx]
                    self._idx += 1
                    # Huy ngay giua luong download sau chunk 1
                    token.cancel()
                    return data
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            local_voice.urllib.request,
            "urlopen",
            lambda req, **kw: StreamResponse(),
        )

        mgr = PiperVoiceManager(tmp_path)
        with pytest.raises(RuntimeError, match="da bi huy"):
            mgr.download_voice(test_id, cancel_token=token)

        vdir = tmp_path / test_id
        # Dam bao .part bi xoa va model khong bao gio la ready
        assert not (vdir / "cancel.onnx").exists()
        assert not (vdir / "cancel.onnx.part").exists()
        assert mgr.get_status(test_id) == STATUS_NOT_DOWNLOADED

    def test_download_cancellation_and_retry(self, tmp_path, monkeypatch):
        test_id = "test-dl-cancel-retry"
        valid_content = b"valid data for retry test"
        valid_hash = hashlib.sha256(valid_content).hexdigest()
        art = VoiceArtifact("retry.onnx", "https://mock/retry.onnx", valid_hash, len(valid_content))
        register_voice(VoiceInfo(id=test_id, name="Test", language="en", artifacts=(art,)))

        # Lan 1: Huy bang CancelToken thuc te cua ung dung
        token = CancelToken()
        token.cancel()

        class MockResponse:
            def read(self, amt: int = -1):
                return b"some data"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            local_voice.urllib.request,
            "urlopen",
            lambda req, **kw: MockResponse(),
        )

        mgr = PiperVoiceManager(tmp_path)
        with pytest.raises(RuntimeError, match="da bi huy"):
            mgr.download_voice(test_id, cancel_token=token)

        vdir = tmp_path / test_id
        assert not (vdir / "retry.onnx").exists()
        assert not (vdir / "retry.onnx.part").exists()
        assert mgr.get_status(test_id) == STATUS_NOT_DOWNLOADED

        # Lan 2: Thu lai (retry) thanh cong
        class SuccessResponse:
            def __init__(self):
                self._read = False

            def read(self, amt: int = -1):
                if not self._read:
                    self._read = True
                    return valid_content
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            local_voice.urllib.request,
            "urlopen",
            lambda req, **kw: SuccessResponse(),
        )

        mgr.download_voice(test_id)
        assert (vdir / "retry.onnx").read_bytes() == valid_content
        assert mgr.get_status(test_id) == STATUS_READY


class TestSynthesisCommandAndControls:
    """Kiem tra cau truc lenh synthesis, anh xa toc do, khong sua toc do per-cue."""

    def test_synthesis_command_structure(self):
        cmd = build_synthesis_command(
            "C:/piper/piper.exe",
            "C:/models/voice.onnx",
            "C:/models/voice.onnx.json",
            "C:/output/out.wav",
        )
        assert cmd == [
            "C:/piper/piper.exe",
            "-m",
            "C:/models/voice.onnx",
            "-c",
            "C:/models/voice.onnx.json",
            "-f",
            "C:/output/out.wav",
        ]

    def test_global_speed_mapped_to_length_scale(self):
        # speed = 1.25 -> length_scale = 1.0 / 1.25 = 0.8
        cmd_fast = build_synthesis_command(
            "piper.exe",
            "m.onnx",
            "m.json",
            "out.wav",
            length_scale=0.8,
        )
        assert "--length_scale" in cmd_fast
        idx = cmd_fast.index("--length_scale")
        assert cmd_fast[idx + 1] == "0.8000"

        # speed = 0.8 -> length_scale = 1.0 / 0.8 = 1.25
        cmd_slow = build_synthesis_command(
            "piper.exe",
            "m.onnx",
            "m.json",
            "out.wav",
            length_scale=1.25,
        )
        assert "--length_scale" in cmd_slow
        idx = cmd_slow.index("--length_scale")
        assert cmd_slow[idx + 1] == "1.2500"

        # speed = 1.0 -> khong can truyen --length_scale
        cmd_normal = build_synthesis_command(
            "piper.exe",
            "m.onnx",
            "m.json",
            "out.wav",
            length_scale=1.0,
        )
        assert "--length_scale" not in cmd_normal

    def test_no_pitch_argument_in_command(self):
        # Pitch khong ho tro boi Piper CLI, dam bao khong chen co pitch vo nghia
        cmd = build_synthesis_command("piper.exe", "m.onnx", "m.json", "out.wav")
        for arg in cmd:
            assert "pitch" not in arg.lower()

    def test_speaker_id_support(self):
        cmd = build_synthesis_command(
            "piper.exe",
            "m.onnx",
            "m.json",
            "out.wav",
            speaker_id=42,
        )
        assert "-s" in cmd
        assert cmd[cmd.index("-s") + 1] == "42"

    def test_no_per_cue_speed_mutation(self):
        # Goi synthesis nhieu lan voi cac gia tri speed doc lap khong lam bien doi trang thai
        cmd1 = build_synthesis_command("piper.exe", "m.onnx", "m.json", "1.wav", length_scale=0.8)
        cmd2 = build_synthesis_command("piper.exe", "m.onnx", "m.json", "2.wav", length_scale=1.25)
        cmd3 = build_synthesis_command("piper.exe", "m.onnx", "m.json", "3.wav", length_scale=0.8)

        assert cmd1[cmd1.index("--length_scale") + 1] == "0.8000"
        assert cmd2[cmd2.index("--length_scale") + 1] == "1.2500"
        assert cmd3[cmd3.index("--length_scale") + 1] == "0.8000"

    def test_synthesize_piper_execution_mocked(self, tmp_path, monkeypatch):
        # Mock binary va model ready
        test_id = "test-synth-voice"
        onnx_file = tmp_path / "model.onnx"
        json_file = tmp_path / "model.onnx.json"
        exe_file = tmp_path / "piper.exe"
        onnx_file.write_bytes(b"onnx-data")
        json_file.write_bytes(b"{}")
        exe_file.write_bytes(b"fake-exe")

        monkeypatch.setattr(paths, "piper_bin_path", lambda: exe_file)
        monkeypatch.setattr(local_voice, "piper_bin_path", lambda: exe_file)

        mgr = MagicMock()
        mgr.get_model_paths.return_value = (onnx_file, json_file)

        recorded_commands = []

        def mock_subprocess_run(cmd, **kwargs):
            recorded_commands.append(cmd)
            # Tao file out.wav gia lap thanh cong
            out_file = Path(cmd[cmd.index("-f") + 1])
            out_file.write_bytes(b"RIFF" + b"\x00" * 100)
            res = MagicMock()
            res.returncode = 0
            return res

        monkeypatch.setattr(local_voice.subprocess, "run", mock_subprocess_run)

        out_wav = tmp_path / "output.wav"
        result = synthesize_piper(
            "Hello world",
            out_wav,
            voice_id=test_id,
            speed=1.25,
            speaker_id=3,
            manager=mgr,
        )

        assert result == out_wav
        assert out_wav.is_file()
        assert len(recorded_commands) == 1
        cmd = recorded_commands[0]
        assert "-m" in cmd
        assert "-s" in cmd
        assert cmd[cmd.index("-s") + 1] == "3"
        assert "--length_scale" in cmd
        assert cmd[cmd.index("--length_scale") + 1] == "0.8000"

    def test_runtime_absent_readiness_reason(self, monkeypatch):
        monkeypatch.setattr(paths, "piper_bin_path", lambda: None)
        monkeypatch.setattr(local_voice, "piper_bin_path", lambda: None)
        ready, reason = piper_runtime_ready()
        assert not ready
        assert "piper.exe" in reason
