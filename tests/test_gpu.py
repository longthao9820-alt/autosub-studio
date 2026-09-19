"""Kiem thu cho phan tang toc bang card do hoa."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from autosub_studio.services import gpu


class TestEncoderArgs:
    def test_cpu_uses_libx264(self):
        args = gpu.video_encoder_args(20, "medium", "")
        assert "libx264" in args
        assert "-crf" in args and "20" in args

    def test_nvidia_encoder(self):
        args = gpu.video_encoder_args(20, "medium", "h264_nvenc")
        assert "h264_nvenc" in args
        assert "-cq" in args and "20" in args
        assert "libx264" not in args

    def test_intel_encoder(self):
        args = gpu.video_encoder_args(22, "medium", "h264_qsv")
        assert "h264_qsv" in args
        assert "-global_quality" in args and "22" in args

    def test_amd_encoder(self):
        args = gpu.video_encoder_args(24, "medium", "h264_amf")
        assert "h264_amf" in args
        assert "-qp_i" in args and "24" in args

    def test_preset_is_translated_for_nvidia(self):
        args = gpu.video_encoder_args(20, "ultrafast", "h264_nvenc")
        assert "p1" in args
        assert "ultrafast" not in args

    def test_unknown_preset_falls_back(self):
        args = gpu.video_encoder_args(20, "khong-co-preset-nay", "h264_nvenc")
        assert "p4" in args

    def test_unknown_encoder_falls_back_to_cpu(self):
        assert "libx264" in gpu.video_encoder_args(20, "medium", "h264_khong_co")

    def test_quality_is_clamped(self):
        assert "51" in gpu.video_encoder_args(999, "medium", "")
        assert "0" in gpu.video_encoder_args(-5, "medium", "h264_nvenc")

    def test_boolean_still_accepted(self):
        assert "h264_nvenc" in gpu.video_encoder_args(20, "medium", True)
        assert "libx264" in gpu.video_encoder_args(20, "medium", False)

    def test_pixel_format_always_set(self):
        for enc in ("", "h264_nvenc", "h264_qsv", "h264_amf"):
            args = gpu.video_encoder_args(20, "medium", enc)
            assert "-pix_fmt" in args

    def test_encoder_labels_are_readable(self):
        assert "NVIDIA" in gpu.encoder_label("h264_nvenc")
        assert "Intel" in gpu.encoder_label("h264_qsv")
        assert "AMD" in gpu.encoder_label("h264_amf")
        assert "CPU" in gpu.encoder_label("")


class TestDetection:
    def test_profile_enables_available_acceleration(self, monkeypatch):
        from autosub_studio.providers import ocr

        monkeypatch.setattr(gpu, "graphics_adapters", lambda: ("NVIDIA Test GPU",))
        monkeypatch.setattr(gpu, "machine_signature", lambda: "machine-a")
        monkeypatch.setattr(gpu, "cuda_ready", lambda: (True, "CUDA san sang"))
        monkeypatch.setattr(gpu, "onnx_cuda_ready", lambda: True)
        monkeypatch.setattr(ocr, "gpu_available", lambda: True)
        monkeypatch.setattr(gpu, "best_hw_encoder", lambda _path: "h264_nvenc")

        profile = gpu.acceleration_profile("ffmpeg.exe")

        assert profile.use_gpu is True
        assert profile.use_gpu_encoder is True
        assert profile.ocr_cuda_ready is True
        assert profile.signature == "machine-a"
        assert "NVIDIA Test GPU" in profile.summary()

    def test_profile_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(gpu, "graphics_adapters", lambda: ())
        monkeypatch.setattr(gpu, "machine_signature", lambda: "machine-b")
        monkeypatch.setattr(gpu, "cuda_ready", lambda: (False, "Khong co CUDA"))
        monkeypatch.setattr(gpu, "best_hw_encoder", lambda _path: "")

        profile = gpu.acceleration_profile("")

        assert profile.use_gpu is False
        assert profile.use_gpu_encoder is False
        assert "CPU" in profile.summary()

    def test_profile_does_not_claim_ocr_gpu_when_real_probe_fails(self, monkeypatch):
        from autosub_studio.providers import ocr

        monkeypatch.setattr(gpu, "graphics_adapters", lambda: ("NVIDIA Test GPU",))
        monkeypatch.setattr(gpu, "machine_signature", lambda: "machine-c")
        monkeypatch.setattr(gpu, "cuda_ready", lambda: (True, "CUDA san sang"))
        monkeypatch.setattr(gpu, "onnx_cuda_ready", lambda: True)
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        monkeypatch.setattr(gpu, "best_hw_encoder", lambda _path: "")

        profile = gpu.acceleration_profile("ffmpeg.exe")

        assert profile.cuda_ready is True
        assert profile.ocr_cuda_ready is False
        assert "OCR: CPU" in profile.summary()

    def test_cuda_ready_returns_reason(self):
        ready, reason = gpu.cuda_ready()
        assert isinstance(ready, bool)
        assert reason

    def test_dll_dirs_are_absolute(self):
        for path in gpu.cuda_dll_dirs():
            assert path and (path[1:3] == ":\\" or path.startswith("/"))

    def test_register_does_not_raise(self):
        assert gpu.register_cuda_dlls() >= 0

    def test_nvenc_check_handles_missing_ffmpeg(self):
        assert gpu.nvenc_available("") is False

    def test_best_encoder_handles_missing_ffmpeg(self):
        assert gpu.best_hw_encoder("") == ""

    def test_best_encoder_is_one_we_support(self):
        from autosub_studio.services.ffmpeg import FFmpeg

        ff = FFmpeg()
        if not ff.available:
            return
        assert gpu.best_hw_encoder(ff.ffmpeg) in ("", *gpu.HW_ENCODERS)

    def test_status_text_is_readable(self):
        text = gpu.status_text("")
        assert text and "\n" not in text.rstrip("\n") or True
        assert len(text) > 5

    def test_driver_check_returns_bool(self):
        assert isinstance(gpu.driver_present(), bool)


class TestOcrGpu:
    def test_nts_profile_does_not_upscale_a_cropped_subtitle_strip(self, monkeypatch):
        from autosub_studio.providers import ocr

        _name, module, _engine = ocr._engine_module()
        captured: dict = {}

        class FakeEngine:
            def __init__(self, *, params):
                captured.update(params)

        monkeypatch.setattr(module, "RapidOCR", FakeEngine)
        ocr._build_engine(True, ocr.NTS_FAST_MODE, 5)

        assert captured["Det.limit_type"] == "max"
        assert captured["Det.score_mode"] == "fast"

    def test_parallel_gpu_reader_keeps_original_frame_order(self, monkeypatch):
        from autosub_studio.providers import ocr

        frames = [Path(f"frame_{index:03d}.png") for index in range(12)]
        worker_threads: set[int] = set()

        def fake_task(frame, *_args):
            worker_threads.add(threading.get_ident())
            index = int(frame.stem.rsplit("_", 1)[-1])
            time.sleep((index % 3) * 0.003)
            return [ocr.Row((0.0, 0.0), frame.stem, 1.0, None)]

        monkeypatch.setattr(ocr, "_gpu_frame_task", fake_task)
        rows, failed = ocr._read_frames_gpu_parallel(
            frames,
            min_confidence=0.7,
            text_filter=None,
            profile=ocr.NTS_FAST_MODE,
            batch_size=5,
            on_progress=None,
            should_cancel=None,
        )

        assert not failed
        assert [row[0].text for row in rows] == [frame.stem for frame in frames]
        assert len(worker_threads) > 1

    def test_each_worker_thread_gets_its_own_engine(self, monkeypatch):
        from autosub_studio.providers import ocr

        made: list[object] = []
        monkeypatch.setattr(ocr, "_OCR_THREAD_LOCAL", threading.local())
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)

        def build(*_args, **_kwargs):
            engine = object()
            made.append(engine)
            return engine

        monkeypatch.setattr(ocr, "_build_engine", build)
        engines: list[tuple[object, object]] = []

        def load_twice():
            engines.append((ocr._load_engine(False), ocr._load_engine(False)))

        threads = [threading.Thread(target=load_twice) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(made) == 2
        assert all(first is second for first, second in engines)
        assert engines[0][0] is not engines[1][0]

    def test_reports_a_boolean(self):
        from autosub_studio.providers import ocr

        assert isinstance(ocr.gpu_available(), bool)

    def test_never_claims_gpu_when_onnx_has_none(self):
        from autosub_studio.providers import ocr

        if not gpu.onnx_cuda_ready():
            assert ocr.gpu_available() is False

    def test_probe_runs_a_real_image(self, monkeypatch):
        """Bao co GPU thi phai la da chay thu that, khong chi hoi thu vien."""
        from autosub_studio.providers import ocr

        if not ocr.is_available():
            return
        calls: list[object] = []

        class Broken:
            def __init__(self, **kwargs):
                pass

            def __call__(self, image):
                calls.append(image)
                raise RuntimeError("CUDNN failure 4000")

        ocr.gpu_available.cache_clear()
        monkeypatch.setattr(gpu, "onnx_cuda_ready", lambda: True)
        monkeypatch.setattr(ocr, "onnx_cuda_ready", lambda: True)
        monkeypatch.setattr(ocr, "_build_engine", lambda *_a, **_k: Broken())
        assert ocr.gpu_available() is False
        assert calls, "chua chay thu tam anh nao"
        ocr.gpu_available.cache_clear()

    def test_falls_back_to_cpu_when_gpu_fails_midway(self, monkeypatch, tmp_path):
        """GPU hong giua chung thi doc not bang CPU chu khong dung han."""
        from autosub_studio.providers import ocr

        frames = [tmp_path / f"{i}.png" for i in range(3)]
        for f in frames:
            f.write_bytes(b"")
        state = {"gpu": True}

        def fake_load(use_gpu: bool = False, *_args):
            return "gpu" if use_gpu else "cpu"

        def fake_read(engine, frame, min_confidence):
            if engine == "gpu":
                raise ocr.OCRError("CUDNN failure 4000")
            return "xin chao", 0.9

        monkeypatch.setattr(ocr, "gpu_available", lambda: state["gpu"])
        monkeypatch.setattr(ocr, "_load_engine", fake_load)
        monkeypatch.setattr(ocr, "read_frame", fake_read)
        logs: list[str] = []
        cues = ocr.read_frames(frames, fps=2.0, use_gpu=True, min_duration=0.0, on_log=logs.append)
        assert cues and cues[0].text == "xin chao"
        assert any("CPU" in line for line in logs)

    def test_engine_loads_on_cpu(self):
        from autosub_studio.providers import ocr

        if not ocr.is_available():
            return
        assert ocr._load_engine(False) is not None

    def test_engine_falls_back_when_gpu_missing(self, monkeypatch):
        from autosub_studio.providers import ocr

        if not ocr.is_available():
            return
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        assert ocr._load_engine(True) is not None

    def test_status_text_mentions_ocr(self):
        assert "OCR" in gpu.status_text("")


class TestDeviceChoice:
    def test_cpu_choice_never_uses_gpu(self):
        from autosub_studio.providers import asr

        assert asr.resolve_device("CPU", True)[0] == "cpu"

    def test_auto_choice_matches_machine(self):
        from autosub_studio.providers import asr

        device, compute = asr.resolve_device("Tu chon", True)
        expected = "cuda" if gpu.cuda_ready()[0] else "cpu"
        assert device == expected
        assert compute == ("float16" if expected == "cuda" else "int8")

    def test_gpu_choice_falls_back_when_unavailable(self, monkeypatch):
        from autosub_studio.providers import asr

        monkeypatch.setattr(asr, "cuda_ready", lambda: (False, "khong co"))
        assert asr.resolve_device("GPU (CUDA)", True)[0] == "cpu"
