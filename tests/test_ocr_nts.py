"""Kiem thu cau hinh OCR toc do cao dung model PP-OCRv4 cua NTS."""

from pathlib import Path

import numpy as np

from autosub_studio.providers import ocr
from autosub_studio.providers.ocr_filter import TextFilter


def test_nts_profile_names_are_detected():
    assert ocr.is_nts_profile(ocr.NTS_FAST_MODE)
    assert ocr.is_nts_profile(ocr.NTS_FAST_SERVER)
    assert not ocr.is_nts_profile("PP-OCRv6 Medium (Chuẩn nhất)")


def test_nts_models_are_bundled_in_source_tree():
    paths = ocr._nts_model_paths()
    assert set(paths) == {"det", "cls", "rec"}
    assert all(isinstance(path, Path) and path.stat().st_size > 500_000 for path in paths.values())


def test_nts_engine_uses_mobile_v4_and_skips_second_read(monkeypatch):
    captured = {}

    class Enum:
        MOBILE = "mobile"
        PPOCRV4 = "PP-OCRv4"

    class FakeEngine:
        def __init__(self, *, params):
            captured.update(params)

    class Module:
        ModelType = Enum
        OCRVersion = Enum

    monkeypatch.setattr(ocr, "_engine_module", lambda: ("rapidocr", Module, FakeEngine))
    engine = ocr._build_engine(True, ocr.NTS_FAST_SERVER, 5)

    assert captured["Det.model_type"] == "mobile"
    assert captured["Det.ocr_version"] == "PP-OCRv4"
    assert captured["Rec.rec_batch_num"] == 5
    assert captured["EngineConfig.onnxruntime.use_cuda"] is True
    assert captured["EngineConfig.onnxruntime.intra_op_num_threads"] == 1
    assert captured["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
    assert engine._autosub_fast_profile is True


def test_nts_reads_original_image_instead_of_artificial_mask(monkeypatch, tmp_path):
    image = np.full((80, 300, 3), 127, dtype=np.uint8)
    artificial = np.zeros_like(image)
    seen = []

    class Result:
        boxes = None
        txts = None
        scores = None

    class Engine:
        _autosub_fast_profile = True

        def __call__(self, source):
            seen.append(source)
            return Result()

    monkeypatch.setattr(ocr.ocr_filter, "available", lambda: True)
    monkeypatch.setattr(ocr.ocr_filter, "load_image", lambda _path: image)
    monkeypatch.setattr(ocr.ocr_filter, "color_mask", lambda *_args: artificial[:, :, 0])
    monkeypatch.setattr(
        ocr.ocr_filter,
        "grow_mask",
        lambda _mask: np.ones((80, 300), dtype=np.uint8),
    )
    monkeypatch.setattr(ocr.ocr_filter, "has_text_pixels", lambda _mask: True)
    monkeypatch.setattr(ocr.ocr_filter, "masked_image", lambda *_args: artificial)

    flt = TextFilter(color="#FFFFFF", use_color=True)
    ocr._engine_rows(Engine(), tmp_path / "frame.png", flt)

    assert seen and seen[0] is image


def test_progressive_chinese_caption_is_one_cue():
    rows = [
        [ocr.Row((0.0, 0.0), "没说您", 0.99, None)],
        [ocr.Row((0.0, 0.0), "没说您您坐", 0.98, None)],
        [ocr.Row((0.0, 0.0), "没说您您坐", 0.99, None)],
    ]
    cues = ocr._merge_rows(
        rows,
        fps=5.0,
        start_offset=0.0,
        stamps=None,
        total=3,
        similarity=0.8,
        min_duration=0.0,
        ignore=frozenset(),
        consensus=1,
        on_log=None,
    )

    assert len(cues) == 1
    assert cues[0].text == "没说您您坐"


def test_repeated_chinese_read_beats_one_rare_high_score_mistake():
    correct = "明天我必须在我们部门开除一个人立威"
    wrong = "明天我必须在我们部门开除余一个人立威"
    text, score = ocr._best_caption(
        [(correct, 0.96), (correct, 0.97), (wrong, 0.999)], 0.8
    )

    assert text == correct
    assert score == 0.965


def test_nts_merge_uses_the_read_supported_by_more_frames():
    correct = "但是据我得到消息"
    wrong = "但是据我得刭消息"
    rows = [
        [ocr.Row((0.0, 0.0), correct, 0.96, None)],
        [ocr.Row((0.0, 0.0), wrong, 0.999, None)],
        [ocr.Row((0.0, 0.0), correct, 0.97, None)],
    ]

    cues = ocr._merge_rows(
        rows,
        fps=15.0,
        start_offset=0.0,
        stamps=None,
        total=3,
        similarity=0.8,
        min_duration=0.0,
        ignore=frozenset(),
        consensus=1,
        on_log=None,
    )

    assert len(cues) == 1
    assert cues[0].text == correct
