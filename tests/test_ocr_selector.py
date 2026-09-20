"""Kiem thu cho trinh chon khung hinh va phan doan phu de ocr_selector.

Kiem tra:
- Khong import OCR/RapidOCR/AI (pure local deterministic non-recognition qua kiem tra AST).
- Crop correctness fixture (anh da crop san va helper crop_image).
- Loc phan doan trung lap (duplicate filtering).
- Ranh gioi phan doan (change boundaries, blank ends segment, final frame gap).
- Khung nhieu/near-duplicate giu nguyen phan doan, chu khac nhau (AAA/BBB) tach phan doan.
- Tranh khung mo/fade va chon khung dai dien ro net nhat (transition avoidance).
- Consensus 1, 2, 3 khung hinh (one/two/three consensus).
- Gioi han kich thuoc, chat luong va dung luong payload (max dims/quality/payload).
- Tinh xac dinh cua ID va content_hash (deterministic IDs/hash).
- Phat hien khung trong (blank detection).
- Huy bo tien trinh (cancellation voi should_cancel va CancelToken).
- Quan ly bo nho va dong handle anh (open image count <= small constant).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from autosub_studio.providers import ocr_selector
from autosub_studio.providers.ocr_selector import (
    FrameSample,
    PreparedImage,
    VisualSegment,
    coerce_sample,
    compute_clarity_score,
    compute_dhash,
    compute_phash,
    filter_duplicate_segments,
    hamming_distance,
    is_blank_frame,
    prepare_image,
    select_visual_segments,
)
from autosub_studio.services import media
from autosub_studio.services.ffmpeg import CancelToken

# --------------------------------------------------------------------------- Fixtures


def _create_blank_image(size: tuple[int, int] = (320, 80), color: int = 15) -> Image.Image:
    """Tao anh don sac khong chua chu."""
    return Image.new("RGB", size, (color, color, color))


def _create_text_image(
    text: str,
    size: tuple[int, int] = (320, 80),
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    """Tao anh co chu ro net."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=24)
        draw.text((20, 20), text, fill=fg, font=font)
    except Exception:
        draw.text((20, 20), text, fill=fg)
    return img


def _create_faded_image(
    text: str,
    alpha: float,
    size: tuple[int, int] = (320, 80),
    bg: tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    """Tao anh co chu mo dan hoac fade."""
    base = Image.new("RGB", size, bg)
    full = _create_text_image(text, size=size, bg=bg)
    blended = Image.blend(base, full, alpha)
    if alpha < 0.6:
        blended = blended.filter(ImageFilter.GaussianBlur(radius=1.2))
    return blended


@pytest.fixture
def frame_factory(tmp_path: Path):
    """Factory tao cac file anh tam de kiem thu."""
    def _make(name: str, img: Image.Image) -> Path:
        p = tmp_path / f"{name}.jpg"
        img.save(p, format="JPEG", quality=95)
        return p
    return _make


# --------------------------------------------------------------------------- Tests


class TestInvariantsAndDecoupling:
    """Kiem tra tinh doc lap: khong import RapidOCR, AI hoac bo nhan dien chu."""

    def test_no_ai_or_rapidocr_imports(self):
        # Kiem tra cay AST cua ocr_selector de dam bao khong import bat ky module AI/OCR nao
        code_path = Path(ocr_selector.__file__)
        tree = ast.parse(code_path.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        forbidden = {"ocr", "ocr_ai", "rapidocr", "ai_gateway", "openai"}
        for mod in imported_modules:
            parts = mod.split(".")
            assert not (forbidden & set(parts)), f"Forbidden module imported: {mod}"


class TestCropCorrectness:
    """Kiem tra cat anh: anh da crop san va helper crop_image."""

    def test_already_cropped_image_preserved(self, tmp_path: Path):
        orig = _create_text_image("ALREADY_CROPPED", size=(200, 60))
        path = tmp_path / "cropped.jpg"
        orig.save(path)

        prep = prepare_image(path, max_dimension=1280)
        assert isinstance(prep, PreparedImage)
        assert prep.width == 200
        assert prep.height == 60

    def test_crop_helper_in_media(self, tmp_path: Path):
        full = Image.new("RGB", (640, 360), (30, 30, 30))
        draw = ImageDraw.Draw(full)
        draw.text((50, 300), "SUBTITLE IN LOWER REGION", fill=(255, 255, 255))
        full_path = tmp_path / "full.jpg"
        full.save(full_path)

        region = (40, 280, 400, 70)  # x, y, w, h
        out_crop = tmp_path / "cropped_via_helper.jpg"
        result_path = media.crop_image(full_path, region, out_path=out_crop)

        assert Path(result_path).exists()
        with Image.open(result_path) as cropped:
            assert cropped.size == (400, 70)

    def test_prepare_image_with_crop_region(self, tmp_path: Path):
        full = Image.new("RGB", (500, 500), (0, 0, 0))
        draw = ImageDraw.Draw(full)
        draw.rectangle([100, 100, 300, 200], fill=(255, 255, 255))
        path = tmp_path / "box.jpg"
        full.save(path)

        prep = prepare_image(path, crop_region=(100, 100, 200, 100))
        assert prep.width == 200
        assert prep.height == 100


class TestBlankDetection:
    """Kiem tra phat hien khung trong: don sac, khong co chi tiet chu."""

    def test_blank_frames_detected(self):
        black = _create_blank_image(color=0)
        gray = _create_blank_image(color=128)
        white = _create_blank_image(color=255)

        assert is_blank_frame(black) is True
        assert is_blank_frame(gray) is True
        assert is_blank_frame(white) is True

    def test_text_frame_not_blank(self):
        text_img = _create_text_image("HELLO WORLD")
        assert is_blank_frame(text_img) is False

    def test_blank_ends_prior_segment(self, frame_factory):
        f1 = frame_factory("t1", _create_text_image("TEXT1"))
        f2 = frame_factory("t2", _create_text_image("TEXT1"))
        b1 = frame_factory("b1", _create_blank_image())
        b2 = frame_factory("b2", _create_blank_image())

        samples = [
            (0.0, f1),
            (0.5, f2),
            (1.0, b1),
            (1.5, b2),
        ]
        segments = select_visual_segments(samples)
        assert len(segments) == 1
        assert isinstance(segments[0], VisualSegment)
        assert segments[0].start == pytest.approx(0.0)
        # Blank ends prior segment: moc end la 1.0 (blank stamp)
        assert segments[0].end == pytest.approx(1.0)
        assert segments[0].frame_count == 2


class TestChangeBoundariesAndSegmentation:
    """Kiem tra ranh gioi phan doan: visual change, blank, va final frame gap."""

    def test_change_and_blank_boundaries(self, frame_factory):
        f_a1 = frame_factory("a1", _create_text_image("AAA"))
        f_a2 = frame_factory("a2", _create_text_image("AAA"))
        f_b1 = frame_factory("b1", _create_text_image("BBB"))
        f_b2 = frame_factory("b2", _create_text_image("BBB"))
        f_blank = frame_factory("blank", _create_blank_image())
        f_c1 = frame_factory("c1", _create_text_image("CCC"))

        samples = [
            (0.0, f_a1),
            (0.5, f_a2),
            (1.0, f_b1),  # Visual change tai default threshold
            (1.5, f_b2),
            (2.0, f_blank),  # Blank
            (3.0, f_c1),  # New after blank
        ]

        segments = select_visual_segments(samples, final_frame_gap=0.5)
        assert len(segments) == 3

        # Segment A: 0.0 -> 1.0 (ended by visual change at 1.0)
        assert segments[0].start == pytest.approx(0.0)
        assert segments[0].end == pytest.approx(1.0)

        # Segment B: 1.0 -> 2.0 (ended by blank at 2.0)
        assert segments[1].start == pytest.approx(1.0)
        assert segments[1].end == pytest.approx(2.0)

        # Segment C: 3.0 -> 3.5 (final frame gap applied)
        assert segments[2].start == pytest.approx(3.0)
        assert segments[2].end == pytest.approx(3.5)

    def test_near_duplicate_and_noise_kept_in_same_segment(self, frame_factory):
        base = _create_text_image("NOISY_SAMPLE")
        arr = np.array(base, dtype=np.int16)
        np.random.seed(123)
        # Mo phong nhieu nen video / nen JPEG nhe (+/- 5 gia tri do sang)
        arr_noisy = np.clip(arr + np.random.randint(-5, 6, arr.shape), 0, 255).astype(np.uint8)
        noisy = Image.fromarray(arr_noisy)

        f1 = frame_factory("noise_base", base)
        f2 = frame_factory("noise_variant", noisy)

        samples = [(0.0, f1), (0.5, f2)]
        segments = select_visual_segments(samples)
        assert len(segments) == 1
        assert segments[0].frame_count == 2


class TestTransitionAvoidanceAndClearestRepresentative:
    """Kiem tra tranh khung mo/fade va chon khung ro net nhat."""

    def test_representative_is_clearest_middle_frame(self, frame_factory):
        # Fade-in -> Crisp -> Fade-out
        f_fade_in = frame_factory("fade_in", _create_faded_image("SUBTITLE", alpha=0.25))
        f_crisp = frame_factory("crisp", _create_faded_image("SUBTITLE", alpha=1.0))
        f_fade_out = frame_factory("fade_out", _create_faded_image("SUBTITLE", alpha=0.35))

        samples = [
            (1.0, f_fade_in),
            (1.5, f_crisp),
            (2.0, f_fade_out),
        ]

        segments = select_visual_segments(samples, diff_threshold=15.0)
        assert len(segments) == 1
        seg = segments[0]

        # Khung dai dien phai la khung tai 1.5 (crisp, khong lay fade)
        assert seg.representative.timestamp == pytest.approx(1.5)
        assert seg.representative.score > 0

    def test_compute_clarity_score_ordering(self):
        fade = _create_faded_image("TEXT", alpha=0.25)
        crisp = _create_faded_image("TEXT", alpha=1.0)

        sh_fade, con_fade, sc_fade = compute_clarity_score(fade)
        sh_crisp, con_crisp, sc_crisp = compute_clarity_score(crisp)

        assert sc_crisp > sc_fade
        assert con_crisp > con_fade
        assert sh_crisp >= sh_fade


class TestConsensusModes:
    """Kiem tra che do 1, 2, 3 consensus duoc phan bo cach nhau du xa."""

    def test_one_two_three_consensus(self, frame_factory):
        frames = [
            frame_factory(f"seq_{i}", _create_text_image("CONSENSUS"))
            for i in range(5)
        ]
        samples = [(float(i) * 0.5, frames[i]) for i in range(5)]  # 0.0, 0.5, 1.0, 1.5, 2.0

        # Consensus 1
        seg1 = select_visual_segments(samples, consensus_frames=1)[0]
        assert len(seg1.samples) == 1
        assert seg1.samples[0] == seg1.representative

        # Consensus 2
        seg2 = select_visual_segments(samples, consensus_frames=2, min_separation_seconds=0.4)[0]
        assert len(seg2.samples) == 2
        assert abs(seg2.samples[0].timestamp - seg2.samples[1].timestamp) >= 0.4

        # Consensus 3
        seg3 = select_visual_segments(samples, consensus_frames=3, min_separation_seconds=0.4)[0]
        assert len(seg3.samples) == 3
        stamps = [s.timestamp for s in seg3.samples]
        assert stamps[1] - stamps[0] >= 0.4
        assert stamps[2] - stamps[1] >= 0.4

    def test_accuracy_mode_defaults_to_multiple(self, frame_factory):
        frames = [
            frame_factory(f"acc_{i}", _create_text_image("ACCURACY"))
            for i in range(4)
        ]
        samples = [(float(i) * 0.5, frames[i]) for i in range(4)]

        seg = select_visual_segments(samples, accuracy_mode=True)[0]
        assert len(seg.samples) >= 2


class TestImageConstraintsAndPayload:
    """Kiem tra gioi han kich thuoc, ti le, chat luong JPEG va payload bytes."""

    def test_max_dimension_preserving_aspect(self, tmp_path: Path):
        large = Image.new("RGB", (2000, 1000), (50, 50, 50))
        p = tmp_path / "large.jpg"
        large.save(p)

        prep = prepare_image(p, max_dimension=1000)
        assert prep.width == 1000
        assert prep.height == 500  # Giu ti le 2:1

    def test_jpeg_quality_and_payload_budget(self, tmp_path: Path):
        img = _create_text_image("PAYLOAD_TEST", size=(800, 300))
        p = tmp_path / "test_payload.jpg"
        img.save(p)

        prep = prepare_image(p, max_dimension=800, max_payload_bytes=4000)
        assert len(prep.payload) <= 4000
        assert prep.payload[:2] == b"\xff\xd8"
        assert prep.content_hash == hashlib_sha256(prep.payload)


class TestDeterministicIDsAndHash:
    """Kiem tra tinh xac dinh va nhat quan cua ID va content_hash."""

    def test_deterministic_across_repeated_runs(self, frame_factory):
        f1 = frame_factory("det1", _create_text_image("DETERMINISTIC"))
        f2 = frame_factory("det2", _create_text_image("DETERMINISTIC"))
        samples = [(0.0, f1), (0.5, f2)]

        run1 = select_visual_segments(samples)
        run2 = select_visual_segments(samples)

        assert len(run1) == len(run2) == 1
        assert run1[0].id == run2[0].id
        assert run1[0].content_hash == run2[0].content_hash
        assert run1[0].representative.payload == run2[0].representative.payload


class TestDuplicateFiltering:
    """Kiem tra loc phan doan trung lap dung content_hash."""

    def test_duplicate_filtering_by_content_hash(self, frame_factory):
        f_sub1_a = frame_factory("sub1_a", _create_text_image("SUBTITLE 1"))
        f_sub1_b = frame_factory("sub1_b", _create_text_image("SUBTITLE 1"))
        f_blank = frame_factory("d_blank", _create_blank_image())
        f_sub2 = frame_factory("sub2", _create_text_image("SUBTITLE 2"))

        samples = [
            (0.0, f_sub1_a),
            (0.5, f_sub1_b),
            (1.5, f_blank),
            (2.0, f_sub1_a),
            (2.5, f_sub1_b),
            (3.5, f_blank),
            (4.0, f_sub2),
        ]

        segments = select_visual_segments(samples, final_frame_gap=0.5)
        assert len(segments) == 3

        # Segment 0 va Segment 1 co noi dung giong nhau -> content_hash trung nhau
        assert segments[0].content_hash == segments[1].content_hash
        assert segments[0].content_hash != segments[2].content_hash

        deduped = filter_duplicate_segments(segments)
        assert len(deduped) == 2
        assert deduped[0].id == segments[0].id
        assert deduped[1].id == segments[2].id


class TestCancellation:
    """Kiem tra dung kip thoi khi co yeu cau huy bo."""

    def test_cancellation_with_should_cancel(self, frame_factory):
        frames = [
            frame_factory(f"c_{i}", _create_text_image(f"TEXT_{i}"))
            for i in range(10)
        ]
        samples = [(float(i) * 0.5, frames[i]) for i in range(10)]

        call_count = 0

        def cancel_after_3() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        segments = select_visual_segments(samples, should_cancel=cancel_after_3)
        assert len(segments) < 10

    def test_cancellation_with_cancel_token(self, frame_factory):
        frames = [
            frame_factory(f"ct_{i}", _create_text_image(f"TOKEN_{i}"))
            for i in range(10)
        ]
        samples = [(float(i) * 0.5, frames[i]) for i in range(10)]

        token = CancelToken()
        token.cancel()
        segments = select_visual_segments(samples, token=token)
        assert len(segments) == 0


class TestMemoryBehaviorBoundedOpenImages:
    """Kiem tra bo nho: open image count luon <= small constant (<= 2)."""

    def test_open_image_count_bounded(self, frame_factory, monkeypatch):
        frames = [
            frame_factory(f"mem_{i}", _create_text_image(f"MEM_{i}"))
            for i in range(12)
        ]
        samples = [(float(i) * 0.5, frames[i]) for i in range(12)]

        open_count = 0
        max_open_count = 0
        real_open = Image.open

        def tracked_open(*args, **kwargs):
            nonlocal open_count, max_open_count
            img = real_open(*args, **kwargs)
            open_count += 1
            max_open_count = max(max_open_count, open_count)
            real_close = img.close

            def tracked_close():
                nonlocal open_count
                open_count -= 1
                real_close()

            monkeypatch.setattr(img, "close", tracked_close)
            return img

        monkeypatch.setattr(Image, "open", tracked_open)

        segments = select_visual_segments(samples)
        assert len(segments) > 0

        # Max open count khong duoc vuot qua hang so nho (1 hoac 2)
        assert max_open_count <= 2
        # Cuoi cung tat ca file handles phai duoc dong
        assert open_count == 0


class TestHashingAndMetricsHelpers:
    """Kiem tra cac ham bam thi giac va do sai khac."""

    def test_dhash_and_phash_sensitivity(self):
        img_a = _create_text_image("ALPHA")
        img_b = _create_text_image("OMEGA")

        dh_a = compute_dhash(img_a)
        dh_b = compute_dhash(img_b)
        dist_dhash = hamming_distance(dh_a, dh_b)

        ph_a = compute_phash(img_a)
        ph_b = compute_phash(img_b)
        dist_phash = hamming_distance(ph_a, ph_b)

        assert dist_dhash > 0
        assert dist_phash > 0

    def test_coerce_sample_forms(self, tmp_path: Path):
        p = tmp_path / "f.jpg"
        p.touch()

        s1 = coerce_sample((1.5, p))
        assert s1.timestamp == 1.5
        assert s1.path == p

        s2 = coerce_sample(FrameSample(timestamp=2.0, path=p, index=5))
        assert s2.timestamp == 2.0
        assert s2.index == 5

    def test_media_chunk_sequence_helper(self):
        items = list(range(10))
        chunks = media.chunk_sequence(items, 3)
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def hashlib_sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()
