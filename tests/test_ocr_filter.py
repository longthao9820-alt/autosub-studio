"""Kiem thu bo loc chu theo mau va chieu cao truoc khi doc chu tren hinh."""

from __future__ import annotations

from pathlib import Path

import pytest

from autosub_studio.providers import ocr, ocr_filter
from autosub_studio.providers.ocr_filter import TextFilter

np = pytest.importorskip("numpy")


def _picture(size: tuple[int, int] = (60, 200)) -> object:
    """Anh nen xanh dam, co mot vach chu trang va mot vach chu vang."""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = (40, 60, 30)  # BGR: xanh dam
    image[20:32, 10:80] = (255, 255, 255)  # chu trang
    image[20:32, 120:190] = (0, 220, 255)  # chu vang
    return image


class TestColorText:
    def test_reads_hex_and_triplet(self):
        assert ocr_filter.parse_color("#FFFFFF") == (255, 255, 255)
        assert ocr_filter.parse_color("#fff") == (255, 255, 255)
        assert ocr_filter.parse_color("255, 222, 0") == (255, 222, 0)

    def test_refuses_nonsense(self):
        for text in ("", "  ", "khong phai mau", "#12345", "300,0,0", "#GGGGGG"):
            assert ocr_filter.parse_color(text) is None

    def test_writes_hex_back(self):
        assert ocr_filter.format_color((255, 222, 0)) == "#FFDE00"
        assert ocr_filter.format_color(None) == ""

    def test_names_the_common_colors(self):
        assert ocr_filter.color_name((255, 255, 255)) == "trang"
        assert ocr_filter.color_name((250, 220, 10)) == "vang"
        assert ocr_filter.color_name((20, 20, 20)) == "den"
        assert ocr_filter.color_name(None) == "chua ro"


class TestFilterRules:
    def test_empty_color_does_not_trigger_automatic_color_measurement(self):
        flt = TextFilter(color="", min_height=25, max_height=200)
        assert flt.rgb is None
        assert flt.needs_probe is False

    def test_missing_height_can_still_be_measured(self):
        assert TextFilter(min_height=0, max_height=0).needs_probe is True

    def test_turning_color_off_stops_the_color_check(self):
        flt = TextFilter(color="#FFFFFF", use_color=False)
        assert flt.rgb is None
        assert flt.touches_image is False

    def test_height_limits(self):
        flt = TextFilter(min_height=20, max_height=60)
        assert flt.height_ok(40) is True
        assert flt.height_ok(19) is False
        assert flt.height_ok(61) is False

    def test_no_limit_when_zero(self):
        flt = TextFilter(min_height=0, max_height=0)
        assert flt.height_ok(2) is True
        assert flt.height_ok(2000) is True

    def test_full_block_of_color_is_not_text(self):
        flt = TextFilter()
        assert flt.fill_ok(0.2) is True
        assert flt.fill_ok(0.0) is False
        assert flt.fill_ok(0.99) is False

    def test_description_is_readable(self):
        text = TextFilter(
            color="#FFFFFF", use_color=True, min_height=20, max_height=60
        ).describe()
        assert "#FFFFFF" in text and "trang" in text and "20..60" in text


class TestColorMask:
    def test_keeps_only_the_wanted_color(self):
        image = _picture()
        mask = ocr_filter.color_mask(image, (255, 255, 255), 12.0)
        assert mask[25, 40] == 1  # cho co chu trang
        assert mask[25, 150] == 0  # cho co chu vang
        assert mask[5, 5] == 0  # nen

    def test_wide_tolerance_starts_taking_other_colors(self):
        image = _picture()
        narrow = ocr_filter.color_mask(image, (255, 255, 255), 12.0).sum()
        wide = ocr_filter.color_mask(image, (255, 255, 255), 70.0).sum()
        assert wide > narrow

    def test_masked_picture_is_black_text_on_white(self):
        image = _picture()
        mask = ocr_filter.color_mask(image, (255, 255, 255), 12.0)
        out = ocr_filter.masked_image(image, mask)
        assert tuple(out[25, 40]) == (0, 0, 0)  # net chu thanh den
        assert tuple(out[25, 150]) == (255, 255, 255)  # chu khac mau bien mat
        assert tuple(out[5, 5]) == (255, 255, 255)

    def test_grow_mask_thickens_the_strokes(self):
        image = _picture()
        mask = ocr_filter.color_mask(image, (255, 255, 255), 12.0)
        assert ocr_filter.grow_mask(mask, 3).sum() > mask.sum()

    def test_fill_ratio_of_a_box(self):
        image = _picture()
        mask = ocr_filter.color_mask(image, (255, 255, 255), 12.0)
        assert ocr_filter.rect_fill(mask, (10, 20, 79, 31)) == pytest.approx(1.0, abs=0.05)
        assert ocr_filter.rect_fill(mask, (120, 20, 189, 31)) == 0.0
        assert ocr_filter.rect_fill(mask, None) == 1.0


class TestBrightness:
    def test_brightening_raises_every_pixel(self):
        image = _picture()
        out = ocr_filter.adjust(image, 30, 0)
        assert int(out[5, 5, 0]) > int(image[5, 5, 0])

    def test_zero_change_returns_the_same_picture(self):
        image = _picture()
        assert ocr_filter.adjust(image, 0, 0) is image


class TestMeasuring:
    def test_finds_the_color_inside_the_letters(self):
        """Chu trang co vien den tren nen mau: phai do ra mau trang."""
        image = np.zeros((40, 200, 3), dtype=np.uint8)
        image[:, :] = (120, 40, 180)  # nen tim
        for x in range(8, 190, 16):  # cac chu hinh chu 'H', co vien den bao quanh
            image[6:34, x - 3 : x + 15] = (0, 0, 0)
            image[8:32, x : x + 2] = (255, 255, 255)
            image[8:32, x + 10 : x + 12] = (255, 255, 255)
            image[19:21, x : x + 12] = (255, 255, 255)
        found = ocr_filter.fill_color(image, (5.0, 4.0, 195.0, 36.0))
        assert found is not None
        assert min(found) >= 200

    def test_ignores_boxes_too_small_to_judge(self):
        image = _picture()
        assert ocr_filter.fill_color(image, (0.0, 0.0, 5.0, 4.0)) is None

    def test_picks_the_color_with_the_most_area(self):
        samples = [((255, 255, 255), 100.0), ((252, 250, 250), 80.0), ((255, 220, 0), 30.0)]
        found = ocr_filter.pick_color(samples)
        assert found is not None and min(found) >= 200

    def test_height_range_wraps_the_middle_value(self):
        low, high = ocr_filter.height_range([40, 42, 44, 40, 41])
        assert low < 41 < high
        assert low > 0

    def test_height_range_of_nothing_is_no_limit(self):
        assert ocr_filter.height_range([]) == (0.0, 0.0)


class TestFilterInsideOcr:
    """Bo loc phai lam viec trong luong doc chu, khong chi rieng le."""

    class Engine:
        """Bo doc chu gia: mot dong chu cao 40 px va mot dong chu cao 12 px."""

        def __call__(self, image):
            tall = [[[10.0, 10.0], [300.0, 10.0], [300.0, 50.0], [10.0, 50.0]], "cau thoai", 0.95]
            small = [[[10.0, 60.0], [120.0, 60.0], [120.0, 72.0], [10.0, 72.0]], "logo abc", 0.95]
            return [tall, small], 0.0

    def test_height_filter_drops_the_small_line(self, tmp_path: Path):
        frame = tmp_path / "a.png"
        frame.write_bytes(b"")
        flt = TextFilter(use_color=False, min_height=20, max_height=80)
        text, _score = ocr.read_frame(self.Engine(), frame, 0.5, flt)
        assert text == "cau thoai"

    def test_without_the_filter_both_lines_come_out(self, tmp_path: Path):
        frame = tmp_path / "a.png"
        frame.write_bytes(b"")
        text, _score = ocr.read_frame(self.Engine(), frame, 0.5)
        assert "logo abc" in text

    def test_probe_result_fills_the_missing_numbers(self):
        flt = TextFilter(color="", use_color=True, min_height=0, max_height=0)
        ocr.apply_probe(flt, ocr.Probe("#FFDE00", 25.0, 70.0, 40.0, 12))
        assert flt.color == "#FFDE00"
        assert (flt.min_height, flt.max_height) == (25.0, 70.0)

    def test_probe_never_overwrites_what_the_user_chose(self):
        flt = TextFilter(color="#FFFFFF", min_height=30, max_height=90)
        ocr.apply_probe(flt, ocr.Probe("#FFDE00", 10.0, 20.0, 15.0, 9))
        assert flt.color == "#FFFFFF"
        assert (flt.min_height, flt.max_height) == (30, 90)

    def test_a_failed_probe_changes_nothing(self):
        flt = TextFilter(color="", min_height=0, max_height=0)
        ocr.apply_probe(flt, ocr.Probe("", 0.0, 0.0, 0.0, 0))
        assert flt.color == ""


class TestWithRealEngine:
    """Chay that bo doc chu tren mot khung hinh dung: chu khac mau phai bi bo."""

    FONT = Path("C:/Windows/Fonts/arialbd.ttf")

    def _frame(self, path: Path) -> bool:
        """Ve mot khung hinh: phu de trang o duoi, chu vang va chu do lam nhieu."""
        pil = pytest.importorskip("PIL")
        if not self.FONT.is_file():
            return False
        from PIL import Image, ImageDraw, ImageFont

        assert pil is not None
        image = Image.new("RGB", (900, 260), (70, 90, 60))
        draw = ImageDraw.Draw(image)
        big = ImageFont.truetype(str(self.FONT), 44)
        small = ImageFont.truetype(str(self.FONT), 40)
        draw.text(
            (450, 200),
            "xin chao cac ban",
            font=big,
            fill=(255, 255, 255),
            anchor="mm",
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        draw.text(
            (450, 60),
            "kenh quang cao",
            font=small,
            fill=(255, 215, 0),
            anchor="mm",
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        image.save(path)
        return True

    def test_only_the_white_line_survives(self, tmp_path: Path):
        if not ocr.is_available() or not ocr_filter.available():
            return
        frame = tmp_path / "khung.png"
        if not self._frame(frame):
            return
        engine = ocr._load_engine(False)
        plain, _score = ocr.read_frame(engine, frame, 0.5)
        assert "quang" in plain.casefold()  # khong loc thi bat ca chu vang

        flt = TextFilter(color="#FFFFFF", tolerance=12.0, use_color=True)
        clean, _score = ocr.read_frame(engine, frame, 0.5, flt)
        assert "quang" not in clean.casefold()
        assert "xin chao" in clean.casefold()

    def test_measuring_a_frame_finds_white_text_and_its_height(self, tmp_path: Path):
        if not ocr.is_available() or not ocr_filter.available():
            return
        frame = tmp_path / "khung.png"
        if not self._frame(frame):
            return
        probe = ocr.probe_frames([frame], samples=4)
        assert probe.found
        rgb = ocr_filter.parse_color(probe.color)
        assert rgb is not None
        assert probe.min_height < probe.height < probe.max_height
        assert probe.height > 20


class TestSkippingEmptyFrames:
    def test_frame_without_the_right_color_is_skipped(self):
        empty = np.zeros((60, 200), dtype=np.uint8)
        assert ocr_filter.has_text_pixels(empty) is False
        empty[10:20, 10:30] = 1
        assert ocr_filter.has_text_pixels(empty) is True

    def test_no_mask_means_do_not_skip(self):
        assert ocr_filter.has_text_pixels(None) is True

    def test_engine_is_not_called_for_a_frame_with_no_matching_color(self, tmp_path: Path):
        """Khung hinh khong co chu dung mau thi khong goi bo doc chu."""
        import cv2

        frame = tmp_path / "trong.png"
        image = np.zeros((80, 240, 3), dtype=np.uint8)
        image[:, :] = (30, 60, 40)  # chi co nen, khong co chu trang
        cv2.imwrite(str(frame), image)
        calls: list[object] = []

        def engine(source):
            calls.append(source)
            return [], 0.0

        flt = TextFilter(color="#FFFFFF", tolerance=10.0, use_color=True)
        text, score = ocr.read_frame(engine, frame, 0.5, flt)
        assert (text, score) == ("", 0.0)
        assert calls == []
