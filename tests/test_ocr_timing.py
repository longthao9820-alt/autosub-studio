"""Kiem thu do chinh xac cua moc thoi gian va cach chon ban doc trong OCR."""

from __future__ import annotations

from pathlib import Path

from autosub_studio.core.models import Cue
from autosub_studio.pipeline import steps
from autosub_studio.providers import ocr, ocr_filter


class FakeEngine:
    """Bo doc chu gia: moi tep anh mang san noi dung o ngay trong ten tep."""

    def __call__(self, path):
        name = Path(path).stem
        text = name.split("__", 1)[1].replace("_", " ") if "__" in name else ""
        if not text:
            return [], 0.0
        box = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
        return [[box, text, 0.9]], 0.0


def frames(tmp_path: Path, texts: list[str], prefix: str = "f") -> list[Path]:
    out: list[Path] = []
    for i, text in enumerate(texts):
        slug = text.replace(" ", "_")
        p = tmp_path / f"{prefix}{i:03d}__{slug}.png"
        p.write_bytes(b"")
        out.append(p)
    return out


class TestNtsSamplingRate:
    def test_uses_every_second_frame_for_25_fps_video(self):
        assert steps._ocr_frame_rate(15.0, 25.0, True) == 12.5

    def test_keeps_15_fps_for_30_fps_video(self):
        assert steps._ocr_frame_rate(15.0, 30.0, True) == 15.0

    def test_respects_a_lower_user_limit(self):
        assert steps._ocr_frame_rate(5.0, 30.0, True) == 5.0

    def test_other_profiles_keep_the_configured_rate(self):
        assert steps._ocr_frame_rate(15.0, 25.0, False) == 15.0


class TestReadOrdering:
    def test_rapid_similarity_handles_one_wrong_chinese_character(self):
        assert ocr._text_ratio("今天妈妈带我去了游乐场", "今天妈妈带我去游乐场") > 0.9

    def test_lines_come_out_top_to_bottom(self, tmp_path):
        class TwoLines:
            def __call__(self, path):
                low = [[[0.0, 90.0], [9.0, 90.0], [9.0, 99.0], [0.0, 99.0]], "dong duoi", 0.9]
                high = [[[0.0, 10.0], [9.0, 10.0], [9.0, 19.0], [0.0, 19.0]], "dong tren", 0.9]
                return [low, high], 0.0

        text, score = ocr.read_frame(TwoLines(), tmp_path / "a.png", 0.5)
        assert text == "dong tren dong duoi"
        assert score == 0.9

    def test_low_confidence_lines_are_dropped(self, tmp_path):
        class Mixed:
            def __call__(self, path):
                box = [[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [0.0, 9.0]]
                return [[box, "chac chan", 0.95], [box, "doan mo", 0.2]], 0.0

        text, _ = ocr.read_frame(Mixed(), tmp_path / "a.png", 0.5)
        assert text == "chac chan"


class TestSplitBoxes:
    def test_overlapping_boxes_do_not_duplicate_characters(self, tmp_path):
        """Bo do chu cat mot dong lam doi thi khong duoc doc thua ky tu."""

        class Split:
            def __call__(self, path):
                left = [[[348.0, 33.0], [449.0, 33.0], [449.0, 78.0], [348.0, 78.0]], "Hai b", 1.0]
                right = [
                    [[424.0, 33.0], [679.0, 33.0], [679.0, 81.0], [424.0, 81.0]],
                    "ba bon nam",
                    1.0,
                ]
                return [left, right], 0.0

        text, _ = ocr.read_frame(Split(), tmp_path / "a.png", 0.5)
        assert text == "Hai ba bon nam"

    def test_separate_words_keep_their_space(self, tmp_path):
        class TwoWords:
            def __call__(self, path):
                a = [[[10.0, 30.0], [90.0, 30.0], [90.0, 70.0], [10.0, 70.0]], "Xin", 1.0]
                b = [[[120.0, 30.0], [220.0, 30.0], [220.0, 70.0], [120.0, 70.0]], "chao", 1.0]
                return [a, b], 0.0

        text, _ = ocr.read_frame(TwoWords(), tmp_path / "a.png", 0.5)
        assert text == "Xin chao"

    def test_a_box_inside_another_is_dropped(self, tmp_path):
        class Nested:
            def __call__(self, path):
                big = [[[10.0, 10.0], [300.0, 10.0], [300.0, 80.0], [10.0, 80.0]], "Xin chao", 1.0]
                inner = [[[20.0, 20.0], [90.0, 20.0], [90.0, 70.0], [20.0, 70.0]], "Xin", 0.9]
                return [big, inner], 0.0

        text, _ = ocr.read_frame(Nested(), tmp_path / "a.png", 0.5)
        assert text == "Xin chao"

    def test_two_lines_are_not_merged_as_one(self, tmp_path):
        class TwoLines:
            def __call__(self, path):
                top = [[[10.0, 10.0], [200.0, 10.0], [200.0, 50.0], [10.0, 50.0]], "dong tren", 1.0]
                low = [[[10.0, 60.0], [200.0, 60.0], [200.0, 99.0], [10.0, 99.0]], "dong duoi", 1.0]
                return [top, low], 0.0

        text, _ = ocr.read_frame(TwoLines(), tmp_path / "a.png", 0.5)
        assert text == "dong tren dong duoi"


class TestExactStamps:
    def test_second_run_uses_sqlite_frame_cache_without_loading_model(
        self, tmp_path, monkeypatch
    ):
        paths = frames(tmp_path, ["Cau mot", "Cau mot", ""])
        calls = []

        def load(*_args, **_kwargs):
            calls.append(1)
            return FakeEngine()

        monkeypatch.setattr(ocr, "_load_engine", load)
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        cache = tmp_path / "ocr_cache.sqlite3"
        first = ocr.read_frames(
            paths,
            fps=2.0,
            min_duration=0.0,
            cache_path=cache,
            cache_key="video-a",
        )
        assert calls == [1]

        monkeypatch.setattr(
            ocr,
            "_load_engine",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("loaded model")),
        )
        second = ocr.read_frames(
            paths,
            fps=2.0,
            min_duration=0.0,
            cache_path=cache,
            cache_key="video-a",
        )

        assert [(cue.start, cue.end, cue.text) for cue in second] == [
            (cue.start, cue.end, cue.text) for cue in first
        ]

    def test_uses_the_real_frame_times(self, tmp_path, monkeypatch):
        paths = frames(tmp_path, ["", "Cau mot", "Cau mot", ""])
        monkeypatch.setattr(ocr, "_load_engine", lambda *_a, **_k: FakeEngine())
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        cues = ocr.read_frames(
            paths,
            fps=2.0,
            stamps=[0.0, 0.52, 1.04, 1.56],
            min_duration=0.0,
        )
        assert len(cues) == 1
        assert cues[0].start == 0.52
        assert cues[0].end == 1.56

    def test_falls_back_when_stamp_count_is_wrong(self, tmp_path, monkeypatch):
        paths = frames(tmp_path, ["", "Cau mot", "Cau mot", ""])
        monkeypatch.setattr(ocr, "_load_engine", lambda *_a, **_k: FakeEngine())
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        cues = ocr.read_frames(paths, fps=2.0, stamps=[0.0, 0.52], min_duration=0.0)
        assert cues[0].start == 0.5  # quay ve moc uoc luong theo so khung/giay


class TestBestRead:
    def test_keeps_the_most_confident_read(self):
        assert ocr._better_read("Xin chao", 0.95, "Xin chdo", 0.60) is True
        assert ocr._better_read("Xin chdo", 0.60, "Xin chao", 0.95) is False

    def test_prefers_fuller_text_when_confidence_ties(self):
        assert ocr._better_read("Xin chao ban", 0.90, "Xin chao", 0.90) is True
        assert ocr._better_read("Xin chao", 0.90, "Xin chao ban", 0.90) is False

    def test_does_not_prefer_a_longer_but_less_confident_chinese_read(self):
        correct = "明天我必须在我们部门开除一个人立威"
        wrong = "明天我必须在我们部门开除余一个人立威"
        assert ocr._better_read(wrong, 0.971, correct, 0.998) is False

    def test_read_frames_keeps_the_clean_version(self, tmp_path, monkeypatch):
        paths = frames(tmp_path, ["Xin chao ban", "Xin chao ban", "Xin chao ban"])
        scores = {0: 0.4, 1: 0.98, 2: 0.5}
        texts = {0: "Xln chao bcn", 1: "Xin chao ban", 2: "Xin chao bon"}
        order = {p: i for i, p in enumerate(paths)}

        def fake_read(engine, frame, min_confidence):
            i = order[Path(frame)]
            return texts[i], scores[i]

        monkeypatch.setattr(ocr, "_load_engine", lambda *_a, **_k: FakeEngine())
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        monkeypatch.setattr(ocr, "read_frame", fake_read)
        cues = ocr.read_frames(paths, fps=2.0, min_duration=0.0, similarity=0.7)
        assert cues[0].text == "Xin chao ban"


class TestRefineBoundaries:
    def _refiner(self, tmp_path, monkeypatch, timeline):
        """timeline: ham tra ve chu hien tren man hinh tai mot moc giay."""
        monkeypatch.setattr(ocr, "_load_engine", lambda *_a, **_k: FakeEngine())
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        calls: list[tuple[float, float]] = []

        def frames_around(a: float, b: float) -> list[tuple[float, Path]]:
            calls.append((a, b))
            span = max(0.05, b - a)
            out: list[tuple[float, Path]] = []
            for i in range(8):
                stamp = a + span * i / 8
                slug = (timeline(stamp) or "").replace(" ", "_")
                p = tmp_path / f"w{len(calls)}_{i}__{slug}.png"
                p.write_bytes(b"")
                out.append((stamp, p))
            return out

        return frames_around, calls

    def test_moves_start_and_end_closer_to_the_truth(self, tmp_path, monkeypatch):
        # Chu that su hien tu 1,30s den 2,70s.
        def timeline(t: float) -> str:
            return "Cau mot" if 1.30 <= t < 2.70 else ""

        frames_around, _ = self._refiner(tmp_path, monkeypatch, timeline)
        # Do tho o 2 khung/giay chi bat duoc 1,5s -> 3,0s.
        cues = [Cue(1.5, 3.0, "Cau mot")]
        out = ocr.refine_boundaries(
            cues, frames_around=frames_around, coarse_step=0.5, similarity=0.8
        )
        assert 1.25 <= out[0].start <= 1.45
        assert 2.65 <= out[0].end <= 2.85
        # Sai so phai nho hon nhieu so voi truoc khi do lai.
        assert abs(out[0].start - 1.30) < abs(1.5 - 1.30)
        assert abs(out[0].end - 2.70) < abs(3.0 - 2.70)

    def test_leaves_cue_alone_when_nothing_matches(self, tmp_path, monkeypatch):
        frames_around, _ = self._refiner(tmp_path, monkeypatch, lambda _t: "")
        cues = [Cue(1.5, 3.0, "Cau mot")]
        out = ocr.refine_boundaries(
            cues, frames_around=frames_around, coarse_step=0.5, similarity=0.8
        )
        assert out[0].start == 1.5
        assert out[0].end == 3.0

    def test_never_overlaps_the_previous_cue(self, tmp_path, monkeypatch):
        def timeline(t: float) -> str:
            return "Cau hai"  # luon khop, ep buoc do lai keo start ve rat som

        frames_around, _ = self._refiner(tmp_path, monkeypatch, timeline)
        cues = [Cue(0.0, 2.0, "Cau mot"), Cue(2.0, 4.0, "Cau hai")]
        out = ocr.refine_boundaries(
            cues, frames_around=frames_around, coarse_step=0.5, similarity=0.8
        )
        assert out[1].start >= out[0].end
        assert out[1].start < out[1].end

    def test_skips_the_first_cue_start_at_zero(self, tmp_path, monkeypatch):
        frames_around, calls = self._refiner(tmp_path, monkeypatch, lambda _t: "Cau mot")
        ocr.refine_boundaries(
            [Cue(0.0, 2.0, "Cau mot")],
            frames_around=frames_around,
            coarse_step=0.5,
            similarity=0.8,
        )
        assert len(calls) == 1  # chi do lai moc ket thuc

    def test_empty_list_is_safe(self, tmp_path, monkeypatch):
        frames_around, _ = self._refiner(tmp_path, monkeypatch, lambda _t: "")
        assert ocr.refine_boundaries([], frames_around=frames_around, coarse_step=0.5) == []

    def test_stops_when_cancelled(self, tmp_path, monkeypatch):
        frames_around, calls = self._refiner(tmp_path, monkeypatch, lambda _t: "Cau mot")
        cues = [Cue(i * 2.0, i * 2.0 + 1.5, "Cau mot") for i in range(1, 5)]
        ocr.refine_boundaries(
            cues,
            frames_around=frames_around,
            coarse_step=0.5,
            similarity=0.8,
            should_cancel=lambda: len(calls) >= 2,
        )
        assert len(calls) <= 4


class TestStaticOverlay:
    """Logo, watermark, chu dan co dinh trong vung khoanh phai bi bo."""

    def rows(self, items: list[tuple[str, float, float]]) -> list[ocr.Row]:
        out: list[ocr.Row] = []
        for text, x, y in items:
            rect = (x, y, x + 10.0 * len(text), y + 30.0)
            out.append(ocr.Row((y, x), text, 0.95, rect))
        return out

    def test_finds_a_logo_that_never_moves(self):
        per_frame = [
            self.rows([("KENH ABC", 900.0, 10.0), (f"cau thoai so {i}", 100.0, 80.0)])
            for i in range(12)
        ]
        found = ocr.static_texts(per_frame)
        assert "kenh abc" in found
        assert not any("cau thoai" in text for text in found)

    def test_keeps_dialogue_even_when_it_sits_in_one_place(self):
        per_frame = [self.rows([(f"cau thoai khac nhau so {i}", 100.0, 80.0)]) for i in range(12)]
        assert ocr.static_texts(per_frame) == frozenset()

    def test_ignores_short_videos(self):
        per_frame = [self.rows([("KENH ABC", 900.0, 10.0)]) for _ in range(4)]
        assert ocr.static_texts(per_frame) == frozenset()

    def test_text_that_moves_around_is_not_a_logo(self):
        per_frame = [self.rows([("chay ngang man hinh", 100.0 + i * 40, 10.0)]) for i in range(12)]
        assert ocr.static_texts(per_frame) == frozenset()

    def test_logo_is_dropped_from_the_final_cues(self, tmp_path, monkeypatch):
        """Doc ca video: cau thoai con lai phai sach, khong con ten kenh."""

        class WithLogo:
            def __init__(self) -> None:
                self.turn = -1

            def __call__(self, image):
                self.turn += 1
                line = "cau mot" if self.turn < 6 else "cau hai"
                logo = [
                    [[900.0, 8.0], [1100.0, 8.0], [1100.0, 34.0], [900.0, 34.0]],
                    "KENH ABC",
                    0.97,
                ]
                talk = [
                    [[120.0, 80.0], [600.0, 80.0], [600.0, 118.0], [120.0, 118.0]],
                    line,
                    0.97,
                ]
                return [logo, talk], 0.0

        frames = []
        for i in range(12):
            p = tmp_path / f"f{i:03d}.png"
            p.write_bytes(b"")
            frames.append(p)
        monkeypatch.setattr(ocr, "_load_engine", lambda *_a, **_k: WithLogo())
        monkeypatch.setattr(ocr, "gpu_available", lambda: False)
        flt = ocr_filter.TextFilter(
            use_color=False, drop_static=True, min_height=20, max_height=60
        )
        cues = ocr.read_frames(frames, fps=2.0, min_duration=0.0, text_filter=flt)
        assert [c.text for c in cues] == ["cau mot", "cau hai"]
        # Nho lai ten kenh de buoc do lai moc thoi gian cung bo dung dong do.
        assert "kenh abc" in flt.ignore_texts

    def test_known_lines_are_dropped_even_when_read_slightly_differently(self):
        assert ocr.is_ignored("TIN TUC 24H", ["tintuc24h"]) is True
        assert ocr.is_ignored("cau thoai binh thuong", ["tintuc24h"]) is False
        assert ocr.is_ignored("", ["tintuc24h"]) is False


class TestFlicker:
    def test_exact_caption_split_by_short_detector_gap_is_rejoined(self):
        cues = [
            Cue(1.0, 2.0, "cau khong doi"),
            Cue(2.1, 3.0, "cau khong doi"),
            Cue(3.4, 4.0, "cau khong doi"),
        ]

        out = ocr._merge_exact_repeats(cues, max_gap=0.2)

        assert [(cue.start, cue.end, cue.text) for cue in out] == [
            (1.0, 3.0, "cau khong doi"),
            (3.4, 4.0, "cau khong doi"),
        ]

    def test_similar_captions_are_not_rejoined(self):
        cues = [Cue(1.0, 2.0, "那强哥在呢"), Cue(2.1, 3.0, "强哥在呢")]
        assert ocr._merge_exact_repeats(cues, max_gap=0.2) == cues

    def test_consensus_replaces_one_bad_chinese_read(self):
        reads = [
            ("欢迎来到中国", 0.93),
            ("欢迎来到中囯", 0.66),
            ("欢迎来到中国", 0.95),
        ]
        out = ocr._stabilize_reads(reads, 0.80, 2)
        assert [text for text, _score in out] == ["欢迎来到中国"] * 3

    def test_consensus_does_not_fill_empty_boundary(self):
        reads = [("", 0.0), ("你好世界", 0.9), ("你好世界", 0.95)]
        out = ocr._stabilize_reads(reads, 0.80, 2)
        assert out[0] == ("", 0.0)

    def test_chinese_boxes_are_joined_left_to_right_without_spaces(self):
        rows = [
            ocr.Row((37.0, 745.0), "一起学习。", 0.99, (745.0, 37.0, 1082.0, 159.0)),
            ocr.Row((39.0, 581.0), "我们-", 0.84, (581.0, 39.0, 780.0, 158.0)),
            ocr.Row((33.0, 72.0), "欢迎来到中国，", 0.99, (72.0, 33.0, 579.0, 161.0)),
        ]
        text, _score = ocr.join_rows(rows)
        assert text == "欢迎来到中国，我们一起学习。"

    def test_generated_join_hyphen_is_removed_without_box_overlap(self):
        rows = [
            ocr.Row((10.0, 10.0), "我们-", 0.99, (10.0, 10.0, 80.0, 40.0)),
            ocr.Row((10.0, 82.0), "一起学习", 0.99, (82.0, 10.0, 180.0, 40.0)),
        ]

        text, _score = ocr.join_rows(rows)

        assert text == "我们一起学习"

    def test_rereading_a_split_line_does_not_disable_detection(self):
        """RapidOCR v3 must not receive sticky recognition-only flags."""
        import numpy as np

        class Result:
            boxes = [
                [[0, 0], [100, 0], [100, 30], [0, 30]],
                [[101, 0], [180, 0], [180, 30], [101, 30]],
            ]
            txts = ["欢迎来到", "中国"]
            scores = [0.99, 0.98]

        class StickyEngine:
            def __init__(self):
                self.detection_enabled = True
                self.kwargs = []

            def __call__(self, _source, **kwargs):
                self.kwargs.append(kwargs)
                if "use_det" in kwargs:
                    self.detection_enabled = bool(kwargs["use_det"])
                return Result()

        engine = StickyEngine()
        rows = [
            [[[0, 0], [90, 0], [90, 30], [0, 30]], "欢迎来到", 0.98],
            [[[85, 0], [180, 0], [180, 30], [85, 30]], "中国", 0.97],
        ]
        image = np.zeros((40, 200, 3), dtype=np.uint8)

        reread = ocr._reread_subtitle_lines(engine, rows, image)

        assert reread[0][1] == "欢迎来到中国"
        assert engine.detection_enabled is True
        assert engine.kwargs == [{}]

    def test_one_odd_frame_between_two_alike_frames_is_fixed(self):
        reads = [
            ("xin chao cac ban", 0.9),
            ("KENH ABC 24H", 0.5),
            ("xin chao cac ban", 0.9),
        ]
        out = ocr._drop_flicker(reads, 0.82)
        assert [t for t, _s in out] == ["xin chao cac ban"] * 3

    def test_a_real_change_is_left_alone(self):
        reads = [("cau mot", 0.9), ("cau hai", 0.9), ("cau ba", 0.9)]
        assert ocr._drop_flicker(reads, 0.82) == reads

    def test_a_short_line_between_two_empty_frames_is_kept(self):
        reads = [("", 0.0), ("cau rat ngan", 0.9), ("", 0.0)]
        assert ocr._drop_flicker(reads, 0.82) == reads

    def test_a_real_blank_between_related_chinese_lines_is_kept(self):
        reads = [("那强哥在呢", 0.99), ("", 0.0), ("强哥在呢", 0.99)]
        assert ocr._drop_flicker(reads, 0.82) == reads

    def test_a_scrap_between_two_different_lines_is_dropped(self):
        reads = [
            ("Sau do la phan hoi dap cuoi buoi", 0.9),
            ("C.15", 0.6),
            ("Cam on cac ban da theo doi", 0.9),
        ]
        out = ocr._drop_flicker(reads, 0.82)
        assert [t for t, _s in out] == [
            "Sau do la phan hoi dap cuoi buoi",
            "",
            "Cam on cac ban da theo doi",
        ]

    def test_a_full_line_between_two_others_is_kept(self):
        reads = [
            ("cau thoai dau tien khá dai", 0.9),
            ("cau thoai thu hai cung dai", 0.9),
            ("cau thoai thu ba cung the", 0.9),
        ]
        assert ocr._drop_flicker(reads, 0.82) == reads
