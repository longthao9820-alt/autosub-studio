"""Kiem thu cho khung khoanh vung keo duoc tren video."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, QRectF

from autosub_studio.ui.player import (
    H_BOTTOM,
    H_BOTTOM_RIGHT,
    H_LEFT,
    H_TOP_LEFT,
    MIN_REGION,
    RegionItem,
    VideoPlayer,
    clamp_region,
    default_region,
)

FRAME = QRectF(0, 0, 1920, 1080)


@pytest.fixture
def item(qapp):
    region = RegionItem()
    region.set_bounds(FRAME)
    region.set_region(QRectF(400, 800, 1000, 200))
    return region


class TestDefaultRegion:
    def test_sits_in_lower_part_of_frame(self):
        rect = default_region(1920, 1080)
        assert rect.top() > 1080 * 0.5
        assert rect.bottom() <= 1080

    def test_is_horizontally_centred(self):
        rect = default_region(1920, 1080)
        assert rect.center().x() == pytest.approx(960, abs=1)

    def test_fits_inside_small_frame(self):
        rect = default_region(320, 180)
        assert rect.left() >= 0 and rect.top() >= 0
        assert rect.right() <= 320 and rect.bottom() <= 180

    def test_never_smaller_than_minimum(self):
        rect = default_region(60, 40)
        assert rect.width() >= MIN_REGION or rect.width() == 60


class TestRegionItem:
    def test_keeps_the_region_it_is_given(self, item):
        assert item.region() == QRectF(400, 800, 1000, 200)

    def test_clamps_region_inside_frame(self, item):
        item.set_region(QRectF(-500, -500, 400, 300))
        rect = item.region()
        assert rect.left() >= 0 and rect.top() >= 0

    def test_clamps_oversized_region(self, item):
        item.set_region(QRectF(0, 0, 5000, 5000))
        rect = item.region()
        assert rect.width() <= FRAME.width()
        assert rect.height() <= FRAME.height()

    def test_rejects_region_below_minimum(self, item):
        item.set_region(QRectF(100, 100, 2, 2))
        rect = item.region()
        assert rect.width() >= MIN_REGION
        assert rect.height() >= MIN_REGION

    def test_move_keeps_size(self, item):
        before = item.region().size()
        item._start_rect = item.region()
        item._grab = -2
        item._grab_offset = QPointF(10, 10)
        item.set_region(QRectF(QPointF(600, 400), before))
        assert item.region().size() == before

    def test_resize_from_bottom_right(self, item):
        item._start_rect = item.region()
        new = item._resized(H_BOTTOM_RIGHT, QPointF(1700, 1000))
        assert new.right() == pytest.approx(1700)
        assert new.bottom() == pytest.approx(1000)
        assert new.left() == pytest.approx(400)

    def test_resize_from_top_left(self, item):
        item._start_rect = item.region()
        new = item._resized(H_TOP_LEFT, QPointF(200, 600))
        assert new.left() == pytest.approx(200)
        assert new.top() == pytest.approx(600)
        assert new.right() == pytest.approx(1400)

    def test_resize_cannot_invert_horizontally(self, item):
        item._start_rect = item.region()
        new = item._resized(H_LEFT, QPointF(9000, 0))
        assert new.width() >= MIN_REGION

    def test_resize_cannot_invert_vertically(self, item):
        item._start_rect = item.region()
        new = item._resized(H_BOTTOM, QPointF(0, -9000))
        assert new.height() >= MIN_REGION

    def test_handle_hit_test_finds_corner(self, item):
        item.set_view_scale(1.0)
        assert item._handle_at(item.region().topLeft()) == H_TOP_LEFT

    def test_handle_hit_test_misses_middle(self, item):
        item.set_view_scale(1.0)
        assert item._handle_at(item.region().center()) == -1

    def test_handles_grow_when_video_shown_small(self, item):
        item.set_view_scale(0.25)
        big = item._handle
        item.set_view_scale(1.0)
        assert big > item._handle

    def test_bounds_change_reclamps_region(self, item):
        item.set_bounds(QRectF(0, 0, 640, 360))
        rect = item.region()
        assert rect.right() <= 640 and rect.bottom() <= 360

    def test_handle_size_has_an_upper_limit(self, item):
        item.set_view_scale(0.0001)
        assert item._handle <= 40.0


class TestClampRegion:
    def test_keeps_a_valid_region(self):
        assert clamp_region([100, 100, 400, 200], 1920, 1080) == [100, 100, 400, 200]

    def test_pulls_region_back_inside(self):
        assert clamp_region([1800, 1000, 400, 200], 1920, 1080) == [1520, 880, 400, 200]

    def test_shrinks_oversized_region(self):
        fixed = clamp_region([0, 0, 5000, 5000], 1280, 720)
        assert fixed == [0, 0, 1280, 720]

    def test_rejects_bad_input(self):
        assert clamp_region([1, 2, 3], 1920, 1080) == []
        assert clamp_region([0, 0, 10, 10], 0, 0) == []

    def test_region_saved_for_big_video_fits_small_one(self):
        fixed = clamp_region([192, 799, 1536, 216], 1280, 720)
        assert fixed[0] + fixed[2] <= 1280
        assert fixed[1] + fixed[3] <= 720


class TestPlayerFrameSize:
    def test_scene_has_a_sane_rect_before_any_video(self, qapp):
        player = VideoPlayer()
        rect = player.scene.sceneRect()
        assert rect.width() == 1920 and rect.height() == 1080
        assert rect.left() == 0 and rect.top() == 0

    def test_set_frame_size_updates_scene(self, qapp):
        player = VideoPlayer()
        player.set_frame_size(1280, 720)
        assert player.frame_size == (1280, 720)
        assert player.scene.sceneRect().width() == 1280

    def test_default_region_matches_real_video_size(self, qapp):
        player = VideoPlayer()
        player.set_frame_size(1280, 720)
        x, y, w, h = player.begin_region("ocr", None)
        assert (w, h) == (1024, 144)
        assert x + w <= 1280 and y + h <= 720

    def test_begin_region_reuses_saved_region(self, qapp):
        player = VideoPlayer()
        player.set_frame_size(1280, 720)
        assert player.begin_region("ocr", [10, 20, 300, 100]) == (10, 20, 300, 100)

    def test_end_region_hides_and_returns_value(self, qapp):
        player = VideoPlayer()
        player.set_frame_size(1280, 720)
        player.begin_region("blur", [10, 20, 300, 100])
        assert player.end_region() == (10, 20, 300, 100)
        assert player.region_mode == "none"

    def test_ignores_bad_frame_size(self, qapp):
        player = VideoPlayer()
        player.set_frame_size(0, 0)
        assert player.frame_size == (1920, 1080)
