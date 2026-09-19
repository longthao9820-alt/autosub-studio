"""Kiem thu cho moc thoi gian, dinh dang phu de va thao tac bien tap."""

from __future__ import annotations

import pytest

from autosub_studio.core import editing, formats
from autosub_studio.core.models import Cue, SubtitleDoc
from autosub_studio.core.timecode import (
    TimecodeError,
    format_ass,
    format_srt,
    format_vtt,
    parse_timecode,
)

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:03,500
Xin chao the gioi

2
00:00:04,000 --> 00:00:06,250
Dong thu hai
va dong thu ba
"""

VTT_SAMPLE = """WEBVTT

NOTE ghi chu bi bo qua

00:00:01.000 --> 00:00:02.000 line:90%
Cau mot

00:00:03.000 --> 00:00:04.000
Cau hai
"""

ASS_SAMPLE = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,Nam,0,0,0,,{\\an8}Cau ASS\\Ndong hai
Comment: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,Bi bo qua
"""


class TestTimecode:
    def test_parse_srt_style(self):
        assert parse_timecode("00:00:01,500") == pytest.approx(1.5)

    def test_parse_vtt_style(self):
        assert parse_timecode("01:02:03.250") == pytest.approx(3723.25)

    def test_parse_ass_centiseconds(self):
        assert parse_timecode("0:00:02.05") == pytest.approx(2.05)

    def test_parse_plain_seconds(self):
        assert parse_timecode("12.5") == pytest.approx(12.5)

    def test_reject_garbage(self):
        with pytest.raises(TimecodeError):
            parse_timecode("khong phai thoi gian")

    def test_reject_empty(self):
        with pytest.raises(TimecodeError):
            parse_timecode("")

    def test_reject_minutes_over_59(self):
        with pytest.raises(TimecodeError):
            parse_timecode("00:75:00,000")

    def test_format_round_trip(self):
        assert format_srt(3723.25) == "01:02:03,250"
        assert format_vtt(3723.25) == "01:02:03.250"
        assert format_ass(3723.25) == "1:02:03.25"

    def test_negative_clamped(self):
        assert format_srt(-5) == "00:00:00,000"


class TestFormats:
    def test_parse_srt(self):
        result = formats.parse_srt(SRT_SAMPLE)
        assert len(result.doc.cues) == 2
        assert result.doc.cues[0].text == "Xin chao the gioi"
        assert result.doc.cues[1].text == "Dong thu hai\nva dong thu ba"
        assert result.doc.cues[1].start == pytest.approx(4.0)

    def test_parse_srt_skips_broken_block(self):
        broken = SRT_SAMPLE + "\n3\nkhong co thoi gian\nnoi dung\n"
        result = formats.parse_srt(broken)
        assert len(result.doc.cues) == 2
        assert result.warnings

    def test_parse_vtt_ignores_note_and_settings(self):
        result = formats.parse_vtt(VTT_SAMPLE)
        assert [c.text for c in result.doc.cues] == ["Cau mot", "Cau hai"]
        assert result.doc.cues[0].end == pytest.approx(2.0)

    def test_parse_ass(self):
        result = formats.parse_ass(ASS_SAMPLE)
        assert len(result.doc.cues) == 1
        cue = result.doc.cues[0]
        assert cue.text == "Cau ASS\ndong hai"
        assert cue.speaker == "Nam"

    def test_write_srt_round_trip(self):
        doc = formats.parse_srt(SRT_SAMPLE).doc
        again = formats.parse_srt(formats.write_srt(doc)).doc
        assert [c.text for c in again.cues] == [c.text for c in doc.cues]
        assert again.cues[0].start == pytest.approx(doc.cues[0].start)

    def test_write_vtt_has_header(self):
        doc = formats.parse_srt(SRT_SAMPLE).doc
        assert formats.write_vtt(doc).startswith("WEBVTT")

    def test_write_ass_has_style_and_dialogue(self):
        doc = formats.parse_srt(SRT_SAMPLE).doc
        text = formats.write_ass(doc, font="Arial", font_size=40)
        assert "[V4+ Styles]" in text
        assert text.count("Dialogue:") == 2
        assert "\\N" in text

    def test_write_translation_mode_falls_back(self):
        doc = SubtitleDoc(cues=[Cue(0, 1, "goc", "")])
        assert "goc" in formats.write_srt(doc, text_mode="translation")

    def test_both_mode_joins_lines(self):
        cue = Cue(0, 1, "goc", "dich")
        assert cue.display_text("both") == "goc\ndich"

    def test_save_and_load_file(self, tmp_path):
        doc = formats.parse_srt(SRT_SAMPLE).doc
        target = tmp_path / "out.srt"
        formats.save_subtitle(target, doc)
        loaded = formats.load_subtitle(target)
        assert len(loaded.doc.cues) == 2

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(formats.SubtitleFormatError):
            formats.load_subtitle(tmp_path / "khong-co.srt")

    def test_load_plain_text(self, tmp_path):
        target = tmp_path / "text.txt"
        target.write_text("dong mot\ndong hai\n", encoding="utf-8")
        result = formats.load_subtitle(target)
        assert len(result.doc.cues) == 2
        assert result.warnings

    @pytest.mark.parametrize(
        ("encoding", "subtitle"),
        [("gb18030", "中文字幕测试"), ("big5", "中文字幕測試")],
    )
    def test_loads_legacy_chinese_srt_encodings(self, tmp_path, encoding, subtitle):
        target = tmp_path / f"chinese-{encoding}.srt"
        text = f"1\n00:00:01,000 --> 00:00:03,000\n{subtitle}\n"
        target.write_bytes(text.encode(encoding))

        result = formats.load_subtitle(target)

        assert result.doc.cues[0].text == subtitle


class TestEditing:
    def _doc(self) -> SubtitleDoc:
        return SubtitleDoc(
            cues=[
                Cue(0.0, 2.0, "cau mot day du", "ban dich mot"),
                Cue(2.5, 4.0, "cau hai", "ban dich hai"),
                Cue(5.0, 7.0, "cau ba", ""),
            ]
        )

    def test_split(self):
        doc = self._doc()
        index = editing.split_cue(doc, 0, 1.0)
        assert index == 1
        assert len(doc.cues) == 4
        assert doc.cues[0].end == pytest.approx(1.0)
        assert doc.cues[1].start == pytest.approx(1.0)
        assert doc.cues[0].text and doc.cues[1].text

    def test_split_rejects_edge(self):
        doc = self._doc()
        with pytest.raises(ValueError):
            editing.split_cue(doc, 0, 0.0)

    def test_merge(self):
        doc = self._doc()
        index = editing.merge_cues(doc, [0, 1])
        assert index == 0
        assert len(doc.cues) == 2
        assert doc.cues[0].start == pytest.approx(0.0)
        assert doc.cues[0].end == pytest.approx(4.0)
        assert "cau mot" in doc.cues[0].text and "cau hai" in doc.cues[0].text

    def test_merge_requires_adjacent(self):
        doc = self._doc()
        with pytest.raises(ValueError):
            editing.merge_cues(doc, [0, 2])

    def test_shift_all(self):
        doc = self._doc()
        editing.shift_cues(doc, 1.5)
        assert doc.cues[0].start == pytest.approx(1.5)

    def test_shift_cannot_go_negative(self):
        doc = self._doc()
        editing.shift_cues(doc, -100)
        assert doc.cues[0].start == 0.0

    def test_scale(self):
        doc = self._doc()
        editing.scale_cues(doc, 2.0)
        assert doc.cues[1].start == pytest.approx(5.0)

    def test_remove_overlaps(self):
        doc = SubtitleDoc(cues=[Cue(0, 3, "a"), Cue(2, 4, "b")])
        fixed = editing.remove_overlaps(doc)
        assert fixed == 1
        assert doc.cues[0].end <= doc.cues[1].start

    def test_check_timing_finds_overlap(self):
        doc = SubtitleDoc(cues=[Cue(0, 3, "a"), Cue(2, 4, "b")])
        kinds = {i.kind for i in editing.check_timing(doc)}
        assert "overlap" in kinds

    def test_check_timing_finds_empty_and_gap(self):
        doc = SubtitleDoc(cues=[Cue(0, 2, ""), Cue(30, 32, "b")])
        kinds = {i.kind for i in editing.check_timing(doc)}
        assert "empty" in kinds
        assert "gap" in kinds

    def test_check_timing_flags_long_translation(self):
        doc = SubtitleDoc(cues=[Cue(0, 3, "ngan", "ban dich rat rat rat dai hon nhieu")])
        kinds = {i.kind for i in editing.check_timing(doc)}
        assert "ratio" in kinds

    def test_check_timing_clean_document(self):
        doc = SubtitleDoc(cues=[Cue(0, 2, "mot hai ba"), Cue(2.5, 4.5, "bon nam sau")])
        assert editing.check_timing(doc) == []


class TestModels:
    def test_cps_and_ratio(self):
        cue = Cue(0, 2, "abcd", "abcdefgh")
        assert cue.cps == pytest.approx(2.0)
        assert cue.length_ratio == pytest.approx(2.0)

    def test_index_at(self):
        doc = SubtitleDoc(cues=[Cue(0, 1, "a"), Cue(2, 3, "b")])
        assert doc.index_at(0.5) == 0
        assert doc.index_at(1.5) == -1
        assert doc.index_at(2.5) == 1

    def test_serialisation_round_trip(self):
        doc = SubtitleDoc(cues=[Cue(1, 2, "a", "b", "Nam")], language="en")
        again = SubtitleDoc.from_dict(doc.to_dict())
        assert again.language == "en"
        assert again.cues[0].speaker == "Nam"
