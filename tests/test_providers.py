"""Kiem thu cho cac trinh cam: dich, chia cau, giong doc, phan tach giong."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosub_studio.core.models import Cue
from autosub_studio.providers import asr, diarize, translate, tts


class TestTranslateHelpers:
    def test_chunk_splits_evenly(self):
        assert translate.chunk(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]

    def test_chunk_minimum_size(self):
        assert len(translate.chunk(["a", "b"], 0)) == 2

    def test_glossary_applies(self):
        out = translate.apply_glossary("toi thich anthropic", {"anthropic": "Anthropic"})
        assert "Anthropic" in out

    def test_glossary_ignores_empty(self):
        assert translate.apply_glossary("giu nguyen", {"": "x"}) == "giu nguyen"

    def test_cache_key_stable_and_distinct(self):
        a = translate.cache_key("xin chao", "en", "vi", "Google")
        b = translate.cache_key("xin chao", "en", "vi", "Google")
        c = translate.cache_key("xin chao", "en", "ja", "Google")
        assert a == b and a != c

    def test_none_provider_returns_input(self):
        request = translate.TranslationRequest(texts=["a", "b"])
        assert translate.translate_batch(translate.PROVIDER_NONE, request) == ["a", "b"]

    def test_empty_input(self):
        assert (
            translate.translate_batch(
                translate.PROVIDER_GOOGLE, translate.TranslationRequest(texts=[])
            )
            == []
        )

    def test_unknown_provider(self):
        with pytest.raises(translate.TranslationError):
            translate.translate_batch("khong co", translate.TranslationRequest(texts=["a"]))

    def test_claude_needs_key(self):
        ready, reason = translate.provider_ready(translate.PROVIDER_CLAUDE, "")
        assert not ready or "khoa API" in reason.lower() or reason

    def test_parse_claude_json(self):
        payload = json.dumps({"translations": [{"id": 1, "text": "hai"}, {"id": 0, "text": "mot"}]})
        assert translate._parse_claude_json(payload, 2) == ["mot", "hai"]

    def test_parse_claude_json_in_code_fence(self):
        payload = "```json\n" + json.dumps({"translations": [{"id": 0, "text": "a"}]}) + "\n```"
        assert translate._parse_claude_json(payload, 1) == ["a"]

    def test_parse_claude_json_missing_entries(self):
        payload = json.dumps({"translations": [{"id": 0, "text": "a"}]})
        assert translate._parse_claude_json(payload, 3) == ["a", "", ""]

    def test_parse_claude_json_invalid(self):
        with pytest.raises(translate.TranslationError):
            translate._parse_claude_json("khong phai json", 1)

    def test_parse_claude_json_wrong_shape(self):
        with pytest.raises(translate.TranslationError):
            translate._parse_claude_json(json.dumps({"khac": []}), 1)


class TestAsrSegmentation:
    def test_short_cue_untouched(self):
        cues = [Cue(0, 2, "cau ngan")]
        out = asr.resegment(cues, max_chars=40, max_lines=2)
        assert out[0].text == "cau ngan"

    def test_long_cue_is_split(self):
        text = " ".join(["tu"] * 60)
        out = asr.resegment([Cue(0, 12, text)], max_chars=20, max_lines=1)
        assert len(out) > 1
        assert all(len(c.text) <= 40 for c in out)

    def test_split_keeps_time_order(self):
        text = " ".join(["tu"] * 40)
        out = asr.resegment([Cue(0, 8, text)], max_chars=20, max_lines=1)
        for a, b in zip(out, out[1:], strict=False):
            assert a.end <= b.start + 1e-6

    def test_wrap_respects_line_limit(self):
        wrapped = asr._wrap("mot hai ba bon nam sau bay", 10, 2)
        assert wrapped.count("\n") <= 1

    def test_merge_short_neighbours(self):
        out = asr.resegment(
            [Cue(0, 1.0, "mot"), Cue(1.0, 1.2, "hai")],
            max_chars=40,
            max_lines=2,
            min_duration=0.5,
        )
        assert len(out) == 1
        assert "mot" in out[0].text and "hai" in out[0].text

    def test_word_timestamps_trim_silence_around_segment(self):
        segment = SimpleNamespace(
            start=10.0,
            end=15.0,
            words=[
                SimpleNamespace(word=" 你好", start=10.6, end=11.2),
                SimpleNamespace(word=" 世界", start=11.3, end=12.1),
            ],
        )

        assert asr._segment_bounds(segment) == pytest.approx((10.6, 12.1))

    def test_chinese_text_splits_without_spaces(self):
        text = "这是第一句话。这是第二句话！这是第三句话，需要继续切分。"

        parts = asr._split_by_length(text, 12)

        assert "".join(parts) == text
        assert all(len(part) <= 12 for part in parts)

    def test_chinese_text_wraps_to_requested_lines(self):
        wrapped = asr._wrap("这是一个没有空格的中文字幕测试句子", 8, 2)
        assert wrapped.count("\n") == 1
        assert all(len(line) <= 8 for line in wrapped.splitlines())

    def test_device_resolution(self):
        assert asr.resolve_device("CPU", True)[0] == "cpu"
        assert asr.resolve_device("GPU (CUDA)", False)[0] == "cuda"
        assert asr.resolve_device("Tu chon", True)[0] == "cuda"

    def test_model_is_local_false_for_missing(self, tmp_path):
        assert asr.model_is_local("small", str(tmp_path)) is False

    def test_model_is_local_true_when_present(self, tmp_path):
        (tmp_path / "model.bin").write_bytes(b"x")
        assert asr.model_is_local("small", str(tmp_path)) is True

    def test_install_hint_mentions_package(self):
        assert "faster-whisper" in asr.install_hint()

    def test_resolve_model_source_uses_given_folder(self, tmp_path):
        (tmp_path / "model.bin").write_bytes(b"x")
        source, local = asr.resolve_model_source("small", str(tmp_path))
        assert local is True
        assert source == str(tmp_path)

    def test_resolve_model_source_finds_nested_folder(self, tmp_path):
        nested = tmp_path / "faster-whisper-small"
        nested.mkdir()
        (nested / "model.bin").write_bytes(b"x")
        source, local = asr.resolve_model_source("small", str(tmp_path))
        assert local is True
        assert source == str(nested)

    def test_resolve_model_source_prefers_bundled(self, tmp_path, monkeypatch):
        bundled = tmp_path / "models" / "faster-whisper-base"
        bundled.mkdir(parents=True)
        (bundled / "model.bin").write_bytes(b"x")
        monkeypatch.setattr(asr, "bundled_dir", lambda name: tmp_path / name)
        source, local = asr.resolve_model_source("base", "")
        assert local is True
        assert Path(source).name == "faster-whisper-base"

    def test_resolve_model_source_falls_back_to_name(self, monkeypatch):
        monkeypatch.setattr(asr, "bundled_dir", lambda name: None)
        monkeypatch.setattr(asr, "model_is_local", lambda size, d="": False)
        source, local = asr.resolve_model_source("medium", "")
        assert source == "medium"
        assert local is False

    def test_bundled_model_dirs_empty_without_bundle(self, monkeypatch):
        monkeypatch.setattr(asr, "bundled_dir", lambda name: None)
        assert asr.bundled_model_dirs() == []


class TestDiarize:
    def _tone(self, path, freq: float, seconds: float = 1.0, rate: int = 16000):
        import math

        frames = bytearray()
        for i in range(int(rate * seconds)):
            value = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            frames += value.to_bytes(2, "little", signed=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(bytes(frames))

    def test_detects_low_pitch(self, tmp_path):
        target = tmp_path / "nam.wav"
        self._tone(target, 110)
        assert diarize.estimate_pitch(target) == pytest.approx(110, abs=12)

    def test_detects_high_pitch(self, tmp_path):
        target = tmp_path / "nu.wav"
        self._tone(target, 220)
        assert diarize.estimate_pitch(target) == pytest.approx(220, abs=20)

    def test_labels(self):
        assert diarize.label_from_pitch(110) == diarize.LABEL_MALE
        assert diarize.label_from_pitch(230) == diarize.LABEL_FEMALE
        assert diarize.label_from_pitch(0) == diarize.LABEL_UNKNOWN

    def test_silence_returns_zero(self, tmp_path):
        target = tmp_path / "im.wav"
        with wave.open(str(target), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        assert diarize.estimate_pitch(target) == 0.0

    def test_missing_file_returns_zero(self, tmp_path):
        assert diarize.estimate_pitch(tmp_path / "khong-co.wav") == 0.0

    def test_assign_speakers(self, tmp_path):
        low = tmp_path / "a.wav"
        high = tmp_path / "b.wav"
        self._tone(low, 110)
        self._tone(high, 230)
        cues = [Cue(0, 1, "a"), Cue(1, 2, "b")]
        count = diarize.assign_speakers(cues, {0: low, 1: high})
        assert count == 2
        assert cues[0].speaker == diarize.LABEL_MALE
        assert cues[1].speaker == diarize.LABEL_FEMALE


class TestTts:
    def test_provider_list_contains_only_local_voice(self):
        assert tts.available_providers() == [tts.PROVIDER_LOCAL]
        assert tts.PROVIDER_LOCAL == "Local Voice"

    def test_provider_ready_checks_piper_runtime(self, monkeypatch):
        monkeypatch.setattr(tts, "piper_runtime_ready", lambda: (True, "Ready"))
        ready, reason = tts.provider_ready(tts.PROVIDER_LOCAL)
        assert ready is True
        assert reason == "Ready"

    def test_list_voices_from_local_catalog(self):
        voices = tts.list_voices(tts.PROVIDER_LOCAL)
        assert "en_US-bryce-medium" in voices
        assert "vi_VN-vais1000-medium" in voices

        vi_voices = tts.list_voices(tts.PROVIDER_LOCAL, language="vi-VN")
        assert "vi_VN-vais1000-medium" in vi_voices
        assert "en_US-bryce-medium" not in vi_voices

    def test_unknown_provider_rejected(self):
        ready, reason = tts.provider_ready("khong co")
        assert not ready and reason

    def test_empty_text_rejected(self, tmp_path):
        with pytest.raises(tts.TTSError):
            tts.synthesize(tts.PROVIDER_LOCAL, "   ", tmp_path / "a.wav")

    def test_dictionary_and_punctuation_are_applied_before_speech(self):
        text = tts.apply_dictionary("OpenAI, xin chao。", "OpenAI=ô-pần ây-ai")
        assert text.startswith("ô-pần ây-ai")
        assert tts.normalize_punctuation(text) == "ô-pần ây-ai, xin chao."

    def test_voice_cache_round_trip(self, tmp_path):
        target = tmp_path / "voices.json"
        tts.save_voice_cache(target, {"piper": ["en_US-bryce-medium"]})
        assert tts.load_voice_cache(target) == {"piper": ["en_US-bryce-medium"]}

    def test_voice_cache_missing_file(self, tmp_path):
        assert tts.load_voice_cache(tmp_path / "khong-co.json") == {}

    def test_synthesis_cache_reuses_the_same_voice(self, tmp_path, monkeypatch):
        calls = []

        def fake_synthesize(_provider, _text, out_path, **_kwargs):
            calls.append(1)
            out = Path(out_path).with_suffix(".wav")
            out.write_bytes(b"voice" * 20)
            return out

        monkeypatch.setattr(tts, "synthesize", fake_synthesize)
        kwargs = {
            "voice": "en_US-bryce-medium",
            "cache_dir": tmp_path / "cache",
        }
        first = tts.synthesize_cached(tts.PROVIDER_LOCAL, "Xin chao", tmp_path / "a.wav", **kwargs)
        second = tts.synthesize_cached(tts.PROVIDER_LOCAL, "Xin chao", tmp_path / "b.wav", **kwargs)

        assert first == second
        assert first.is_file()
        assert first.suffix == ".wav"
        assert len(calls) == 1

    def test_deprecated_constants_safely_accessible_via_getattr(self):
        available = tts.available_providers()
        assert tts.PROVIDER_VOICESTUDIO not in available
        assert tts.PROVIDER_EDGE not in available
        assert tts.PROVIDER_SAPI not in available
