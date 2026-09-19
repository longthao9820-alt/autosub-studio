"""Doc project/config tu SQLite cua NTS AutoSub ma khong sua du lieu goc."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..core.models import Cue, SubtitleDoc
from ..core.timecode import parse_timecode
from ..services.settings import Settings


class NTSImportError(RuntimeError):
    pass


def _connect(path: str | Path) -> sqlite3.Connection:
    db = Path(path)
    if not db.is_file():
        raise NTSImportError(f"Khong tim thay NTS database: {db}")
    try:
        return sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise NTSImportError(f"Khong mo duoc NTS database: {exc}") from exc


def project_names(path: str | Path) -> list[str]:
    try:
        with _connect(path) as connection:
            rows = connection.execute(
                "SELECT ten_cau_hinh FROM configeditsubmodel ORDER BY id DESC"
            ).fetchall()
    except sqlite3.Error as exc:
        raise NTSImportError(f"Day khong phai database project NTS: {exc}") from exc
    return [str(row[0]) for row in rows if "không xài" not in str(row[0]).casefold()]


def load_project(path: str | Path, name: str) -> tuple[str, SubtitleDoc, dict[str, Any]]:
    try:
        with _connect(path) as connection:
            row = connection.execute(
                "SELECT value FROM configeditsubmodel WHERE ten_cau_hinh=?", (name,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise NTSImportError(f"Khong doc duoc project NTS: {exc}") from exc
    if row is None:
        raise NTSImportError(f"Khong co project NTS ten '{name}'.")
    try:
        config = json.loads(row[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise NTSImportError("Cau hinh project NTS bi hong.") from exc
    cues = []
    for item in config.get("data_table") or []:
        if not isinstance(item, list) or len(item) < 4:
            continue
        try:
            left, right = str(item[1]).split("-->", 1)
            start, end = parse_timecode(left.strip()), parse_timecode(right.strip())
        except (ValueError, TypeError):
            continue
        original = str(item[3] or "").strip()
        translation = str(item[4] or "").strip() if len(item) > 4 else ""
        speaker = str(item[2] or "").strip()
        if original and end > start:
            cues.append(Cue(start, end, original, translation, speaker))
    document = SubtitleDoc(
        language="auto",
        target_language="vi",
        cues=cues,
    )
    return str(config.get("video_file") or ""), document, config


def load_global_settings(path: str | Path) -> dict[str, Any]:
    try:
        with _connect(path) as connection:
            row = connection.execute(
                "SELECT value FROM cauhinhtuychonmodel ORDER BY id LIMIT 1"
            ).fetchone()
    except sqlite3.Error:
        return {}
    if row is None:
        return {}
    try:
        value = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def apply_global_settings(settings: Settings, config: dict[str, Any]) -> None:
    """Anh xa cac tham so chung co ten/ngu nghia ro rang cua NTS."""
    settings.use_gpu = bool(config.get("use_gpu", settings.use_gpu))
    settings.max_workers = max(1, min(8, int(config.get("thread_ocr_v2", 2) or 2)))
    settings.ocr_batch_size = max(1, min(64, int(config.get("frame_size_ocr_v2", 5) or 5)))
    settings.ocr_confidence = float(config.get("confidence_ocr_v2", 70) or 70)
    settings.translate_batch = max(1, int(config.get("chunk_split_trans", 8) or 8))
    settings.asr_max_chars = max(5, int(config.get("max_chars_asr", 15) or 15))
    settings.asr_max_duration = max(0.5, float(config.get("max_duration_asr", 2) or 2))
    settings.render_crf = max(0, int(config.get("he_so_crf", 19) or 19))
    settings.render_preset = str(config.get("he_so_preset", "fast") or "fast")
    settings.render_fps = str(config.get("he_so_fps", 30) or 30)
    settings.music_volume = max(0, min(100, int(config.get("am_luong_nhac_nen", 17) or 17)))
    settings.tts_short_threshold_ms = max(
        0, int(config.get("check_sub_time_ngan", 300) or 300)
    )
    settings.tts_pause_period_ms = max(0, int(float(config.get("dau_cham", 0.3)) * 1000))
    settings.tts_pause_comma_ms = max(0, int(float(config.get("dau_phay", 0.2)) * 1000))
    settings.tts_pause_newline_ms = max(
        0, int(float(config.get("dau_xuong_dong", 0.4)) * 1000)
    )
    settings.timeline_workers = max(1, min(8, int(config.get("so_luong_render", 2) or 2)))
    settings.limit_cpu = bool(config.get("chong_full_cpu", settings.limit_cpu))
    settings.fast_concat = bool(config.get("concat_fast", settings.fast_concat))
    settings.smart_cut = bool(config.get("smart_cut", settings.smart_cut))
    settings.play_on_edit = bool(config.get("auto_play", settings.play_on_edit))
    settings.edit_volume = max(0, min(100, int(config.get("volume_phat", 42) or 42)))
    settings.enter_newline = bool(config.get("use_enter_xuong_hang", settings.enter_newline))


def apply_project_settings(settings: Settings, config: dict[str, Any]) -> None:
    """Anh xa cac tuy chon NTS co nghia tuong duong sang Toolsub."""
    settings.tts_end_pause_ms = int(config.get("khoang_nghi_o_cuoi", 250) or 0)
    settings.tts_allow_overlap = bool(config.get("chong_tieng", False))
    settings.tts_store_voice = bool(config.get("save_cache", False))
    settings.tts_cache_enabled = bool(config.get("save_cache", True))
    settings.keep_original_audio = bool(config.get("thuyet_minh", False))
    settings.dub_use_original_video = bool(config.get("use_video_origin", False))
    settings.original_audio_volume = int(config.get("volume_video_goc_thuyet_minh", 20) or 20)
    settings.dub_source = (
        "translation"
        if str(config.get("sub_long_tieng", "origin")) == "translation"
        else "original"
    )
    settings.render_scale = str(config.get("chat_luong_video", "giu nguyen")).replace("|", "x")
    settings.style.font = str(config.get("font_family", settings.style.font))
    size = str(config.get("font_size", settings.style.font_size)).removesuffix("px")
    with contextlib.suppress(ValueError):
        settings.style.font_size = int(float(size))
    settings.style.bold = bool(config.get("font_sub", settings.style.bold))
    settings.style.primary_color = str(config.get("mau_sub", settings.style.primary_color))
    settings.style.outline_color = str(config.get("mau_vien_sub", settings.style.outline_color))
    settings.style.back_color = str(config.get("mau_nen_sub", settings.style.back_color))
