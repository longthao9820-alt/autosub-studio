"""Kiem thu nhap project va cau hinh tu NTS AutoSub."""

from __future__ import annotations

import json
import sqlite3

from autosub_studio.providers import nts_import
from autosub_studio.services.settings import Settings


def _database(path):
    config = {
        "video_file": "D:/video/demo.mp4",
        "data_table": [
            [1, "00:00:00,133 --> 00:00:02,600", "Nam", "Câu gốc", "Bản dịch"],
            [1, "00:00:02,666 --> 00:00:04,200", "", "Câu hai", ""],
        ],
        "khoang_nghi_o_cuoi": 250,
        "chong_tieng": True,
        "save_cache": True,
        "thuyet_minh": True,
        "use_video_origin": False,
        "volume_video_goc_thuyet_minh": 20,
        "sub_long_tieng": "translation",
        "font_family": "Roboto",
        "font_size": "50px",
        "font_sub": True,
        "mau_sub": "#ffff00",
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE configeditsubmodel (id INTEGER, ten_cau_hinh TEXT, value TEXT)"
        )
        connection.execute(
            "INSERT INTO configeditsubmodel VALUES(1, ?, ?)",
            ("demo", json.dumps(config, ensure_ascii=False)),
        )


def test_reads_nts_project_and_maps_settings(tmp_path):
    db = tmp_path / "app.db"
    _database(db)

    assert nts_import.project_names(db) == ["demo"]
    video, document, config = nts_import.load_project(db, "demo")
    settings = Settings()
    nts_import.apply_project_settings(settings, config)

    assert video == "D:/video/demo.mp4"
    assert len(document.cues) == 2
    assert document.cues[0].start == 0.133
    assert document.cues[0].translation == "Bản dịch"
    assert document.cues[0].speaker == "Nam"
    assert settings.tts_end_pause_ms == 250
    assert settings.tts_allow_overlap is True
    assert settings.tts_store_voice is True
    assert settings.tts_cache_enabled is True
    assert settings.keep_original_audio is True
    assert settings.dub_source == "translation"
    assert settings.style.font == "Roboto"
    assert settings.style.font_size == 50
    assert settings.style.primary_color == "#ffff00"


def test_maps_global_nts_settings():
    settings = Settings()

    nts_import.apply_global_settings(
        settings,
        {
            "thread_ocr_v2": 2,
            "frame_size_ocr_v2": 5,
            "confidence_ocr_v2": 70,
            "chunk_split_trans": 8,
            "max_chars_asr": 15,
            "max_duration_asr": 2,
            "he_so_crf": 19,
            "he_so_preset": "fast",
            "he_so_fps": 30,
            "am_luong_nhac_nen": 17,
            "check_sub_time_ngan": 300,
            "dau_cham": 0.3,
            "dau_phay": 0.2,
            "dau_xuong_dong": 0.4,
            "so_luong_render": 2,
        },
    )

    assert settings.max_workers == 2
    assert settings.ocr_batch_size == 5
    assert settings.ocr_confidence == 70
    assert settings.translate_batch == 8
    assert settings.asr_max_chars == 15
    assert settings.asr_max_duration == 2
    assert settings.render_crf == 19
    assert settings.render_preset == "fast"
    assert settings.render_fps == "30"
    assert settings.music_volume == 17
    assert settings.tts_short_threshold_ms == 300
    assert settings.tts_pause_period_ms == 300
    assert settings.tts_pause_comma_ms == 200
    assert settings.tts_pause_newline_ms == 400
    assert settings.timeline_workers == 2
