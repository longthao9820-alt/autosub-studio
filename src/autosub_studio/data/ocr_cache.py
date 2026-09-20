"""Quan ly bo nho dem (cache) va diem luu (checkpoint) cua OCR noi bo va OCR AI.

Luu tru tren SQLite o che do WAL de ghi nhanh, an toan da luong voi busy timeout.
Khong bao gio luu khoa API vao database hay cache key.
Ho tro doc/ghi ca dinh dang cache cu de tuong thich nguoc hoan toan.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.ocr_common import Rect, Row

_DB_LOCK = threading.RLock()


@dataclass
class SegmentRecord:
    """Ban ghi checkpoint cho mot doan phu de thi giac."""

    segment_id: str
    start: float
    end: float
    content_hash: str
    text: str
    confidence: float = 1.0
    uncertain: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def hash_endpoint(endpoint: str) -> str:
    """Bam dinh danh endpoint de luu vao cache ma khong bao gio dinh toi API key."""
    clean = endpoint.strip().rstrip("/")
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def file_identity(path: Path | str) -> list[Any]:
    """Dinh danh tep gom duong dan tuyet doi, kich thuoc va thoi gian sua."""
    p = Path(path)
    try:
        stat = p.stat()
        return [str(p.resolve()), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(p), 0, 0]


def make_ai_ocr_cache_key(
    *,
    video: Path | str,
    region: Sequence[int],
    fps: float,
    endpoint: str,
    model_alias: str,
    actual_model: str,
    thinking: str = "",
    prompt_version: str = "v1",
    custom_prompt: str = "",
    preprocessing_version: str = "v1",
    diff_threshold: float = 4.0,
    image_quality: int = 88,
    consensus_mode: str = "disabled",
    consensus_frames: int = 1,
    engine: str = "ai_gateway",
) -> str:
    """Tao cache key xac dinh cho OCR AI tu cac tham so co ban.

    Bao gom dinh danh endpoint da bam (khong chua key), model, prompt, region,
    video va cac thong so lay mau. Bat ky thay doi nao cung lam doi cache key.
    """
    payload = {
        "engine": str(engine),
        "endpoint_hash": hash_endpoint(endpoint),
        "model_alias": str(model_alias),
        "actual_model": str(actual_model),
        "thinking": str(thinking).strip().lower(),
        "prompt_version": str(prompt_version),
        "custom_prompt": str(custom_prompt).strip(),
        "preprocessing_version": str(preprocessing_version),
        "diff_threshold": round(float(diff_threshold), 4),
        "image_quality": int(image_quality),
        "consensus_mode": str(consensus_mode),
        "consensus_frames": int(consensus_frames),
        "fps": round(float(fps), 6),
        "region": list(region),
        "video": file_identity(video),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_local_ocr_cache_key(
    video: Path | str,
    region: Sequence[int],
    fps: float,
    settings: Any,
    text_filter: Any,
) -> str:
    """Nhan dien dung video + vung + cau hinh cho OCR noi bo."""
    payload = {
        "video": file_identity(video),
        "region": list(region),
        "fps": round(float(fps), 6),
        "profile": getattr(settings, "ocr_server", "") or getattr(settings, "ocr_mode", ""),
        "confidence": getattr(settings, "ocr_confidence", 70.0),
        "filter": dict(vars(text_filter)) if hasattr(text_filter, "__dict__") else {},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=list)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _init_db(conn: sqlite3.Connection) -> None:
    """Khoi tao database o che do WAL va tao bang neu chua co."""
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS frames ("
        "cache_key TEXT NOT NULL, "
        "frame_idx INTEGER NOT NULL, "
        "rows_json TEXT NOT NULL, "
        "content_hash TEXT, "
        "PRIMARY KEY(cache_key, frame_idx))"
    )
    cols = [row[1] for row in conn.execute("PRAGMA table_info(frames)").fetchall()]
    if "content_hash" not in cols:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE frames ADD COLUMN content_hash TEXT")

    conn.execute(
        "CREATE TABLE IF NOT EXISTS ai_segments ("
        "cache_key TEXT NOT NULL, "
        "segment_id TEXT NOT NULL, "
        "start REAL NOT NULL, "
        "end REAL NOT NULL, "
        "content_hash TEXT, "
        "text TEXT NOT NULL, "
        "confidence REAL NOT NULL DEFAULT 1.0, "
        "uncertain INTEGER NOT NULL DEFAULT 0, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "PRIMARY KEY(cache_key, segment_id))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_segments_hash ON ai_segments(cache_key, content_hash);"
    )

    conn.execute(
        "CREATE TABLE IF NOT EXISTS ai_cache_meta ("
        "cache_key TEXT PRIMARY KEY, "
        "engine TEXT NOT NULL, "
        "endpoint_hash TEXT NOT NULL, "
        "model_alias TEXT NOT NULL, "
        "actual_model TEXT NOT NULL, "
        "thinking TEXT NOT NULL, "
        "prompt_version TEXT NOT NULL, "
        "preprocessing_version TEXT NOT NULL, "
        "video_identity TEXT NOT NULL, "
        "region TEXT NOT NULL, "
        "sampling_settings TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )


def load_frame_cache(
    cache_path: str | Path | None, cache_key: str, total: int
) -> dict[int, list[Row]]:
    """Doc ket qua OCR tung frame tu SQLite; tep hong thi coi nhu cache rong."""
    if not cache_path or not cache_key:
        return {}
    path = Path(cache_path)
    if not path.is_file():
        return {}
    try:
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            connection.execute("PRAGMA busy_timeout=5000;")
            rows = connection.execute(
                "SELECT frame_idx, rows_json FROM frames WHERE cache_key=? AND frame_idx<?",
                (cache_key, int(total)),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}
    out: dict[int, list[Row]] = {}
    for index, raw in rows:
        try:
            data = json.loads(raw)
            parsed: list[Row] = []
            for item in data:
                rect: Rect | None = None
                if item[3] is not None:
                    rect = (
                        float(item[3][0]),
                        float(item[3][1]),
                        float(item[3][2]),
                        float(item[3][3]),
                    )
                parsed.append(
                    Row(
                        (float(item[0][0]), float(item[0][1])),
                        str(item[1]),
                        float(item[2]),
                        rect,
                    )
                )
        except (TypeError, ValueError, IndexError, json.JSONDecodeError):
            continue
        out[int(index)] = parsed
    return out


def save_frame_batch(
    cache_path: str | Path | None,
    cache_key: str,
    rows_by_index: dict[int, list[Row]],
    content_hashes: dict[int, str] | None = None,
) -> None:
    """Luu ket qua mot lo hoac toan bo khung hinh xuong SQLite WAL ngay lap tuc."""
    if not cache_path or not cache_key or not rows_by_index:
        return
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            _init_db(connection)
            payload = []
            for index, rows in rows_by_index.items():
                encoded = [
                    [list(row.order), row.text, row.score, list(row.rect) if row.rect else None]
                    for row in rows
                ]
                ch = content_hashes.get(index) if content_hashes else None
                payload.append((cache_key, int(index), json.dumps(encoded, ensure_ascii=False), ch))
            connection.executemany(
                "INSERT OR REPLACE INTO frames(cache_key, frame_idx, rows_json, content_hash) "
                "VALUES(?,?,?,?)",
                payload,
            )
            connection.commit()
    except (OSError, sqlite3.Error):
        return


save_frame_cache = save_frame_batch


def load_segment_checkpoint(
    cache_path: str | Path | None,
    cache_key: str,
) -> dict[str, SegmentRecord]:
    """Doc cac doan phu de da hoan thanh tu checkpoint SQLite."""
    if not cache_path or not cache_key:
        return {}
    path = Path(cache_path)
    if not path.is_file():
        return {}
    try:
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            connection.execute("PRAGMA busy_timeout=5000;")
            _init_db(connection)
            rows = connection.execute(
                "SELECT segment_id, start, end, content_hash, text, confidence, uncertain "
                "FROM ai_segments WHERE cache_key=?",
                (cache_key,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}
    out: dict[str, SegmentRecord] = {}
    for seg_id, start, end, ch, text, conf, unc in rows:
        out[str(seg_id)] = SegmentRecord(
            segment_id=str(seg_id),
            start=float(start),
            end=float(end),
            content_hash=str(ch or ""),
            text=str(text or ""),
            confidence=float(conf if conf is not None else 1.0),
            uncertain=bool(unc),
        )
    return out


load_segment_cache = load_segment_checkpoint


def load_segment_by_hash(
    cache_path: str | Path | None,
    cache_key: str,
) -> dict[str, SegmentRecord]:
    """Doc cac doan phu de theo content_hash tu checkpoint SQLite."""
    if not cache_path or not cache_key:
        return {}
    path = Path(cache_path)
    if not path.is_file():
        return {}
    try:
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            connection.execute("PRAGMA busy_timeout=5000;")
            _init_db(connection)
            rows = connection.execute(
                "SELECT segment_id, start, end, content_hash, text, confidence, uncertain "
                "FROM ai_segments WHERE cache_key=? AND content_hash IS NOT NULL "
                "AND content_hash != ''",
                (cache_key,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}
    out: dict[str, SegmentRecord] = {}
    for seg_id, start, end, ch, text, conf, unc in rows:
        if ch:
            out[str(ch)] = SegmentRecord(
                segment_id=str(seg_id),
                start=float(start),
                end=float(end),
                content_hash=str(ch),
                text=str(text or ""),
                confidence=float(conf if conf is not None else 1.0),
                uncertain=bool(unc),
            )
    return out


def save_segment_checkpoint(
    cache_path: str | Path | None,
    cache_key: str,
    segments: Sequence[Any],
) -> None:
    """Luu ngay lap tuc mot lo doan phu de vao checkpoint SQLite WAL."""
    if not cache_path or not cache_key or not segments:
        return
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            _init_db(connection)
            payload = []
            for seg in segments:
                if isinstance(seg, SegmentRecord):
                    payload.append((
                        cache_key,
                        seg.segment_id,
                        float(seg.start),
                        float(seg.end),
                        seg.content_hash,
                        seg.text,
                        float(seg.confidence),
                        1 if seg.uncertain else 0,
                    ))
                elif isinstance(seg, dict):
                    payload.append((
                        cache_key,
                        str(seg.get("segment_id") or seg.get("id", "")),
                        float(seg.get("start", 0.0)),
                        float(seg.get("end", 0.0)),
                        str(seg.get("content_hash", "")),
                        str(seg.get("text", "")),
                        float(seg.get("confidence", 1.0)),
                        1 if seg.get("uncertain") else 0,
                    ))
                else:
                    seg_id = getattr(seg, "segment_id", None) or getattr(seg, "id", "")
                    start = getattr(seg, "start", 0.0)
                    end = getattr(seg, "end", 0.0)
                    ch = getattr(seg, "content_hash", "")
                    text = getattr(seg, "text", "")
                    conf = getattr(seg, "confidence", 1.0)
                    unc = getattr(seg, "uncertain", False)
                    payload.append((
                        cache_key,
                        str(seg_id),
                        float(start),
                        float(end),
                        str(ch),
                        str(text),
                        float(conf),
                        1 if unc else 0,
                    ))
            connection.executemany(
                "INSERT OR REPLACE INTO ai_segments("
                "cache_key, segment_id, start, end, content_hash, text, confidence, uncertain) "
                "VALUES(?,?,?,?,?,?,?,?)",
                payload,
            )
            connection.commit()
    except (OSError, sqlite3.Error):
        return


save_segment_batch = save_segment_checkpoint


def save_cache_meta(
    cache_path: str | Path | None,
    cache_key: str,
    meta: dict[str, Any],
) -> None:
    """Ghi thong tin metadata cua phien OCR xuong SQLite."""
    if not cache_path or not cache_key or not meta:
        return
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            _init_db(connection)
            connection.execute(
                "INSERT OR REPLACE INTO ai_cache_meta("
                "cache_key, engine, endpoint_hash, model_alias, actual_model, "
                "thinking, prompt_version, preprocessing_version, video_identity, "
                "region, sampling_settings) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cache_key,
                    str(meta.get("engine", "ai_gateway")),
                    str(meta.get("endpoint_hash", "")),
                    str(meta.get("model_alias", "")),
                    str(meta.get("actual_model", "")),
                    str(meta.get("thinking", "")),
                    str(meta.get("prompt_version", "")),
                    str(meta.get("preprocessing_version", "")),
                    json.dumps(meta.get("video_identity", []), ensure_ascii=False),
                    json.dumps(meta.get("region", []), ensure_ascii=False),
                    json.dumps(meta.get("sampling_settings", {}), ensure_ascii=False),
                ),
            )
            connection.commit()
    except (OSError, sqlite3.Error):
        return


def get_cache_meta(cache_path: str | Path | None, cache_key: str) -> dict[str, Any] | None:
    """Doc metadata cua phien OCR tu SQLite."""
    if not cache_path or not cache_key:
        return None
    path = Path(cache_path)
    if not path.is_file():
        return None
    try:
        with _DB_LOCK, sqlite3.connect(str(path), timeout=30.0) as connection:
            _init_db(connection)
            row = connection.execute(
                "SELECT engine, endpoint_hash, model_alias, actual_model, thinking, "
                "prompt_version, preprocessing_version, video_identity, region, "
                "sampling_settings FROM ai_cache_meta WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            return {
                "engine": row[0],
                "endpoint_hash": row[1],
                "model_alias": row[2],
                "actual_model": row[3],
                "thinking": row[4],
                "prompt_version": row[5],
                "preprocessing_version": row[6],
                "video_identity": json.loads(row[7]),
                "region": json.loads(row[8]),
                "sampling_settings": json.loads(row[9]),
            }
    except (OSError, sqlite3.Error):
        return None
