"""Kiem ke read-only mot ban NTS AutoSub de doi chieu tinh nang.

Script khong sua NTS va khong dua khoa API/thong tin dang nhap vao bao cao.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SENSITIVE = re.compile(r"api.?key|token|password|passwd|login|cookie|proxy", re.I)


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _safe(key: str, value: Any) -> Any:
    if SENSITIVE.search(key):
        return "<redacted>"
    if key == "data_table" and isinstance(value, list):
        return {"rows": len(value), "columns": max((len(row) for row in value), default=0)}
    if isinstance(value, dict):
        return {str(child): _safe(str(child), item) for child, item in value.items()}
    if isinstance(value, list):
        return [_safe(key, item) for item in value]
    if isinstance(value, str) and len(value) > 500:
        return {"type": "string", "length": len(value)}
    return value


def _json_rows(db: Path, table: str, name_column: str) -> list[dict[str, Any]]:
    with _connect(db) as connection:
        rows = connection.execute(f'SELECT "{name_column}", value FROM "{table}"').fetchall()
    out = []
    for name, raw in rows:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            value = raw
        if isinstance(value, dict):
            value = {key: _safe(key, item) for key, item in value.items()}
        out.append({"name": str(name), "value": value})
    return out


def _database_schema(path: Path) -> dict[str, Any]:
    with _connect(path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {
            table: {
                "rows": connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0],
                "columns": [
                    {"name": row[1], "type": row[2]}
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                ],
            }
            for table in tables
        }


def _wav_summary(folder: Path) -> dict[str, Any]:
    formats: Counter[str] = Counter()
    durations = []
    for path in folder.glob("*.wav"):
        try:
            with wave.open(str(path), "rb") as stream:
                rate = stream.getframerate()
                channels = stream.getnchannels()
                width = stream.getsampwidth()
                formats[f"{rate}Hz/{channels}ch/{width * 8}bit"] += 1
                durations.append(stream.getnframes() / max(1, rate))
        except (OSError, wave.Error):
            formats["unreadable"] += 1
    return {
        "files": sum(formats.values()),
        "formats": dict(formats.most_common()),
        "duration_min": round(min(durations), 3) if durations else 0,
        "duration_max": round(max(durations), 3) if durations else 0,
    }


def audit(root: Path) -> dict[str, Any]:
    db = root / "db" / "app.db"
    global_profiles = _json_rows(db, "cauhinhtuychonmodel", "ten_cau_hinh")
    project_profiles = _json_rows(db, "configeditsubmodel", "ten_cau_hinh")
    project_keys: dict[str, dict[str, Any]] = defaultdict(lambda: {"projects": 0, "samples": []})
    for profile in project_profiles:
        value = profile["value"]
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            entry = project_keys[key]
            entry["projects"] += 1
            shown = _safe(key, item)
            if shown not in entry["samples"] and len(entry["samples"]) < 3:
                entry["samples"].append(shown)

    raw_databases = {}
    for path in sorted((root / "project").glob("*/sub_raw_*.db")):
        signature = json.dumps(_database_schema(path), sort_keys=True, ensure_ascii=False)
        raw_databases.setdefault(signature, {"count": 0, "example": str(path), "schema": None})
        raw_databases[signature]["count"] += 1
        raw_databases[signature]["schema"] = json.loads(signature)

    extension_counts = Counter()
    top_files = []
    for path in root.iterdir():
        if path.is_file():
            top_files.append({"name": path.name, "bytes": path.stat().st_size})
    for folder_name in ("font", "images", "LUT", "model", "theme", "tools", "voice"):
        folder = root / folder_name
        for path in folder.rglob("*"):
            if path.is_file():
                extension_counts[f"{folder_name}:{path.suffix.lower() or '[none]'}"] += 1

    return {
        "root": str(root.resolve()),
        "top_files": top_files,
        "asset_counts": dict(sorted(extension_counts.items())),
        "voice_samples": _wav_summary(root / "voice"),
        "voice_cache": _wav_summary(root / "audio"),
        "main_database": _database_schema(db),
        "global_profiles": global_profiles,
        "project_setting_index": dict(sorted(project_keys.items())),
        "raw_subtitle_database_variants": list(raw_databases.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("NTS_AUDIT.json"))
    args = parser.parse_args()
    report = audit(args.root)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
