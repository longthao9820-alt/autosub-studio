"""Doc chu tren hinh bang AI Gateway (chuan OpenAI vision HTTP).

Cac buoc crop, loc mau/sang va nhan dien keyframe duoc xu ly cuc bo tren may.
AI chi doc chu tren cac keyframe thuc su thay doi, khong upload toan bo moi frame.
Moc thoi gian va gop dong do he thong timeline/cache san co quan ly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.models import Cue
from ..services import ai_gateway
from ..services.settings import Settings
from . import ocr
from .ocr import Row
from .ocr_filter import TextFilter

if TYPE_CHECKING:
    pass


def _image_to_small_gray(
    frame_path: Path, flt: TextFilter | None = None
) -> tuple[Any, bytes | None]:
    """Doc anh, ap dung bo loc neu co, tra ve anh thu nho xam de so sanh va bytes goc."""
    try:
        import cv2

        img = cv2.imread(str(frame_path))
        if img is None:
            return None, None
        if flt is not None and flt.touches_image:
            from . import ocr_filter

            img = ocr_filter.adjust(img, flt.brightness, flt.contrast)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        _, buf = cv2.imencode(".jpg", img)
        return small, buf.tobytes()
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(frame_path) as pil_img:
            rgb = pil_img.convert("RGB")
            small_pil = rgb.convert("L").resize((64, 64))
            raw_bytes = frame_path.read_bytes()
            return small_pil, raw_bytes
    except Exception:
        return None, None


def _is_blank_frame(small_img: Any) -> bool:
    """Kiem tra khung hinh co phai don sac/khong co chi tiet nao khong."""
    if small_img is None:
        return True
    try:
        import numpy as np

        arr = np.asarray(small_img)
        if arr.std() < 2.0 or arr.mean() < 3.0:
            return True
    except Exception:
        pass
    return False


def _diff_metric(a: Any, b: Any) -> float:
    """Do sai khac trung binh giua hai khung hinh thu nho."""
    if a is None or b is None:
        return 999.0
    try:
        import numpy as np

        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        return float(np.mean(np.abs(arr_a - arr_b)))
    except Exception:
        return 999.0


def read_frames_ai(
    frames: Sequence[Path],
    *,
    fps: float = 2.0,
    start_offset: float = 0.0,
    stamps: Sequence[float] | None = None,
    similarity: float = 0.82,
    min_duration: float = 0.4,
    model: str = "sub",
    endpoint: str = "",
    api_key: str = "",
    thinking: str = "",
    consensus: int = 1,
    text_filter: TextFilter | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    cache_path: str | Path | None = None,
    cache_key: str = "",
) -> list[Cue]:
    """Doc chu tren cac khung hinh bang AI vision.

    Su dung local keyframe deduplication: chi upload cac khung hinh co su thay doi,
    tranh upload mu toan bo cac frame giong nhau. Ket qua duoc luu vao SQLite cache
    va gop timeline bang pipeline chuan.
    """
    if not frames:
        return []

    settings = Settings.load()
    ep = endpoint.strip() or settings.ai_endpoint.strip()
    if not ep:
        raise ai_gateway.AIGatewayError("Chưa cấu hình Endpoint AI Gateway cho OCR AI.")
    key = api_key.strip() or Settings.get_secret("ai_gateway_key")
    if not key:
        raise ai_gateway.AIGatewayError("Chưa có khóa API cho AI Gateway.")

    actual_model, default_thinking = ai_gateway.resolve_model(
        model or settings.ocr_ai_model, settings
    )
    actual_thinking = thinking if thinking else default_thinking

    total = len(frames)
    cached = ocr._load_frame_cache(cache_path, cache_key, total)
    rows_by_index: dict[int, list[Row]] = dict(cached)
    missing_indices = [i for i in range(total) if i not in rows_by_index]

    if on_log:
        if missing_indices:
            on_log(
                f"OCR AI ({actual_model}): xử lý {len(missing_indices)}/{total} khung hinh "
                "(chỉ upload keyframe mới)..."
            )
        else:
            on_log("Tất cả khung hình đã có trong cache OCR AI.")
        if cached:
            on_log(f"Dùng lại cache OCR AI: {len(cached)}/{total} khung hình.")

    fresh: dict[int, list[Row]] = {}
    last_keyframe_small: Any = None
    last_keyframe_rows: list[Row] = []

    for done, index in enumerate(missing_indices):
        if should_cancel is not None and should_cancel():
            break

        frame_path = frames[index]
        small, img_bytes = _image_to_small_gray(frame_path, text_filter)

        if _is_blank_frame(small):
            fresh[index] = []
            last_keyframe_small = small
            last_keyframe_rows = []
        elif last_keyframe_small is not None and _diff_metric(small, last_keyframe_small) < 4.0:
            # Khung hinh giong het hoac rat gan voi keyframe truoc do -> tai su dung cuc bo
            fresh[index] = list(last_keyframe_rows)
        else:
            # Keyframe moi -> goi AI Gateway doc chu
            text = ""
            if img_bytes:
                text = ai_gateway.ocr_image_with_ai(
                    ep,
                    key,
                    model=actual_model,
                    image_data=img_bytes,
                    thinking=actual_thinking,
                )

            clean_text = text.strip()
            if clean_text:
                rows = [Row((0.0, 0.0), clean_text, 1.0, None)]
            else:
                rows = []

            fresh[index] = rows
            last_keyframe_small = small
            last_keyframe_rows = rows

        if on_progress:
            completed = len(cached) + done + 1
            on_progress(max(0, min(96, int(completed / total * 96))))

    rows_by_index.update(fresh)
    ocr._save_frame_cache(cache_path, cache_key, fresh)

    per_frame: list[list[Row]] = []
    for index in range(total):
        if index not in rows_by_index:
            break
        per_frame.append(rows_by_index[index])

    ignore: frozenset[str] = frozenset()
    if text_filter is not None and text_filter.drop_static:
        ignore = ocr.static_texts(per_frame)
        if ignore and on_log:
            shown = ", ".join(sorted(ignore)[:4])
            on_log(f"Bỏ {len(ignore)} dòng chữ cố định: {shown}")

    processed = len(per_frame)
    used_stamps = list(stamps[:processed]) if stamps is not None else None
    cues = ocr._merge_rows(
        per_frame,
        fps=fps,
        start_offset=start_offset,
        stamps=used_stamps,
        total=processed,
        similarity=similarity,
        min_duration=min_duration,
        ignore=ignore,
        consensus=consensus,
        on_log=on_log,
    )
    cues = ocr._merge_exact_repeats(cues, max_gap=0.2)
    if on_progress:
        on_progress(100)
    return cues
