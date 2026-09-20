"""Doc chu chay san tren hinh (hardsub) bang OCR chay tren may.

Mac dinh NTS dua anh goc vao PP-OCRv4 va chi gioi han chieu cao hop chu. Loc
mau va bo chu dung yen la tuy chon rieng khi vung khoanh con logo/quang cao.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from ..core.models import Cue
from ..core.ocr_common import (
    Rect,
    Row,
    _area,
    _best_caption,
    _better_read,
    _box_order,
    _box_rect,
    _covered,
    _drop_flicker,
    _drop_overlaps,
    _is_cjk,
    _join_parts,
    _merge_exact_repeats,
    _merge_rows,
    _overlap_x,
    _parse_row,
    _reading_order,
    _same_caption_version,
    _same_line,
    _separator,
    _similar,
    _stabilize_reads,
    _text_ratio,
    _trim_repeat,
    clean_cues,
    is_ignored,
    join_rows,
    make_continuous,
    static_texts,
)
from ..data.ocr_cache import (
    load_frame_cache as _load_frame_cache,
)
from ..data.ocr_cache import (
    save_frame_cache as _save_frame_cache,
)
from ..services.gpu import onnx_cuda_ready
from ..services.paths import app_root, bundled_dir
from . import ocr_filter
from .ocr_filter import TextFilter

__all__ = [
    "NTS_FAST_MODE",
    "NTS_FAST_SERVER",
    "OCRError",
    "OCRUnavailable",
    "Probe",
    "Rect",
    "Row",
    "_area",
    "_best_caption",
    "_better_read",
    "_box_order",
    "_box_rect",
    "_build_engine",
    "_covered",
    "_drop_flicker",
    "_drop_overlaps",
    "_engine_module",
    "_engine_rows",
    "_engine_uses_cuda",
    "_first_match",
    "_gpu_frame_task",
    "_is_cjk",
    "_join_parts",
    "_load_engine",
    "_load_frame_cache",
    "_merge_exact_repeats",
    "_merge_rows",
    "_nts_model_paths",
    "_overlap_x",
    "_parse_row",
    "_quality_key",
    "_read_frames_gpu_parallel",
    "_reading_order",
    "_rect_points",
    "_reread_subtitle_lines",
    "_same_caption_version",
    "_same_line",
    "_save_frame_cache",
    "_separator",
    "_similar",
    "_spread",
    "_stabilize_reads",
    "_text_ratio",
    "_trim_repeat",
    "apply_probe",
    "clean_cues",
    "frame_rows",
    "gpu_available",
    "install_hint",
    "is_available",
    "is_ignored",
    "is_nts_profile",
    "join_rows",
    "make_continuous",
    "probe_frames",
    "read_frame",
    "read_frames",
    "refine_boundaries",
    "static_texts",
]

_OCR_THREAD_LOCAL = threading.local()
# Bon worker cho tong toan ung dung: du mot project hay nhieu project cung
# chay thi GPU cung khong bi tao vo han session va het bo nho. Cac thread song
# suot phien lam viec nen model duoc tai mot lan, giong server OCR cua NTS.
_OCR_GPU_WORKERS = 4
_OCR_GPU_PENDING = _OCR_GPU_WORKERS * 2
_OCR_GPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=_OCR_GPU_WORKERS, thread_name_prefix="autosub-ocr-gpu"
)
NTS_FAST_MODE = "Nhanh Như NTS"
NTS_FAST_SERVER = "PP-OCRv4 Mobile (Nhanh Như NTS)"
_NTS_MODEL_FILES = {
    "det": "ch_PP-OCRv4_det_infer.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "rec": "ch_PP-OCRv4_rec_infer.onnx",
}


class Probe(NamedTuple):
    """Ket qua do thu mau chu va chieu cao chu tren vai khung hinh."""

    color: str
    min_height: float
    max_height: float
    height: float
    lines: int
    static: tuple[str, ...] = ()  # cac dong chu dan co dinh do duoc trong vung

    @property
    def found(self) -> bool:
        return self.lines > 0


class OCRUnavailable(RuntimeError):
    """Chua cai thu vien OCR."""


class OCRError(RuntimeError):
    """OCR that bai."""


def is_available() -> bool:
    for name in ("rapidocr", "rapidocr_onnxruntime"):
        try:
            __import__(name)
        except ImportError:
            continue
        return True
    return False


def install_hint() -> str:
    return (
        "Chua cai bo doc chu tren hinh. Chay: pip install rapidocr onnxruntime "
        "(chay hoan toan tren may)."
    )


def _engine_module():
    """Uu tien RapidOCR moi; giu ban cu lam duong lui cho ban cai dat cu."""
    last: Exception | None = None
    for name in ("rapidocr", "rapidocr_onnxruntime"):
        try:
            module = __import__(name, fromlist=["RapidOCR"])
        except ImportError as exc:
            last = exc
            continue
        engine_cls = getattr(module, "RapidOCR", None)
        if engine_cls is not None:
            return name, module, engine_cls
    raise OCRUnavailable(install_hint()) from last


def _quality_key(profile: str) -> str:
    name = str(profile).casefold()
    if "nhanh" in name or "small" in name:
        return "fast"
    if "cân" in name or "can " in name:
        return "balanced"
    return "accurate"


def is_nts_profile(profile: str) -> bool:
    """Che do dung bo PP-OCRv4 Mobile va tham so nhanh giong NTS."""
    name = str(profile).casefold()
    return "nts" in name or "pp-ocrv4" in name or "ppocrv4" in name


def _nts_model_paths() -> dict[str, Path]:
    """Tim model kem theo o ca ban ma nguon va ban portable."""
    packaged = bundled_dir("ocr")
    root = packaged if packaged is not None else app_root() / "assets" / "ocr"
    paths = {key: root / filename for key, filename in _NTS_MODEL_FILES.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise OCRUnavailable("Thiếu model OCR Nhanh Như NTS: " + ", ".join(missing))
    return paths


def _build_engine(use_gpu: bool, profile: str = "Chính Xác", batch_size: int = 6):
    """Tao engine OCR tieng Trung theo che do toc do/doi chieu dang chon."""
    name, module, engine_cls = _engine_module()
    if name == "rapidocr":
        model_type = module.ModelType
        ocr_version = module.OCRVersion
        if is_nts_profile(profile):
            models = _nts_model_paths()
            engine = engine_cls(
                params={
                    # Dung dung model va kich thuoc detector cua NTS: PP-OCRv4
                    # Mobile, gioi han canh 736 px, nhan dang theo lo 5 dong.
                    "Global.log_level": "error",
                    "Global.use_cls": True,
                    "Det.lang_type": "ch",
                    "Det.model_type": model_type.MOBILE,
                    "Det.ocr_version": ocr_version.PPOCRV4,
                    "Det.model_path": str(models["det"]),
                    "Det.limit_side_len": 736,
                    # Vung OCR da duoc cat sat dong subtitle. Kieu "min" mac
                    # dinh phong dai chieu cao 120 px len 736 px va bien anh
                    # 1728x120 thanh gan 3000x736, ton hon 2,5 lan thoi gian.
                    # "max" giu nguyen do phan giai goc, khong lam mat net chu.
                    "Det.limit_type": "max",
                    "Det.score_mode": "fast",
                    "Cls.model_type": model_type.MOBILE,
                    "Cls.ocr_version": ocr_version.PPOCRV4,
                    "Cls.model_path": str(models["cls"]),
                    "Rec.lang_type": "ch",
                    "Rec.model_type": model_type.MOBILE,
                    "Rec.ocr_version": ocr_version.PPOCRV4,
                    "Rec.model_path": str(models["rec"]),
                    "Rec.rec_batch_num": max(1, min(64, int(batch_size))),
                    "EngineConfig.onnxruntime.use_cuda": bool(use_gpu),
                    # Bon engine GPU chay song song. Neu de ONNX Runtime tu
                    # tao pool theo toan bo so nhan CPU cho ca 12 session
                    # det/cls/rec, ung dung co the vuot 200 thread va cac
                    # engine tranh CPU voi nhau. GPU chi can mot thread dieu
                    # phoi cho moi session; phan hau xu ly OpenCV van chay o
                    # worker rieng.
                    "EngineConfig.onnxruntime.intra_op_num_threads": 1
                    if use_gpu
                    else -1,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1
                    if use_gpu
                    else -1,
                }
            )
            # Phu de nam ngang va PP-OCRv4 da tra ca dong on dinh. Danh dau de
            # khong doc lai tung mien cat lan hai trong duong chay nhanh.
            engine._autosub_fast_profile = True
            return engine
        quality = _quality_key(profile)
        model = model_type.MEDIUM if quality == "accurate" else model_type.SMALL
        return engine_cls(
            params={
                "Det.lang_type": "ch",
                "Det.model_type": model,
                "Det.ocr_version": ocr_version.PPOCRV6,
                "Rec.lang_type": "ch",
                "Rec.model_type": model,
                "Rec.ocr_version": ocr_version.PPOCRV6,
                "Rec.rec_batch_num": max(1, min(64, int(batch_size))),
                "EngineConfig.onnxruntime.use_cuda": bool(use_gpu),
                "EngineConfig.onnxruntime.intra_op_num_threads": 1 if use_gpu else -1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1 if use_gpu else -1,
            }
        )
    return engine_cls(
        det_use_cuda=bool(use_gpu),
        cls_use_cuda=bool(use_gpu),
        rec_use_cuda=bool(use_gpu),
        rec_batch_num=max(1, min(64, int(batch_size))),
    )


def _engine_uses_cuda(engine: Any) -> bool:
    """Xac nhan RapidOCR moi khong am tham lui ve CPU khi thieu DLL CUDA."""
    for name in ("text_det", "text_cls", "text_rec"):
        component = getattr(engine, name, None)
        wrapper = getattr(component, "session", None)
        session = getattr(wrapper, "session", None)
        getter = getattr(session, "get_providers", None)
        if not callable(getter):
            continue
        if "CUDAExecutionProvider" not in getter():
            return False
    # Engine cu khong cho xem provider; phep thu anh ben duoi van la kiem tra chinh.
    return True


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Co chay duoc OCR bang card do hoa khong.

    Chi hoi thu vien la chua du: co may nap duoc CUDA nhung thieu mot phan
    cuDNN, luc do phai chay that mot tam anh nho moi lo ra. Chay thu mot lan
    roi nho ket qua, nen khong lam cham qua trinh tach sub.
    """
    if not onnx_cuda_ready():
        return False
    try:
        import numpy as np

        engine = _build_engine(True, NTS_FAST_MODE, 1)
        probe = np.zeros((96, 320, 3), dtype=np.uint8)
        probe[36:60, 40:280] = 255  # mot vach trang de bo doc chu co viec lam
        engine(probe)
    except Exception:
        return False
    return _engine_uses_cuda(engine)


def _load_engine(use_gpu: bool = False, profile: str = "Chính Xác", batch_size: int = 6):
    """Nap mot bo doc chu rieng cho moi luong xu ly.

    RapidOCR co mot vai tuy chon noi bo thay doi trong luc nhan dang, vi vay
    khong duoc dung chung cung mot engine cho hai video dang chay song song.
    Moi worker giu cache rieng de van tranh nap lai model o tung khung hinh.
    """
    key = (bool(use_gpu), str(profile), max(1, min(64, int(batch_size))))
    engines = getattr(_OCR_THREAD_LOCAL, "engines", None)
    if engines is None:
        engines = {}
        _OCR_THREAD_LOCAL.engines = engines
    if key in engines:
        return engines[key]
    on_gpu = bool(use_gpu) and gpu_available()
    last: Exception | None = None
    if on_gpu:
        try:
            engine = _build_engine(True, profile, batch_size)
            with contextlib.suppress(AttributeError, TypeError):
                engine._autosub_uses_cuda = _engine_uses_cuda(engine)
            engines[key] = engine
            return engine
        except Exception as exc:
            last = exc
    try:
        engine = _build_engine(False, profile, batch_size)
        with contextlib.suppress(AttributeError, TypeError):
            engine._autosub_uses_cuda = False
        engines[key] = engine
        engines[(False, key[1], key[2])] = engine
        return engine
    except Exception as exc:
        raise OCRUnavailable(install_hint()) from (last or exc)


# --------------------------------------------------------------------------- doc


def _gpu_frame_task(
    frame: Path,
    min_confidence: float,
    text_filter: TextFilter | None,
    profile: str,
    batch_size: int,
    gpu_failed: threading.Event,
) -> list[Row]:
    """Doc mot frame trong pool; neu CUDA loi thi worker tu lui ve CPU."""
    use_gpu = not gpu_failed.is_set()
    engine = _load_engine(use_gpu, profile, batch_size)
    if use_gpu and getattr(engine, "_autosub_uses_cuda", True) is False:
        gpu_failed.set()
    try:
        return frame_rows(engine, frame, min_confidence, text_filter)
    except OCRError:
        if not use_gpu:
            raise
        gpu_failed.set()
        engine = _load_engine(False, profile, batch_size)
        return frame_rows(engine, frame, min_confidence, text_filter)


def _read_frames_gpu_parallel(
    frames: Sequence[Path],
    *,
    min_confidence: float,
    text_filter: TextFilter | None,
    profile: str,
    batch_size: int,
    on_progress: Callable[[int], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> tuple[list[list[Row]], bool]:
    """Chia frame cho pool GPU dung chung, giu nguyen thu tu dau vao."""
    total = len(frames)
    gpu_failed = threading.Event()
    pending: dict[int, Future[list[Row]]] = {}
    submitted = 0
    collected = 0
    output: list[list[Row]] = []
    try:
        while collected < total:
            while submitted < total and len(pending) < _OCR_GPU_PENDING:
                if should_cancel is not None and should_cancel():
                    break
                pending[submitted] = _OCR_GPU_EXECUTOR.submit(
                    _gpu_frame_task,
                    frames[submitted],
                    min_confidence,
                    text_filter,
                    profile,
                    batch_size,
                    gpu_failed,
                )
                submitted += 1
            if collected not in pending:
                break
            output.append(pending.pop(collected).result())
            collected += 1
            if on_progress:
                on_progress(max(0, min(96, int(collected / total * 96))))
            if should_cancel is not None and should_cancel():
                break
    finally:
        for future in pending.values():
            future.cancel()
    return output, gpu_failed.is_set()


def read_frames(
    frames: Sequence[Path],
    *,
    fps: float = 2.0,
    start_offset: float = 0.0,
    stamps: Sequence[float] | None = None,
    similarity: float = 0.82,
    min_confidence: float = 0.5,
    min_duration: float = 0.4,
    use_gpu: bool = False,
    profile: str = "Chính Xác",
    batch_size: int = 6,
    consensus: int = 3,
    text_filter: TextFilter | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    cache_path: str | Path | None = None,
    cache_key: str = "",
) -> list[Cue]:
    """Doc chu tren loat khung hinh, gop cac khung giong nhau thanh mot cau.

    Doc xong het moi khung roi moi gop, vi phai xem toan bo video moi biet dong
    chu nao dung im mot cho (logo, watermark) de bo di.
    """
    if not frames:
        return []
    total = len(frames)
    cached = _load_frame_cache(cache_path, cache_key, total)
    rows_by_index = dict(cached)
    missing_indices = [index for index in range(total) if index not in rows_by_index]
    on_gpu = bool(missing_indices) and bool(use_gpu) and gpu_available()
    if on_log:
        if missing_indices:
            on_log("Doc chu bang card do hoa." if on_gpu else "Doc chu bang CPU (cham hon).")
            if is_nts_profile(profile):
                on_log("Che do Nhanh Nhu NTS: PP-OCRv4 Mobile tieng Trung, batch 5.")
            if text_filter is not None and text_filter.active:
                on_log(f"Bo loc chu: {text_filter.describe()}")
        else:
            on_log("Tat ca khung hinh da co trong cache OCR, khong nap lai model.")
    if cached and on_log:
        on_log(f"Dung lai cache OCR: {len(cached)}/{total} khung hinh.")

    def report_missing(percent: int) -> None:
        if not on_progress:
            return
        completed = int(max(0, min(96, percent)) / 96 * len(missing_indices))
        on_progress(max(0, min(96, int((len(cached) + completed) / total * 96))))

    if on_gpu:
        missing_rows, gpu_failed = _read_frames_gpu_parallel(
            [frames[index] for index in missing_indices],
            min_confidence=min_confidence,
            text_filter=text_filter,
            profile=profile,
            batch_size=batch_size,
            on_progress=report_missing,
            should_cancel=should_cancel,
        )
        if gpu_failed and on_log:
            on_log("Card do hoa bao loi, da chuyen cac frame con lai sang CPU.")
        fresh = dict(zip(missing_indices, missing_rows, strict=False))
        rows_by_index.update(fresh)
        _save_frame_cache(cache_path, cache_key, fresh)
    else:
        fresh = {}
        if missing_indices:
            engine = _load_engine(False, profile, batch_size)
            for done, index in enumerate(missing_indices):
                if should_cancel is not None and should_cancel():
                    break
                fresh[index] = frame_rows(engine, frames[index], min_confidence, text_filter)
                if on_progress:
                    completed = len(cached) + done + 1
                    on_progress(max(0, min(96, int(completed / total * 96))))
        rows_by_index.update(fresh)
        _save_frame_cache(cache_path, cache_key, fresh)

    per_frame: list[list[Row]] = []
    for index in range(total):
        if index not in rows_by_index:
            break
        per_frame.append(rows_by_index[index])

    ignore: frozenset[str] = frozenset()
    if text_filter is not None and text_filter.drop_static:
        ignore = static_texts(per_frame)
        if ignore:
            if on_log:
                shown = ", ".join(sorted(ignore)[:4])
                on_log(f"Bo {len(ignore)} dong chu co dinh trong vung: {shown}")
            # Nho lai de buoc do lai moc thoi gian cung bo dung nhung dong nay.
            known = list(text_filter.ignore_texts)
            known += [text for text in sorted(ignore) if text not in known]
            text_filter.ignore_texts = tuple(known)

    processed = len(per_frame)
    used_stamps = list(stamps[:processed]) if stamps is not None else None
    cues = _merge_rows(
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
    cues = _merge_exact_repeats(cues, max_gap=0.2)
    if on_progress:
        on_progress(100)
    return cues


def refine_boundaries(
    cues: list[Cue],
    *,
    frames_around: Callable[[float, float], list[tuple[float, Path]]],
    coarse_step: float,
    similarity: float = 0.82,
    min_confidence: float = 0.5,
    use_gpu: bool = False,
    profile: str = "Chính Xác",
    batch_size: int = 6,
    text_filter: TextFilter | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[Cue]:
    """Do lai moc bat dau va ket thuc cho sat voi luc chu that su xuat hien.

    Buoc trich khung hinh chi lay vai khung moi giay nen moc thoi gian ban dau
    chi chinh xac trong khoang do. O day ta lay them cac khung nam giua hai lan
    do cu, tim dung khung ma chu bat dau hien ra va khung ma chu bien mat.
    """
    if not cues or coarse_step <= 0:
    return cues


    engine = _load_engine(bool(use_gpu) and gpu_available(), profile, batch_size)
    total = max(1, len(cues))

    for i, cue in enumerate(cues):
        if should_cancel is not None and should_cancel():
            break
        window = max(0.0, cue.start - coarse_step)
        samples = frames_around(window, cue.start) if cue.start > 0 else []
        appeared = _first_match(
            engine, samples, cue.text, similarity, min_confidence, True, text_filter
        )
        if appeared is not None:
            floor = cues[i - 1].end if i > 0 else 0.0
            cue.start = max(floor, min(appeared, cue.end - 0.05))
        samples = frames_around(max(0.0, cue.end - coarse_step), cue.end)
        gone = _first_match(
            engine, samples, cue.text, similarity, min_confidence, False, text_filter
        )
        if gone is not None and gone > cue.start + 0.05:
            cue.end = gone
        if on_progress:
            on_progress(max(0, min(100, int((i + 1) / total * 100))))
    if on_log:
        on_log(f"Da do lai moc thoi gian cho {len(cues)} cau.")
    return cues


def _first_match(
    engine,
    samples: list[tuple[float, Path]],
    target: str,
    similarity: float,
    min_confidence: float,
    want_match: bool,
    text_filter: TextFilter | None = None,
) -> float | None:
    """Moc dau tien trong cua so ma chu khop (hoac het khop) voi cau dang xet."""
    seen_opposite = False
    for stamp, path in samples:
        text = read_frame(engine, path, min_confidence, text_filter)[0]
        matched = _similar(text, target, similarity)
        if matched == want_match:
            if want_match or seen_opposite:
                return stamp
        else:
            seen_opposite = True
    return None


# ------------------------------------------------------------------ do thu mau chu


def probe_frames(
    frames: Sequence[Path],
    *,
    min_confidence: float = 0.5,
    use_gpu: bool = False,
    profile: str = "Chính Xác",
    batch_size: int = 6,
    samples: int = 14,
    base: TextFilter | None = None,
    on_log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Probe:
    """Do mau chu va chieu cao chu phu de tu vai khung hinh trong video.

    Doc thu vai khung hinh khong loc gi ca, do mau ruot chu cua tung dong doc
    duoc, bo cac dong dung im mot cho (logo), roi lay mau chiem nhieu dien tich
    chu nhat lam mau phu de. Chieu cao lay theo trung vi cua chinh cac dong do.
    """
    picked = _spread(frames, max(4, int(samples)))
    if not picked or not ocr_filter.available():
        return Probe("", 0.0, 0.0, 0.0, 0)
    raw = TextFilter(
        use_color=False,
        brightness=base.brightness if base else 0,
        contrast=base.contrast if base else 0,
    )
    engine = _load_engine(bool(use_gpu) and gpu_available(), profile, batch_size)
    per_frame: list[list[Row]] = []
    measured: list[list[tuple[Row, tuple[int, int, int]]]] = []

    for frame in picked:
        if should_cancel is not None and should_cancel():
            break
        rows_raw, image, _mask = _engine_rows(engine, frame, raw, need_image=True)
        rows: list[Row] = []
        found: list[tuple[Row, tuple[int, int, int]]] = []
        for row in rows_raw:
            parsed = _parse_row(row, min_confidence)
            if parsed is None:
                continue
            rows.append(parsed)
            color = ocr_filter.fill_color(image, parsed.rect)
            if color is not None and parsed.rect is not None:
                found.append((parsed, color))
        per_frame.append(rows)
        measured.append(found)

    ignore = static_texts(per_frame, ratio=0.7, min_frames=4)
    weights: list[tuple[tuple[int, int, int], float]] = []
    heights: list[float] = []
    for found in measured:
        for row, color in found:
            if row.rect is None or is_ignored(row.text, ignore):
                continue
            width = row.rect[2] - row.rect[0]
            height = row.rect[3] - row.rect[1]
            weights.append((color, max(1.0, width * height)))
            heights.append(height)
    if not heights:
        if on_log:
            on_log("Khong do duoc mau chu: chua thay dong chu nao trong vung da khoanh.")
        return Probe("", 0.0, 0.0, 0.0, 0, tuple(sorted(ignore)))

    rgb = ocr_filter.pick_color(weights)
    low, high = ocr_filter.height_range(heights)
    middle = sorted(heights)[len(heights) // 2]
    if on_log:
        name = ocr_filter.color_name(rgb)
        on_log(
            f"Do tren {len(per_frame)} khung hinh: mau chu {ocr_filter.format_color(rgb)} "
            f"({name}), chu cao khoang {middle:.0f} px."
        )
    if ignore and on_log:
        on_log(f"Thay {len(ignore)} dong chu dan co dinh: {', '.join(sorted(ignore)[:4])}")
    return Probe(
        ocr_filter.format_color(rgb), low, high, float(middle), len(heights), tuple(sorted(ignore))
    )


def _spread(frames: Sequence[Path], count: int) -> list[Path]:
    """Lay cac khung hinh rai deu tu dau den cuoi de do cho dai dien."""
    items = list(frames)
    if len(items) <= count:
        return items
    stride = (len(items) - 1) / max(1, count - 1)
    return [items[min(len(items) - 1, int(round(i * stride)))] for i in range(count)]


def apply_probe(flt: TextFilter, probe: Probe) -> TextFilter:
    """Dien nhung so lieu con thieu cua bo loc bang ket qua do duoc."""
    if probe.static and flt.drop_static:
        merged = list(flt.ignore_texts) + [t for t in probe.static if t not in flt.ignore_texts]
        flt.ignore_texts = tuple(merged)
    if not probe.found:
        return flt
    if flt.use_color and not flt.rgb and probe.color:
        flt.color = probe.color
    if flt.min_height <= 0 and flt.max_height <= 0:
        flt.min_height = probe.min_height
        flt.max_height = probe.max_height
    return flt


def _engine_rows(
    engine, frame: Path, flt: TextFilter | None, *, need_image: bool = False
) -> tuple[list[Any], Any, Any]:
    """Chay bo doc chu tren mot khung hinh, tra ve (cac dong tho, anh, mat na mau).

    PP-OCRv6 co the dung anh da loc mau. PP-OCRv4 theo NTS luon doc anh mau
    that; mat na mau chi dung de loai cac hop khong phai phu de sau khi doc.
    """
    source: Any = str(frame)
    image = None
    mask = None
    if flt is not None and (flt.touches_image or need_image) and ocr_filter.available():
        image = ocr_filter.load_image(frame)
        if image is not None:
            image = ocr_filter.adjust(image, flt.brightness, flt.contrast)
            source = image
            rgb = flt.rgb
            if rgb is not None:
                mask = ocr_filter.grow_mask(ocr_filter.color_mask(image, rgb, flt.tolerance))
                if not ocr_filter.has_text_pixels(mask):
                    return ([], image, mask)  # khung nay khong co chu dung mau, doc lam gi
                # NTS dua anh mau goc vao PP-OCRv4. Ban cu cua ta to den moi
                # diem khong trung mau phu de; o canh chuyen dong, phep to den
                # nay vo tinh tao ra net chu gia nhu "!..", "11", "n.".
                # Voi profile NTS, mat na chi dung de kiem tra hop OCR sau khi
                # doc, con model luon nhin anh that nhu NTS.
                if not getattr(engine, "_autosub_fast_profile", False):
                    source = ocr_filter.masked_image(image, mask)
            else:
                source = image
    try:
        result = engine(source)
    except Exception as exc:
        raise OCRError(f"Doc chu that bai o {Path(frame).name}: {exc}") from exc
    if hasattr(result, "txts"):
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = [] if boxes is None else list(boxes)
        texts = [] if texts is None else list(texts)
        scores = [] if scores is None else list(scores)
        rows = [
            [box, text, scores[i] if i < len(scores) else 1.0]
            for i, (box, text) in enumerate(zip(boxes, texts, strict=False))
        ]
        if not getattr(engine, "_autosub_fast_profile", False):
            rows = _reread_subtitle_lines(engine, rows, source)
    else:
        rows = result[0] if isinstance(result, tuple) else result
    return (list(rows or []), image, mask)


def _reread_subtitle_lines(engine, rows: list[Any], source: Any) -> list[Any]:
    """Doc lai ca dong thay vi tin vao cac manh hop bi cat nho.

    Tren phu de Trung Quoc, bo phat hien doi khi cat mot dong thanh 2-3 hop
    chong nhau. Nhan dang tung hop de sai mot chu o diem noi. PP-OCRv6 doc ca
    dong mot lan se giu duoc ngu canh va cho ket qua on dinh hon.
    """
    parsed = [item for item in (_parse_row(row, 0.0) for row in rows) if item is not None]
    if len(parsed) < 2:
        return rows
    lines: list[list[Row]] = []
    for row in _reading_order(parsed):
        if row.rect is None:
            continue
        for line in lines:
            if any(_same_line(row.rect, item.rect) for item in line):
                line.append(row)
                break
        else:
            lines.append([row])
    if not any(len(line) > 1 for line in lines):
        return rows
    frame = source if not isinstance(source, (str, Path)) else ocr_filter.load_image(source)
    if frame is None or not hasattr(frame, "shape"):
        return rows
    height, width = frame.shape[:2]
    out: list[Any] = []
    for line in lines:
        if len(line) == 1 or any(item.rect is None for item in line):
            item = line[0]
            out.append([_rect_points(item.rect), item.text, item.score])
            continue
        rects = [item.rect for item in line if item.rect is not None]
        x1 = max(0, int(min(rect[0] for rect in rects)) - 4)
        y1 = max(0, int(min(rect[1] for rect in rects)) - 4)
        x2 = min(width, int(max(rect[2] for rect in rects)) + 5)
        y2 = min(height, int(max(rect[3] for rect in rects)) + 5)
        crop = frame[y1:y2, x1:x2]
        try:
            # Khong truyen ``use_det=False`` vao chinh engine dang quet video.
            # RapidOCR v3 ghi nho tuy chon nay cho cac lan goi sau; sau lan doc
            # lai dau tien, detector bi tat vinh vien va moi khung tiep theo deu
            # tra rong. Chay day du tren mien cat hep van nhanh, dong thoi khong
            # lam thay doi trang thai cua engine dung chung.
            reread = engine(crop)
            boxes = list(getattr(reread, "boxes", None) or ())
            texts = list(getattr(reread, "txts", None) or ())
            scores = list(getattr(reread, "scores", None) or ())
        except Exception:
            boxes, texts, scores = [], [], []
        reread_rows = [
            parsed
            for parsed in (
                _parse_row(
                    [box, text, scores[i] if i < len(scores) else 1.0],
                    0.0,
                )
                for i, (box, text) in enumerate(zip(boxes, texts, strict=False))
            )
            if parsed is not None
        ]
        reread_text, reread_score = join_rows(reread_rows)
        original_text, _original_score = join_rows(line)
        enough_text = len("".join(reread_text.split())) >= max(
            2, int(len("".join(original_text.split())) * 0.85)
        )
        if reread_text and reread_score >= 0.35 and enough_text:
            out.append(
                [
                    _rect_points((float(x1), float(y1), float(x2), float(y2))),
                    reread_text,
                    reread_score,
                ]
            )
        else:
            out.extend([_rect_points(item.rect), item.text, item.score] for item in line)
    return out


def _rect_points(rect: Rect | None) -> list[list[float]]:
    if rect is None:
        return []
    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def frame_rows(
    engine, frame: Path, min_confidence: float, text_filter: TextFilter | None = None
) -> list[Row]:
    """Cac vung chu dat yeu cau tren mot khung hinh."""
    if text_filter is None or not text_filter.active:
        text, score = read_frame(engine, frame, min_confidence)
        return [Row((0.0, 0.0), text, score, None)] if text else []
    rows, _image, mask = _engine_rows(engine, frame, text_filter)
    good: list[Row] = []
    for row in rows:
        parsed = _parse_row(row, min_confidence)
        if parsed is None:
            continue
        if is_ignored(parsed.text, text_filter.ignore_texts):
            continue
        if parsed.rect is not None:
            height = parsed.rect[3] - parsed.rect[1]
            if not text_filter.height_ok(height):
                continue
            if mask is not None:
                fill = ocr_filter.rect_fill(mask, parsed.rect)
                if not text_filter.fill_ok(fill):
                    continue
        good.append(parsed)
    return _drop_overlaps(good)


def read_frame(
    engine, frame: Path, min_confidence: float, text_filter: TextFilter | None = None
) -> tuple[str, float]:
    """Doc mot khung hinh, tra ve chu va do tin cay trung binh.

    Do tin cay dung de chon ban doc sach nhat trong nhieu khung cung mot cau,
    thay vi chi lay ban dai nhat.
    """
    if text_filter is not None and text_filter.active:
        return join_rows(frame_rows(engine, frame, min_confidence, text_filter))
    rows, _image, _mask = _engine_rows(engine, frame, None)
    good = [r for r in (_parse_row(row, min_confidence) for row in rows) if r is not None]
    return join_rows(_drop_overlaps(good))

