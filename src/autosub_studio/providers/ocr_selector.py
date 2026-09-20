"""Trinh chon khung hinh va phan doan phu de cuc bo (khong dung AI/OCR).

Loc va phan doan dua tren dHash/pHash, luma/edge difference, do net (sharpness),
do tuong phan (contrast), phat hien khung trong (blank), va tranh frame chuyen canh/mo.
Thuc hien hoan toan cuc bo, xac dinh, khong nhan dien chu va khong phu thuoc AI/RapidOCR.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass
class FrameSample:
    """Mot mau khung hinh kem moc thoi gian."""

    timestamp: float
    path: Path | str | None = None
    image: Any = None
    index: int = 0
    crop_region: Sequence[int] | None = None

    def __repr__(self) -> str:
        return f"FrameSample(idx={self.index}, t={self.timestamp:.3f}, path={self.path})"


@dataclass
class PreparedImage:
    """Anh da duoc chuan hoa, resize, nen JPEG va san sang lam payload."""

    payload: bytes
    width: int
    height: int
    timestamp: float
    format: str = "JPEG"
    content_hash: str = ""
    sharpness: float = 0.0
    contrast: float = 0.0
    score: float = 0.0
    frame_index: int = 0
    path: Path | str | None = None

    def __post_init__(self) -> None:
        if not self.content_hash and self.payload:
            self.content_hash = hashlib.sha256(self.payload).hexdigest()

    @property
    def data(self) -> bytes:
        return self.payload

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass
class VisualSegment:
    """Phan doan phu de thi giac on dinh voi ID va anh dai dien."""

    id: str
    start: float
    end: float
    representative: PreparedImage
    samples: list[PreparedImage] = field(default_factory=list)
    content_hash: str = ""
    frame_count: int = 0
    start_index: int = 0
    end_index: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash and self.representative is not None:
            self.content_hash = self.representative.content_hash
        if not self.samples and self.representative is not None:
            self.samples = [self.representative]

    @property
    def representatives(self) -> list[PreparedImage]:
        return self.samples if self.samples else [self.representative]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class _FrameMeta:
    """Sieu du lieu nhe cua mot khung hinh de phan doan ma khong giu anh trong bo nho."""

    index: int
    timestamp: float
    path: Path | str | None
    is_blank: bool
    dhash: int
    phash: int
    luma_thumb: np.ndarray
    sharpness: float
    contrast: float
    score: float
    sample: FrameSample | None = None


def coerce_sample(
    item: FrameSample | tuple[float, Path | str] | tuple[float, Any] | Path | str | Any,
    index: int = 0,
    fps: float | None = None,
) -> FrameSample:
    """Chuyen doi linh hoat cac kieu dau vao thanh FrameSample."""
    if isinstance(item, FrameSample):
        if item.index == 0 and index != 0:
            item.index = index
        return item
    if isinstance(item, tuple):
        if len(item) == 2:
            ts, val = item
            if isinstance(val, (str, Path)):
                return FrameSample(timestamp=float(ts), path=Path(val), index=index)
            return FrameSample(timestamp=float(ts), image=val, index=index)
        if len(item) >= 3:
            ts, val, idx = item[0], item[1], item[2]
            if isinstance(val, (str, Path)):
                return FrameSample(timestamp=float(ts), path=Path(val), index=int(idx))
            return FrameSample(timestamp=float(ts), image=val, index=int(idx))
    if isinstance(item, (str, Path)):
        ts = float(index) / float(fps) if fps and fps > 0 else float(index) * 0.5
        return FrameSample(timestamp=ts, path=Path(item), index=index)
    raise TypeError(f"Khong the chuyen kieu {type(item)} thanh FrameSample")


def _open_image(
    source: FrameSample | Path | str | Image.Image | np.ndarray,
) -> tuple[Image.Image, bool]:
    """Mo anh va tra ve cap (PIL Image, can_dong_file)."""
    if isinstance(source, FrameSample):
        if source.path is not None:
            return Image.open(source.path), True
        if source.image is not None:
            if isinstance(source.image, Image.Image):
                return source.image, False
            if isinstance(source.image, np.ndarray):
                return Image.fromarray(source.image), True
        raise ValueError("FrameSample khong co duong dan hoac du lieu anh")
    if isinstance(source, (str, Path)):
        return Image.open(source), True
    if isinstance(source, Image.Image):
        return source, False
    if isinstance(source, np.ndarray):
        return Image.fromarray(source), True
    raise TypeError(f"Nguon anh khong hop le: {type(source)}")


@contextmanager
def open_frame_image(
    source: FrameSample | Path | str | Image.Image | np.ndarray,
    crop_region: Sequence[int] | None = None,
) -> Iterator[Image.Image]:
    """Context manager mo anh, crop neu co yeu cau, va dong handle ngay khi xong."""
    img, should_close = _open_image(source)
    try:
        if crop_region and len(crop_region) == 4 and crop_region[2] > 0 and crop_region[3] > 0:
            x, y, w, h = (int(v) for v in crop_region)
            cw = max(1, min(w, img.width - max(0, x)))
            ch = max(1, min(h, img.height - max(0, y)))
            cropped = img.crop((max(0, x), max(0, y), max(0, x) + cw, max(0, y) + ch))
            yield cropped
        else:
            yield img
    finally:
        if should_close:
            img.close()


def compute_dhash(img: Image.Image, hash_size: int = 16) -> int:
    """Tinh Difference Hash (dHash) kich thuoc hash_size (mac dinh 16x16 = 256 bit)."""
    thumb = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.int32)
    diff = arr[:, 1:] > arr[:, :-1]
    val = 0
    for bit in diff.flat:
        val = (val << 1) | int(bit)
    return val


def _normalize_structural_topology(img: Image.Image) -> Image.Image:
    """Chuan hoa tuong phan cuc bo va cau truc glyph de tach biet chu khoi do sang/fade."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    ptp = float(np.ptp(gray))
    if ptp < 8.0:
        return Image.fromarray(np.zeros_like(gray, dtype=np.uint8))

    bg = float(np.median(gray))
    diff = gray - bg
    if np.abs(np.percentile(diff, 99)) > np.abs(np.percentile(diff, 1)):
        mag = np.maximum(0.0, diff)
    else:
        mag = np.maximum(0.0, -diff)

    peak = float(np.max(mag))
    if peak < 8.0:
        return Image.fromarray(np.zeros_like(gray, dtype=np.uint8))

    noise_floor = max(8.0, peak * 0.15)
    cleaned = np.where(mag > noise_floor, mag, 0.0)
    norm = np.clip(cleaned / peak * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(norm)


def compute_phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Tinh Perceptual Hash (pHash) su dung 2D DCT."""
    img_size = hash_size * highfreq_factor
    thumb = img.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32)
    n = img_size
    k = np.arange(n, dtype=np.float32).reshape((1, n))
    i = np.arange(n, dtype=np.float32).reshape((n, 1))
    dct_matrix = np.cos((np.pi * (2 * i + 1) * k) / (2.0 * n))
    dct_matrix[0, :] *= 1.0 / np.sqrt(2.0)
    dct_matrix *= np.sqrt(2.0 / n)
    dct = dct_matrix @ arr @ dct_matrix.T
    dct_low = dct[:hash_size, :hash_size]
    med = float(np.median(dct_low))
    diff = dct_low > med
    val = 0
    for bit in diff.flat:
        val = (val << 1) | int(bit)
    return val


def hamming_distance(h1: int, h2: int) -> int:
    """Khoang cach Hamming giua hai ma hash."""
    return bin(h1 ^ h2).count("1")


def is_blank_frame(
    img: Image.Image | np.ndarray,
    std_threshold: float = 3.0,
    ptp_threshold: float = 8.0,
) -> bool:
    """Phat hien khung hinh trong: don sac, nen tinh, hoac khong co chi tiet chu."""
    if isinstance(img, Image.Image):
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    else:
        gray = np.asarray(img, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)

    if gray.size == 0:
        return True
    std = float(np.std(gray))
    if std < std_threshold:
        return True
    ptp = float(np.ptp(gray))
    return ptp < ptp_threshold


def compute_clarity_score(img: Image.Image | np.ndarray) -> tuple[float, float, float]:
    """Tinh (sharpness, contrast, score) de danh gia do ro net va tranh khung mo/fade.

    - Sharpness: Do bien thien Laplacian (Laplacian variance), the hien canh chu sac net.
    - Contrast: Do lech chuan do sang pixel.
    - Score: Diem chat luong ket hop, cuc dai o khung hien thi tron ven ro nhat.
    """
    if isinstance(img, Image.Image):
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    else:
        gray = np.asarray(img, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)

    if gray.size == 0:
        return 0.0, 0.0, 0.0

    contrast = float(np.std(gray))
    ptp = float(np.ptp(gray))

    # Laplacian variance for edge sharpness
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        sharpness = float(np.var(lap))
    else:
        sharpness = 0.0

    score = sharpness + (contrast * 10.0) + (ptp * 2.0)
    return sharpness, contrast, score


def compute_visual_difference(
    meta1: _FrameMeta,
    meta2: _FrameMeta,
    phash_weight: float = 1.0,
    block_weight: float = 100.0,
    mean_weight: float = 50.0,
) -> float:
    """Do sai khac thi giac giua hai khung hinh dua tren pHash va luma block difference."""
    ph_raw = float(hamming_distance(meta1.phash, meta2.phash))
    if meta1.phash.bit_length() > 64 or meta2.phash.bit_length() > 64:
        ph_dist = ph_raw * 0.25
    else:
        ph_dist = ph_raw
    luma1 = meta1.luma_thumb.astype(np.float32) / 255.0
    luma2 = meta2.luma_thumb.astype(np.float32) / 255.0
    p_diff = np.abs(luma1 - luma2)
    mean_diff = float(np.mean(p_diff))
    # 4x8 blocks (each block is 8x8 pixels for 32x64 thumbnail)
    block_diff = float(np.max(p_diff.reshape(4, 8, 8, 8).mean(axis=(2, 3))))
    return (ph_dist * phash_weight) + (block_diff * block_weight) + (mean_diff * mean_weight)


def prepare_image(
    source: FrameSample | Path | str | Image.Image | np.ndarray,
    *,
    max_dimension: int = 1280,
    jpeg_quality: int = 88,
    max_payload_bytes: int | None = None,
    normalize: bool = False,
    crop_region: Sequence[int] | None = None,
    timestamp: float = 0.0,
    frame_index: int = 0,
) -> PreparedImage:
    """Chuan hoa anh, resize giu ti le va canh, nen JPEG chat luong va payload gioi han."""
    source_path: Path | str | None = None
    if isinstance(source, FrameSample):
        source_path = source.path
        if timestamp == 0.0:
            timestamp = source.timestamp
        if frame_index == 0:
            frame_index = source.index
        if crop_region is None:
            crop_region = source.crop_region
    elif isinstance(source, (str, Path)):
        source_path = str(source)

    with open_frame_image(source, crop_region=crop_region) as raw_img:
        img = raw_img.convert("RGB")

    if normalize:
        img = ImageOps.autocontrast(img)

    orig_w, orig_h = img.size
    if max_dimension > 0 and max(orig_w, orig_h) > max_dimension:
        ratio = float(max_dimension) / float(max(orig_w, orig_h))
        new_w = max(1, int(round(orig_w * ratio)))
        new_h = max(1, int(round(orig_h * ratio)))
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    sharpness, contrast, score = compute_clarity_score(img)

    q = max(1, min(100, int(jpeg_quality)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    payload = buf.getvalue()

    if max_payload_bytes is not None and max_payload_bytes > 0:
        while len(payload) > max_payload_bytes and q > 20:
            q = max(10, q - 15)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            payload = buf.getvalue()

        while len(payload) > max_payload_bytes and max(img.size) > 64:
            cur_w, cur_h = img.size
            new_w = max(32, int(round(cur_w * 0.75)))
            new_h = max(32, int(round(cur_h * 0.75)))
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=min(q, 75), optimize=True)
            payload = buf.getvalue()

    final_w, final_h = img.size
    content_hash = hashlib.sha256(payload).hexdigest()

    return PreparedImage(
        payload=payload,
        width=final_w,
        height=final_h,
        timestamp=timestamp,
        format="JPEG",
        content_hash=content_hash,
        sharpness=sharpness,
        contrast=contrast,
        score=score,
        frame_index=frame_index,
        path=source_path,
    )


def check_cancelled(
    should_cancel: Callable[[], bool] | None = None,
    token: Any | None = None,
) -> bool:
    """Kiem tra yeu cau huy bo tu callback hoac CancelToken (ho tro ca bool va callable)."""
    if should_cancel is not None and should_cancel():
        return True
    if token is not None:
        val = getattr(token, "cancelled", None)
        if callable(val):
            if val():
                return True
        elif bool(val):
            return True

        val_is = getattr(token, "is_cancelled", None)
        if callable(val_is):
            if val_is():
                return True
        elif bool(val_is):
            return True
    return False


def select_visual_segments(
    frames: Sequence[FrameSample | tuple[float, Path | str] | tuple[float, Any] | Path | str | Any],
    *,
    fps: float | None = None,
    diff_threshold: float = 4.5,
    blank_threshold: float = 3.0,
    max_dimension: int = 1280,
    jpeg_quality: int = 88,
    max_payload_bytes: int | None = None,
    normalize: bool = False,
    crop_region: Sequence[int] | None = None,
    consensus_frames: int = 1,
    accuracy_mode: bool = False,
    min_separation_seconds: float = 0.4,
    final_frame_gap: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
    token: Any | None = None,
) -> list[VisualSegment]:
    """Phan doan cac khung hinh thanh cac phan doan thi giac on dinh.

    Dac diem:
    - Tai va xu ly tung khung hinh mot (bounded memory, mo khong qua 1 anh mot luc).
    - Khung trong (blank) ket thuc phan doan truoc do.
    - Khung thay doi thi giac ket thuc phan doan truoc do va bat dau phan doan moi.
    - Chon khung dai dien ro net nhat (tranh khung mo/fade).
    - Ho tro che do consensus 1, 2, hoac 3 khung hinh phan bo cach nhau du xa.
    - ID xac dinh, ben vung tu timeline/index.
    """
    if not frames:
        return []

    # Chuyen doi dau vao sang danh sach FrameSample
    samples: list[FrameSample] = []
    for i, raw_item in enumerate(frames):
        samples.append(coerce_sample(raw_item, index=i, fps=fps))

    if not samples:
        return []

    # Uoc tinh frame gap cho moc ket thuc khung cuoi
    if final_frame_gap is not None and final_frame_gap > 0:
        inferred_gap = float(final_frame_gap)
    elif len(samples) >= 2:
        diffs = [
            samples[i].timestamp - samples[i - 1].timestamp
            for i in range(1, len(samples))
            if samples[i].timestamp > samples[i - 1].timestamp
        ]
        inferred_gap = float(np.median(diffs)) if diffs else 0.5
    elif fps and fps > 0:
        inferred_gap = 1.0 / float(fps)
    else:
        inferred_gap = 0.5

    # Buoc 1: Quet tung khung hinh trich xuat sieu du lieu nhe, dong handle ngay
    metas: list[_FrameMeta] = []
    for sample in samples:
        if check_cancelled(should_cancel, token):
            break
        cr = crop_region if crop_region is not None else sample.crop_region
        with open_frame_image(sample, crop_region=cr) as img:
            blank = is_blank_frame(img, std_threshold=blank_threshold)
            dhash = compute_dhash(img)
            norm_img = _normalize_structural_topology(img)
            phash = compute_phash(norm_img, hash_size=16)
            thumb = np.asarray(
                norm_img.resize((64, 32), Image.Resampling.BILINEAR), dtype=np.uint8
            )
            sharpness, contrast, score = compute_clarity_score(img)

        metas.append(
            _FrameMeta(
                index=sample.index,
                timestamp=sample.timestamp,
                path=sample.path,
                is_blank=blank,
                dhash=dhash,
                phash=phash,
                luma_thumb=thumb,
                sharpness=sharpness,
                contrast=contrast,
                score=score,
                sample=sample,
            )
        )

    if not metas:
        return []

    # Buoc 2: Phan doan theo ranh gioi: blank ket thuc phan doan truoc;
    # segment bat dau tu matching frame dau tien, ket thuc tai visual change / blank tiep theo.
    raw_segments: list[dict[str, Any]] = []
    current_frames: list[_FrameMeta] = []
    current_ref_meta: _FrameMeta | None = None
    seg_start_ts: float = 0.0

    def close_segment(end_ts: float) -> None:
        nonlocal current_frames, current_ref_meta
        if current_frames:
            raw_segments.append(
                {
                    "start": seg_start_ts,
                    "end": end_ts,
                    "frames": list(current_frames),
                }
            )
            current_frames = []
            current_ref_meta = None

    for meta in metas:
        if meta.is_blank:
            if current_frames:
                close_segment(end_ts=meta.timestamp)
            continue

        if not current_frames:
            seg_start_ts = meta.timestamp
            current_frames.append(meta)
            current_ref_meta = meta
        else:
            assert current_ref_meta is not None
            diff = compute_visual_difference(current_ref_meta, meta)
            if diff <= diff_threshold:
                current_frames.append(meta)
            else:
                close_segment(end_ts=meta.timestamp)
                seg_start_ts = meta.timestamp
                current_frames.append(meta)
                current_ref_meta = meta

    if current_frames:
        last_ts = current_frames[-1].timestamp
        close_segment(end_ts=last_ts + inferred_gap)

    # Buoc 3: Chon representative va consensus frames, tao PreparedImage cho tung frame duoc chon
    effective_consensus = consensus_frames
    if accuracy_mode and consensus_frames <= 1:
        effective_consensus = 3

    segments: list[VisualSegment] = []
    for seg_idx, seg_info in enumerate(raw_segments):
        if check_cancelled(should_cancel, token):
            break

        seg_frames: list[_FrameMeta] = seg_info["frames"]
        # Sap xep theo diem chat luong giam dan
        sorted_by_score = sorted(seg_frames, key=lambda f: f.score, reverse=True)
        best_meta = sorted_by_score[0]

        # Chon top 1, 2, hoac 3 khung hinh phan bo cach nhau du xa
        chosen_metas: list[_FrameMeta] = [best_meta]
        if effective_consensus > 1:
            for cand in sorted_by_score[1:]:
                if len(chosen_metas) >= effective_consensus:
                    break
                is_separated = all(
                    abs(cand.timestamp - c.timestamp) >= min_separation_seconds
                    for c in chosen_metas
                )
                if is_separated:
                    chosen_metas.append(cand)

            if len(chosen_metas) < effective_consensus:
                for cand in sorted_by_score:
                    if len(chosen_metas) >= min(effective_consensus, len(seg_frames)):
                        break
                    if cand not in chosen_metas:
                        chosen_metas.append(cand)

        # Sap xep chosen_metas theo thoi gian
        chosen_metas_chrono = sorted(chosen_metas, key=lambda f: f.timestamp)

        # Tao PreparedImage cho tung khung da chon (mo va dong tung khung mot)
        prepared_samples: list[PreparedImage] = []
        best_prepared: PreparedImage | None = None

        for meta_item in chosen_metas_chrono:
            source_item: Any = meta_item.path if meta_item.path is not None else meta_item.sample
            cr = crop_region
            if cr is None and meta_item.sample is not None:
                cr = meta_item.sample.crop_region

            prep = prepare_image(
                source_item,
                max_dimension=max_dimension,
                jpeg_quality=jpeg_quality,
                max_payload_bytes=max_payload_bytes,
                normalize=normalize,
                crop_region=cr,
                timestamp=meta_item.timestamp,
                frame_index=meta_item.index,
            )
            prepared_samples.append(prep)
            if meta_item == best_meta:
                best_prepared = prep

        if best_prepared is None and prepared_samples:
            best_prepared = prepared_samples[0]

        assert best_prepared is not None

        seg_id = f"seg_{seg_idx:04d}_{seg_info['start']:.3f}_{seg_info['end']:.3f}"
        segment = VisualSegment(
            id=seg_id,
            start=seg_info["start"],
            end=seg_info["end"],
            representative=best_prepared,
            samples=prepared_samples,
            content_hash=best_prepared.content_hash,
            frame_count=len(seg_frames),
            start_index=seg_frames[0].index,
            end_index=seg_frames[-1].index,
        )
        segments.append(segment)

    return segments


segment_frames = select_visual_segments


def filter_duplicate_segments(
    segments: Sequence[VisualSegment],
    *,
    by: str = "content_hash",
) -> list[VisualSegment]:
    """Loc bo cac phan doan trung lap dung content_hash de tan dung cache."""
    seen: set[str] = set()
    unique: list[VisualSegment] = []
    for seg in segments:
        h = seg.content_hash if by == "content_hash" else seg.representative.content_hash
        if h not in seen:
            seen.add(h)
            unique.append(seg)
    return unique


deduplicate_segments = filter_duplicate_segments

