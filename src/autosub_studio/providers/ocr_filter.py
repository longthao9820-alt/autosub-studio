"""Loc chu theo mau va chieu cao truoc khi doc, de OCR khong bat nham chu khac.

Phu de chay san tren hinh gan nhu luon co mot mau rieng (hay gap nhat la mau
trang) va chieu cao chu on dinh tu dau den cuoi video. Con logo, chu quang cao,
chu tren bang hieu trong canh thi khac mau hoac khac co. Bo loc o day lam hai
viec:

1. To den nhung diem anh khong dung mau chu phu de roi moi dua cho bo doc chu.
   Chu khac mau bien mat truoc khi bo doc chu kip nhin thay, nen khong con bi
   ghep lan vao cau thoai.
2. Bo cac vung chu qua thap hoac qua cao so voi chieu cao chu phu de.

Ca hai muc deu do duoc tu dong tu chinh video: xem 'fill_color' (do mau ruot
chu) va 'height_range' (khoang chieu cao chap nhan duoc).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_DISTANCE = 441.673  # khoang cach xa nhat giua hai mau trong khong gian RGB
WHITE = "#FFFFFF"


def _cv2() -> Any:
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _np() -> Any:
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def available() -> bool:
    """Co du thu vien xu ly anh de loc theo mau khong."""
    return _cv2() is not None and _np() is not None


# --------------------------------------------------------------------------- mau


def parse_color(text: str) -> tuple[int, int, int] | None:
    """Doc mau dang #RRGGBB hoac R,G,B. Sai dinh dang thi tra ve None."""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("#"):
        raw = raw[1:]
        if len(raw) == 3:
            raw = "".join(c * 2 for c in raw)
        if len(raw) != 6:
            return None
        try:
            return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
        except ValueError:
            return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    if len(parts) != 3:
        return None
    try:
        values = [int(float(p)) for p in parts]
    except ValueError:
        return None
    if any(v < 0 or v > 255 for v in values):
        return None
    return (values[0], values[1], values[2])


def format_color(rgb: tuple[int, int, int] | None) -> str:
    if not rgb:
        return ""
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, int(v))) for v in rgb))


def color_name(rgb: tuple[int, int, int] | None) -> str:
    """Ten mau tieng Viet de ghi vao nhat ky cho de hieu."""
    if not rgb:
        return "chua ro"
    r, g, b = (int(v) for v in rgb)
    high, low = max(r, g, b), min(r, g, b)
    if high - low <= 32:
        if high >= 200:
            return "trang"
        if high <= 60:
            return "den"
        return "xam"
    if r >= 180 and g >= 150 and b <= 120:
        return "vang"
    if r >= 150 and g <= 120 and b <= 120:
        return "do"
    if g >= 150 and r <= 140:
        return "xanh la"
    if b >= 150 and r <= 150:
        return "xanh duong"
    if r >= 150 and b >= 150:
        return "tim"
    return "mau khac"


# --------------------------------------------------------------------------- bo loc


@dataclass
class TextFilter:
    """Dieu kien de mot vung chu duoc coi la phu de.

    color de rong nghia la chua do mau, luc do buoc doc chu se tu do tu video.
    min_height / max_height bang 0 nghia la chua gioi han chieu cao.
    """

    color: str = ""
    tolerance: float = 15.0  # phan tram sai lech mau cho phep
    min_height: float = 0.0  # px
    max_height: float = 0.0  # px
    brightness: int = 0  # -100..100
    contrast: int = 0  # -100..100
    use_color: bool = False
    drop_static: bool = False  # tuy chon rieng; NTS mac dinh khong bo
    ignore_texts: tuple[str, ...] = ()  # cac dong chu da biet chac la khong phai thoai
    min_fill: float = 0.008  # ti le diem anh dung mau chu it nhat trong mot vung
    max_fill: float = 0.92  # cao hon muc nay la mang mau dac, khong phai chu

    # ------------------------------------------------------------------ tien ich

    @property
    def rgb(self) -> tuple[int, int, int] | None:
        return parse_color(self.color) if self.use_color else None

    @property
    def touches_image(self) -> bool:
        """Co can mo tep anh ra xu ly khong."""
        return bool(self.rgb) or bool(self.brightness) or bool(self.contrast)

    @property
    def active(self) -> bool:
        if self.touches_image or self.ignore_texts:
            return True
        return self.min_height > 0 or self.max_height > 0

    @property
    def needs_probe(self) -> bool:
        """Con thieu chieu cao chu phai do tu video khong.

        Mau chu la tuy chon chi dinh cua nguoi dung, khong tu suy doan trong
        luong OCR mac dinh. Nut do rieng van co the dien mau khi nguoi dung can.
        """
        return self.min_height <= 0 and self.max_height <= 0

    def height_ok(self, height: float) -> bool:
        if self.min_height > 0 and height < self.min_height:
            return False
        return not (self.max_height > 0 and height > self.max_height)

    def fill_ok(self, ratio: float) -> bool:
        return self.min_fill <= ratio <= self.max_fill

    def describe(self) -> str:
        parts: list[str] = []
        if self.use_color:
            rgb = self.rgb
            if rgb:
                parts.append(f"mau chu {format_color(rgb)} ({color_name(rgb)})")
                parts.append(f"sai lech {self.tolerance:.0f}%")
            else:
                parts.append("mau chu: se tu do tu video")
        low = f"{self.min_height:.0f}" if self.min_height > 0 else "-"
        high = f"{self.max_height:.0f}" if self.max_height > 0 else "-"
        parts.append(f"cao {low}..{high} px")
        if self.brightness:
            parts.append(f"sang {self.brightness:+d}")
        if self.contrast:
            parts.append(f"tuong phan {self.contrast:+d}")
        if self.drop_static:
            parts.append("bo chu co dinh")
        if self.ignore_texts:
            parts.append(f"bo san {len(self.ignore_texts)} dong chu dan")
        return ", ".join(parts)


# --------------------------------------------------------------------------- anh


def load_image(path: str | Path) -> Any:
    """Doc tep anh thanh mang BGR. Doc qua numpy de duong dan tieng Viet van chay."""
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        return None
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def adjust(image: Any, brightness: int, contrast: int) -> Any:
    """Chinh sang va tuong phan (-100..100, 0 la giu nguyen)."""
    if image is None or (not brightness and not contrast):
        return image
    np = _np()
    if np is None:
        return image
    gain = 1.0 + max(-95, min(100, int(contrast))) / 100.0
    shift = max(-100, min(100, int(brightness))) * 1.28
    out = image.astype(np.float32)
    out = (out - 128.0) * gain + 128.0 + shift
    return np.clip(out, 0, 255).astype(np.uint8)


def color_mask(image: Any, rgb: tuple[int, int, int], tolerance: float) -> Any:
    """Mang 0/1 danh dau nhung diem anh dung mau chu."""
    np = _np()
    if image is None or np is None:
        return None
    target = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)  # doi sang BGR
    diff = image.astype(np.float32) - target
    dist = np.sqrt((diff * diff).sum(axis=2))
    limit = max(2.0, min(100.0, float(tolerance))) / 100.0 * MAX_DISTANCE
    return (dist <= limit).astype(np.uint8)


def grow_mask(mask: Any, size: int = 3) -> Any:
    """No nhe net chu cho lien mach lai sau khi bi cat bot o vien."""
    cv2, np = _cv2(), _np()
    if mask is None or cv2 is None or np is None or size < 2:
        return mask
    return cv2.dilate(mask, np.ones((size, size), np.uint8))


def has_text_pixels(mask: Any, least: int = 40) -> bool:
    """Trong khung hinh nay co du diem anh dung mau chu de coi la co chu khong.

    Khung hinh khong co phu de thi sau khi loc mau chi con vai diem anh le. Biet
    truoc nhu vay thi khoi phai goi bo doc chu, doc ca video nhanh hon dang ke.
    """
    if mask is None:
        return True
    return int(mask.sum()) >= max(1, int(least))


def masked_image(image: Any, mask: Any) -> Any:
    """Anh chi con chu: chu den tren nen trang, cho bo doc chu ket qua sach hon.

    Bo doc chu quen voi chu den tren giay trang hon la chu sang tren nen den,
    nen ta dao mau lai chu khong giu nguyen anh goc.
    """
    np = _np()
    if image is None or mask is None or np is None:
        return image
    out = np.full(image.shape, 255, dtype=np.uint8)
    out[mask > 0] = 0
    return out


def rect_fill(mask: Any, rect: tuple[float, float, float, float] | None) -> float:
    """Ti le diem anh dung mau chu nam trong mot vung chu."""
    if mask is None or rect is None:
        return 1.0
    height, width = mask.shape[:2]
    x1 = max(0, int(rect[0]))
    y1 = max(0, int(rect[1]))
    x2 = min(width, int(rect[2]) + 1)
    y2 = min(height, int(rect[3]) + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


# ------------------------------------------------------------------ do mau chu


def _color_options(crop: Any, groups: int = 5) -> Any:
    """Vai mau dai dien cho mot vung chu, lay bang cach gom nhom diem anh."""
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        return []
    pixels = crop.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 60:
        return []
    if len(pixels) > 20000:  # lay thua du de gom nhom cho nhanh
        pixels = pixels[np.linspace(0, len(pixels) - 1, 20000).astype(np.int32)]
    rule = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, _, centers = cv2.kmeans(pixels, groups, None, rule, 3, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return []
    return centers


def _looks_like_text(crop: Any, color: Any, tolerance: float = 45.0) -> float:
    """Diem cho biet mot mau la net chu hay chi la mang mau nen.

    Net chu mong va keo dai gan het chieu ngang cua vung nen bi an di khi ta
    mai mon anh; con mang nen thi to, mai mon xong van con lai.
    """
    cv2, np = _cv2(), _np()
    if cv2 is None or np is None:
        return 0.0
    diff = crop.astype(np.float32) - color
    dist = np.sqrt((diff * diff).sum(axis=2))
    mask = (dist <= tolerance).astype(np.uint8)
    cover = float(mask.mean())
    if cover < 0.02 or cover > 0.45:
        return 0.0
    span = float((mask.max(axis=0) > 0).mean())
    eroded = float(cv2.erode(mask, np.ones((7, 7), np.uint8)).mean())
    thin = 1.0 - eroded / max(cover, 1e-6)
    return span * max(0.0, thin)


def fill_color(
    image: Any, rect: tuple[float, float, float, float] | None
) -> tuple[int, int, int] | None:
    """Mau ruot chu trong mot vung chu doc duoc.

    Chu phu de thuong co vien den bao quanh, nen trong mot vung chu co ba lop
    mau: nen video, vien den va ruot chu. Ta lay lop nao vua giong net chu vua
    sang nhat, vi ruot chu luon sang hon vien.
    """
    np = _np()
    if image is None or rect is None or np is None:
        return None
    height, width = image.shape[:2]
    x1 = max(0, int(rect[0]))
    y1 = max(0, int(rect[1]))
    x2 = min(width, int(rect[2]) + 1)
    y2 = min(height, int(rect[3]) + 1)
    if x2 - x1 < 12 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2]
    best: tuple[float, Any] | None = None
    for color in _color_options(crop):
        if _looks_like_text(crop, color) < 0.45:
            continue
        light = 0.114 * float(color[0]) + 0.587 * float(color[1]) + 0.299 * float(color[2])
        if best is None or light > best[0]:
            best = (light, color)
    if best is None:
        return None
    bgr = best[1]
    return (int(round(float(bgr[2]))), int(round(float(bgr[1]))), int(round(float(bgr[0]))))


def pick_color(samples: list[tuple[tuple[int, int, int], float]]) -> tuple[int, int, int] | None:
    """Chon mau phu de tu nhieu mau do duoc, moi mau kem trong so dien tich.

    Gom cac mau gan nhau thanh mot nhom roi lay nhom co tong dien tich lon
    nhat: chu phu de chiem nhieu dien tich hon logo hay chu phu khac.
    """
    np = _np()
    if not samples or np is None:
        return None
    groups: list[dict[str, Any]] = []
    for rgb, weight in samples:
        base = np.array(rgb, dtype=np.float32)
        share = max(0.0, float(weight))
        for group in groups:
            if float(np.linalg.norm(base - group["center"])) <= 60.0:
                total = group["weight"] + share
                group["center"] = (group["center"] * group["weight"] + base * share) / max(
                    total, 1e-6
                )
                group["weight"] = total
                break
        else:
            groups.append({"center": base, "weight": share})
    if not groups:
        return None
    groups.sort(key=lambda g: -g["weight"])
    center = groups[0]["center"]
    return (
        int(round(float(center[0]))),
        int(round(float(center[1]))),
        int(round(float(center[2]))),
    )


def height_range(heights: list[float]) -> tuple[float, float]:
    """Khoang chieu cao chu chap nhan duoc, tinh tu cac chieu cao do duoc."""
    np = _np()
    if not heights or np is None:
        return (0.0, 0.0)
    middle = float(np.median(heights))
    if middle <= 0:
        return (0.0, 0.0)
    return (round(middle * 0.62, 1), round(middle * 1.75, 1))
