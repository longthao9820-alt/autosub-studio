"""Phat hien va bat tang toc bang card do hoa NVIDIA.

Thu vien CUDA duoc mang theo ung dung nen chep sang may khac van dung duoc,
mien la may do co card NVIDIA va da cai driver. Neu khong co, moi thu tu dong
lui ve chay bang CPU chu khong bao loi.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .paths import app_root, bundled_dir

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CUDA_DIR_NAME = "cuda"

# Anh xa muc nen cua libx264 sang muc nen tuong duong cua bo ma hoa NVIDIA.
_NVENC_PRESETS = {
    "ultrafast": "p1",
    "veryfast": "p2",
    "fast": "p3",
    "medium": "p4",
    "slow": "p6",
}


@lru_cache(maxsize=1)
def cuda_dll_dirs() -> tuple[str, ...]:
    """Cac thu muc chua DLL CUDA di kem ung dung hoac trong moi truong phat trien."""
    found: list[str] = []
    bundled = bundled_dir(CUDA_DIR_NAME)
    if bundled is not None:
        found.append(str(bundled))
    for extra in (app_root() / "assets" / CUDA_DIR_NAME,):
        if extra.is_dir():
            found.append(str(extra))
    # Khi chay tu ma nguon, lay tu cac goi nvidia-* da cai trong moi truong ao.
    try:
        import nvidia  # type: ignore[import-not-found]

        root = Path(next(iter(nvidia.__path__)))
        for child in sorted(root.rglob("bin")):
            if child.is_dir() and any(child.glob("*.dll")):
                found.append(str(child.resolve()))
    except Exception:
        pass
    seen: list[str] = []
    for item in found:
        if item not in seen:
            seen.append(item)
    return tuple(seen)


def register_cuda_dlls() -> int:
    """Bao cho Windows biet cho tim DLL CUDA. Goi mot lan luc khoi dong."""
    if os.name != "nt":
        return 0
    count = 0
    for path in cuda_dll_dirs():
        try:
            os.add_dll_directory(path)
            count += 1
        except (OSError, AttributeError):
            continue
    if count:
        os.environ["PATH"] = os.pathsep.join((*cuda_dll_dirs(), os.environ.get("PATH", "")))
    return count


@lru_cache(maxsize=1)
def driver_present() -> bool:
    """May co card NVIDIA kem driver hay khong."""
    if os.name != "nt":
        return False
    try:
        ctypes.WinDLL("nvcuda.dll")  # type: ignore[attr-defined]
    except OSError:
        return False
    return True


@lru_cache(maxsize=1)
def gpu_name() -> str:
    """Ten card do hoa, chuoi rong neu khong doc duoc."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    line = (res.stdout or "").strip().splitlines()
    return line[0].strip() if res.returncode == 0 and line else ""


@lru_cache(maxsize=1)
def graphics_adapters() -> tuple[str, ...]:
    """Danh sach card man hinh tren Windows, gom ca NVIDIA, Intel va AMD."""
    if os.name != "nt":
        return ()
    command = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { if ($_.Name) { $_.Name.Trim() } }"
    )
    try:
        res = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        res = None
    names: list[str] = []
    if res is not None and res.returncode == 0:
        for line in (res.stdout or "").splitlines():
            name = line.strip()
            if name and name not in names:
                names.append(name)
    fallback = gpu_name()
    if not names and fallback:
        names.append(fallback)
    return tuple(names)


def machine_signature() -> str:
    """Chu ky ngan de nhan ra thu muc tool da duoc chuyen sang may khac."""
    computer = os.environ.get("COMPUTERNAME", "").strip().lower()
    raw = "|".join((computer, *[name.lower() for name in graphics_adapters()]))
    return hashlib.sha256((raw or "unknown-machine").encode("utf-8")).hexdigest()[:20]


@lru_cache(maxsize=1)
def machine_id() -> str:
    """Dinh danh may ma khong do card GPU hay khoi tao CUDA."""
    computer = os.environ.get("COMPUTERNAME", "").strip().lower()
    machine_guid = ""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                machine_guid = str(winreg.QueryValueEx(key, "MachineGuid")[0]).strip().lower()
        except OSError:
            pass
    raw = f"{computer}|{machine_guid}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@lru_cache(maxsize=1)
def cuda_libs_present() -> bool:
    """Da co du thu vien CUDA de chay nhan dang giong noi tren GPU chua."""
    needed = ("cublas64_12.dll", "cudnn64_9.dll")
    for path in cuda_dll_dirs():
        folder = Path(path)
        if all((folder / name).is_file() for name in needed):
            return True
    return False


def cuda_ready() -> tuple[bool, str]:
    """Co chay duoc nhan dang giong noi tren GPU khong, kem ly do neu khong."""
    if os.name != "nt":
        return False, "Chi ho tro tang toc GPU tren Windows."
    if not driver_present():
        return False, (
            "Khong thay card NVIDIA hoac chua cai driver. "
            "Cai driver tai nvidia.com/drivers roi mo lai phan mem."
        )
    if not cuda_libs_present():
        return False, (
            "Thieu thu vien CUDA di kem (thu muc 'cuda'). "
            "Co the ban da chep thieu tep khi mang thu muc sang may khac."
        )
    name = gpu_name()
    return True, f"Dung GPU: {name}" if name else "Dung GPU NVIDIA."


@lru_cache(maxsize=1)
def onnx_cuda_ready() -> bool:
    """Bo doc chu tren hinh co chay duoc bang card do hoa khong."""
    if not driver_present() or not cuda_libs_present():
        return False
    try:
        import onnxruntime  # type: ignore[import-not-found]
    except ImportError:
        return False
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


@lru_cache(maxsize=1)
def _encoders(ffmpeg_path: str) -> frozenset[str]:
    if not ffmpeg_path:
        return frozenset()
    try:
        res = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in (res.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


# Cac bo ma hoa bang phan cung, thu theo thu tu nay.
HW_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf")
HW_LABELS = {
    "h264_nvenc": "card NVIDIA (NVENC)",
    "h264_qsv": "chip Intel (Quick Sync)",
    "h264_amf": "card AMD (AMF)",
    "": "bo xu ly (CPU)",
}


def _probe_encoder(ffmpeg_path: str, name: str) -> bool:
    """Ma hoa thu vai khung hinh de biet may nay dung duoc bo ma hoa do khong.

    Chi xem danh sach cua FFmpeg thi chua du, vi danh sach van liet ke ca bo
    ma hoa ma may khong co phan cung tuong ung.
    """
    try:
        res = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x144:d=0.2",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=40,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


@lru_cache(maxsize=4)
def best_hw_encoder(ffmpeg_path: str) -> str:
    """Bo ma hoa phan cung tot nhat may nay dung duoc, chuoi rong neu khong co."""
    if not ffmpeg_path:
        return ""
    listed = _encoders(ffmpeg_path)
    for name in HW_ENCODERS:
        if name in listed and _probe_encoder(ffmpeg_path, name):
            return name
    return ""


def nvenc_available(ffmpeg_path: str) -> bool:
    """May co ma hoa video bang phan cung khong (ten giu lai cho quen thuoc)."""
    return bool(best_hw_encoder(ffmpeg_path))


def encoder_label(name: str) -> str:
    return HW_LABELS.get(name, name or "bo xu ly (CPU)")


@dataclass(frozen=True)
class AccelerationProfile:
    """Ket qua kiem tra tang toc phu hop voi mot may cu the."""

    adapters: tuple[str, ...]
    signature: str
    cuda_ready: bool
    cuda_reason: str
    ocr_cuda_ready: bool
    ocr_cuda_reason: str
    video_encoder: str

    @property
    def use_gpu(self) -> bool:
        return self.cuda_ready or self.ocr_cuda_ready

    @property
    def use_gpu_encoder(self) -> bool:
        return bool(self.video_encoder)

    def summary(self) -> str:
        cards = ", ".join(self.adapters) if self.adapters else "Không phát hiện card màn hình"
        recognition = "GPU NVIDIA (CUDA)" if self.cuda_ready else "CPU"
        ocr_mode = (
            "GPU NVIDIA (CUDA, đã chạy thử thành công)"
            if self.ocr_cuda_ready
            else f"CPU ({self.ocr_cuda_reason})"
        )
        render = encoder_label(self.video_encoder)
        return (
            f"Card màn hình: {cards}\n"
            f"Nhận dạng giọng nói: {recognition}\n"
            f"OCR: {ocr_mode}\n"
            f"Render video: {render}"
        )


def acceleration_profile(ffmpeg_path: str) -> AccelerationProfile:
    """Kiem tra va de xuat cau hinh GPU; tac vu nay nen chay o luong nen."""
    adapters = graphics_adapters()
    cuda_ok, cuda_reason = cuda_ready()
    ocr_cuda_ok = False
    if not cuda_ok:
        ocr_cuda_reason = cuda_reason
    elif not onnx_cuda_ready():
        ocr_cuda_reason = "ONNX Runtime không nạp được CUDA trên máy này"
    else:
        from ..providers import ocr  # nap muon de tranh vong nhap khau

        ocr_cuda_ok = ocr.gpu_available()
        ocr_cuda_reason = (
            "Đã chạy thử OCR bằng CUDA thành công"
            if ocr_cuda_ok
            else "chạy thử OCR bằng CUDA thất bại; kiểm tra driver NVIDIA"
        )
    encoder = best_hw_encoder(ffmpeg_path)
    return AccelerationProfile(
        adapters=adapters,
        signature=machine_signature(),
        cuda_ready=cuda_ok,
        cuda_reason=cuda_reason,
        ocr_cuda_ready=ocr_cuda_ok,
        ocr_cuda_reason=ocr_cuda_reason,
        video_encoder=encoder,
    )


def refresh_detection() -> None:
    """Buoc bam kiem tra phai do lai that, khong dung ket qua cache cu."""
    for cached in (
        cuda_dll_dirs,
        driver_present,
        gpu_name,
        graphics_adapters,
        cuda_libs_present,
        onnx_cuda_ready,
        _encoders,
        best_hw_encoder,
    ):
        cached.cache_clear()
    try:
        from ..providers import ocr

        ocr.gpu_available.cache_clear()
    except ImportError:
        pass


def video_encoder_args(crf: int, preset: str, encoder: str | bool = "") -> list[str]:
    """Tham so ma hoa video cho bo ma hoa da chon.

    `encoder` la ten bo ma hoa phan cung; de rong hoac False thi dung CPU.
    """
    quality = max(0, min(51, int(crf)))
    if encoder is True:
        encoder = "h264_nvenc"
    elif encoder is False:
        encoder = ""

    if encoder == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            _NVENC_PRESETS.get(preset, "p4"),
            "-rc",
            "vbr",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v",
            "h264_qsv",
            "-preset",
            preset,
            "-global_quality",
            str(quality),
            "-pix_fmt",
            "nv12",
        ]
    if encoder == "h264_amf":
        return [
            "-c:v",
            "h264_amf",
            "-quality",
            "balanced",
            "-rc",
            "cqp",
            "-qp_i",
            str(quality),
            "-qp_p",
            str(quality),
            "-pix_fmt",
            "yuv420p",
        ]
    return ["-c:v", "libx264", "-crf", str(quality), "-preset", preset, "-pix_fmt", "yuv420p"]


def status_text(ffmpeg_path: str = "") -> str:
    """Mot doan mo ta ngan cho giao dien."""
    ready, reason = cuda_ready()
    lines = [reason if ready else f"Nhan dang giong noi: chay bang CPU. {reason}"]
    from ..providers import ocr  # nap muon de tranh vong nhap khau

    lines.append(
        "Doc chu tren hinh (OCR): dung card do hoa"
        if ocr.gpu_available()
        else "Doc chu tren hinh (OCR): chay bang CPU"
    )
    if ffmpeg_path:
        encoder = best_hw_encoder(ffmpeg_path)
        lines.append(
            f"Render bang phan cung: {encoder_label(encoder)}"
            if encoder
            else "Render bang phan cung: khong co, se render bang CPU"
        )
    return "\n".join(lines)
