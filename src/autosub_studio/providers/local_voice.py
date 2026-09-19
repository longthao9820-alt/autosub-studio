"""Bo quan ly va thuc thi giong doc offline Piper Local V2."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..services.paths import (
    piper_bin_path,
    piper_espeak_data_dir,
    piper_models_dir,
)

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

STATUS_NOT_DOWNLOADED = "not_downloaded"
STATUS_DOWNLOADING = "downloading"
STATUS_READY = "ready"


class TTSError(RuntimeError):
    """Loi khi tao giong doc hoac xu ly model."""


class ChecksumMismatchError(TTSError):
    """Loi ma kiem tra SHA256 khong khop."""


@dataclass(frozen=True)
class VoiceArtifact:
    """Thong tin mot tep thanh phan cua model."""

    name: str
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class VoiceInfo:
    """Thong tin mo ta va metadata ban quyen cua mot giong doc trong catalog."""

    id: str
    name: str
    language: str
    language_name: str = ""
    quality: str = "medium"
    num_speakers: int = 1
    speaker_id_map: dict[str, int] = field(default_factory=dict)
    license: str = ""
    commercial_use: bool = True
    model_card_url: str = ""
    dataset_url: str = ""
    sample_rate: int = 22050
    artifacts: tuple[VoiceArtifact, ...] = field(default_factory=tuple)


# Catalog ban dau da kiem chung voi day du URL, SHA256 va metadata giay phep
_DEFAULT_VOICES: dict[str, VoiceInfo] = {
    "en_US-bryce-medium": VoiceInfo(
        id="en_US-bryce-medium",
        name="Bryce (English US)",
        language="en-US",
        language_name="English (US)",
        quality="medium",
        num_speakers=1,
        license="Public Domain",
        commercial_use=True,
        model_card_url="https://huggingface.co/rhasspy/piper-voices/raw/main/en/en_US/bryce/medium/MODEL_CARD",
        dataset_url="https://creativecommons.org/publicdomain/mark/1.0/",
        sample_rate=22050,
        artifacts=(
            VoiceArtifact(
                name="en_US-bryce-medium.onnx",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx",
                sha256="dc9caa6c313199ffb5ac698b6e542fa6cba388aeaf2731e25262e33b9810aef1",
                size_bytes=63531379,
            ),
            VoiceArtifact(
                name="en_US-bryce-medium.onnx.json",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx.json",
                sha256="7ceb1bc4af6d4e41b6d1edbb86c67e91e01eaa71f66db4cd0ae92ac704d415be",
                size_bytes=4966,
            ),
        ),
    ),
    "vi_VN-vais1000-medium": VoiceInfo(
        id="vi_VN-vais1000-medium",
        name="VAIS 1000 (Tiếng Việt)",
        language="vi-VN",
        language_name="Tiếng Việt",
        quality="medium",
        num_speakers=1,
        license="CC BY 4.0",
        commercial_use=True,
        model_card_url="https://huggingface.co/rhasspy/piper-voices/raw/main/vi/vi_VN/vais1000/medium/MODEL_CARD",
        dataset_url="https://ieee-dataport.org/documents/vais-1000-vietnamese-speech-synthesis-corpus",
        sample_rate=22050,
        artifacts=(
            VoiceArtifact(
                name="vi_VN-vais1000-medium.onnx",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
                sha256="ec7c89e2c85f4d1edc24b6120c18aaf1bda614f06b511567eb9c7c0de15e2dab",
                size_bytes=63201294,
            ),
            VoiceArtifact(
                name="vi_VN-vais1000-medium.onnx.json",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
                sha256="fafb9da1354ed4b77c31af228ed41fb41cd825c14cffa105454b25e6ae751ee0",
                size_bytes=4860,
            ),
        ),
    ),
    "zh_CN-chaowen-medium": VoiceInfo(
        id="zh_CN-chaowen-medium",
        name="Chaowen (中文)",
        language="zh-CN",
        language_name="中文",
        quality="medium",
        num_speakers=1,
        license="CC0",
        commercial_use=True,
        model_card_url="https://huggingface.co/rhasspy/piper-voices/raw/main/zh/zh_CN/chaowen/medium/MODEL_CARD",
        dataset_url="https://github.com/OHF-Voice/voice-datasets",
        sample_rate=22050,
        artifacts=(
            VoiceArtifact(
                name="zh_CN-chaowen-medium.onnx",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/chaowen/medium/zh_CN-chaowen-medium.onnx",
                sha256="820d64ac16048fbcf38dd0823d37fab5f5e0c2bd71b01ca5a50f553fac19e746",
                size_bytes=63221984,
            ),
            VoiceArtifact(
                name="zh_CN-chaowen-medium.onnx.json",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/chaowen/medium/zh_CN-chaowen-medium.onnx.json",
                sha256="a6bb2caafa0645642f13cbf7e2f6fbbb16fded66e51109fc26d622f6472fa16f",
                size_bytes=2927,
            ),
        ),
    ),
    "en_US-libritts_r-medium": VoiceInfo(
        id="en_US-libritts_r-medium",
        name="LibriTTS-R Multi-Speaker (English US)",
        language="en-US",
        language_name="English (US)",
        quality="medium",
        num_speakers=904,
        license="CC BY 4.0",
        commercial_use=True,
        model_card_url="https://huggingface.co/rhasspy/piper-voices/raw/main/en/en_US/libritts_r/medium/MODEL_CARD",
        dataset_url="https://www.openslr.org/141/",
        sample_rate=22050,
        artifacts=(
            VoiceArtifact(
                name="en_US-libritts_r-medium.onnx",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
                sha256="10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195",
                size_bytes=78580914,
            ),
            VoiceArtifact(
                name="en_US-libritts_r-medium.onnx.json",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json",
                sha256="b471dc60d2d8335e819c393d196d6fbf792817f40051257b269878505bc9afb3",
                size_bytes=20123,
            ),
        ),
    ),
}

_catalog: dict[str, VoiceInfo] = dict(_DEFAULT_VOICES)
_catalog_lock = threading.Lock()


def register_voice(info: VoiceInfo) -> None:
    """Dang ky them giong doc vao catalog (cho phep mo rong sau nay)."""
    with _catalog_lock:
        _catalog[info.id] = info


def get_voice_info(voice_id: str) -> VoiceInfo | None:
    """Lay thong tin mot giong doc theo ID."""
    with _catalog_lock:
        return _catalog.get(voice_id)


def get_catalog() -> dict[str, VoiceInfo]:
    """Lay toan bo danh muc giong doc."""
    with _catalog_lock:
        return dict(_catalog)


def list_catalog(language: str = "") -> list[VoiceInfo]:
    """Danh sach giong doc trong catalog, co the loc theo ngon ngu."""
    with _catalog_lock:
        items = list(_catalog.values())
    if not language:
        return items
    lang_clean = language.casefold().replace("_", "-")
    return [
        v
        for v in items
        if v.language.casefold().replace("_", "-") == lang_clean
        or v.language.casefold().startswith(lang_clean.split("-")[0])
    ]


def get_voice_metadata(voice_id: str) -> dict[str, Any]:
    """Xuat metadata ban quyen va nguon goc cua giong doc."""
    info = get_voice_info(voice_id)
    if not info:
        return {}
    return {
        "id": info.id,
        "name": info.name,
        "language": info.language,
        "language_name": info.language_name,
        "quality": info.quality,
        "num_speakers": info.num_speakers,
        "license": info.license,
        "commercial_use": info.commercial_use,
        "model_card_url": info.model_card_url,
        "dataset_url": info.dataset_url,
        "sample_rate": info.sample_rate,
        "artifacts": [
            {
                "name": a.name,
                "url": a.url,
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
            }
            for a in info.artifacts
        ],
    }


def compute_sha256(path: str | Path) -> str:
    """Tinh ma SHA256 cua mot tep tren dia theo tung khoi."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_cancelled(token: Any) -> bool:
    """Kiem tra trang thai huy tu nhieu loai token: CancelToken (.cancelled), event, callable."""
    if token is None:
        return False
    val = getattr(token, "cancelled", None)
    if val is not None:
        return bool(val() if callable(val) else val)
    val = getattr(token, "is_cancelled", None)
    if val is not None:
        return bool(val() if callable(val) else val)
    val = getattr(token, "is_set", None)
    if callable(val):
        return bool(val())
    if callable(token):
        return bool(token())
    return False


class PiperVoiceManager:
    """Quan ly luu tru, trang thai, tai xuong va xoa model Piper doc lap."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self._models_dir = Path(models_dir) if models_dir is not None else piper_models_dir()
        self._active_downloads: set[str] = set()
        self._lock = threading.Lock()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def get_voice_dir(self, voice_id: str) -> Path:
        """Thu muc luu rieng cho tung voice: Data/models/piper/<voice_id>."""
        return self._models_dir / voice_id

    def verify_voice_integrity(self, voice_id: str) -> bool:
        """Kiem tra tinh toan ven SHA256 cua tat ca tep artifact.

        Tep thieu hoac loi hash deu bi coi la khong hop le.
        """
        info = get_voice_info(voice_id)
        if not info or not info.artifacts:
            return False
        vdir = self.get_voice_dir(voice_id)
        if not vdir.is_dir():
            return False
        for art in info.artifacts:
            fpath = vdir / art.name
            if not fpath.is_file():
                return False
            if fpath.stat().st_size == 0:
                return False
            if art.size_bytes > 0 and fpath.stat().st_size != art.size_bytes:
                return False
            actual_sha = compute_sha256(fpath)
            if actual_sha.lower() != art.sha256.lower():
                return False
        return True

    def get_status(self, voice_id: str, verify_checksum: bool = True) -> str:
        """Trang thai model: not_downloaded | downloading | ready.

        Partial hoac corrupt khong bao gio la ready.
        """
        with self._lock:
            if voice_id in self._active_downloads:
                return STATUS_DOWNLOADING

        if verify_checksum:
            if self.verify_voice_integrity(voice_id):
                return STATUS_READY
            return STATUS_NOT_DOWNLOADED

        info = get_voice_info(voice_id)
        if not info or not info.artifacts:
            return STATUS_NOT_DOWNLOADED
        vdir = self.get_voice_dir(voice_id)
        if not vdir.is_dir():
            return STATUS_NOT_DOWNLOADED
        for art in info.artifacts:
            fpath = vdir / art.name
            if not fpath.is_file() or fpath.stat().st_size == 0:
                return STATUS_NOT_DOWNLOADED
        return STATUS_READY

    def download_voice(
        self,
        voice_id: str,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_token: Any = None,
        timeout: float = 60.0,
    ) -> Path:
        """Tai toan bo artifact cua voice ve thu muc.

        Moi artifact duoc ghi ra .part, tinh SHA256 trong khi tai,
        va chi doi ten nguyen tu (atomic rename) sang ten chinh sau khi hash khop.
        Neu bi huy hoac sai hash thi .part bi xoa ngay lap tuc.
        """
        info = get_voice_info(voice_id)
        if not info:
            raise KeyError(f"Khong tim thay giong doc trong catalog: {voice_id}")

        if self.verify_voice_integrity(voice_id):
            return self.get_voice_dir(voice_id)

        with self._lock:
            if voice_id in self._active_downloads:
                raise RuntimeError(f"Giong doc {voice_id} dang trong qua trinh tai.")
            self._active_downloads.add(voice_id)

        vdir = self.get_voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=True)

        total_expected_bytes = sum(a.size_bytes for a in info.artifacts)
        bytes_downloaded = 0

        try:
            for art in info.artifacts:
                target_file = vdir / art.name
                part_file = vdir / f"{art.name}.part"

                # Neu target da co va dung hash thi bo qua artifact nay
                if target_file.is_file() and target_file.stat().st_size > 0:
                    if compute_sha256(target_file).lower() == art.sha256.lower():
                        bytes_downloaded += target_file.stat().st_size
                        if on_progress:
                            on_progress(bytes_downloaded, total_expected_bytes)
                        continue
                    else:
                        target_file.unlink(missing_ok=True)

                if part_file.is_file():
                    part_file.unlink(missing_ok=True)

                req = urllib.request.Request(
                    art.url,
                    headers={"User-Agent": "AutoSubStudio/2.0 PiperLocalDownloader"},
                )
                hasher = hashlib.sha256()

                with (
                    urllib.request.urlopen(req, timeout=timeout) as resp,
                    open(part_file, "wb") as f,
                ):
                    while True:
                        if _is_cancelled(cancel_token):
                            raise RuntimeError("Tai giong doc da bi huy.")

                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        bytes_downloaded += len(chunk)
                        if on_progress:
                            on_progress(bytes_downloaded, total_expected_bytes)

                computed_hash = hasher.hexdigest().lower()
                if computed_hash != art.sha256.lower():
                    raise ChecksumMismatchError(
                        f"Kiem tra SHA256 that bai cho {art.name}: "
                        f"ky vong {art.sha256}, nhan duoc {computed_hash}"
                    )

                # Chi doi ten sang ten that sau khi SHA256 da kiem chung thanh cong
                part_file.replace(target_file)

            return vdir
        except Exception:
            # Xoa cac tep .part con dang do
            for art in info.artifacts:
                (vdir / f"{art.name}.part").unlink(missing_ok=True)
            raise
        finally:
            with self._lock:
                self._active_downloads.discard(voice_id)

    def delete_voice(self, voice_id: str) -> bool:
        """Xoa an toan toan bo model cua giong doc."""
        with self._lock:
            self._active_downloads.discard(voice_id)
        vdir = self.get_voice_dir(voice_id)
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)
            return True
        return False

    def get_model_paths(self, voice_id: str) -> tuple[Path, Path]:
        """Tra ve (onnx_path, config_path) neu model da san sang tren dia."""
        info = get_voice_info(voice_id)
        if not info:
            raise TTSError(f"Khong tim thay giong doc {voice_id} trong danh muc.")
        vdir = self.get_voice_dir(voice_id)
        onnx_file = next(
            (vdir / a.name for a in info.artifacts if a.name.endswith(".onnx")),
            None,
        )
        json_file = next(
            (vdir / a.name for a in info.artifacts if a.name.endswith(".onnx.json")),
            None,
        )
        if not onnx_file or not onnx_file.is_file() or not json_file or not json_file.is_file():
            msg = f"Giong doc '{voice_id}' chua san sang tren may. Hay tai truoc khi dung."
            raise TTSError(msg)
        return onnx_file, json_file


_default_manager: PiperVoiceManager | None = None
_manager_lock = threading.Lock()


def get_default_manager() -> PiperVoiceManager:
    """Lay hoac tao quan ly model Piper mac dinh."""
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = PiperVoiceManager()
        return _default_manager


def piper_runtime_ready() -> tuple[bool, str]:
    """Kiem tra runtime piper.exe da san sang hay chua.

    Khi chay tu ma nguon dev ma chua co binary thi tra ve ly do ro rang,
    khong lam sap ung dung.
    """
    exe = piper_bin_path()
    if not exe or not exe.is_file():
        return (
            False,
            "Chưa tìm thấy piper.exe. Đặt piper vào assets/piper hoặc dùng bản đóng gói đầy đủ.",
        )
    return True, "Piper Local sẵn sàng."


def build_synthesis_command(
    piper_exe: str | Path,
    onnx_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
    *,
    speaker_id: int | None = None,
    length_scale: float | None = None,
    espeak_data_dir: str | Path | None = None,
) -> list[str]:
    """Tao lenh goi piper.exe voi cac tham so hop le.

    Chi toc do tong the (global speed) duoc anh xa thanh length_scale.
    Pitch khong duoc ho tro boi Piper CLI (xu ly o buoc sau bang media.to_wav).
    """
    cmd = [
        str(piper_exe),
        "-m",
        str(onnx_path),
        "-c",
        str(config_path),
    ]
    if speaker_id is not None:
        cmd.extend(["-s", str(int(speaker_id))])
    if length_scale is not None and abs(length_scale - 1.0) > 1e-4:
        cmd.extend(["--length_scale", f"{length_scale:.4f}"])
    if espeak_data_dir:
        p = Path(espeak_data_dir)
        if p.is_dir():
            cmd.extend(["--espeak_data", str(p)])
    cmd.extend(["-f", str(out_path)])
    return cmd


def synthesize_piper(
    text: str,
    out_path: str | Path,
    *,
    voice_id: str = "en_US-bryce-medium",
    speed: float = 1.0,
    length_scale: float | None = None,
    speaker_id: int | None = None,
    timeout: float = 120.0,
    manager: PiperVoiceManager | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Goi piper.exe de tao file am thanh WAV tu van ban.

    Gia tri toc do duoc anh xa duy nhat sang length_scale:
      length_scale = 1.0 / speed.
    Khong ho tro tham so pitch tai day (FFmpeg media xu ly sau).
    """
    clean = " ".join((text or "").split())
    if not clean:
        raise TTSError("Cau khong co noi dung de doc.")

    exe = piper_bin_path()
    if not exe or not exe.is_file():
        raise TTSError(
            "Chưa tìm thấy piper.exe. Đặt piper vào assets/piper hoặc dùng bản đóng gói đầy đủ."
        )

    mgr = manager or get_default_manager()
    onnx_path, config_path = mgr.get_model_paths(voice_id)

    out = Path(out_path).with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Anh xa global speed -> length_scale
    if length_scale is None:
        clamped_speed = max(0.25, min(4.0, float(speed)))
        effective_length_scale = 1.0 / clamped_speed
    else:
        effective_length_scale = max(0.25, min(4.0, float(length_scale)))

    espeak_dir = piper_espeak_data_dir()
    cmd = build_synthesis_command(
        exe,
        onnx_path,
        config_path,
        out,
        speaker_id=speaker_id,
        length_scale=effective_length_scale,
        espeak_data_dir=espeak_dir,
    )

    if on_log:
        on_log(f"Piper: {voice_id} (length_scale={effective_length_scale:.2f})")

    try:
        proc = subprocess.run(
            cmd,
            input=clean.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise TTSError(f"Khong the thuc thi piper.exe: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TTSError("Tao giong doc qua lau, da dung lai.") from exc

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise TTSError(f"Piper bao loi (ma {proc.returncode}): {err[:300]}")

    if not out.is_file() or out.stat().st_size < 64:
        raise TTSError("Piper khong tao duoc tep am thanh hop le.")

    return out
