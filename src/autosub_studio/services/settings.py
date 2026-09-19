"""Cau hinh ung dung va noi luu khoa API (ma hoa theo tai khoan Windows)."""

from __future__ import annotations

import base64
import contextlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import config_dir, default_workspace, is_portable, write_text_atomic

CONFIG_NAME = "config.json"
SECRETS_NAME = "secrets.dat"
SCHEMA_VERSION = 12


def _is_moved_portable_workspace(value: str) -> bool:
    """Nhan duong dan Data/workspace duoc luu tuyet doi tren may cu."""
    if not value:
        return False
    parts = [part.casefold() for part in Path(value).parts]
    return len(parts) >= 2 and parts[-2:] == ["data", "workspace"]


@dataclass
class SubtitleStyle:
    """Kieu chu phu de dung khi hien thi va khi render."""

    font: str = "Arial"
    font_size: int = 16
    bold: bool = True
    primary_color: str = "#e0e196"
    outline_color: str = "#000000"
    back_color: str = "#000000"
    back_opacity: int = 50  # phan tram, 0 = khong nen
    outline: float = 2.0
    shadow: float = 1.0
    alignment: int = 2  # 1..9 theo chuan ASS, 2 = duoi giua
    margin_v: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleStyle:
        base = cls()
        for k, v in (data or {}).items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base


@dataclass
class Settings:
    """Toan bo cau hinh chung cua ung dung."""

    schema_version: int = SCHEMA_VERSION
    workspace: str = ""
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    model_dir: str = ""
    max_workers: int = 2
    task_timeout_minutes: int = 120
    use_gpu: bool = True  # nhan dang giong noi tren card do hoa
    use_gpu_encoder: bool = True  # render video bang bo ma hoa cua card do hoa
    keep_temp: bool = False
    auto_export_srt: bool = True  # tach sub xong thi luu luon .srt canh video
    language: str = "vi"  # ngon ngu giao dien
    autosave_seconds: int = 60
    limit_cpu: bool = False
    fast_concat: bool = True
    format_capcut: bool = False
    add_background_music: bool = False
    smart_cut: bool = False
    play_on_edit: bool = False
    edit_volume: int = 50
    enter_newline: bool = False
    left_screen: bool = False
    hardware_signature: str = ""
    hardware_machine_id: str = ""

    # Nhan dang giong noi
    asr_model: str = "small"
    asr_language: str = "auto"
    asr_vad: bool = True
    asr_max_chars: int = 42
    asr_max_lines: int = 2
    asr_min_duration: float = 0.6
    asr_max_duration: float = 7.0

    # Doc chu tren hinh
    # Mac dinh chat luong cao nhu NTS: PP-OCRv4 Mobile tieng Trung va doc toi
    # da moi 2 frame nguon (video 30 fps thanh 15 hinh/giay). Toc do duoc xu ly
    # o pool GPU/ONNX, khong ha tan suat lay mau de tranh bo sot subtitle ngan.
    # Khong do lai moc thoi gian lan hai.
    ocr_mode: str = "Nhanh Như NTS"
    ocr_server: str = "PP-OCRv4 Mobile (Nhanh Như NTS)"
    ocr_language: str = "Simplified Chinese"
    ocr_batch_size: int = 5
    ocr_consensus: int = 1
    ocr_fps: float = 15.0
    ocr_confidence: float = 70.0
    ocr_similarity: float = 0.80
    ocr_min_duration: float = 0.20
    ocr_continuous: bool = False
    ocr_refine: bool = False  # che do chuan moi do lai; NTS bo buoc nay de chay nhanh
    ocr_drop_chars: str = ""
    ocr_drop_words: str = ""

    # Loc chu de khong bat nham logo, chu quang cao... trong vung da khoanh
    ocr_color_filter: bool = False  # chi bat khi nguoi dung chu dong chi dinh mau
    ocr_text_color: str = ""  # mau tuy chon; de trong nghia la khong loc mau
    ocr_color_tolerance: float = 15.0  # phan tram sai lech mau cho phep
    ocr_min_height: float = 25.0  # px
    ocr_max_height: float = 200.0  # px
    ocr_brightness: int = 0  # -100..100
    ocr_contrast: int = 0  # -100..100; 0 giu nguyen anh, hop nhat cho PP-OCRv6
    ocr_drop_static: bool = False  # tuy chon rieng, NTS mac dinh khong bat
    ocr_cache_enabled: bool = True

    # Dich
    translate_provider: str = "Google (mien phi)"
    translate_batch: int = 8
    translate_prompt: str = ""
    source_language: str = "auto"
    target_language: str = "vi"
    translate_context: int = 2
    llm_model: str = "claude-opus-5"

    # Long tieng
    tts_provider: str = "VoiceStudio Local (English US)"
    tts_language: str = "en-US"
    tts_voice: str = "Kitten English Male 2|kittentts|expr-voice-2-m"
    tts_rate: int = 0
    tts_volume: int = 100
    tts_speed_percent: int = 100
    tts_pitch_percent: int = 100
    tts_dictionary: str = ""
    tts_pause_period_ms: int = 300
    tts_pause_comma_ms: int = 200
    tts_pause_newline_ms: int = 400
    tts_fit_timing: bool = True
    tts_store_voice: bool = False
    tts_cache_enabled: bool = True
    tts_allow_overlap: bool = False
    tts_short_threshold_ms: int = 300
    tts_end_pause_ms: int = 250
    tts_voice_profiles: list[dict[str, Any]] = field(default_factory=list)
    dub_source: str = "translation"  # "original" hoac "translation"
    dub_output_mode: str = "video"  # "video" hoac "audio"
    dub_timing_mode: str = "voice"  # "voice" hoac "subtitle"
    dub_use_original_video: bool = True
    original_audio_volume: int = 20
    timeline_workers: int = 2
    timeline_cpu_percent: int = 50
    timeline_decode_library: str = "FFmpeg"
    music_volume: int = 35
    ducking: bool = True
    eq_bass: int = 0
    eq_mid: int = 0
    eq_treble: int = 0
    keep_original_audio: bool = True

    # Render
    render_crf: int = 20
    render_preset: str = "medium"
    render_scale: str = "giu nguyen"
    render_fps: str = "giu nguyen"

    # AI Gateway
    ai_endpoint: str = ""
    ai_model_sub: str = "sub"
    ai_thinking_sub: str = "low"
    ai_model_prime: str = "prime"
    ai_thinking_prime: str = "medium"

    # Export & Update
    output_folder: str = ""
    auto_check_update: bool = True

    # Voice & Preset
    local_voice: str = ""
    default_preset: str = "DEFAULT"

    # Cau hinh tuy chinh giong NTS: moi profile la mot ban chup setting, khong
    # chua duong dan may, dau van tay phan cung hay chinh danh sach profile.
    active_config_profile: str = "default"
    config_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    capcut_path: str = ""
    capcut_template_draft: str = ""
    download_folder: str = ""
    download_proxy: str = ""

    # Du lieu mo rong tu ban cu hoac tuy bien khong bi mat
    extra: dict[str, Any] = field(default_factory=dict)

    style: SubtitleStyle = field(default_factory=SubtitleStyle)

    def __post_init__(self) -> None:
        if not self.workspace:
            self.workspace = str(default_workspace())
        if not self.capcut_path:
            self.capcut_path = str(Path.home() / "AppData" / "Local" / "CapCut")

    def profile_snapshot(self) -> dict[str, Any]:
        """Ban chup setting co the luu/doi qua lai nhu cau hinh APP cua NTS."""
        excluded = {
            "schema_version",
            "workspace",
            "hardware_signature",
            "hardware_machine_id",
            "active_config_profile",
            "config_profiles",
            "extra",
            "output_folder",
            "auto_check_update",
            "ai_endpoint",
            "ai_model_sub",
            "ai_thinking_sub",
            "ai_model_prime",
            "ai_thinking_prime",
        }
        return {key: value for key, value in asdict(self).items() if key not in excluded}

    def remember_active_profile(self) -> None:
        name = self.active_config_profile.strip() or "default"
        self.active_config_profile = name
        self.config_profiles[name] = self.profile_snapshot()

    def apply_config_profile(self, name: str) -> bool:
        profile = self.config_profiles.get(str(name))
        if not isinstance(profile, dict):
            return False
        for key, value in profile.items():
            if key == "style" and isinstance(value, dict):
                self.style = SubtitleStyle.from_dict(value)
                continue
            if not hasattr(self, key) or key in {"config_profiles", "active_config_profile"}:
                continue
            current = getattr(self, key)
            try:
                setattr(self, key, type(current)(value) if current is not None else value)
            except (TypeError, ValueError):
                continue
        self.active_config_profile = str(name)
        return True

    # ------------------------------------------------------------------ luu/doc

    @staticmethod
    def config_path() -> Path:
        return config_dir() / CONFIG_NAME

    @classmethod
    def load(cls) -> Settings:
        """Doc cau hinh. Neu tep hong thi dung mac dinh, khong lam sap ung dung."""
        path = cls.config_path()
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        settings = cls()
        old_schema = int(data.get("schema_version", 1) or 1)
        legacy_ocr = "ocr_mode" not in data
        style = data.pop("style", None)
        for key, value in data.items():
            if hasattr(settings, key) and key not in {"style", "extra"}:
                try:
                    current = getattr(settings, key)
                    setattr(settings, key, type(current)(value) if current is not None else value)
                except (TypeError, ValueError):
                    continue
            elif key != "style":
                settings.extra[key] = value
        if isinstance(style, dict):
            settings.style = SubtitleStyle.from_dict(style)
        # Nang cap mot lan sang bo PP-OCRv4 Mobile giong NTS. Viec nay cung
        # dam bao ban portable dang co san cau hinh cu se dung che do moi ngay.
        if legacy_ocr or old_schema < 3:
            settings.ocr_mode = "Nhanh Như NTS"
            settings.ocr_server = "PP-OCRv4 Mobile (Nhanh Như NTS)"
            settings.ocr_language = "Simplified Chinese"
            settings.ocr_batch_size = 5
            settings.ocr_consensus = 1
            settings.ocr_fps = 15.0
            settings.ocr_confidence = 70.0
            settings.ocr_similarity = 0.80
            settings.ocr_min_duration = 0.20
            settings.ocr_refine = False
        # NTS dung nguong 70%. Ban schema 3 cua ta de 50%, khien cac net trang
        # o chuyen canh co diem 0.50-0.69 bi giu nham thanh chu Latin/so.
        if old_schema < 4 and (
            "nts" in settings.ocr_mode.casefold() or "nts" in settings.ocr_server.casefold()
        ):
            settings.ocr_confidence = 70.0
        # Ban schema 4 gan nhan "Nhanh Nhu NTS" nhung moi doc 5 hinh/giay,
        # dong thoi tu do mau trang va bat bo chu co dinh. NTS that su doc moi
        # 2 khung hinh nguon va chi loc mau khi nguoi dung chu dong chi dinh.
        # Dua ca cau hinh portable da luu ve dung mac dinh nay mot lan.
        if old_schema < 5 and (
            "nts" in settings.ocr_mode.casefold() or "nts" in settings.ocr_server.casefold()
        ):
            settings.ocr_fps = 15.0
            settings.ocr_color_filter = False
            settings.ocr_text_color = ""
            settings.ocr_drop_static = False
        # Schema 6 tam ha tan suat NTS xuong 3 hinh/giay de chua toc do, nhung
        # cach do co the bo sot caption ngan o video khac. Loi goc nam o moi
        # ONNX session tu tao ca pool CPU cho tung engine GPU; schema 7 sua tai
        # engine va dua tan suat chat luong cao ve lai 15 hinh/giay.
        if old_schema < 7 and (
            "nts" in settings.ocr_mode.casefold() or "nts" in settings.ocr_server.casefold()
        ):
            settings.ocr_fps = 15.0
        # Ban cu dung SAPI voi cac giong Windows tieng Anh (David/Zira), nen
        # van tao duoc WAV nhung khong doc dung tieng Viet. Nang cap mot lan
        # sang dung "Giong Free" cua NTS; sau do nguoi dung van co the chon
        # lai SAPI neu muon.
        migrated_voice = False
        if old_schema < 11:
            settings.tts_provider = "VoiceStudio Local (English US)"
            settings.tts_language = "en-US"
            settings.tts_voice = "Kitten English Male 2|kittentts|expr-voice-2-m"
            settings.target_language = "en"
            settings.tts_voice_profiles = []
            migrated_voice = True
            settings.tts_fit_timing = settings.dub_timing_mode == "subtitle"
        if old_schema < 12:
            if not settings.default_preset:
                settings.default_preset = "DEFAULT"
            if not settings.ai_model_sub:
                settings.ai_model_sub = "sub"
            if not settings.ai_model_prime:
                settings.ai_model_prime = "prime"
            if settings.translate_provider in {
                "Claude (can khoa API)",
                "Claude",
                "claude",
            }:
                settings.translate_provider = "AI Gateway"
            if old_schema == 11 and settings.tts_provider in {
                "VoiceStudio Local (English US)",
                "Windows SAPI (offline)",
                "Edge TTS (can Internet)",
                "VoiceStudio",
                "SAPI",
                "Edge",
            }:
                settings.tts_provider = "Local Voice"
                migrated_voice = True
            for prof in settings.config_profiles.values():
                if isinstance(prof, dict):
                    if old_schema == 11 and prof.get("tts_provider") in {
                        "VoiceStudio Local (English US)",
                        "Windows SAPI (offline)",
                        "Edge TTS (can Internet)",
                        "VoiceStudio",
                        "SAPI",
                        "Edge",
                    }:
                        prof["tts_provider"] = "Local Voice"
                    if prof.get("translate_provider") in {
                        "Claude (can khoa API)",
                        "Claude",
                        "claude",
                    }:
                        prof["translate_provider"] = "AI Gateway"
        # Ban schema 1 dat nham +100 (muc cuc dai) lam mac dinh, khien net
        # chu Trung Quoc bi bet/mat khi loc mau. Dua gia tri mac dinh cu ve
        # trung tinh; cac gia tri nguoi dung chon khac van duoc giu nguyen.
        if old_schema < 2 and settings.ocr_contrast == 100:
            settings.ocr_contrast = 0
        portable_paths_changed = False
        if is_portable():
            current_workspace = str(default_workspace())
            if (
                _is_moved_portable_workspace(settings.workspace)
                and Path(settings.workspace) != Path(current_workspace)
            ):
                settings.workspace = current_workspace
                portable_paths_changed = True
            # Cac duong dan tuy chon cua may cu khong duoc lam ban portable
            # loi. De trong se tu dong dung FFmpeg/model da dong goi kem tool.
            for field_name in ("ffmpeg_path", "ffprobe_path", "model_dir"):
                value = str(getattr(settings, field_name) or "").strip()
                if value and not Path(value).exists():
                    setattr(settings, field_name, "")
                    portable_paths_changed = True
        # Ket qua GPU chi duoc phep dung lai tren dung may da bam kiem tra.
        # Dinh danh nay khong do card va khong khoi tao CUDA, nen giao dien van
        # hien "Chua kiem tra" tren may moi cho den khi nguoi dung bam nut.
        from .gpu import machine_id

        current_machine_id = machine_id()
        hardware_machine_changed = settings.hardware_machine_id != current_machine_id
        if hardware_machine_changed:
            settings.hardware_machine_id = current_machine_id
            settings.hardware_signature = ""
            settings.use_gpu = False
            settings.use_gpu_encoder = False
        settings.schema_version = SCHEMA_VERSION
        if not settings.config_profiles:
            settings.config_profiles = {"default": settings.profile_snapshot()}
        if settings.active_config_profile not in settings.config_profiles:
            settings.active_config_profile = "default"
        if migrated_voice:
            settings.config_profiles[settings.active_config_profile] = settings.profile_snapshot()
        if (
            legacy_ocr
            or old_schema < SCHEMA_VERSION
            or portable_paths_changed
            or hardware_machine_changed
        ):
            with contextlib.suppress(OSError):
                settings.save()
        return settings

    def save(self) -> Path:
        data = asdict(self)
        extra = data.pop("extra", {})
        data["style"] = self.style.to_dict()
        for k, v in extra.items():
            if k not in data:
                data[k] = v
        return write_text_atomic(self.config_path(), json.dumps(data, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------ khoa API

    @staticmethod
    def _secrets_path() -> Path:
        return config_dir() / SECRETS_NAME

    @classmethod
    def load_secrets(cls) -> dict[str, str]:
        path = cls._secrets_path()
        if not path.is_file():
            return {}
        try:
            blob = path.read_bytes()
            raw = _unprotect(blob)
            data = json.loads(raw.decode("utf-8"))
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    @classmethod
    def save_secrets(cls, secrets: dict[str, str]) -> None:
        raw = json.dumps(secrets, ensure_ascii=False).encode("utf-8")
        path = cls._secrets_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(_protect(raw))
        tmp.replace(path)

    @classmethod
    def get_secret(cls, name: str) -> str:
        return cls.load_secrets().get(name, "")

    @classmethod
    def set_secret(cls, name: str, value: str) -> None:
        secrets = cls.load_secrets()
        if value:
            secrets[name] = value
        else:
            secrets.pop(name, None)
        cls.save_secrets(secrets)


# --------------------------------------------------------------------------- DPAPI


def _protect(raw: bytes) -> bytes:
    """Ma hoa bang DPAPI cua Windows, chi tai khoan hien tai giai duoc."""
    try:
        return b"DPAPI" + _dpapi(raw, protect=True)
    except (OSError, ImportError, RuntimeError):
        return b"B64__" + base64.b64encode(raw)


def _unprotect(blob: bytes) -> bytes:
    if blob.startswith(b"DPAPI"):
        return _dpapi(blob[5:], protect=False)
    if blob.startswith(b"B64__"):
        return base64.b64decode(blob[5:])
    raise ValueError("Tep khoa API khong dung dinh dang")


def _dpapi(data: bytes, *, protect: bool) -> bytes:
    import ctypes
    import ctypes.wintypes as wt

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.WinDLL("crypt32.dll")  # type: ignore[attr-defined]
    kernel32 = ctypes.WinDLL("kernel32.dll")  # type: ignore[attr-defined]
    buf = ctypes.create_string_buffer(data, len(data))
    src = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = Blob()
    func = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    args = (
        (ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out))
        if protect
        else (ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out))
    )
    if not func(*args):
        raise RuntimeError("DPAPI that bai")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)
