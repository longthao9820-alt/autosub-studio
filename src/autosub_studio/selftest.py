"""Tu kiem tra moi truong: dung khi vua chep phan mem sang mot may khac."""

from __future__ import annotations

import difflib
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import APP_NAME, APP_VERSION
from .providers import asr, local_voice, ocr, ocr_filter, separate, tts
from .services import gpu, media
from .services.ffmpeg import FFmpeg
from .services.paths import app_root, config_dir, human_size, is_portable
from .services.settings import Settings

OK = "DAT"
WARN = "THIEU"
FAIL = "HONG"


@dataclass
class CheckResult:
    """Ket qua mot muc kiem tra."""

    name: str
    status: str
    detail: str

    def line(self) -> str:
        return f"[{self.status:5}] {self.name}\n         {self.detail}"


def _check_ffmpeg(settings: Settings) -> list[CheckResult]:
    ff = FFmpeg(settings.ffmpeg_path, settings.ffprobe_path)
    if not ff.available:
        return [
            CheckResult(
                "FFmpeg",
                FAIL,
                "Khong tim thay ffmpeg.exe. Hau het chuc nang se khong chay.\n"
                "         Cach sua: chep ffmpeg.exe va ffprobe.exe vao thu muc 'ffmpeg' "
                "canh tep chay,\n         hoac vao tab Cai dat chung de chon duong dan.",
            )
        ]
    results = [CheckResult("FFmpeg", OK, f"{ff.ffmpeg}\n         {ff.version()}")]
    if ff.ffprobe and Path(ff.ffprobe).is_file():
        results.append(CheckResult("FFprobe", OK, ff.ffprobe))
    else:
        results.append(
            CheckResult(
                "FFprobe",
                FAIL,
                "Thieu ffprobe.exe nen khong doc duoc thong tin video.",
            )
        )
    return results


def _check_asr(settings: Settings) -> list[CheckResult]:
    if not asr.is_available():
        return [CheckResult("Nhan dang giong noi", FAIL, asr.install_hint())]
    source, local = asr.resolve_model_source(settings.asr_model, settings.model_dir)
    bundled = asr.bundled_model_dirs()
    if local and Path(source).is_dir():
        size = sum(f.stat().st_size for f in Path(source).rglob("*") if f.is_file())
        detail = (
            f"Model kem theo: {Path(source).name} ({human_size(size)})\n"
            f"         Chay duoc ngay, khong can Internet."
        )
        return [CheckResult("Nhan dang giong noi", OK, detail)]
    if local:
        return [
            CheckResult(
                "Nhan dang giong noi", OK, f"Model '{source}' da co trong bo nho dem tren may."
            )
        ]
    names = ", ".join(p.name for p in bundled) or "khong co"
    return [
        CheckResult(
            "Nhan dang giong noi",
            WARN,
            f"Model '{settings.asr_model}' chua co san (model kem theo: {names}).\n"
            "         Lan chay dau se tai ve, buoc nay can Internet.",
        )
    ]


def _check_ocr() -> list[CheckResult]:
    if not ocr.is_available():
        return [CheckResult("Doc chu tren hinh (OCR)", WARN, ocr.install_hint())]
    try:
        ocr._load_engine()
    except Exception as exc:
        return [
            CheckResult(
                "Doc chu tren hinh (OCR)", FAIL, f"Cai roi nhung khong khoi dong duoc: {exc}"
            )
        ]
    results = [CheckResult("Doc chu tren hinh (OCR)", OK, "San sang, chay hoan toan tren may.")]
    if ocr_filter.available():
        results.append(
            CheckResult(
                "Loc chu theo mau va chieu cao",
                OK,
                "San sang. Mac dinh dung anh goc; chi loc mau khi ban chi dinh.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Loc chu theo mau va chieu cao",
                WARN,
                "Thieu opencv-python nen OCR co the bat nham logo, chu quang cao. "
                "Chay: pip install opencv-python",
            )
        )
    return results


def _check_tts() -> list[CheckResult]:
    """Kiem tra runtime Piper Local, catalog va cac model giong doc da tai."""
    results: list[CheckResult] = []
    ready, reason = local_voice.piper_runtime_ready()
    catalog = local_voice.list_catalog()
    manager = local_voice.get_default_manager()
    ready_voices = [v for v in catalog if manager.get_status(v.id) == local_voice.STATUS_READY]

    if not ready:
        results.append(
            CheckResult(
                "Giong doc offline (Piper Local)",
                WARN,
                f"{reason}\n         Co the chep piper vao assets/piper hoac dung ban dong goi.",
            )
        )
    elif not ready_voices:
        results.append(
            CheckResult(
                "Giong doc offline (Piper Local)",
                WARN,
                f"Runtime san sang. Danh muc co {len(catalog)} giong, chua tai model nao ve may.\n"
                "         Co the tai model tai tab B3 hoac dung mang mot lan de tai.",
            )
        )
    else:
        names = ", ".join(v.name for v in ready_voices)
        results.append(
            CheckResult(
                "Giong doc offline (Piper Local)",
                OK,
                f"Runtime san sang. Da co {len(ready_voices)}/{len(catalog)} model: {names}",
            )
        )
    return results


def _check_gpu(settings: Settings) -> list[CheckResult]:
    """Kiem tra tang toc bang card do hoa."""
    results: list[CheckResult] = []
    ready, reason = gpu.cuda_ready()
    if ready:
        results.append(CheckResult("Nhan dang giong noi bang GPU", OK, reason))
    elif gpu.driver_present():
        results.append(CheckResult("Nhan dang giong noi bang GPU", WARN, reason))
    else:
        results.append(
            CheckResult(
                "Nhan dang giong noi bang GPU",
                WARN,
                f"{reason}\n         Van chay binh thuong bang CPU, chi cham hon.",
            )
        )
    if ocr.is_available():
        results.append(
            CheckResult(
                "Doc chu tren hinh bang GPU",
                OK if ocr.gpu_available() else WARN,
                "Dung card do hoa, nhanh hon nhieu lan."
                if ocr.gpu_available()
                else "Chay bang CPU. Can card NVIDIA va thu vien CUDA di kem.",
            )
        )
    ff = FFmpeg(settings.ffmpeg_path, settings.ffprobe_path)
    encoder = gpu.best_hw_encoder(ff.ffmpeg) if ff.available else ""
    if encoder:
        results.append(
            CheckResult(
                "Render video bang phan cung",
                OK,
                f"Dung {gpu.encoder_label(encoder)}, render nhanh hon nhieu.",
            )
        )
    else:
        results.append(
            CheckResult("Render video bang phan cung", WARN, "Khong co, se render bang CPU.")
        )
    return results


def _check_storage(settings: Settings) -> list[CheckResult]:
    from .services.paths import free_space

    workspace = Path(settings.workspace)
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        probe = workspace / ".ghi-thu.tmp"
        probe.write_text("x", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return [CheckResult("Thu muc lam viec", FAIL, f"Khong ghi duoc vao {workspace}: {exc}")]
    free = free_space(workspace)
    status = OK if free > 5 * 1024**3 else WARN
    return [
        CheckResult(
            "Thu muc lam viec",
            status,
            f"{workspace}\n         Con trong {human_size(free)} tren o dia.",
        )
    ]


def run_checks() -> tuple[list[CheckResult], str]:
    """Chay het cac muc kiem tra. Tra ve (danh sach ket qua, ket luan chung)."""
    settings = Settings.load()
    results: list[CheckResult] = []
    results += _check_ffmpeg(settings)
    results += _check_asr(settings)
    results += _check_gpu(settings)
    results += _check_ocr()
    results += _check_tts()
    results += _check_storage(settings)
    results.append(
        CheckResult(
            "Tach nhac chat luong cao",
            OK if separate.demucs_available() else WARN,
            "San sang."
            if separate.demucs_available()
            else "Chua cai Demucs. Van tach duoc bang FFmpeg o muc co ban.",
        )
    )

    if any(r.status == FAIL for r in results):
        verdict = "CHUA CHAY DUOC - hay xem cac muc danh dau HONG o tren."
    elif any(r.status == WARN for r in results):
        verdict = "CHAY DUOC - mot vai tinh nang phu chua san sang (muc THIEU)."
    else:
        verdict = "CHAY DUOC DAY DU - tat ca deu san sang."
    return results, verdict


SAMPLE_TEXT = "欢迎来到中国，我们一起学习。"
SPOKEN_TEXT = "The quick brown fox jumps over the lazy dog."


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _deep_check_ocr(ff: FFmpeg, work: Path) -> CheckResult:
    """Ve mot dong phu de tieng Trung roi doc lai bang che do Nhanh Nhu NTS."""
    if not ocr.is_available():
        return CheckResult("Chay thu OCR", WARN, "Chua cai bo doc chu tren hinh.")
    font = next(
        (
            p
            for p in (
                Path("C:/Windows/Fonts/msyh.ttc"),
                Path("C:/Windows/Fonts/msyhbd.ttc"),
                Path("C:/Windows/Fonts/simsun.ttc"),
            )
            if p.is_file()
        ),
        None,
    )
    if font is None:
        return CheckResult("Chay thu OCR", WARN, "Khong tim thay font tieng Trung de tao anh thu.")
    frame = work / "frame.png"
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGB", (1280, 180), "black")
        draw = ImageDraw.Draw(image)
        face = ImageFont.truetype(str(font), 72)
        box = draw.textbbox((0, 0), SAMPLE_TEXT, font=face, stroke_width=2)
        x = max(10, (image.width - (box[2] - box[0])) // 2)
        draw.text(
            (x, 42),
            SAMPLE_TEXT,
            font=face,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        image.save(frame)
        on_gpu = ocr.gpu_available()
        started = time.monotonic()
        engine = ocr._load_engine(on_gpu, ocr.NTS_FAST_SERVER, 5)
        got = ocr.read_frame(engine, frame, 0.5)[0]
        elapsed = time.monotonic() - started
    except Exception as exc:
        return CheckResult("Chay thu OCR", FAIL, f"Khong doc duoc chu tren anh: {exc}")
    accuracy = difflib.SequenceMatcher(None, got, SAMPLE_TEXT).ratio()
    where = "card do hoa" if on_gpu else "CPU"
    if accuracy >= 0.9:
        return CheckResult(
            "Chay thu OCR",
            OK,
            f"Chay tren {where} het {elapsed:.1f} giay, doc tieng Trung "
            f"dat {accuracy:.0%}: {got!r}",
        )
    return CheckResult("Chay thu OCR", FAIL, f"Doc tieng Trung chi dat {accuracy:.0%}: {got!r}")


def _deep_check_tts(work: Path) -> CheckResult:
    """Chay thu giong doc Piper Local neu runtime va model da san sang."""
    ready, reason = local_voice.piper_runtime_ready()
    if not ready:
        return CheckResult("Chay thu giong doc", WARN, f"Bo qua: {reason}")
    manager = local_voice.get_default_manager()
    catalog = local_voice.list_catalog()
    ready_voice = next(
        (v for v in catalog if manager.get_status(v.id) == local_voice.STATUS_READY),
        None,
    )
    if not ready_voice:
        return CheckResult(
            "Chay thu giong doc",
            WARN,
            "Bo qua: chua co model Piper nao duoc tai ve may de chay thu.",
        )
    spoken = work / "piper_test.wav"
    try:
        started = time.monotonic()
        tts.synthesize(tts.PROVIDER_LOCAL, SPOKEN_TEXT, spoken, voice=ready_voice.id)
        elapsed = time.monotonic() - started
        size = spoken.stat().st_size if spoken.is_file() else 0
        if size < 64:
            return CheckResult("Chay thu giong doc", FAIL, "Tep am thanh tao ra rong.")
        return CheckResult(
            "Chay thu giong doc",
            OK,
            f"Da tao am thanh bang {ready_voice.name} het {elapsed:.1f} giay ({human_size(size)}).",
        )
    except Exception as exc:
        return CheckResult("Chay thu giong doc", FAIL, f"Loi khi tao giong doc: {exc}")


def _deep_check_asr(ff: FFmpeg, work: Path, settings: Settings) -> CheckResult:
    """Nhan dang lai cau thu bang model Whisper kem theo."""
    if not asr.is_available():
        return CheckResult("Chay thu nhan dang giong noi", FAIL, asr.install_hint())
    ready, _ = local_voice.piper_runtime_ready()
    manager = local_voice.get_default_manager()
    catalog = local_voice.list_catalog()
    ready_voice = next(
        (v for v in catalog if manager.get_status(v.id) == local_voice.STATUS_READY),
        None,
    )
    if not ready or not ready_voice:
        return CheckResult(
            "Chay thu nhan dang giong noi",
            WARN,
            "Chua co runtime Piper hoac model giong doc tren may de tao cau thu.",
        )
    spoken = work / "spoken.wav"
    wav16 = work / "spoken_16k.wav"
    try:
        tts.synthesize(tts.PROVIDER_LOCAL, SPOKEN_TEXT, spoken, voice=ready_voice.id)
        media.extract_audio(ff, spoken, wav16, rate=16000, channels=1)
    except Exception as exc:
        return CheckResult("Chay thu nhan dang giong noi", FAIL, f"Khong tao duoc cau thu: {exc}")
    device = "GPU (CUDA)" if gpu.cuda_ready()[0] else "CPU"
    started = time.monotonic()
    try:
        cues, lang = asr.transcribe(
            wav16,
            model_size=settings.asr_model,
            model_dir=settings.model_dir,
            language="en",
            device=device,
            use_gpu=settings.use_gpu,
            duration=media.wav_duration(wav16),
        )
    except Exception as exc:
        return CheckResult("Chay thu nhan dang giong noi", FAIL, str(exc))
    elapsed = time.monotonic() - started
    got = " ".join(c.text for c in cues)
    hit = len(_words(got) & _words(SPOKEN_TEXT))
    total = len(_words(SPOKEN_TEXT))
    where = "card do hoa" if device.startswith("GPU") else "CPU"
    detail = (
        f"Chay tren {where} het {elapsed:.1f} giay, nhan ra {hit}/{total} tu "
        f"(ngon ngu {lang}):\n         {got.strip()!r}"
    )
    return CheckResult("Chay thu nhan dang giong noi", OK if hit >= total * 0.6 else FAIL, detail)


def deep_checks() -> list[CheckResult]:
    """Kiem tra sau: chay that OCR, giong doc, nhan dang giong noi va render."""
    settings = Settings.load()
    ff = FFmpeg(settings.ffmpeg_path, settings.ffprobe_path)
    if not ff.available:
        return [CheckResult("Kiem tra sau", FAIL, "Khong co FFmpeg nen bo qua cac buoc chay thu.")]
    work = Path(tempfile.mkdtemp(prefix="autosub_deep_"))
    try:
        results = [
            _deep_check_ocr(ff, work),
            _deep_check_tts(work),
            _deep_check_asr(ff, work, settings),
        ]
        results.append(_deep_check_render_real(ff, work))
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _deep_check_render_real(ff: FFmpeg, work: Path) -> CheckResult:
    """Tao video ngan, ghep phu de cung roi doc lai tep ket qua."""
    from .core.formats import write_ass
    from .core.models import Cue, SubtitleDoc

    source = work / "src.mp4"
    ass = work / "sub.ass"
    out = work / "out.mp4"
    use_nvenc = media.use_gpu_encoder(ff, True)
    try:
        ff.run(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=15:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=200:duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "32",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ]
        )
        doc = SubtitleDoc(cues=[Cue(0.2, 1.8, "Cau thu nghiem", "Cau thu nghiem")])
        ass.write_text(
            write_ass(doc, font_size=24, play_res_x=320, play_res_y=180), encoding="utf-8"
        )
        media.burn_subtitles(
            ff, source, ass, out, crf=32, preset="ultrafast", gpu=use_nvenc, duration=2.0
        )
        info = ff.probe(out)
    except Exception as exc:
        return CheckResult("Chay thu render", FAIL, f"Khong render duoc: {exc}")
    if not out.is_file() or out.stat().st_size < 2000:
        return CheckResult("Chay thu render", FAIL, "Tep ket qua rong.")
    return CheckResult(
        "Chay thu render",
        OK,
        f"Da render video thu bang {gpu.encoder_label(use_nvenc)} "
        f"({info.resolution}, {human_size(out.stat().st_size)}).",
    )


def report_text(deep: bool = False) -> str:
    """Ban bao cao day du dang van ban."""
    results, verdict = run_checks()
    if deep:
        extra = deep_checks()
        results = results + extra
        if any(r.status == FAIL for r in results):
            verdict = "CHUA CHAY DUOC - hay xem cac muc danh dau HONG o tren."
        elif any(r.status == WARN for r in results):
            verdict = "CHAY DUOC - mot vai tinh nang phu chua san sang (muc THIEU)."
        else:
            verdict = "CHAY DUOC DAY DU - da chay thu that va deu dat."
    head = [
        f"{APP_NAME} {APP_VERSION} - KET QUA TU KIEM TRA",
        f"Thoi diem : {datetime.now():%d/%m/%Y %H:%M:%S}",
        f"May       : {platform.node()} | Windows {platform.release()}",
        f"Thu muc   : {app_root()}",
        f"Che do    : {'di dong (du lieu nam canh tep chay)' if is_portable() else 'thuong'}",
        f"Cau hinh  : {config_dir()}",
        "",
        "-" * 68,
    ]
    body = [r.line() for r in results]
    tail = ["-" * 68, "", f"KET LUAN: {verdict}"]
    return "\n".join(head + body + tail)


def main(deep: bool = False) -> int:
    """Chay tu kiem tra, ghi bao cao ra tep va hien bang thong bao."""
    text = report_text(deep=deep)
    target = app_root() / "ket-qua-kiem-tra.txt"
    try:
        target.write_text(text, encoding="utf-8")
        saved = f"\n\nDa luu bao cao vao:\n{target}"
    except OSError:
        saved = ""

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        existing = QApplication.instance()
        app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
        box = QMessageBox()
        box.setWindowTitle(f"{APP_NAME} - Tu kiem tra")
        box.setText("Ket qua kiem tra may nay:")
        box.setDetailedText(text)
        box.setInformativeText(text.split("KET LUAN: ")[-1] + saved)
        box.exec()
        del app
    except Exception:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--full" in sys.argv or "--sau" in sys.argv))
