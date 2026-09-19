"""Cac buoc xu ly cua ung dung, dung chung cho nut bam le va kich ban tu dong."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..core import editing, formats
from ..core.models import Cue
from ..data.project import ProjectData, ProjectStore
from ..providers import asr, capcut, diarize, ocr, ocr_ai, ocr_filter, separate, translate, tts
from ..services import gpu, media
from ..services.ffmpeg import CancelledError, CancelToken, FFmpeg, FFmpegError
from ..services.paths import safe_name, unique_path
from ..services.settings import Settings, SubtitleStyle
from ..services.tasks import TaskContext

STEP_NORMALIZE = "Chuan hoa video goc"
STEP_KEEP_MUSIC = "Xoa loi thoai, giu nhac nen"
STEP_KEEP_VOICE = "Xoa nhac nen, giu loi thoai"
STEP_OCR = "Lay phu de bang chu tren hinh (OCR)"
STEP_OCR_MEASURE = "Do mau chu va chieu cao chu (OCR)"
STEP_ASR = "Lay phu de bang giong noi"
STEP_TRANSLATE = "Dich phu de"
STEP_DIARIZE = "Phan tach giong nam / nu"
STEP_DUB = "Long tieng theo ban dich"
STEP_CAPCUT = "Tao CapCut draft"
STEP_BLUR = "Che mo phu de goc"
STEP_EXPORT = "Xuat goi du an"
STEP_RENDER = "Render video"

ALL_STEPS: tuple[str, ...] = (
    STEP_NORMALIZE,
    STEP_KEEP_MUSIC,
    STEP_KEEP_VOICE,
    STEP_OCR_MEASURE,
    STEP_OCR,
    STEP_ASR,
    STEP_TRANSLATE,
    STEP_DIARIZE,
    STEP_DUB,
    STEP_CAPCUT,
    STEP_BLUR,
    STEP_EXPORT,
    STEP_RENDER,
)

DEFAULT_SCRIPT_STEPS: tuple[str, ...] = (
    STEP_ASR,
    STEP_TRANSLATE,
    STEP_DUB,
    STEP_RENDER,
)


class StepError(RuntimeError):
    """Buoc xu ly khong hoan tat duoc."""


@dataclass
class PipelineContext:
    """Moi thu mot buoc can de chay."""

    ff: FFmpeg
    settings: Settings
    store: ProjectStore
    project: ProjectData
    task: TaskContext
    api_key: str = ""
    glossary: dict[str, str] = field(default_factory=dict)

    @property
    def token(self) -> CancelToken:
        return self.task.token

    def log(self, message: str) -> None:
        self.task.log(message)

    def progress(self, percent: int) -> None:
        self.task.progress(percent)

    def check(self) -> None:
        self.task.check_cancel()

    def save(self) -> None:
        self.store.save(self.project)

    def require_video(self) -> Path:
        path = Path(self.project.video_path or self.project.original_video)
        if not path.is_file():
            raise StepError(
                "Du an chua co video. Bam 'Chon video' o tab B1 truoc khi chay buoc nay."
            )
        return path

    def require_cues(self) -> list[Cue]:
        if not self.project.doc.cues:
            raise StepError(
                "Du an chua co phu de. Hay chay buoc lay phu de hoac nhap tep SRT truoc."
            )
        return self.project.doc.cues


# --------------------------------------------------------------------------- am thanh


def ensure_audio(pc: PipelineContext) -> Path:
    """Bao dam co tep WAV 16 kHz de nhan dang giong noi."""
    existing = Path(pc.project.audio_path) if pc.project.audio_path else None
    if existing and existing.is_file():
        return existing
    video = pc.require_video()
    out = pc.project.sub_dir("audio") / "source_16k.wav"
    pc.log("Tach am thanh tu video...")
    media.extract_audio(
        pc.ff,
        video,
        out,
        duration=pc.project.duration,
        token=pc.token,
        on_progress=pc.progress,
        on_log=None,
    )
    pc.project.audio_path = str(out)
    pc.save()
    return out


def step_normalize(pc: PipelineContext) -> str:
    """Chuan hoa do phan giai, khung hinh va ma hoa cua video goc."""
    video = pc.require_video()
    out = pc.project.sub_dir("video") / "normalized.mp4"
    scale = pc.settings.render_scale
    fps = pc.settings.render_fps
    pc.log(f"Chuan hoa video: {scale}, {fps} hinh/giay")
    media.normalize_video(
        pc.ff,
        video,
        out,
        scale=scale,
        fps=fps,
        crf=pc.settings.render_crf,
        preset=pc.settings.render_preset,
        gpu=pc.settings.use_gpu_encoder,
        duration=pc.project.duration,
        token=pc.token,
        on_progress=pc.progress,
    )
    if not pc.project.original_video:
        pc.project.original_video = str(video)
    pc.project.video_path = str(out)
    info = pc.ff.probe(out)
    pc.project.duration = info.duration or pc.project.duration
    pc.save()
    return f"Da chuan hoa video: {out.name}"


def step_keep_music(pc: PipelineContext) -> str:
    """Xoa loi thoai, giu lai nhac nen."""
    video = pc.require_video()
    out = pc.project.sub_dir("audio") / "music.wav"
    pc.log("Dang khu loi thoai...")
    media.remove_vocals(
        pc.ff, video, out, duration=pc.project.duration, token=pc.token, on_progress=pc.progress
    )
    pc.project.music_path = str(out)
    pc.save()
    return f"Da tao nhac nen: {out.name}"


def step_keep_voice(pc: PipelineContext) -> str:
    """Giu loi thoai, giam nhac nen."""
    video = pc.require_video()
    out_dir = pc.project.sub_dir("audio")
    mode = separate.MODE_DEMUCS if separate.demucs_available() else separate.MODE_FFMPEG
    pc.log(f"Che do tach: {mode}")
    voice, music = separate.separate(
        pc.ff,
        video,
        out_dir,
        mode=mode,
        duration=pc.project.duration,
        token=pc.token,
        on_progress=pc.progress,
        on_log=pc.log,
    )
    pc.project.voice_path = str(voice)
    if not pc.project.music_path:
        pc.project.music_path = str(music)
    pc.save()
    return f"Da tach giong noi: {Path(voice).name}"


# --------------------------------------------------------------------------- lay phu de


def auto_srt_target(project: ProjectData) -> Path:
    """Cho biet tep .srt tu dong se nam o dau: ngay canh video nguoi dung da them.

    Lan tach dau tien, neu ben canh video da san co tep trung ten thi lay ten
    khac de khong de len tep cua nguoi dung. Nhung lan sau ghi de dung tep cu.
    """
    if project.auto_srt_path:
        remembered = Path(project.auto_srt_path)
        if remembered.parent.is_dir():
            return remembered
    source = project.original_video or project.video_path
    if source:
        beside = Path(source).with_suffix(".srt")
        if beside.parent.is_dir():
            return beside if not beside.exists() else unique_path(beside)
    return project.sub_dir("exports") / f"{safe_name(project.name, 'phu-de')}.srt"


def auto_export_srt(pc: PipelineContext) -> str:
    """Tach sub xong thi luu luon tep .srt canh video, khoi phai bam xuat tay."""
    doc = pc.project.doc
    if not pc.settings.auto_export_srt or not doc.cues:
        return ""
    mode = "translation" if doc.translated_count else "original"
    target = auto_srt_target(pc.project)
    try:
        formats.save_subtitle(target, doc, text_mode=mode)
    except OSError as exc:
        spare = pc.project.sub_dir("exports") / target.name
        try:
            formats.save_subtitle(spare, doc, text_mode=mode)
        except OSError:
            pc.log(f"Khong tu luu duoc tep .srt: {exc}")
            return ""
        pc.project.auto_srt_path = str(spare)
        pc.log(f"Thu muc video khong ghi duoc nen luu tep .srt vao: {spare}")
        return str(spare)
    pc.project.auto_srt_path = str(target)
    pc.log(f"Da tu dong luu phu de: {target}")
    return str(target)


def step_asr(pc: PipelineContext) -> str:
    """Lay phu de bang nhan dang giong noi."""
    if not asr.is_available():
        raise StepError(asr.install_hint())
    source = Path(pc.project.voice_path) if pc.project.voice_path else None
    audio = source if (source and source.is_file()) else ensure_audio(pc)
    s = pc.settings
    _source, local = asr.resolve_model_source(s.asr_model, s.model_dir)
    if not local:
        pc.log(f"Model '{s.asr_model}' chua co tren may, se tai ve lan dau (can Internet).")
    cues, detected = asr.transcribe(
        audio,
        model_size=s.asr_model,
        model_dir=s.model_dir,
        language=s.asr_language,
        device="GPU (CUDA)" if s.use_gpu else "CPU",
        use_gpu=s.use_gpu,
        vad=s.asr_vad,
        duration=pc.project.duration,
        on_progress=pc.progress,
        on_log=pc.log,
        should_cancel=lambda: pc.token.cancelled,
    )
    pc.check()
    if not cues:
        raise StepError(
            "Khong nhan ra cau noi nao. Kiem tra video co tieng khong, "
            "hoac thu model lon hon trong Cai dat chung."
        )
    cues = asr.resegment(
        cues,
        max_chars=s.asr_max_chars,
        max_lines=s.asr_max_lines,
        min_duration=s.asr_min_duration,
        max_duration=s.asr_max_duration,
    )
    pc.project.doc.cues = cues
    pc.project.doc.language = detected or s.asr_language
    editing.remove_overlaps(pc.project.doc)
    saved = auto_export_srt(pc)
    pc.save()
    note = f" Da luu {Path(saved).name} canh video." if saved else ""
    return f"Da lay {len(cues)} cau phu de (ngon ngu: {detected or 'khong ro'}).{note}"


def _frame_gap(stamps: list[float], fallback: float) -> float:
    """Khoang cach that giua hai khung hinh lien tiep, dung de do lai moc thoi gian."""
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
    if not gaps:
        return fallback
    return min(max(gaps), fallback * 2.0)


def _ocr_frame_rate(configured: float, source_fps: float, fast_nts: bool) -> float:
    """Che do NTS doc toi da moi hai frame nguon, khong ep video 25 fps thanh 15."""
    wanted = max(0.5, float(configured))
    if fast_nts and source_fps > 1.0:
        return min(wanted, max(0.5, float(source_fps) / 2.0))
    return wanted


def _ocr_cache_key(
    video: Path,
    region: list[int],
    fps: float,
    settings: Settings,
    text_filter: ocr_filter.TextFilter,
) -> str:
    """Nhan dien dung video + vung + cau hinh; doi mot muc thi khong dung cache cu."""
    try:
        stat = video.stat()
        file_identity = [str(video.resolve()), stat.st_size, stat.st_mtime_ns]
    except OSError:
        file_identity = [str(video), 0, 0]
    payload = {
        "video": file_identity,
        "region": list(region),
        "fps": round(float(fps), 6),
        "profile": settings.ocr_server or settings.ocr_mode,
        "confidence": settings.ocr_confidence,
        "filter": dict(vars(text_filter)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=list)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def text_filter_for(settings: Settings) -> ocr_filter.TextFilter:
    """Bo loc mau chu va chieu cao chu lay tu cai dat nguoi dung."""
    return ocr_filter.TextFilter(
        color=settings.ocr_text_color.strip(),
        tolerance=float(settings.ocr_color_tolerance),
        min_height=max(0.0, float(settings.ocr_min_height)),
        max_height=max(0.0, float(settings.ocr_max_height)),
        brightness=int(settings.ocr_brightness),
        contrast=int(settings.ocr_contrast),
        use_color=bool(settings.ocr_color_filter),
        drop_static=bool(settings.ocr_drop_static),
    )


def _remember_filter(pc: PipelineContext, flt: ocr_filter.TextFilter) -> None:
    """Ghi lai so lieu vua do duoc de lan sau khong phai do lai va giao dien thay."""
    if flt.use_color:
        pc.settings.ocr_text_color = flt.color
    pc.settings.ocr_min_height = round(flt.min_height, 1)
    pc.settings.ocr_max_height = round(flt.max_height, 1)
    with contextlib.suppress(OSError):
        pc.settings.save()


def _prepare_filter(pc: PipelineContext, frames: list[Path]) -> ocr_filter.TextFilter:
    """Chuan bi bo loc tuy chon truoc khi doc.

    NTS khong tu loc mau trong cau hinh mac dinh. Mau chi duoc ap dung khi
    nguoi dung bat bo loc va da chi dinh mot mau cu the.
    """
    flt = text_filter_for(pc.settings)
    if flt.use_color and not flt.rgb:
        flt.use_color = False
        pc.log("Chua chi dinh mau phu de, OCR dung anh goc nhu mac dinh NTS.")
    if not ocr_filter.available():
        if flt.active:
            pc.log("Thieu thu vien xu ly anh nen khong loc duoc theo mau chu.")
        return ocr_filter.TextFilter(use_color=False, drop_static=False)
    profile = pc.settings.ocr_server or pc.settings.ocr_mode
    # Voi che do NTS, neu da do duoc mau/chiều cao o lan truoc thi khong can
    # OCR lai 14 anh mau. Chu dan co dinh van duoc loai tren toan bo video o
    # read_frames, nen bo phep do lap nay khong lam giam ket qua.
    need_static_probe = flt.drop_static and not ocr.is_nts_profile(profile)
    if flt.needs_probe or need_static_probe:
        pc.log("Do mau chu va chieu cao chu tren vai khung hinh...")
        probe = ocr.probe_frames(
            frames,
            min_confidence=pc.settings.ocr_confidence / 100.0,
            use_gpu=pc.settings.use_gpu,
            profile=profile,
            batch_size=pc.settings.ocr_batch_size,
            base=flt,
            on_log=pc.log,
            should_cancel=lambda: pc.token.cancelled,
        )
        needed = flt.needs_probe
        ocr.apply_probe(flt, probe)
        if probe.found and needed:
            _remember_filter(pc, flt)
    return flt


def is_ai_ocr(settings: Settings) -> bool:
    mode = (settings.ocr_mode or "").strip().lower()
    server = (settings.ocr_server or "").strip().lower()
    return (
        "ocr ai" in mode
        or "server ai" in mode
        or "server ai" in server
        or "ai gateway" in server
    )


def step_ocr(pc: PipelineContext) -> str:
    """Lay phu de bang cach doc chu chay san tren hinh."""
    s = pc.settings
    ai_mode = is_ai_ocr(s)
    if not ai_mode and not ocr.is_available():
        raise StepError(ocr.install_hint())
    video = pc.require_video()
    region = pc.project.ocr_region
    if not region or len(region) != 4:
        raise StepError("Chua khoanh vung chu tren hinh. Vao tab B1, bam 'Khoanh vung OCR' truoc.")
    frames_dir = pc.project.sub_dir("temp") / "ocr_frames"
    fast_nts = ocr.is_nts_profile(s.ocr_server or s.ocr_mode) and not ai_mode
    source_fps = 0.0
    if fast_nts:
        with contextlib.suppress(FFmpegError, OSError):
            source_fps = pc.ff.probe(video).fps
    fps = _ocr_frame_rate(s.ocr_fps, source_fps, fast_nts)
    pc.log("Trich khung hinh de doc chu...")
    if fast_nts:
        pc.log(f"Nhanh Nhu NTS: doc {fps:g} hinh/giay de giu caption ngan va moc thoi gian.")
    timed = media.extract_frames(
        pc.ff,
        video,
        frames_dir,
        fps=fps,
        region=region,
        duration=pc.project.duration,
        token=pc.token,
        on_progress=lambda p: pc.progress(int(p * 0.4)),
    )
    pc.check()
    if not timed:
        raise StepError("Khong trich duoc khung hinh nao tu video.")
    stamps = [t for t, _p in timed]
    frames = [p for _t, p in timed]
    if fast_nts:
        pc.log("Dang dung cau hinh Nhanh Nhu NTS cho phu de tieng Trung.")
    flt = _prepare_filter(pc, frames)
    pc.check()
    pc.log(f"Doc chu tren {len(frames)} khung hinh...")
    refine = bool(s.ocr_refine) and not fast_nts and not ai_mode
    weight = 0.42 if refine else 0.55
    if ai_mode:
        key = pc.api_key or Settings.get_secret("ai_gateway_key")
        if not s.ai_endpoint.strip():
            raise StepError("Chưa cấu hình Endpoint AI Gateway cho OCR AI.")
        if not key.strip():
            raise StepError("Chưa có khóa API cho AI Gateway.")
        cues = ocr_ai.read_frames_ai(
            frames,
            fps=fps,
            stamps=stamps,
            similarity=s.ocr_similarity,
            min_duration=s.ocr_min_duration,
            model=getattr(s, "ocr_ai_model", "sub") or "sub",
            endpoint=s.ai_endpoint,
            api_key=key,
            consensus=s.ocr_consensus,
            text_filter=flt,
            on_progress=lambda p: pc.progress(43 + int(p * weight)),
            on_log=pc.log,
            should_cancel=lambda: pc.token.cancelled,
            cache_path=(Path(pc.project.folder) / "ocr_cache.sqlite3")
            if s.ocr_cache_enabled
            else None,
            cache_key=_ocr_cache_key(video, region, fps, s, flt)
            if s.ocr_cache_enabled
            else "",
        )
    else:
        cues = ocr.read_frames(
            frames,
            fps=fps,
            stamps=stamps,
            similarity=s.ocr_similarity,
            min_confidence=s.ocr_confidence / 100.0,
            min_duration=s.ocr_min_duration,
            use_gpu=s.use_gpu,
            profile=s.ocr_server or s.ocr_mode,
            batch_size=s.ocr_batch_size,
            consensus=s.ocr_consensus,
            text_filter=flt,
            on_progress=lambda p: pc.progress(43 + int(p * weight)),
            on_log=pc.log,
            should_cancel=lambda: pc.token.cancelled,
            cache_path=(Path(pc.project.folder) / "ocr_cache.sqlite3")
            if s.ocr_cache_enabled
            else None,
            cache_key=_ocr_cache_key(video, region, fps, s, flt)
            if s.ocr_cache_enabled
            else "",
        )
    pc.check()
    if not cues:
        raise StepError("Khong doc duoc chu nao trong vung da khoanh.")
    if refine:
        pc.log("Do lai moc thoi gian cho tung cau...")
        window_dir = pc.project.sub_dir("temp") / "ocr_window"
        cues = ocr.refine_boundaries(
            cues,
            frames_around=lambda a, b: media.extract_window(
                pc.ff,
                video,
                window_dir,
                a,
                max(0.05, b - a),
                region=region,
                token=pc.token,
            ),
            coarse_step=_frame_gap(stamps, 1.0 / fps),
            similarity=s.ocr_similarity,
            min_confidence=s.ocr_confidence / 100.0,
            use_gpu=s.use_gpu,
            profile=s.ocr_server or s.ocr_mode,
            batch_size=s.ocr_batch_size,
            text_filter=flt,
            on_progress=lambda p: pc.progress(85 + int(p * 0.15)),
            on_log=pc.log,
            should_cancel=lambda: pc.token.cancelled,
        )
        pc.check()
        if not pc.settings.keep_temp:
            shutil.rmtree(window_dir, ignore_errors=True)
    cues = ocr.clean_cues(cues, drop_chars=s.ocr_drop_chars, drop_words=s.ocr_drop_words)
    if s.ocr_continuous:
        cues = ocr.make_continuous(cues)
    if not cues:
        raise StepError("Sau khi loc bo ky tu rac thi khong con cau nao.")
    pc.project.doc.cues = cues
    editing.remove_overlaps(pc.project.doc)
    saved = auto_export_srt(pc)
    pc.save()
    if not pc.settings.keep_temp:
        shutil.rmtree(frames_dir, ignore_errors=True)
    pc.progress(100)
    note = f" Da luu {Path(saved).name} canh video." if saved else ""
    return f"Da doc duoc {len(cues)} cau tu chu tren hinh.{note}"


def step_ocr_measure(pc: PipelineContext) -> str:
    """Do mau chu va chieu cao chu phu de trong vung da khoanh.

    Lay vai khung hinh rai deu tu dau den cuoi video, doc thu, roi ghi lai mau
    chu va khoang chieu cao vao cai dat de buoc lay sub dung ngay.
    """
    if not ocr.is_available():
        raise StepError(ocr.install_hint())
    if not ocr_filter.available():
        raise StepError(
            "Thieu thu vien xu ly anh (opencv-python, numpy) nen khong do duoc mau chu."
        )
    video = pc.require_video()
    region = pc.project.ocr_region
    if not region or len(region) != 4:
        raise StepError("Chua khoanh vung chu tren hinh. Vao tab B1, keo khung xanh tren video.")
    probe_dir = pc.project.sub_dir("temp") / "ocr_probe"
    pc.log("Lay khung hinh mau de do mau chu...")
    timed = media.sample_frames(
        pc.ff,
        video,
        probe_dir,
        count=16,
        duration=pc.project.duration,
        region=region,
        token=pc.token,
        on_progress=lambda p: pc.progress(int(p * 0.45)),
    )
    pc.check()
    if not timed:
        raise StepError("Khong lay duoc khung hinh nao tu video.")
    frames = [path for _stamp, path in timed]
    probe = ocr.probe_frames(
        frames,
        min_confidence=pc.settings.ocr_confidence / 100.0,
        use_gpu=pc.settings.use_gpu,
        profile=pc.settings.ocr_server or pc.settings.ocr_mode,
        batch_size=pc.settings.ocr_batch_size,
        base=text_filter_for(pc.settings),
        on_log=pc.log,
        should_cancel=lambda: pc.token.cancelled,
    )
    pc.progress(95)
    if not pc.settings.keep_temp:
        shutil.rmtree(probe_dir, ignore_errors=True)
    if not probe.found:
        raise StepError(
            "Khong thay dong chu nao trong vung da khoanh. Hay keo khung xanh sat vao "
            "dong phu de roi do lai."
        )
    pc.settings.ocr_text_color = probe.color
    pc.settings.ocr_min_height = probe.min_height
    pc.settings.ocr_max_height = probe.max_height
    with contextlib.suppress(OSError):
        pc.settings.save()
    pc.progress(100)
    name = ocr_filter.color_name(ocr_filter.parse_color(probe.color))
    return (
        f"Mau chu phu de: {probe.color} ({name}), chu cao khoang {probe.height:.0f} px. "
        f"Da dat khoang chieu cao {probe.min_height:.0f}-{probe.max_height:.0f} px."
    )


# --------------------------------------------------------------------------- dich


def step_translate(pc: PipelineContext) -> str:
    """Dich toan bo phu de sang ngon ngu dich."""
    cues = pc.require_cues()
    s = pc.settings
    ready, reason = translate.provider_ready(s.translate_provider, pc.api_key)
    if not ready:
        raise StepError(reason)
    pending = [(i, c) for i, c in enumerate(cues) if c.text.strip()]
    if not pending:
        raise StepError("Khong co cau nao co noi dung de dich.")
    batches = translate.chunk([c.text for _, c in pending], max(1, s.translate_batch))
    index = 0
    done = 0
    for batch_no, batch in enumerate(batches):
        pc.check()
        before = pending[index - 1][1].text if index > 0 else ""
        after_idx = index + len(batch)
        after = pending[after_idx][1].text if after_idx < len(pending) else ""
        request = translate.TranslationRequest(
            texts=batch,
            source=s.source_language,
            target=s.target_language,
            context_before=before if s.translate_context else "",
            context_after=after if s.translate_context else "",
            glossary=pc.glossary,
            extra_prompt=s.translate_prompt,
        )
        try:
            results = translate.translate_batch(
                s.translate_provider,
                request,
                api_key=pc.api_key,
                model=s.llm_model,
                on_log=None,
            )
        except translate.TranslationError as exc:
            raise StepError(str(exc)) from exc
        for offset, text in enumerate(results):
            cue_index, cue = pending[index + offset]
            if text.strip():
                cue.translation = text.strip()
                done += 1
        index += len(batch)
        pc.progress(int((batch_no + 1) / len(batches) * 100))
        pc.log(f"Da dich {index}/{len(pending)} cau.")
    pc.project.doc.target_language = s.target_language
    pc.save()
    lang = translate.LANGUAGES.get(s.target_language, s.target_language)
    return f"Da dich {done} cau sang {lang}."


# --------------------------------------------------------------------------- giong noi


def step_diarize(pc: PipelineContext) -> str:
    """Gan nhan nguoi noi cho tung cau dua tren cao do giong."""
    cues = pc.require_cues()
    audio = Path(pc.project.voice_path or "") if pc.project.voice_path else None
    if not (audio and audio.is_file()):
        audio = ensure_audio(pc)
    seg_dir = pc.project.sub_dir("temp") / "speaker"
    seg_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for i, cue in enumerate(cues):
        pc.check()
        if cue.duration < 0.25:
            continue
        out = seg_dir / f"seg_{i:05d}.wav"
        try:
            media.cut_segment(
                pc.ff, audio, out, cue.start, min(cue.end, cue.start + 4.0), token=pc.token
            )
        except FFmpegError:
            continue
        paths[i] = out
        if i % 10 == 0:
            pc.progress(int(i / max(1, len(cues)) * 50))
    pc.check()
    count = diarize.assign_speakers(
        cues,
        paths,
        on_progress=lambda p: pc.progress(50 + int(p * 0.5)),
        on_log=None,
        should_cancel=lambda: pc.token.cancelled,
    )
    pc.save()
    if not pc.settings.keep_temp:
        shutil.rmtree(seg_dir, ignore_errors=True)
    return f"Da gan nhan nguoi noi cho {count}/{len(cues)} cau."


def step_dub(pc: PipelineContext) -> str:
    """Tao voice, canh thoi gian, tron tieng goc va xuat audio/video hoan chinh."""
    cues = pc.require_cues()
    s = pc.settings
    ready, reason = tts.provider_ready(s.tts_provider)
    if not ready:
        raise StepError(reason)
    mode = s.dub_source if s.dub_source in ("original", "translation") else "translation"
    if mode == "translation" and not any(c.translation.strip() for c in cues):
        pc.log("Chua co ban dich nen long tieng theo ban goc.")
        mode = "original"
    work = pc.project.sub_dir("temp") / "dub"
    work.mkdir(parents=True, exist_ok=True)
    voice_items: dict[int, tuple[Path, float]] = {}
    global_speed = max(0.25, min(4.0, float(s.tts_speed_percent) / 100.0))
    over = 0
    short = 0
    made = 0
    failed = 0
    for i, cue in enumerate(cues):
        pc.check()
        prepared_text = tts.apply_dictionary(cue.display_text(mode), s.tts_dictionary)
        had_newline = "\n" in prepared_text
        text = tts.normalize_punctuation(prepared_text)
        if not text.strip():
            continue
        profile = _voice_profile_for_cue(s, cue)
        voice = str(profile.get("voice") or s.tts_voice)
        volume = int(str(profile.get("volume", s.tts_volume)))
        pitch = max(50, min(200, int(str(profile.get("pitch", s.tts_pitch_percent))))) / 100.0
        raw = work / f"raw_{i:05d}"
        try:
            produced = tts.synthesize_cached(
                s.tts_provider,
                text.replace("\n", " "),
                raw.with_suffix(".wav"),
                voice=voice,
                volume=volume,
                speed=global_speed,
                enabled=s.tts_cache_enabled,
                on_log=None,
            )
        except tts.TTSError as exc:
            pc.log(f"Cau {i + 1}: {exc}")
            failed += 1
            continue
        normalized = work / f"norm_{i:05d}.wav"
        media.to_wav(
            pc.ff,
            produced,
            normalized,
            tempo=1.0,
            pitch=pitch,
            volume=max(0.0, min(2.0, volume / 100.0)),
            token=pc.token,
        )
        profile_bass = int(str(profile.get("bass", 0)))
        profile_mid = int(str(profile.get("mid", 0)))
        profile_treble = int(str(profile.get("treble", 0)))
        if profile_bass or profile_mid or profile_treble:
            toned = work / f"tone_{i:05d}.wav"
            media.apply_equalizer(
                pc.ff,
                normalized,
                toned,
                bass=profile_bass,
                mid=profile_mid,
                treble=profile_treble,
                token=pc.token,
            )
            normalized = toned
        pause_ms = 0
        if had_newline:
            pause_ms = s.tts_pause_newline_ms
        elif text.endswith((".", "!", "?")):
            pause_ms = s.tts_pause_period_ms
        elif text.endswith((",", ";", ":")):
            pause_ms = s.tts_pause_comma_ms
        pause_ms = max(pause_ms, s.tts_end_pause_ms)
        if pause_ms > 0:
            padded = work / f"pause_{i:05d}.wav"
            media.append_silence(normalized, padded, pause_ms / 1000.0)
            normalized = padded
        actual = media.wav_duration(normalized)
        next_start = cues[i + 1].start if i + 1 < len(cues) else cue.end
        slot_end = min(cue.end, next_start) if next_start > cue.start else cue.end
        target = max(0.1, slot_end - cue.start)
        if cue.duration * 1000 < max(0, s.tts_short_threshold_ms):
            short += 1
        if actual > target:
            over += 1
        fitted = normalized
        if s.tts_store_voice:
            saved_dir = pc.project.sub_dir("audio") / "voice_segments"
            saved_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fitted, saved_dir / f"voice_{i + 1:05d}.wav")
        voice_items[i] = (fitted, actual)
        made += 1
        pc.progress(int((i + 1) / len(cues) * 80))

    if not voice_items:
        raise StepError("Khong tao duoc cau doc nao. Kiem tra lai giong doc trong Cai dat.")

    original_total = max(pc.project.duration, max(c.end for c in cues) + 1.0)
    current_video = pc.require_video()
    original = Path(pc.project.original_video) if pc.project.original_video else None
    source_video = (
        original
        if s.dub_use_original_video and original is not None and original.is_file()
        else current_video
    )
    timeline_mode = s.dub_timing_mode == "voice" and s.dub_output_mode == "video"
    video_source = source_video
    if timeline_mode:
        video_spans, segments, timings, total = _voice_timeline(
            cues, voice_items, original_total
        )
        needs_retime = any(
            abs((end - start) - target) > 0.02 for start, end, target in video_spans
        )
        if needs_retime:
            pc.log("Canh timeline video theo do dai tung cau voice...")
            timeline_dir = work / "timeline"
            video_source = work / "timeline_video.mp4"
            media.retime_video_segments(
                pc.ff,
                source_video,
                video_spans,
                timeline_dir,
                video_source,
                crf=s.render_crf,
                preset=s.render_preset,
                gpu=s.use_gpu_encoder,
                workers=min(
                    1 if s.limit_cpu else max(1, s.timeline_workers),
                    max(1, int((os.cpu_count() or 1) * s.timeline_cpu_percent / 100)),
                ),
                token=pc.token,
                on_progress=lambda p: pc.progress(80 + int(p * 0.08)),
            )
        pc.project.dub_timing = timings
    else:
        segments = [(cues[index].start, item[0]) for index, item in sorted(voice_items.items())]
        total = original_total
        pc.project.dub_timing = []

    pc.log("Ghep cac cau doc vao dung moc thoi gian...")
    voice_track = pc.project.sub_dir("audio") / "dub_voice.wav"
    media.build_voice_track(
        segments,
        voice_track,
        total,
        mix_overlaps=s.tts_allow_overlap,
    )
    if not s.tts_voice_profiles and (s.eq_bass or s.eq_mid or s.eq_treble):
        pc.log(
            f"Chinh am sac: tram {s.eq_bass:+d} dB, giua {s.eq_mid:+d} dB, cao {s.eq_treble:+d} dB"
        )
        tuned = pc.project.sub_dir("audio") / "dub_voice_eq.wav"
        media.apply_equalizer(
            pc.ff,
            voice_track,
            tuned,
            bass=s.eq_bass,
            mid=s.eq_mid,
            treble=s.eq_treble,
            token=pc.token,
        )
        voice_track = tuned
    pc.progress(88)

    final = pc.project.sub_dir("audio") / "dub_mixed.wav"
    bed = Path(pc.project.music_path) if pc.project.music_path and not timeline_mode else None
    if s.keep_original_audio and (bed is None or not bed.is_file()):
        candidate = video_source
        bed = candidate if candidate.is_file() else None
    if s.keep_original_audio and bed and bed.is_file():
        pc.log("Tron giong doc voi tieng goc/nhac nen...")
        try:
            media.mix_voice_and_music(
                pc.ff,
                voice_track,
                bed,
                final,
                music_volume=s.original_audio_volume,
                ducking=s.ducking,
                duration=total,
                token=pc.token,
            )
        except FFmpegError as exc:
            pc.log(f"Khong tron duoc tieng goc, da giu rieng giong doc: {exc}")
            shutil.copyfile(voice_track, final)
    else:
        shutil.copyfile(voice_track, final)
    pc.project.dub_path = str(final)
    video_out: Path | None = None
    if s.dub_output_mode == "video":
        video_out = pc.project.sub_dir("exports") / (
            f"{Path(pc.project.name).stem or 'video'}_long_tieng.mp4"
        )
        pc.log("Ghep track long tieng vao video...")
        media.replace_audio(
            pc.ff,
            video_source,
            final,
            video_out,
            duration=total,
            token=pc.token,
            on_progress=lambda p: pc.progress(88 + int(p * 0.12)),
            shortest=False,
        )
        pc.project.dub_video_path = str(video_out)
    pc.save()
    if not pc.settings.keep_temp:
        shutil.rmtree(work, ignore_errors=True)
    warnings = []
    if over:
        warnings.append(f"{over} cau vuot khe time")
    if short:
        warnings.append(f"{short} cau ngan hon {s.tts_short_threshold_ms} ms")
    if failed:
        warnings.append(f"{failed} cau tao voice that bai")
    warn = (" Canh bao: " + "; ".join(warnings) + ".") if warnings else ""
    if video_out is not None:
        return f"Da long tieng {made} cau va xuat video: {video_out}.{warn}"
    return f"Da long tieng {made} cau, da xuat: {final}.{warn}"


def _voice_profile_for_cue(settings: Settings, cue: Cue) -> dict[str, object]:
    """Chon profile giong theo nhan Nam/Nu; lui ve giong mac dinh."""
    profiles = [item for item in settings.tts_voice_profiles if isinstance(item, dict)]
    if not profiles:
        return {}
    speaker = cue.speaker.strip().casefold()
    wanted = "nam" if speaker == "nam" else "nu" if speaker in {"nu", "nữ"} else ""
    if wanted:
        for profile in profiles:
            if str(profile.get("gender", "")).strip().casefold() == wanted:
                return profile
    for profile in profiles:
        if str(profile.get("gender", "")).strip().casefold() in {"", "mac dinh", "default"}:
            return profile
    return profiles[0]


def _voice_timeline(
    cues: list[Cue],
    voice_items: dict[int, tuple[Path, float]],
    source_total: float,
    *,
    gap_threshold: float = 0.5,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, Path]],
    list[list[float]],
    float,
]:
    """Lap timeline canh theo giong doc: target duration bang measured voice duration.

    Moi doan video tuong ung voi mot cau doc se duoc retime bang dung do dai voice
    thuc te (bao gom ca configured pause neu co), ke ca khi voice ngan hon doan goc.
    Video giua cac cau doc (gap) neu lon hon gap_threshold se duoc giu nguyen thoi luong.
    """
    video_spans: list[tuple[float, float, float]] = []
    voice_segments: list[tuple[float, Path]] = []
    timings: list[list[float]] = []
    output_cursor = 0.0

    if not cues:
        if source_total > 0.001:
            video_spans.append((0.0, source_total, source_total))
            output_cursor = source_total
        return video_spans, voice_segments, timings, output_cursor

    if cues[0].start > 0.001:
        prefix = max(0.0, cues[0].start)
        video_spans.append((0.0, prefix, prefix))
        output_cursor = prefix
        source_cursor = prefix
    else:
        source_cursor = 0.0

    for index, cue in enumerate(cues):
        if cue.start > source_cursor + 0.001:
            gap = cue.start - source_cursor
            video_spans.append((source_cursor, cue.start, gap))
            output_cursor += gap
            source_cursor = cue.start

        start = max(source_cursor, cue.start)
        if index + 1 < len(cues):
            next_start = cues[index + 1].start
            gap_to_next = next_start - cue.end
            if 0.0 <= gap_to_next <= gap_threshold and next_start > start:
                end = next_start
            else:
                end = max(cue.end, start + 0.04)
        else:
            end = max(cue.end, start + 0.04)

        source_duration = max(0.04, end - start)
        voice = voice_items.get(index)
        voice_duration = voice[1] if voice is not None else 0.0

        if voice_duration > 0.001:
            target_duration = max(0.04, voice_duration)
        else:
            target_duration = source_duration

        video_spans.append((start, end, target_duration))
        new_start = output_cursor
        new_end = output_cursor + target_duration
        timings.append([round(new_start, 6), round(new_end, 6)])
        if voice is not None and voice_duration > 0.001:
            voice_segments.append((new_start, voice[0]))
        output_cursor = new_end
        source_cursor = end

    if source_cursor < source_total - 0.001:
        tail = source_total - source_cursor
        video_spans.append((source_cursor, source_total, tail))
        output_cursor += tail

    return video_spans, voice_segments, timings, output_cursor


# --------------------------------------------------------------------------- render


def build_ass(
    pc: PipelineContext,
    text_mode: str = "translation",
    timing: list[list[float]] | None = None,
) -> Path:
    """Ghi tep ASS theo kieu chu dang chon, dung cho render va xem truoc."""
    style: SubtitleStyle = pc.settings.style
    info = None
    try:
        info = pc.ff.probe(pc.require_video())
    except (FFmpegError, StepError):
        info = None
    out = pc.project.sub_dir("subtitles") / "render.ass"
    document = pc.project.doc
    if timing and len(timing) == len(document.cues):
        document = copy.deepcopy(document)
        for cue, pair in zip(document.cues, timing, strict=False):
            if len(pair) >= 2 and pair[1] > pair[0]:
                cue.start, cue.end = float(pair[0]), float(pair[1])
    content = formats.write_ass(
        document,
        text_mode=text_mode,
        font=style.font,
        font_size=style.font_size,
        primary_color=_ass_color(style.primary_color),
        outline_color=_ass_color(style.outline_color),
        back_color=_ass_color(style.back_color, style.back_opacity),
        bold=style.bold,
        outline=style.outline,
        shadow=style.shadow,
        alignment=style.alignment,
        margin_v=style.margin_v,
        play_res_x=(info.width if info and info.width else 1920),
        play_res_y=(info.height if info and info.height else 1080),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def _ass_color(hex_color: str, opacity_percent: int = 100) -> str:
    """Doi mau #RRGGBB sang dinh dang &HAABBGGRR cua ASS."""
    value = (hex_color or "#FFFFFF").lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    alpha = int(round((100 - max(0, min(100, opacity_percent))) * 255 / 100))
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def step_blur(pc: PipelineContext) -> str:
    """Che mo vung phu de goc tren video."""
    video = pc.require_video()
    region = pc.project.blur_region
    if not region or len(region) != 4 or region[2] <= 0 or region[3] <= 0:
        raise StepError(
            "Chua khoanh vung phu de goc. Vao tab 'B4' hoac man hinh Render de khoanh vung."
        )
    out = pc.project.sub_dir("video") / "blurred.mp4"
    graph = media.blur_filter(region)
    pc.log("Dang che mo vung phu de goc...")
    encoder = gpu.video_encoder_args(
        pc.settings.render_crf,
        pc.settings.render_preset,
        media.use_gpu_encoder(pc.ff, pc.settings.use_gpu_encoder),
    )
    pc.ff.run(
        ["-i", str(video), "-vf", graph, *encoder, "-c:a", "copy", str(out)],
        duration=pc.project.duration,
        token=pc.token,
        on_progress=pc.progress,
    )
    pc.project.video_path = str(out)
    pc.save()
    return f"Da che mo vung phu de goc: {out.name}"


def step_render(pc: PipelineContext) -> str:
    """Ghep phu de cung vao video, kem giong long tieng neu co."""
    video = pc.require_video()
    dubbed_video = Path(pc.project.dub_video_path) if pc.project.dub_video_path else None
    uses_dubbed_video = bool(dubbed_video and dubbed_video.is_file())
    if uses_dubbed_video and dubbed_video is not None:
        video = dubbed_video
    pc.require_cues()
    mode = "translation" if any(c.translation.strip() for c in pc.project.doc.cues) else "original"
    ass = build_ass(
        pc,
        text_mode=mode,
        timing=pc.project.dub_timing if uses_dubbed_video else None,
    )
    out = pc.project.sub_dir("exports") / f"{Path(pc.project.name).stem or 'video'}_sub.mp4"
    audio = (
        None
        if uses_dubbed_video
        else (Path(pc.project.dub_path) if pc.project.dub_path else None)
    )
    pc.log("Dang render video (buoc nay lau nhat)...")
    media.burn_subtitles(
        pc.ff,
        video,
        ass,
        out,
        audio_path=audio if (audio and audio.is_file()) else None,
        blur_region=pc.project.blur_region or None,
        lut_path=pc.project.lut_path or None,
        crf=pc.settings.render_crf,
        preset=pc.settings.render_preset,
        gpu=pc.settings.use_gpu_encoder,
        duration=pc.project.duration,
        token=pc.token,
        on_progress=pc.progress,
    )
    pc.project.render_path = str(out)
    pc.save()
    return f"Da render xong: {out}"


def step_export(pc: PipelineContext) -> str:
    """Xuat goi du an de mo tiep bang phan mem dung phim khac."""
    pc.require_cues()
    out_dir = pc.project.sub_dir("exports") / "package"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pc.project.doc

    srt = out_dir / "subtitle_goc.srt"
    formats.save_subtitle(srt, doc, text_mode="original")
    files = [srt.name]
    if doc.translated_count:
        trans = out_dir / "subtitle_dich.srt"
        formats.save_subtitle(trans, doc, text_mode="translation")
        files.append(trans.name)
    ass = build_ass(pc, text_mode="translation" if doc.translated_count else "original")
    shutil.copyfile(ass, out_dir / "subtitle.ass")
    files.append("subtitle.ass")

    for key, label in (("video_path", "video.txt"), ("dub_path", "audio_long_tieng.txt")):
        value = getattr(pc.project, key, "")
        if value and Path(value).is_file():
            (out_dir / label).write_text(str(Path(value).resolve()), encoding="utf-8")

    manifest = {
        "format": "autosub-studio-export",
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": pc.project.name,
        "duration": round(pc.project.duration, 3),
        "language": doc.language,
        "target_language": doc.target_language,
        "video": str(Path(pc.project.video_path).resolve()) if pc.project.video_path else "",
        "dub_audio": str(Path(pc.project.dub_path).resolve()) if pc.project.dub_path else "",
        "files": files,
        "cues": [c.to_dict() for c in doc.cues],
    }
    (out_dir / "project_export.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pc.project.exported = True  # type: ignore[attr-defined]
    pc.save()
    pc.progress(100)
    capcut_note = f" {step_capcut(pc)}" if pc.settings.format_capcut else ""
    return f"Da xuat goi du an vao: {out_dir}.{capcut_note}"


def step_capcut(pc: PipelineContext) -> str:
    """Tao native CapCut draft tu template/effect nguoi dung da chon."""
    pc.require_cues()
    template = Path(pc.settings.capcut_template_draft)
    if not template.is_dir():
        raise StepError("Chua chon CapCut draft mau trong Cau Hinh Chung.")
    video_value = pc.project.dub_video_path or pc.project.render_path or pc.project.video_path
    video = Path(video_value)
    if not video.is_file():
        raise StepError("Khong co video de tao CapCut draft.")
    info = pc.ff.probe(video)
    try:
        draft = capcut.export_draft(
            template,
            capcut.drafts_root(pc.settings.capcut_path),
            f"{pc.project.name} - AutoSub",
            video,
            pc.project.doc,
            duration=info.duration or pc.project.duration,
            width=info.width or pc.project.width,
            height=info.height or pc.project.height,
            timing=pc.project.dub_timing if pc.project.dub_video_path else None,
            text_mode="translation" if pc.project.doc.translated_count else "original",
        )
    except capcut.CapCutError as exc:
        raise StepError(str(exc)) from exc
    return f"Da tao CapCut draft: {draft}."


STEP_FUNCTIONS: dict[str, Callable[[PipelineContext], str]] = {
    STEP_NORMALIZE: step_normalize,
    STEP_KEEP_MUSIC: step_keep_music,
    STEP_KEEP_VOICE: step_keep_voice,
    STEP_OCR_MEASURE: step_ocr_measure,
    STEP_OCR: step_ocr,
    STEP_ASR: step_asr,
    STEP_TRANSLATE: step_translate,
    STEP_DIARIZE: step_diarize,
    STEP_DUB: step_dub,
    STEP_CAPCUT: step_capcut,
    STEP_BLUR: step_blur,
    STEP_EXPORT: step_export,
    STEP_RENDER: step_render,
}


def run_step(name: str, pc: PipelineContext) -> str:
    """Chay mot buoc theo ten. Nem StepError khi that bai."""
    func = STEP_FUNCTIONS.get(name)
    if func is None:
        raise StepError(f"Khong biet buoc: {name}")
    try:
        return func(pc)
    except (StepError, CancelledError):
        raise
    except FFmpegError as exc:
        raise StepError(str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise StepError(f"{name} that bai: {exc}") from exc
    finally:
        temp_names: tuple[str, ...] = ()
        if name == STEP_OCR:
            temp_names = ("ocr_frames", "ocr_window")
        elif name == STEP_OCR_MEASURE:
            temp_names = ("ocr_probe",)
        if temp_names:
            freed = pc.store.clean_ocr_temp(pc.project, temp_names)
            if freed:
                pc.log(f"Da tu dong don {freed / (1024 * 1024):.1f} MB anh tam OCR.")


def run_script(steps: list[str], pc: PipelineContext) -> str:
    """Chay lan luot nhieu buoc. Dung ngay khi mot buoc that bai."""
    picked = [s for s in steps if s in STEP_FUNCTIONS]
    if not picked:
        raise StepError("Kich ban chua chon buoc nao.")
    messages: list[str] = []
    for i, name in enumerate(picked):
        pc.check()
        pc.log(f"--- Buoc {i + 1}/{len(picked)}: {name} ---")
        pc.progress(0)
        messages.append(run_step(name, pc))
        pc.log(messages[-1])
    return " | ".join(messages)


def project_store_for(settings: Settings) -> ProjectStore:
    return ProjectStore(settings.workspace)


def load_project(store: ProjectStore, folder: str) -> ProjectData:
    return store.load(folder)
