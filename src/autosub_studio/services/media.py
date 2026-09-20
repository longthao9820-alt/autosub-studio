"""Cac thao tac media dung FFmpeg: tach tieng, ghep phu de, che mo, tron am thanh."""

from __future__ import annotations

import math
import re
import shutil
import sys
import threading
import wave
from array import array
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .ffmpeg import CancelToken, FFmpeg, FFmpegError, MediaInfo
from .gpu import best_hw_encoder, video_encoder_args
from .paths import fonts_dir

VOICE_RATE = 24000  # tan so lay mau cho track long tieng
VOICE_WIDTH = 2  # 16 bit


def extract_audio(
    ff: FFmpeg,
    video: str | Path,
    out_path: str | Path,
    *,
    rate: int = 16000,
    channels: int = 1,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
    on_log=None,
) -> Path:
    """Tach am thanh ra WAV de nhan dang giong noi."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.run(
        [
            "-i",
            str(video),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(rate),
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        duration=duration,
        token=token,
        on_progress=on_progress,
        on_log=on_log,
    )
    return out


def to_wav(
    ff: FFmpeg,
    src: str | Path,
    out_path: str | Path,
    *,
    rate: int = VOICE_RATE,
    channels: int = 1,
    tempo: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
    token: CancelToken | None = None,
) -> Path:
    """Doi sang WAV chuan, co the doi toc do va cao do doc doc lap."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src)]
    speed = max(0.25, min(4.0, float(tempo)))
    tone = max(0.5, min(2.0, float(pitch)))
    filters: list[str] = []
    if abs(tone - 1.0) > 0.01:
        # Tang sample-rate nang cao do nhung cung doi toc do. atempo=speed/tone
        # bu lai phan thoi gian, nen hai nut toc do/cao do hoat dong doc lap.
        pitched_rate = max(8000, int(round(rate * tone)))
        filters += [f"aresample={rate}", f"asetrate={pitched_rate}", f"aresample={rate}"]
        speed /= tone
    if abs(speed - 1.0) > 0.01:
        filters.append(atempo_chain(speed))
    gain = max(0.0, min(2.0, float(volume)))
    if abs(gain - 1.0) > 0.001:
        filters.append(f"volume={gain:.4f}")
    if filters:
        args += ["-filter:a", ",".join(filters)]
    args += ["-ac", str(channels), "-ar", str(rate), "-acodec", "pcm_s16le", str(out)]
    ff.run(args, token=token)
    return out


def blur_filter(region: Sequence[int]) -> str:
    """Chuoi bo loc che mo mot vung hinh chu nhat.

    FFmpeg gioi han ban kinh lam mo kenh mau toi da la 15, nen ta dat rieng
    tham so cho kenh sang va kenh mau thay vi dung chung mot gia tri.
    """
    x, y, w, h = (max(0, int(v)) for v in region)
    luma = max(2, min(20, min(w, h) // 4 or 2))
    chroma = max(1, min(15, luma // 2))
    return (
        f"split[main][sub];[sub]crop={w}:{h}:{x}:{y},"
        f"boxblur=luma_radius={luma}:luma_power=2:"
        f"chroma_radius={chroma}:chroma_power=2[blr];"
        f"[main][blr]overlay={x}:{y}"
    )


def atempo_chain(tempo: float) -> str:
    """Chuoi bo loc atempo hop le (moi buoc chi nhan trong khoang 0.5 - 2.0)."""
    tempo = max(0.25, min(4.0, float(tempo)))
    parts: list[str] = []
    remaining = tempo
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def use_gpu_encoder(ff: FFmpeg, wanted: str | bool) -> str:
    """Ten bo ma hoa phan cung dung cho lan render nay, rong neu dung CPU.

    Nhan vao lua chon cua nguoi dung: True nghia la tu chon bo tot nhat,
    mot chuoi nghia la ep dung dung bo do.
    """
    if isinstance(wanted, str) and wanted:
        return wanted
    return best_hw_encoder(ff.ffmpeg) if wanted else ""


def normalize_video(
    ff: FFmpeg,
    src: str | Path,
    out_path: str | Path,
    *,
    scale: str = "1920x1080",
    fps: str = "30",
    crf: int = 20,
    preset: str = "medium",
    gpu: str | bool = "",
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
    on_log=None,
) -> Path:
    """Chuan hoa video goc ve cung do phan giai, khung hinh va ma hoa."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    if scale and scale != "giu nguyen":
        w, _, h = scale.partition("x")
        filters.append(
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
        )
    args = ["-i", str(src)]
    if filters:
        args += ["-vf", ",".join(filters)]
    if fps and fps != "giu nguyen":
        args += ["-r", str(fps)]
    args += video_encoder_args(crf, preset, use_gpu_encoder(ff, gpu))
    args += ["-c:a", "aac", "-b:a", "192k", str(out)]
    ff.run(args, duration=duration, token=token, on_progress=on_progress, on_log=on_log)
    return out


def burn_subtitles(
    ff: FFmpeg,
    video: str | Path,
    ass_path: str | Path,
    out_path: str | Path,
    *,
    audio_path: str | Path | None = None,
    blur_region: Sequence[int] | None = None,
    lut_path: str | Path | None = None,
    crf: int = 20,
    preset: str = "medium",
    gpu: str | bool = "",
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
    on_log=None,
) -> Path:
    """Ghep phu de cung vao hinh, kem tuy chon che mo vung sub goc va ap LUT."""
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ass = Path(ass_path).resolve()
    work = ass.parent  # chay trong thu muc chua .ass de tranh loi thoat ky tu duong dan

    chain: list[str] = []
    if blur_region and len(blur_region) == 4 and blur_region[2] > 0 and blur_region[3] > 0:
        chain.append(blur_filter(blur_region))
    if lut_path and Path(lut_path).is_file():
        lut = Path(lut_path).resolve().as_posix().replace(":", r"\:")
        chain.append(f"lut3d='{lut}'")
    font_dir = fonts_dir()
    if font_dir is not None:
        fonts = font_dir.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
        chain.append(f"subtitles='{ass.name}':fontsdir='{fonts}'")
    else:
        chain.append(f"subtitles='{ass.name}'")
    vf = ",".join(chain)

    args = ["-i", str(video)]
    if audio_path and Path(audio_path).is_file():
        args += ["-i", str(audio_path)]
    args += ["-vf", vf]
    args += video_encoder_args(crf, preset, use_gpu_encoder(ff, gpu))
    if audio_path and Path(audio_path).is_file():
        args += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        args += ["-c:a", "aac", "-b:a", "192k"]
    args += [str(out)]
    ff.run(args, duration=duration, token=token, on_progress=on_progress, on_log=on_log, cwd=work)
    return out


def mux_soft_subtitle(
    ff: FFmpeg,
    video: str | Path,
    subtitle: str | Path,
    out_path: str | Path,
    *,
    language: str = "vie",
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
) -> Path:
    """Gan phu de mem vao MKV/MP4, nguoi xem co the bat tat."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    codec = "mov_text" if out.suffix.lower() in (".mp4", ".m4v") else "srt"
    ff.run(
        [
            "-i",
            str(video),
            "-i",
            str(subtitle),
            "-map",
            "0",
            "-map",
            "1",
            "-c",
            "copy",
            "-c:s",
            codec,
            "-metadata:s:s:0",
            f"language={language}",
            str(out),
        ],
        duration=duration,
        token=token,
        on_progress=on_progress,
    )
    return out


def replace_audio(
    ff: FFmpeg,
    video: str | Path,
    audio: str | Path,
    out_path: str | Path,
    *,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
    shortest: bool = True,
) -> Path:
    """Thay toan bo tieng cua video bang tep am thanh moi."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    if shortest:
        args.append("-shortest")
    args.append(str(out))
    ff.run(
        args,
        duration=duration,
        token=token,
        on_progress=on_progress,
    )
    return out


def retime_video_segments(
    ff: FFmpeg,
    video: str | Path,
    segments: Sequence[tuple[float, float, float]],
    out_dir: str | Path,
    out_path: str | Path,
    *,
    crf: int = 20,
    preset: str = "fast",
    gpu: str | bool = False,
    workers: int = 2,
    token: CancelToken | None = None,
    on_progress=None,
) -> Path:
    """Cat va co gian timeline video theo tung cau, sau do concat khong ma hoa lai.

    Moi bo ba la (source_start, source_end, target_duration). Quy tac nay trung
    voi render timeline NTS: neu voice dai hon khe sub thi keo video, neu ngan
    hon thi giu thoi luong doan goc.
    """
    valid = [
        (max(0.0, float(start)), float(end), max(0.04, float(target)))
        for start, end, target in segments
        if float(end) - float(start) > 0.001 and float(target) > 0.001
    ]
    if not valid:
        raise FFmpegError("Khong co doan timeline hop le de canh theo giong doc.")
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = ff.probe(video)
    fps = info.fps if info.fps > 1 else 30.0
    encoder = video_encoder_args(crf, preset, use_gpu_encoder(ff, gpu))
    completed = 0
    lock = threading.Lock()

    def render_one(index: int, item: tuple[float, float, float]) -> tuple[int, Path]:
        if token is not None:
            token.raise_if_cancelled()
        start, end, target = item
        source_duration = max(0.04, end - start)
        stretch = target / source_duration
        path = folder / f"timeline_{index:05d}.mp4"
        args = [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{source_duration:.6f}",
            "-i",
            str(video),
        ]
        if info.has_audio:
            audio_tempo = source_duration / target
            graph = (
                f"[0:v]setpts={stretch:.9f}*PTS[v];"
                f"[0:a]{atempo_chain(audio_tempo)},apad=whole_dur={target:.6f}[a]"
            )
            args += [
                "-filter_complex",
                graph,
                "-map",
                "[v]",
                "-map",
                "[a]",
                *encoder,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        else:
            args += ["-vf", f"setpts={stretch:.9f}*PTS", *encoder, "-an"]
        args += [
            "-r",
            f"{fps:.6f}",
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{target:.6f}",
            str(path),
        ]
        ff.run(args, token=token)
        return index, path

    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(8, int(workers)))) as executor:
        futures = {
            executor.submit(render_one, index, item): index for index, item in enumerate(valid)
        }
        for future in as_completed(futures):
            index, path = future.result()
            results[index] = path
            with lock:
                completed += 1
                if on_progress:
                    on_progress(int(completed / len(valid) * 96))

    ordered = [results[index] for index in range(len(valid))]
    concat_file = folder / "timeline_concat.txt"
    concat_lines = [
        f"file '{path.resolve().as_posix().replace(chr(39), chr(39) * 2)}'\n"
        for path in ordered
    ]
    concat_file.write_text(
        "".join(concat_lines),
        encoding="utf-8",
    )
    ff.run(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(out),
        ],
        token=token,
    )
    if on_progress:
        on_progress(100)
    return out


def remove_vocals(
    ff: FFmpeg,
    src: str | Path,
    out_path: str | Path,
    *,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
) -> Path:
    """Xoa loi thoai, giu nhac nen bang cach khu kenh giua (chi hieu qua voi tep stereo)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.run(
        [
            "-i",
            str(src),
            "-af",
            "pan=stereo|c0=c0-c1|c1=c1-c0,highpass=f=60,lowpass=f=16000",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        duration=duration,
        token=token,
        on_progress=on_progress,
    )
    return out


def isolate_voice(
    ff: FFmpeg,
    src: str | Path,
    out_path: str | Path,
    *,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
) -> Path:
    """Giu loi thoai, giam nhac nen bang loc dai tan giong noi."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.run(
        [
            "-i",
            str(src),
            "-af",
            "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=110,lowpass=f=7000,"
            "afftdn=nf=-25,dynaudnorm=f=200:g=5",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        duration=duration,
        token=token,
        on_progress=on_progress,
    )
    return out


def mix_voice_and_music(
    ff: FFmpeg,
    voice: str | Path,
    music: str | Path,
    out_path: str | Path,
    *,
    music_volume: int = 35,
    ducking: bool = True,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
) -> Path:
    """Tron giong doc voi nhac nen, co the tu ha nhac khi co loi thoai."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gain = max(0.0, min(2.0, music_volume / 100.0))
    if ducking:
        graph = (
            f"[1:a]volume={gain:.3f},aformat=channel_layouts=stereo[music];"
            "[0:a]aformat=channel_layouts=stereo[voice];"
            "[voice]asplit=2[v1][vside];"
            "[music][vside]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=400[duck];"
            "[v1][duck]amix=inputs=2:duration=longest:normalize=0[out]"
        )
    else:
        graph = (
            f"[1:a]volume={gain:.3f},aformat=channel_layouts=stereo[music];"
            "[0:a]aformat=channel_layouts=stereo[voice];"
            "[voice][music]amix=inputs=2:duration=longest:normalize=0[out]"
        )
    ff.run(
        [
            "-i",
            str(voice),
            "-i",
            str(music),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        duration=duration,
        token=token,
        on_progress=on_progress,
    )
    return out


def apply_equalizer(
    ff: FFmpeg,
    src: str | Path,
    out_path: str | Path,
    *,
    bass: int = 0,
    mid: int = 0,
    treble: int = 0,
    token: CancelToken | None = None,
) -> Path:
    """Chinh am sac giong doc: tang giam dai tram, dai giua va dai cao (dB)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    if bass:
        parts.append(f"bass=g={max(-20, min(20, int(bass)))}")
    if mid:
        parts.append(f"equalizer=f=1200:t=q:w=1.2:g={max(-20, min(20, int(mid)))}")
    if treble:
        parts.append(f"treble=g={max(-20, min(20, int(treble)))}")
    if not parts:
        shutil.copyfile(src, out)
        return out
    ff.run(
        [
            "-i",
            str(src),
            "-af",
            ",".join(parts),
            "-ar",
            str(VOICE_RATE),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        token=token,
    )
    return out


def cut_segment(
    ff: FFmpeg,
    src: str | Path,
    out_path: str | Path,
    start: float,
    end: float,
    *,
    rate: int = 16000,
    token: CancelToken | None = None,
) -> Path:
    """Cat mot doan am thanh theo moc thoi gian."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    length = max(0.05, end - start)
    ff.run(
        [
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        token=token,
    )
    return out


def extract_frame(
    ff: FFmpeg,
    video: str | Path,
    seconds: float,
    out_path: str | Path,
    *,
    token: CancelToken | None = None,
) -> Path:
    """Lay mot khung hinh ra tep PNG, dung cho khoanh vung OCR."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.run(
        [
            "-ss",
            f"{max(0.0, seconds):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ],
        token=token,
    )
    return out


_PTS_TIME = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def _crop_filter(region: Sequence[int] | None) -> str:
    if region and len(region) == 4 and region[2] > 0 and region[3] > 0:
        x, y, w, h = (int(v) for v in region)
        return f"crop={w}:{h}:{x}:{y}"
    return ""


def _collect_pts(lines: list[str]) -> list[float]:
    """Lay moc thoi gian that cua tung khung hinh tu dong log cua FFmpeg."""
    out: list[float] = []
    for line in lines:
        if "showinfo" not in line:
            continue
        found = _PTS_TIME.search(line)
        if found:
            out.append(float(found.group(1)))
    return out


def extract_frames(
    ff: FFmpeg,
    video: str | Path,
    out_dir: str | Path,
    *,
    fps: float = 2.0,
    region: Sequence[int] | None = None,
    duration: float = 0.0,
    token: CancelToken | None = None,
    on_progress=None,
) -> list[tuple[float, Path]]:
    """Xuat loat khung hinh (co the cat theo vung) de chay OCR.

    Tra ve tung cap (giay that cua khung hinh, tep anh). Dung bo loc 'select'
    chu khong dung 'fps' vi 'fps' danh lai moc thoi gian, khien moc doc duoc
    lech nua nhip so voi video that.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("frame_*.png"):
        old.unlink(missing_ok=True)
    gap = 1.0 / max(0.2, float(fps))
    # PTS la so thuc. Voi video 25 fps, nam khung cach nhau ve ly thuyet
    # 0,200 giay nhung phep tinh co the cho 0,199999; so sanh thang voi 0,2
    # se bo qua khung thu nam va chi lay khoang 4,5 hinh/giay. Tru mot epsilon
    # nho de luon lay du dung tan suat ma khong chon them khung ke canh.
    select_gap = max(0.001, gap - min(0.005, gap * 0.025))
    filters = [
        f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{select_gap:.4f})",
        "showinfo",
    ]
    crop = _crop_filter(region)
    if crop:
        filters.insert(0, crop)
    lines: list[str] = []
    ff.run(
        ["-i", str(video), "-vf", ",".join(filters), "-vsync", "0", str(d / "frame_%06d.png")],
        duration=duration,
        token=token,
        on_progress=on_progress,
        on_log=lines.append,
    )
    paths = sorted(d.glob("frame_*.png"))
    stamps = _collect_pts(lines)
    if len(stamps) != len(paths):  # log khong doc duoc thi quay ve moc uoc luong
        stamps = [i * gap for i in range(len(paths))]
    return list(zip(stamps, paths, strict=False))


def extract_window(
    ff: FFmpeg,
    video: str | Path,
    out_dir: str | Path,
    start: float,
    length: float,
    *,
    count: int = 8,
    region: Sequence[int] | None = None,
    token: CancelToken | None = None,
) -> list[tuple[float, Path]]:
    """Lay nhieu khung hinh sat nhau trong mot khoang ngan de do lai moc thoi gian.

    Tra ve tung cap (giay, tep anh) theo thu tu thoi gian tang dan.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("win_*.png"):
        old.unlink(missing_ok=True)
    span = max(0.05, float(length))
    begin = max(0.0, float(start))
    filters = ["showinfo"]
    crop = _crop_filter(region)
    if crop:
        filters.insert(0, crop)
    lines: list[str] = []
    ff.run(
        [
            "-ss",
            f"{begin:.3f}",
            "-i",
            str(video),
            "-t",
            f"{span:.3f}",
            "-vf",
            ",".join(filters),
            "-vsync",
            "0",
            str(d / "win_%04d.png"),
        ],
        token=token,
        on_log=lines.append,
    )
    paths = sorted(d.glob("win_*.png"))
    stamps = _collect_pts(lines)
    if len(stamps) != len(paths):
        stamps = [span * i / max(1, len(paths)) for i in range(len(paths))]
    pairs = [(begin + s, p) for s, p in zip(stamps, paths, strict=False)]
    keep = max(2, int(count))
    if len(pairs) > keep:  # bo bot cho do ton thoi gian doc chu
        stride = math.ceil(len(pairs) / keep)
        pairs = pairs[::stride]
    return pairs


def sample_frames(
    ff: FFmpeg,
    video: str | Path,
    out_dir: str | Path,
    *,
    count: int = 14,
    duration: float = 0.0,
    region: Sequence[int] | None = None,
    token: CancelToken | None = None,
    on_progress=None,
) -> list[tuple[float, Path]]:
    """Lay vai khung hinh rai deu tu dau den cuoi video de do thu mau va co chu.

    Nhay thang den tung moc thoi gian nen nhanh hon nhieu so voi giai ca video,
    du dung cho viec do thu.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("probe_*.png"):
        old.unlink(missing_ok=True)
    total = max(2, int(count))
    span = float(duration) if duration and duration > 0 else 0.0
    if span <= 0:
        try:
            span = float(ff.probe(video).duration)
        except FFmpegError:
            span = 0.0
    if span <= 0:
        span = 60.0
    begin = min(2.0, span * 0.05)  # bo qua vai giay dau, thuong la hinh hieu
    usable = max(0.5, span - begin - min(2.0, span * 0.05))
    crop = _crop_filter(region)
    out: list[tuple[float, Path]] = []
    for i in range(total):
        if token is not None and token.cancelled:
            break
        stamp = begin + usable * i / max(1, total - 1)
        path = d / f"probe_{i:03d}.png"
        args = ["-ss", f"{stamp:.3f}", "-i", str(video)]
        if crop:
            args += ["-vf", crop]
        args += ["-frames:v", "1", str(path)]
        try:
            ff.run(args, token=token)
        except FFmpegError:
            continue
        if path.is_file():
            out.append((stamp, path))
        if on_progress:
            on_progress(max(0, min(100, int((i + 1) / total * 100))))
    return out


# --------------------------------------------------------------- ghep track giong doc


def build_voice_track(
    segments: Sequence[tuple[float, Path]],
    out_path: str | Path,
    total_duration: float,
    *,
    rate: int = VOICE_RATE,
    mix_overlaps: bool = False,
) -> Path:
    """Dat cac doan giong doc vao dung moc thoi gian tren mot track im lang.

    Cac doan phai la WAV mono 16 bit cung tan so lay mau. Doan nao vuot qua do dai
    track se bi cat bot de khong lam hong tep ket qua.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(math.ceil(max(0.1, total_duration) * rate)))
    buffer = array("h", [0]) * total_frames
    for start, seg_path in segments:
        p = Path(seg_path)
        if not p.is_file():
            continue
        try:
            with wave.open(str(p), "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != VOICE_WIDTH:
                    continue
                if wf.getframerate() != rate:
                    continue
                frames = wf.readframes(wf.getnframes())
        except (wave.Error, OSError):
            continue
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        offset = max(0, int(round(max(0.0, start) * rate)))
        if offset >= len(buffer):
            continue
        end = min(len(buffer), offset + len(samples))
        if mix_overlaps:
            for target, sample in zip(range(offset, end), samples, strict=False):
                buffer[target] = max(-32768, min(32767, buffer[target] + sample))
        else:
            buffer[offset:end] = samples[: end - offset]
    tmp = out.with_name(out.name + ".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(VOICE_WIDTH)
        wf.setframerate(rate)
        if sys.byteorder != "little":
            buffer.byteswap()
        wf.writeframes(buffer.tobytes())
    tmp.replace(out)
    return out


def wav_duration(path: str | Path) -> float:
    """Do dai cua tep WAV theo giay, tra ve 0 khi tep khong doc duoc."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate() or 1
            return wf.getnframes() / rate
    except (wave.Error, OSError):
        return 0.0


def append_silence(src: str | Path, out_path: str | Path, seconds: float) -> Path:
    """Them khoang nghi vao cuoi WAV ma khong ma hoa lai phan voice."""
    source = Path(src)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pause = max(0.0, min(5.0, float(seconds)))
    try:
        with wave.open(str(source), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())
    except (OSError, wave.Error) as exc:
        raise FFmpegError(f"Khong doc duoc WAV de them khoang nghi: {source.name}") from exc
    silent_frames = int(round(params.framerate * pause))
    silence = b"\x00" * silent_frames * params.nchannels * params.sampwidth
    with wave.open(str(out), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames + silence)
    return out


def probe_or_raise(ff: FFmpeg, path: str | Path) -> MediaInfo:
    """Doc thong tin media, doi thong bao loi sang cau de hieu."""
    try:
        return ff.probe(path)
    except FFmpegError as exc:
        raise FFmpegError(f"Khong doc duoc tep {Path(path).name}: {exc}") from exc


def crop_image(
    src: str | Path | Any,
    region: Sequence[int],
    out_path: str | Path | None = None,
) -> Any:
    """Cat mot vung (x, y, w, h) tu anh va luu hoac tra ve PIL Image."""
    from PIL import Image

    if isinstance(src, (str, Path)):
        with Image.open(src) as raw:
            img = raw.convert("RGB")
            cropped = _crop_pil(img, region)
            if out_path:
                out = Path(out_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(out)
                return out
            return cropped
    elif hasattr(src, "crop"):
        cropped = _crop_pil(src, region)
        if out_path:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(out)
            return out
        return cropped
    raise TypeError(f"Unsupported image type: {type(src)}")


def _crop_pil(img: Any, region: Sequence[int]) -> Any:
    if len(region) != 4 or region[2] <= 0 or region[3] <= 0:
        return img.copy()
    x, y, w, h = (int(v) for v in region)
    cw = max(1, min(w, img.width - max(0, x)))
    ch = max(1, min(h, img.height - max(0, y)))
    return img.crop((max(0, x), max(0, y), max(0, x) + cw, max(0, y) + ch))


def chunk_sequence(items: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    """Chia mot danh sach thanh cac doan nho co do dai toi da chunk_size."""
    size = max(1, int(chunk_size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
