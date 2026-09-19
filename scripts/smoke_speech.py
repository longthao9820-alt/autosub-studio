"""Chay thu giong doc Piper Local va nhan dang giong noi.

Cach dung:
    python scripts/smoke_speech.py [--model tiny] [--voice en_US-bryce-medium]

Script doc mot cau bang Local Voice (Piper Local), roi dua chinh tep do cho
faster-whisper nhan dang lai va so sanh ket qua. Neu chua co runtime Piper
hoac chua co model voice tren may, script se bo qua ro rang (skip).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosub_studio.providers import asr, local_voice, tts  # noqa: E402
from autosub_studio.services import media  # noqa: E402
from autosub_studio.services.ffmpeg import FFmpeg  # noqa: E402

SENTENCE = "The quick brown fox jumps over the lazy dog."


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=list(asr.MODEL_SIZES))
    parser.add_argument("--voice", default="", help="Voice ID tu catalog Piper Local")
    args = parser.parse_args()

    ready, reason = local_voice.piper_runtime_ready()
    if not ready:
        print(f"[BO QUA] Piper Local chua san sang: {reason}")
        return 0

    mgr = local_voice.get_default_manager()
    catalog = local_voice.list_catalog()
    ready_voices = [v for v in catalog if mgr.get_status(v.id) == local_voice.STATUS_READY]
    if not ready_voices:
        print("[BO QUA] Chua co model Piper ready tren may de chay thu giong doc.")
        return 0

    target_voice = args.voice
    if not target_voice:
        target_voice = ready_voices[0].id
    elif mgr.get_status(target_voice) != local_voice.STATUS_READY:
        print(f"[BO QUA] Voice '{target_voice}' chua san sang tren may.")
        return 0

    work = Path(tempfile.mkdtemp(prefix="autosub_speech_"))
    ok = True
    try:
        print(f"1) Tao giong doc bang Local Voice ({target_voice})...")
        spoken = work / "spoken.wav"
        try:
            tts.synthesize(tts.PROVIDER_LOCAL, SENTENCE, spoken, voice=target_voice)
        except tts.TTSError as exc:
            print(f"   [HONG] {exc}")
            return 1
        size = spoken.stat().st_size
        print(f"   [OK] Da tao {spoken.name} ({size / 1024:.0f} KB)")

        ff = FFmpeg()
        if not ff.available:
            print("   [BO QUA] Khong co FFmpeg nen khong chuyen duoc dinh dang.")
            return 0
        wav16 = work / "spoken_16k.wav"
        media.extract_audio(ff, spoken, wav16, rate=16000, channels=1)
        print(f"   [OK] Da chuyen sang 16 kHz ({media.wav_duration(wav16):.2f} giay)")

        print(f"2) Nhan dang lai bang faster-whisper (model {args.model})...")
        if not asr.is_available():
            print(f"   [BO QUA] {asr.install_hint()}")
            return 0
        try:
            cues, lang = asr.transcribe(
                wav16,
                model_size=args.model,
                language="en",
                device="CPU",
                duration=media.wav_duration(wav16),
                on_log=lambda m: print(f"   {m}"),
            )
        except (asr.ASRError, asr.ASRUnavailable) as exc:
            print(f"   [HONG] {exc}")
            return 1

        text = " ".join(c.text for c in cues)
        print(f"   Ngon ngu nhan duoc: {lang}")
        print(f"   Van ban nhan duoc : {text.strip()!r}")
        overlap = words(text) & words(SENTENCE)
        ratio = len(overlap) / max(1, len(words(SENTENCE)))
        print(f"   Trung khop tu: {len(overlap)}/{len(words(SENTENCE))} ({ratio:.0%})")
        if ratio < 0.6:
            print("   [HONG] Ket qua nhan dang khac qua nhieu so voi cau goc.")
            ok = False
        else:
            print("   [OK] Nhan dang khop voi cau goc.")

        print("3) Chia lai cau cho vua man hinh...")
        parts = asr.resegment(cues, max_chars=42, max_lines=2)
        print(f"   [OK] {len(cues)} doan -> {len(parts)} cau hien thi")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nKET QUA:", "DAT" if ok else "HONG")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
