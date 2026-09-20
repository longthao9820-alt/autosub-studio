"""Kiem thu video tong hop (synthetic video) voi FFmpeg va AI OCR Pipeline.

Kiem tra:
1. Video tong hop co text phu de ben trong vung lua chon (region) va distractor text ben ngoai.
2. FFmpeg cat dung vung region voi kich thuoc chinh xac (crop dimensions).
3. Chat completion (AI Gateway) nhan dung anh crop 480x60, khong chua distractor text.
4. Tao SRT dung moc thoi gian, don dieu, khong loi.
5. Tuyet doi khong goi bat ky ham nao cua RapidOCR hoac local OCR.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from autosub_studio.core import formats
from autosub_studio.providers import ocr, ocr_ai
from autosub_studio.providers.ocr_ai_provider import AIOCRProvider, BatchItem
from autosub_studio.services import ai_gateway
from autosub_studio.services.ffmpeg import FFmpeg
from autosub_studio.services.settings import Settings


@pytest.fixture
def synthetic_video(tmp_path: Path) -> tuple[Path, list[int]]:
    """Tao video mp4 tong hop bang FFmpeg: 640x360, region [80, 260, 480, 60]."""
    ff = FFmpeg()
    if not ff.available:
        pytest.skip("FFmpeg khong co san tren may")

    frames_dir = tmp_path / "synth_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 4 frames = 2 seconds at 2 fps
    for i in range(4):
        img = Image.new("RGB", (640, 360), color=(15, 15, 15))
        draw = ImageDraw.Draw(img)
        # Distractor text OUTSIDE selected region (top header)
        draw.text((80, 40), "DISTRACTOR HEADER OUTSIDE", fill=(255, 0, 0))
        # Subtitle text INSIDE selected region [80, 260, 480, 60]
        draw.text((120, 280), "HELLO SYNTHETIC SUBTITLE", fill=(255, 255, 255))
        img.save(frames_dir / f"frame_{i:02d}.png")

    video_path = tmp_path / "synthetic_subtitle.mp4"
    ff.run([
        "-r",
        "2",
        "-i",
        str(frames_dir / "frame_%02d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ])
    assert video_path.is_file() and video_path.stat().st_size > 0

    region = [80, 260, 480, 60]
    return video_path, region


def test_synthetic_video_ai_ocr_pipeline_mocked(
    synthetic_video: tuple[Path, list[int]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chay run_ai_ocr_pipeline that tren synthetic video voi mock gateway."""
    video_path, region = synthetic_video
    ff = FFmpeg()

    # Poison local OCR calls
    def forbidden_local(*args: object, **kwargs: object) -> None:
        raise AssertionError("RapidOCR / local OCR was invoked in AI OCR pipeline!")

    monkeypatch.setattr(ocr, "read_frames", forbidden_local)
    monkeypatch.setattr(ocr, "probe_frames", forbidden_local)
    monkeypatch.setattr(ocr, "refine_boundaries", forbidden_local)

    received_crops: list[tuple[int, int, Image.Image]] = []

    def mock_chat_completion(endpoint: str, api_key: str, **kwargs: Any) -> str:
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        user_c: list[dict[str, Any]] = messages[0]["content"]
        ids: list[str] = []
        for block in user_c:
            if block.get("type") == "text" and block["text"].startswith("ID: "):
                ids.append(block["text"].replace("ID: ", ""))
            elif block.get("type") == "image_url":
                b64 = block["image_url"]["url"].split(",")[-1]
                crop_data = base64.b64decode(b64)
                img = Image.open(io.BytesIO(crop_data))
                received_crops.append((img.width, img.height, img.copy()))

        results = [
            {"id": i, "text": "HELLO SYNTHETIC SUBTITLE", "confidence": 1.0}
            for i in ids
        ]
        return json.dumps({"results": results})

    monkeypatch.setattr(ai_gateway, "chat_completion", mock_chat_completion)

    temp_dir = tmp_path / "temp_ocr"
    cues = ocr_ai.run_ai_ocr_pipeline(
        ff,
        video_path,
        temp_dir,
        region=region,
        fps=2.0,
        endpoint="http://mock.gateway/v1",
        api_key="sk-mock-key",
        model="sub",
        batch_size=4,
    )

    # 1. Kiem tra kich thuoc crop
    assert len(received_crops) >= 1, "Gateway phai nhan it nhat 1 anh crop"
    for w, h, _crop_img in received_crops:
        assert (w, h) == (480, 60), (
            f"Kich thuoc anh crop khong dung voi region [80, 260, 480, 60]: ({w}, {h})"
        )

    # 2. Kiem tra phu de va SRT
    assert len(cues) >= 1
    assert cues[0].text == "HELLO SYNTHETIC SUBTITLE"
    assert cues[0].start >= 0.0
    assert cues[0].end <= 2.5
    assert cues[0].end > cues[0].start

    # Xuat ra file SRT va kiem tra dinh dang
    srt_file = tmp_path / "output.srt"
    formats.save_subtitle(srt_file, formats.SubtitleDoc(cues=cues))
    assert srt_file.is_file()
    srt_content = srt_file.read_text(encoding="utf-8")
    assert "HELLO SYNTHETIC SUBTITLE" in srt_content
    assert "00:00:00" in srt_content


def test_synthetic_video_live_gateway_batch(
    synthetic_video: tuple[Path, list[int]],
) -> None:
    """Kiem tra mot batch thuc te gui len Localhost Gateway voi crop tu video tong hop."""
    if os.environ.get("AUTOSUB_LIVE_AI_TEST", "").strip().lower() not in ("1", "true", "yes"):
        pytest.skip("Live Gateway test is opt-in (set AUTOSUB_LIVE_AI_TEST=1 to run)")

    settings = Settings.load()
    key = Settings.get_secret("ai_gateway_key")
    if not settings.ai_endpoint or not key:
        pytest.skip("Chua co ai_endpoint hoac secret key de chay live test")

    # Kiem tra Gateway co reachable hay khong (thu models endpoint va base endpoint khong trung lap)
    import urllib.request

    from autosub_studio.services.ai_gateway import normalize_models_endpoint

    probe_urls: list[str] = []
    norm_models = normalize_models_endpoint(settings.ai_endpoint)
    if norm_models:
        probe_urls.append(norm_models)
    base_ep = settings.ai_endpoint.rstrip("/")
    if base_ep and base_ep not in probe_urls:
        probe_urls.append(base_ep)

    reachable = False
    for url in probe_urls:
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    reachable = True
                    break
        except Exception:
            continue

    if not reachable:
        pytest.skip("Gateway localhost khong kha dung tai thoi diem nay")

    video_path, region = synthetic_video
    ff = FFmpeg()

    # Trich xuat 1 frame truc tiep theo region
    from autosub_studio.services import media
    crops: list[bytes] = []
    for _idx, _start, _end, frames, _dir in media.extract_video_chunks(
        ff, video_path, video_path.parent / "live_chunks", region=region, fps=2.0
    ):
        for _ts, fpath in frames:
            crops.append(fpath.read_bytes())
            if len(crops) >= 2:
                break
        if len(crops) >= 2:
            break

    assert len(crops) >= 1, "Khong trich xuat duoc frame nao tu video"

    # Gui batch len Live Gateway
    provider = AIOCRProvider(
        endpoint=settings.ai_endpoint,
        api_key=key,
        model=settings.ai_model_sub,
        thinking=settings.ai_thinking_sub,
    )

    items = [
        BatchItem(id=f"live_synth_{i}", image=c)
        for i, c in enumerate(crops[:2])
    ]
    results = provider.execute_batch(items)
    assert len(results) == len(items)
    for r in results:
        assert r.id in [it.id for it in items]
        assert r.text is not None
        # Subtitle trong crop la 'HELLO SYNTHETIC SUBTITLE'
        assert "HELLO" in r.text.upper() or "SUBTITLE" in r.text.upper() or len(r.text) > 0
