"""Quan ly Render Preset cho AutoSub Studio (Screen Render)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import config_dir, write_text_atomic
from .settings import SubtitleStyle

PRESETS_CONFIG_FILE = "render_presets.json"


@dataclass
class RenderPreset:
    """Cau hinh render preset luu lai cac thong so Screen Render."""

    name: str
    style: SubtitleStyle = field(default_factory=SubtitleStyle)
    subtitle_visible: bool = True
    render_scale: str = "giu nguyen"
    render_fps: str = "giu nguyen"
    render_crf: int = 20
    render_preset: str = "medium"
    lut_path: str = ""
    builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "style": self.style.to_dict(),
            "subtitle_visible": self.subtitle_visible,
            "render_scale": self.render_scale,
            "render_fps": self.render_fps,
            "render_crf": self.render_crf,
            "render_preset": self.render_preset,
            "lut_path": self.lut_path,
            "builtin": self.builtin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderPreset:
        raw_style = data.get("style")
        style = (
            SubtitleStyle.from_dict(raw_style)
            if isinstance(raw_style, dict)
            else SubtitleStyle()
        )
        return cls(
            name=str(data.get("name", "")).strip(),
            style=style,
            subtitle_visible=bool(data.get("subtitle_visible", True)),
            render_scale=str(data.get("render_scale", "giu nguyen")),
            render_fps=str(data.get("render_fps", "giu nguyen")),
            render_crf=int(data.get("render_crf", 20)),
            render_preset=str(data.get("render_preset", "medium")),
            lut_path=str(data.get("lut_path", "")),
            builtin=bool(data.get("builtin", False)),
        )


BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "DEFAULT": {
        "name": "DEFAULT",
        "style": SubtitleStyle(
            font="Arial",
            font_size=16,
            bold=True,
            primary_color="#e0e196",
            outline_color="#000000",
            back_color="#000000",
            back_opacity=50,
            outline=2.0,
            shadow=1.0,
            alignment=2,
            margin_v=60,
        ).to_dict(),
        "subtitle_visible": True,
        "render_scale": "giu nguyen",
        "render_fps": "giu nguyen",
        "render_crf": 20,
        "render_preset": "medium",
        "lut_path": "",
        "builtin": True,
    },
    "REVIEW ENG": {
        "name": "REVIEW ENG",
        "style": SubtitleStyle(
            font="Arial",
            font_size=18,
            bold=True,
            primary_color="#FFFFFF",
            outline_color="#000000",
            back_color="#000000",
            back_opacity=50,
            outline=2.0,
            shadow=1.0,
            alignment=2,
            margin_v=60,
        ).to_dict(),
        "subtitle_visible": True,
        "render_scale": "giu nguyen",
        "render_fps": "giu nguyen",
        "render_crf": 18,
        "render_preset": "medium",
        "lut_path": "",
        "builtin": True,
    },
    "REVIEW VI": {
        "name": "REVIEW VI",
        "style": SubtitleStyle(
            font="Arial",
            font_size=18,
            bold=True,
            primary_color="#FFFF00",
            outline_color="#000000",
            back_color="#000000",
            back_opacity=50,
            outline=2.0,
            shadow=1.0,
            alignment=2,
            margin_v=60,
        ).to_dict(),
        "subtitle_visible": True,
        "render_scale": "giu nguyen",
        "render_fps": "giu nguyen",
        "render_crf": 18,
        "render_preset": "medium",
        "lut_path": "",
        "builtin": True,
    },
    "BODYCAM": {
        "name": "BODYCAM",
        "style": SubtitleStyle(
            font="Consolas",
            font_size=14,
            bold=True,
            primary_color="#00FF00",
            outline_color="#000000",
            back_color="#000000",
            back_opacity=80,
            outline=1.5,
            shadow=0.0,
            alignment=1,
            margin_v=40,
        ).to_dict(),
        "subtitle_visible": True,
        "render_scale": "giu nguyen",
        "render_fps": "giu nguyen",
        "render_crf": 22,
        "render_preset": "fast",
        "lut_path": "",
        "builtin": True,
    },
    "ANIMAL STORY": {
        "name": "ANIMAL STORY",
        "style": SubtitleStyle(
            font="Arial",
            font_size=20,
            bold=True,
            primary_color="#FFD700",
            outline_color="#1A1A1A",
            back_color="#000000",
            back_opacity=40,
            outline=3.0,
            shadow=1.5,
            alignment=2,
            margin_v=70,
        ).to_dict(),
        "subtitle_visible": True,
        "render_scale": "giu nguyen",
        "render_fps": "giu nguyen",
        "render_crf": 20,
        "render_preset": "medium",
        "lut_path": "",
        "builtin": True,
    },
}


class PresetManager:
    """Quan ly danh sach preset he thong va nguoi dung."""

    def __init__(self, config_file: Path | None = None) -> None:
        self.path = config_file or (config_dir() / PRESETS_CONFIG_FILE)
        self._custom: dict[str, RenderPreset] = {}
        self.load()

    @staticmethod
    def _create_builtin(data: dict[str, Any]) -> RenderPreset:
        preset = RenderPreset.from_dict(data)
        preset.builtin = True
        return preset

    def load(self) -> None:
        self._custom.clear()
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for name, item in raw.items():
                    if isinstance(item, dict) and name not in BUILTIN_PRESETS:
                        preset = RenderPreset.from_dict(item)
                        preset.builtin = False
                        self._custom[preset.name] = preset
        except (OSError, ValueError, json.JSONDecodeError):
            self._custom.clear()

    def save(self) -> Path:
        data = {name: preset.to_dict() for name, preset in self._custom.items()}
        return write_text_atomic(
            self.path,
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def list_presets(self) -> list[RenderPreset]:
        builtins = [
            self._create_builtin(data) for data in BUILTIN_PRESETS.values()
        ]
        customs = [self._custom[k] for k in sorted(self._custom.keys())]
        return builtins + customs

    def list_names(self) -> list[str]:
        return [p.name for p in self.list_presets()]

    def get(self, name: str) -> RenderPreset:
        clean = str(name or "").strip()
        if clean in self._custom:
            return copy.deepcopy(self._custom[clean])
        if clean in BUILTIN_PRESETS:
            return self._create_builtin(BUILTIN_PRESETS[clean])
        return self._create_builtin(BUILTIN_PRESETS["DEFAULT"])

    def create(
        self,
        name: str,
        style: SubtitleStyle | dict[str, Any] | None = None,
        subtitle_visible: bool = True,
        render_scale: str = "giu nguyen",
        render_fps: str = "giu nguyen",
        render_crf: int = 20,
        render_preset: str = "medium",
        lut_path: str = "",
        *,
        allow_overwrite: bool = False,
    ) -> RenderPreset:
        clean = str(name or "").strip()
        if not clean:
            raise ValueError("Ten preset khong duoc de trong.")
        if clean in BUILTIN_PRESETS:
            raise ValueError(f"Khong the dat ten trung voi preset mac dinh: '{clean}'.")
        if clean in self._custom and not allow_overwrite:
            raise ValueError(f"Preset '{clean}' da ton tai.")

        if isinstance(style, SubtitleStyle):
            st = copy.deepcopy(style)
        elif isinstance(style, dict):
            st = SubtitleStyle.from_dict(style)
        else:
            st = SubtitleStyle()

        preset = RenderPreset(
            name=clean,
            style=st,
            subtitle_visible=bool(subtitle_visible),
            render_scale=str(render_scale),
            render_fps=str(render_fps),
            render_crf=int(render_crf),
            render_preset=str(render_preset),
            lut_path=str(lut_path),
            builtin=False,
        )
        self._custom[clean] = preset
        self.save()
        return copy.deepcopy(preset)

    def update(
        self,
        name: str,
        style: SubtitleStyle | dict[str, Any] | None = None,
        subtitle_visible: bool | None = None,
        render_scale: str | None = None,
        render_fps: str | None = None,
        render_crf: int | None = None,
        render_preset: str | None = None,
        lut_path: str | None = None,
    ) -> RenderPreset:
        clean = str(name or "").strip()
        if clean in BUILTIN_PRESETS:
            raise ValueError(f"Khong the cap nhat preset mac dinh: '{clean}'.")
        if clean not in self._custom:
            raise KeyError(f"Khong tim thay preset: '{clean}'.")

        target = self._custom[clean]
        if style is not None:
            target.style = (
                copy.deepcopy(style)
                if isinstance(style, SubtitleStyle)
                else SubtitleStyle.from_dict(style)
            )
        if subtitle_visible is not None:
            target.subtitle_visible = bool(subtitle_visible)
        if render_scale is not None:
            target.render_scale = str(render_scale)
        if render_fps is not None:
            target.render_fps = str(render_fps)
        if render_crf is not None:
            target.render_crf = int(render_crf)
        if render_preset is not None:
            target.render_preset = str(render_preset)
        if lut_path is not None:
            target.lut_path = str(lut_path)

        self.save()
        return copy.deepcopy(target)

    def delete(self, name: str) -> bool:
        clean = str(name or "").strip()
        if clean in BUILTIN_PRESETS:
            raise ValueError(f"Khong the xoa preset mac dinh: '{clean}'.")
        if clean not in self._custom:
            return False
        if len(self.list_presets()) <= 1:
            raise ValueError("Khong the xoa preset cuoi cung.")
        del self._custom[clean]
        self.save()
        return True


__all__ = [
    "BUILTIN_PRESETS",
    "PRESETS_CONFIG_FILE",
    "PresetManager",
    "RenderPreset",
]
