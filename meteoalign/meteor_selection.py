"""流星框选的数据模型与 JSON 读写。

本模块不依赖 Qt，供后续的像素到 ICRS 与渲染流程直接复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METEOR_SELECTION_SCHEMA = "hgastro.meteor_selection"
METEOR_SELECTION_VERSION = 1


@dataclass(frozen=True)
class MeteorBox:
    """图像原始像素坐标系中的一个矩形框。"""

    left: float
    top: float
    right: float
    bottom: float

    def normalized(self) -> "MeteorBox":
        """返回左上、右下顺序规范化后的矩形。"""

        return MeteorBox(
            left=min(float(self.left), float(self.right)),
            top=min(float(self.top), float(self.bottom)),
            right=max(float(self.left), float(self.right)),
            bottom=max(float(self.top), float(self.bottom)),
        )

    def clamped(self, width: int, height: int) -> "MeteorBox":
        """将矩形限制在给定图像边界内。"""

        if width <= 0 or height <= 0:
            raise ValueError("图像尺寸必须为正数。")
        box = self.normalized()
        return MeteorBox(
            left=min(max(box.left, 0.0), float(width)),
            top=min(max(box.top, 0.0), float(height)),
            right=min(max(box.right, 0.0), float(width)),
            bottom=min(max(box.bottom, 0.0), float(height)),
        )

    def to_json(self) -> dict[str, dict[str, int]]:
        """转换为包含左上、右下整数像素点的 JSON 对象。"""

        box = self.normalized()
        return {
            "top_left": {"x": int(round(box.left)), "y": int(round(box.top))},
            "bottom_right": {"x": int(round(box.right)), "y": int(round(box.bottom))},
        }

    @classmethod
    def from_json(cls, payload: object) -> "MeteorBox":
        """从单个 JSON 矩形对象读取坐标。"""

        if not isinstance(payload, dict):
            raise ValueError("流星框必须是对象。")
        top_left = payload.get("top_left")
        bottom_right = payload.get("bottom_right")
        if not isinstance(top_left, dict) or not isinstance(bottom_right, dict):
            raise ValueError("流星框缺少 top_left 或 bottom_right。")
        try:
            return cls(
                left=float(top_left["x"]),
                top=float(top_left["y"]),
                right=float(bottom_right["x"]),
                bottom=float(bottom_right["y"]),
            ).normalized()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("流星框坐标无效。") from exc


def meteor_json_path(image_path: str | Path) -> Path:
    """返回图像同目录下的流星框选 JSON 路径。"""

    path = Path(image_path).expanduser()
    return path.with_name(f"{path.stem}_Meteor.json")


def build_meteor_selection_payload(
    image_path: str | Path,
    image_width: int,
    image_height: int,
    boxes: Iterable[MeteorBox],
) -> dict[str, object]:
    """构建稳定的流星框选 JSON 结构。"""

    image_file = Path(image_path).expanduser()
    normalized_boxes = [box.clamped(image_width, image_height).to_json() for box in boxes]
    return {
        "schema": METEOR_SELECTION_SCHEMA,
        "version": METEOR_SELECTION_VERSION,
        "source_image": image_file.name,
        "source_image_stem": image_file.stem,
        "image_size_px": {"width": int(image_width), "height": int(image_height)},
        "meteor_boxes": normalized_boxes,
    }


def save_meteor_selection(
    image_path: str | Path,
    image_width: int,
    image_height: int,
    boxes: Iterable[MeteorBox],
) -> Path:
    """将一个图像的框选保存到其同目录 JSON 文件。"""

    output_path = meteor_json_path(image_path)
    payload = build_meteor_selection_payload(image_path, image_width, image_height, boxes)
    output_path.write_text(
        # 转义扩展 Unicode 和代理项，避免 Windows 特殊文件名无法编码成 UTF-8。
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_meteor_selection(image_path: str | Path) -> list[MeteorBox]:
    """读取图像同目录已有的流星框选；不存在时返回空列表。"""

    json_path = meteor_json_path(image_path)
    if not json_path.exists():
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取流星框选 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("流星框选 JSON 的根对象必须是对象。")
    box_payloads = payload.get("meteor_boxes")
    if not isinstance(box_payloads, list):
        raise ValueError("流星框选 JSON 缺少 meteor_boxes 列表。")
    return [MeteorBox.from_json(box_payload) for box_payload in box_payloads]
