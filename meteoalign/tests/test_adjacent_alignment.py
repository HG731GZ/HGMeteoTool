from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from meteoalign.adjacent_alignment import (
    ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
    RoughFramingTransform,
    _detect_stars,
    _star_correspondences,
    calculate_adjacent_rough_framing,
)
from meteoalign.alignment.constants import SKY_MATCHING_MODEL_RECTILINEAR
from meteoalign.application.app_alignment import AlignmentMixin
from meteoalign.camera_calibration import CameraCalibrationProfile
from meteoalign.config import AdjacentAlignmentConfig, AdjacentStarAlignmentConfig
from meteoalign.frame_astrometry import FrameAstrometricModel, FramePose


def _draw_feature_image(width: int, height: int, seed: int) -> np.ndarray:
    """生成同时含星点与地景纹理的稳定合成图像。"""

    generator = np.random.default_rng(seed)
    image = generator.integers(0, 8, size=(height, width), dtype=np.uint8)
    for _index in range(90):
        x_value = int(generator.integers(16, width - 16))
        y_value = int(generator.integers(16, height - 16))
        radius = int(generator.integers(1, 4))
        brightness = int(generator.integers(120, 256))
        cv2.circle(image, (x_value, y_value), radius, brightness, -1, cv2.LINE_AA)
    for _index in range(18):
        x_value = int(generator.integers(8, width - 48))
        y_value = int(generator.integers(8, height - 24))
        cv2.rectangle(image, (x_value, y_value), (x_value + 16, y_value + 9), 180, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (0, 0), 0.65)


def _frame_model(width: int, height: int) -> FrameAstrometricModel:
    """构造用于测试的普通透视 Pixel↔ICRS 模型。"""

    profile = CameraCalibrationProfile(
        image_width_px=width,
        image_height_px=height,
        base_projection_type=SKY_MATCHING_MODEL_RECTILINEAR,
        principal_point_x_px=(width - 1.0) * 0.5,
        principal_point_y_px=(height - 1.0) * 0.5,
        scale_x_px=float(width) * 1.15,
        scale_y_px=float(width) * 1.15,
        global_distortion_type="none",
    )
    return FrameAstrometricModel(
        image_width_px=width,
        image_height_px=height,
        frame_pose=FramePose(
            np.asarray(
                (
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0),
                ),
                dtype=np.float64,
            )
        ),
        camera_calibration_profile=profile,
    )


@dataclass(frozen=True)
class _ReferenceStar:
    """用于验证粗略取景参考星刷新流程的最小参考星对象。"""

    star_id: str


class _RoughReferenceRefreshHarness(AlignmentMixin):
    """隔离验证粗略取景刷新列表时的防递归逻辑。"""

    def __init__(self) -> None:
        self._current_reference_stars = (_ReferenceStar("模拟视野星"),)
        self._suspend_alignment_updates = False
        self.selected = (_ReferenceStar("粗略视野星A"), _ReferenceStar("粗略视野星B"))
        self.table_refresh_count = 0
        self.suspension_seen_while_refreshing = False

    def _select_current_reference_stars(self, unused_star_map: object) -> tuple[_ReferenceStar, ...]:
        return self.selected

    def _update_star_pair_table(self, reference_stars: tuple[_ReferenceStar, ...]) -> None:
        self.table_refresh_count += 1
        self.suspension_seen_while_refreshing = self._suspend_alignment_updates
        self._current_reference_stars = reference_stars


def test_rough_framing_refreshes_reference_stars_without_recursive_alignment() -> None:
    """粗略取景成功后，标记与表格应切换到当前估计视野的参考星。"""

    harness = _RoughReferenceRefreshHarness()

    harness._refresh_reference_stars_for_rough_framing(object())

    assert tuple(star.star_id for star in harness._current_reference_stars) == ("粗略视野星A", "粗略视野星B")
    assert harness.table_refresh_count == 1
    assert harness.suspension_seen_while_refreshing
    assert not harness._suspend_alignment_updates


class _StatusBar:
    """记录状态栏文字的轻量测试替身。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:  # noqa: N802 - 保持 Qt 接口名称。
        self.messages.append(message)


class _Ui:
    """仅包含相邻取景工作流所需的 UI 成员。"""

    def __init__(self) -> None:
        self.statusbar = _StatusBar()


class _RoughFramingWorkflowHarness(AlignmentMixin):
    """验证粗略取景和手工匹配的阈值切换规则。"""

    def __init__(self) -> None:
        frame_model = _frame_model(640, 440)
        self.current_image_preview = object()
        self._rough_alignment_transform = RoughFramingTransform(
            frame_model=frame_model,
            pair_count=12,
            rms_px=1.2,
            mode=ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
        )
        self._rough_source_astrometric_model = object()
        self.ui = _Ui()


def test_rough_framing_switches_at_four_manual_pairs_and_returns_after_deletion() -> None:
    """四颗前只用粗略取景，达到阈值后手工求解，删除后应能回退。"""

    harness = _RoughFramingWorkflowHarness()

    assert harness._should_use_rough_framing(0)
    assert harness._should_use_rough_framing(3)
    assert not harness._should_use_rough_framing(4)
    assert harness._should_use_rough_framing(3)

    rough_rotation = harness._rough_alignment_transform.frame_model.frame_pose.icrs_to_camera
    assert np.allclose(harness._manual_projection_initial_rotation_matrix(), rough_rotation)

    harness._show_adjacent_framing_workflow_status(3, using_rough_framing=True)
    harness._show_adjacent_framing_workflow_status(4, using_rough_framing=False)
    assert harness.ui.statusbar.messages[0] == "参考星图同步目前由粗略取景提供"
    assert harness.ui.statusbar.messages[1] == "参考星图同步基于当前匹配求解"


def test_star_correspondences_find_shifted_stars() -> None:
    """星点模式能从相邻小幅平移图像中恢复足够多的对应星。"""

    image_a = _draw_feature_image(640, 440, 20260711)
    matrix = np.float32(((1.0, 0.0, 7.0), (0.0, 1.0, -5.0)))
    image_b = cv2.warpAffine(image_a, matrix, (640, 440), borderValue=0)

    pixels_a, pixels_b, rms_px = _star_correspondences(image_a, image_b, focal_px=700.0)

    assert len(pixels_a) >= 6
    assert len(pixels_a) == len(pixels_b)
    assert rms_px < 4.5


def test_sep_extract_pixstack_overflow_uses_config_and_reports_chinese(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """SEP 缓冲区溢出时应使用配置上限，并只返回中文调参建议。"""

    configured_values: list[int] = []
    monkeypatch.setattr(
        "meteoalign.adjacent_alignment.sep.set_extract_pixstack",
        configured_values.append,
    )

    def raise_pixstack_error(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise Exception(
            "internal pixel buffer full: The limit of 300000 active object pixels over the detection threshold was reached. "
            "If you need to increase the limit, use set_extract_pixstack."
        )

    monkeypatch.setattr("meteoalign.adjacent_alignment.sep.extract", raise_pixstack_error)

    try:
        _detect_stars(
            np.zeros((64, 64), dtype=np.uint8),
            AdjacentStarAlignmentConfig(extract_pixstack=600000),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("SEP 缓冲区溢出必须转换为中文错误")

    assert configured_values == [600000]
    assert "internal pixel buffer full" not in message
    assert "建议调整超参数" in message
    assert "提高检测阈值（背景 RMS）" in message
    assert "把背景网格宽高从 128 调到 64 甚至 32" in message


def test_landscape_mode_builds_current_frame_rough_model(tmp_path) -> None:
    """地景模式能通过 A 的 model.json 建立 B 的可投影粗略 FrameAstrometricModel。"""

    width, height = 640, 440
    image_a = _draw_feature_image(width, height, 20260712)
    matrix = np.float32(((1.0, 0.0, 8.0), (0.0, 1.0, -6.0)))
    image_b = cv2.warpAffine(image_a, matrix, (width, height), borderValue=0)
    image_a_path = tmp_path / "adjacent_a.png"
    image_b_path = tmp_path / "current_b.png"
    assert cv2.imwrite(str(image_a_path), image_a)
    assert cv2.imwrite(str(image_b_path), image_b)

    model = _frame_model(width, height)
    model_path = tmp_path / "adjacent_a.model.json"
    payload = model.to_json_payload(
        source_image={
            "path": str(image_a_path),
            "relative_path": image_a_path.name,
            "file_name": image_a_path.name,
        }
    )
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    result = calculate_adjacent_rough_framing(
        model_path,
        image_b_path,
        ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
        settings=AdjacentAlignmentConfig(),
    )

    expected_radec = model.pixel_to_sky_points(np.asarray(((320.0, 220.0),), dtype=np.float64))
    recovered_radec = result.source_model.pixel_to_radec_points(
        np.asarray(((328.0, 214.0),), dtype=np.float64)
    )
    assert result.correspondence_count >= 4
    assert np.isfinite(result.transform.rms_px)
    assert np.all(np.isfinite(recovered_radec))
    assert np.max(np.abs(expected_radec - recovered_radec)) < 0.1
