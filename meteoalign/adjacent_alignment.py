"""相邻图像配准与粗略取景计算。

本模块复用了 StarAlign 中的 SEP 星点检测、Astroalign/球面三角形初配准、
OpenCV SIFT + RANSAC 地景配准思路。计算结果只用于快速建立当前图像的
初始 Pixel↔ICRS 映射，后续仍可用人工或自动场星匹配进一步精修。
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import astroalign
import cv2
import numpy as np
import sep
from scipy.spatial import KDTree

from .config import (
    AdjacentAlignmentConfig,
    AdjacentLandscapeAlignmentConfig,
    AdjacentStarAlignmentConfig,
    load_adjacent_alignment_config,
)
from .frame_astrometry import FrameAstrometricModel
from .image_path_resolution import associated_image_candidates, expected_image_size, first_matching_image_path
from .source_model import FixedProfilePoseSourceModel, fit_source_astrometric_model_with_fixed_profile


ADJACENT_ALIGNMENT_MODE_STARS = "stars"
ADJACENT_ALIGNMENT_MODE_LANDSCAPE = "landscape"
ADJACENT_ALIGNMENT_MODES = (
    ADJACENT_ALIGNMENT_MODE_STARS,
    ADJACENT_ALIGNMENT_MODE_LANDSCAPE,
)
def adjacent_alignment_mode_display_name(mode: str) -> str:
    """返回与界面下拉列表一致的工作模式名称。"""

    if mode == ADJACENT_ALIGNMENT_MODE_STARS:
        return "星点对齐模式"
    if mode == ADJACENT_ALIGNMENT_MODE_LANDSCAPE:
        return "地景对齐模式"
    return "未知模式"


@dataclass(frozen=True)
class StarCatalog:
    """星点检测结果，仅保留位置与亮度排序依据。"""

    points: np.ndarray
    fluxes: np.ndarray


@dataclass(frozen=True)
class PixelMatch:
    """一对经过几何验证的相邻图像像素坐标。"""

    a_index: int
    b_index: int
    distance_px: float


class PointTransform(Protocol):
    """定义将相邻图像 A 像素预测到当前图像 B 的接口。"""

    def predict(self, points: np.ndarray) -> np.ndarray:
        """返回 B 图中的预测像素位置。"""


@dataclass(frozen=True)
class SimilarityTransform:
    """封装 Astroalign 计算出的平面相似变换。"""

    transform: object

    def predict(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(self.transform(points), dtype=np.float64)


@dataclass(frozen=True)
class SphericalRotationTransform:
    """用近似针孔相机与三维旋转描述星空视向变化。"""

    rotation: np.ndarray
    image_shape: tuple[int, int]
    focal_px: float

    def predict(self, points: np.ndarray) -> np.ndarray:
        directions = _pixels_to_directions(points, self.image_shape, self.focal_px)
        rotated = directions @ self.rotation.T
        height, width = self.image_shape
        center_x = (width - 1.0) * 0.5
        center_y = (height - 1.0) * 0.5
        valid = rotated[:, 2] > 1e-8
        predicted = np.full((len(points), 2), np.nan, dtype=np.float64)
        predicted[valid, 0] = center_x + self.focal_px * rotated[valid, 0] / rotated[valid, 2]
        predicted[valid, 1] = center_y + self.focal_px * rotated[valid, 1] / rotated[valid, 2]
        return predicted


@dataclass(frozen=True)
class HomographyTransform:
    """封装 OpenCV RANSAC 估计的 A→B 单应性。"""

    matrix: np.ndarray

    def predict(self, points: np.ndarray) -> np.ndarray:
        transformed = cv2.perspectiveTransform(np.asarray(points, dtype=np.float64)[None, :, :], self.matrix)
        return transformed[0]


@dataclass(frozen=True)
class RoughFramingTransform:
    """把粗略 FrameAstrometricModel 适配为实时参考星图使用的变换接口。"""

    frame_model: FrameAstrometricModel
    pair_count: int
    rms_px: float
    mode: str

    @property
    def display_name(self) -> str:
        return f"相邻图像{adjacent_alignment_mode_display_name(self.mode)}"

    @property
    def lens_model(self) -> str:
        return str(self.frame_model.camera_calibration_profile.base_projection_type)

    def transform_radec_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        return self.frame_model.sky_to_pixel_points(ra_dec_points)

    def transform_radec(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        return self.frame_model.sky_to_pixel(ra_deg, dec_deg)


@dataclass(frozen=True)
class AdjacentFramingResult:
    """一次粗略取景计算的可显示结果。"""

    model_json_path: Path
    image_a_path: Path
    image_b_path: Path
    mode: str
    correspondence_count: int
    correspondence_rms_px: float
    source_model: FixedProfilePoseSourceModel
    transform: RoughFramingTransform


def resolve_model_source_image_path(payload: object, model_json_path: str | Path) -> Path:
    """从导出 model.json 的 source_image 字段恢复相邻图像 A 的路径。"""

    json_path = Path(model_json_path).expanduser().resolve()
    if not isinstance(payload, dict):
        raise ValueError("相邻图像模型 JSON 根对象必须是对象。")
    source_image = payload.get("source_image")
    if not isinstance(source_image, dict):
        raise ValueError("相邻图像模型 JSON 缺少 source_image，无法定位相邻原图。")

    candidates = associated_image_candidates(source_image, json_path)
    expected_size = expected_image_size(source_image)
    if expected_size is None:
        image_geometry = payload.get("image_geometry")
        if isinstance(image_geometry, dict):
            try:
                width = int(image_geometry.get("width_px", 0))
                height = int(image_geometry.get("height_px", 0))
            except (TypeError, ValueError):
                width, height = 0, 0
            if width > 0 and height > 0:
                expected_size = (width, height)
    candidate = first_matching_image_path(candidates, expected_size)
    if candidate is not None:
        return candidate
    if not candidates:
        raise ValueError("相邻图像模型 JSON 的 source_image 未提供图像路径。")
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到尺寸匹配的相邻图像 A，已尝试：\n{searched}")


def load_adjacent_frame_model(model_json_path: str | Path) -> tuple[FrameAstrometricModel, Path]:
    """读取并验证由软件导出的相邻图像 FrameAstrometricModel。"""

    json_path = Path(model_json_path).expanduser().resolve()
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"无法读取相邻图像模型 JSON：{json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"相邻图像模型 JSON 格式无效：{exc}") from exc
    frame_model = FrameAstrometricModel.from_json_payload(payload)
    image_path = resolve_model_source_image_path(payload, json_path)
    return frame_model, image_path


def _load_image(path: Path) -> np.ndarray:
    """读取图像为 OpenCV 数组，并为 TIFF 读取失败提供 tifffile 兜底。"""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None and path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            image = tifffile.imread(path)
        except Exception as exc:  # noqa: BLE001 - 统一转换为可读的图像错误。
            raise ValueError(f"无法读取图像：{path}") from exc
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    if image.ndim not in (2, 3):
        raise ValueError(f"图像通道数不受支持：{path}")
    return np.ascontiguousarray(image)


def _luminance(image: np.ndarray) -> np.ndarray:
    """把 8/16 位单通道或彩色图像转为连续亮度数组。"""

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(gray)


def _to_feature_image(image: np.ndarray, settings: AdjacentLandscapeAlignmentConfig) -> np.ndarray:
    """把高位深夜景压缩到增强局部对比度的 8 位灰度图。"""

    gray = _luminance(image)
    if gray.dtype != np.uint8:
        low, high = np.percentile(
            gray,
            (settings.normalization_low_percentile, settings.normalization_high_percentile),
        )
        high = max(float(high), float(low) + 1.0)
        gray = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=settings.clahe_clip_limit,
        tileGridSize=(settings.clahe_grid_size, settings.clahe_grid_size),
    )
    return clahe.apply(np.ascontiguousarray(gray))


def _detect_stars(image: np.ndarray, settings: AdjacentStarAlignmentConfig) -> StarCatalog:
    """使用 SEP 背景估计提取并筛选可参与相邻图匹配的恒星。"""

    working = np.ascontiguousarray(_luminance(image).astype(np.float32))
    background = sep.Background(
        working,
        bw=settings.background_bw_px,
        bh=settings.background_bh_px,
        fw=settings.background_fw_px,
        fh=settings.background_fh_px,
    )
    objects = sep.extract(
        working - background.back(),
        settings.detection_sigma * background.globalrms,
        minarea=settings.detection_min_area_px,
        deblend_nthresh=settings.deblend_nthresh,
        deblend_cont=settings.deblend_cont,
    )
    if len(objects) == 0:
        raise RuntimeError("未检测到可用于配准的星点。")
    height, width = working.shape
    x = objects["x"].astype(np.float64)
    y = objects["y"].astype(np.float64)
    major = objects["a"].astype(np.float64)
    minor = objects["b"].astype(np.float64)
    flux = objects["flux"].astype(np.float64)
    axis_ratio = major / np.maximum(minor, 1e-6)
    edge = settings.detection_edge_margin_px
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(flux)
        & (flux > 0.0)
        & (x >= edge)
        & (x < width - edge)
        & (y >= edge)
        & (y < height - edge)
        & (major >= settings.min_major_axis_px)
        & (major <= settings.max_major_axis_px)
        & (minor >= settings.min_minor_axis_px)
        & (axis_ratio <= settings.max_axis_ratio)
    )
    points = np.column_stack((x[valid], y[valid]))
    fluxes = flux[valid]
    if len(points) < 3:
        raise RuntimeError("过滤后可用于配准的星点少于三个。")
    order = np.argsort(fluxes)[::-1][:settings.max_detected_stars]
    return StarCatalog(points=points[order], fluxes=fluxes[order])


def _pixels_to_directions(points: np.ndarray, image_shape: tuple[int, int], focal_px: float) -> np.ndarray:
    """将像素近似反投影为相机坐标系的单位视向。"""

    height, width = image_shape
    center_x = (width - 1.0) * 0.5
    center_y = (height - 1.0) * 0.5
    directions = np.column_stack(
        (
            (points[:, 0] - center_x) / focal_px,
            (points[:, 1] - center_y) / focal_px,
            np.ones(len(points), dtype=np.float64),
        )
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _angular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dot_product = np.sum(left * right, axis=-1)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def _triangle_invariants(
    vectors: np.ndarray,
    minimum_side_deg: float,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """构造对三维相机旋转不敏感的恒星三角形边长描述子。"""

    descriptors: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    minimum_side = math.radians(minimum_side_deg)
    for indices in itertools.combinations(range(len(vectors)), 3):
        triangle = vectors[np.asarray(indices)]
        side_lengths = np.asarray(
            (
                _angular_distance(triangle[0:1], triangle[1:2])[0],
                _angular_distance(triangle[0:1], triangle[2:3])[0],
                _angular_distance(triangle[1:2], triangle[2:3])[0],
            ),
            dtype=np.float64,
        )
        if np.min(side_lengths) < minimum_side:
            continue
        descriptors.append(np.sort(side_lengths))
        triangles.append(indices)
    if not descriptors:
        raise RuntimeError("可用于球面星点匹配的三角形不足。")
    return np.asarray(descriptors), triangles


def _rotation_from_pairs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """使用 Kabsch 算法拟合从 A 相机视向到 B 相机视向的旋转。"""

    covariance = source.T @ target
    left, _values, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T
    return rotation


def _best_triangle_permutation(source: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    """枚举三角形顶点对应，选择边长误差最小的排列。"""

    source_lengths = np.asarray(
        (
            _angular_distance(source[0:1], source[1:2])[0],
            _angular_distance(source[0:1], source[2:3])[0],
            _angular_distance(source[1:2], source[2:3])[0],
        )
    )
    best_error = math.inf
    best_order = (0, 1, 2)
    for order in itertools.permutations(range(3)):
        candidate = target[np.asarray(order)]
        target_lengths = np.asarray(
            (
                _angular_distance(candidate[0:1], candidate[1:2])[0],
                _angular_distance(candidate[0:1], candidate[2:3])[0],
                _angular_distance(candidate[1:2], candidate[2:3])[0],
            )
        )
        error = float(np.sum((source_lengths - target_lengths) ** 2))
        if error < best_error:
            best_error = error
            best_order = order
    return best_order


def _estimate_spherical_rotation(
    source: StarCatalog,
    target: StarCatalog,
    image_shape: tuple[int, int],
    focal_px: float,
    settings: AdjacentStarAlignmentConfig,
) -> SphericalRotationTransform | None:
    """以球面三角形 RANSAC 为较大视向差的星图提供初始旋转。"""

    source_count = min(settings.max_alignment_stars, len(source.points))
    target_count = min(settings.max_alignment_stars, len(target.points))
    source_vectors = _pixels_to_directions(source.points[:source_count], image_shape, focal_px)
    target_vectors = _pixels_to_directions(target.points[:target_count], image_shape, focal_px)
    source_descriptors, source_triangles = _triangle_invariants(
        source_vectors,
        settings.min_triangle_side_deg,
    )
    target_descriptors, target_triangles = _triangle_invariants(
        target_vectors,
        settings.min_triangle_side_deg,
    )
    descriptor_tree = KDTree(target_descriptors)
    tolerance = math.radians(settings.triangle_match_tolerance_deg) * math.sqrt(3.0)
    candidate_pairs: list[tuple[int, int]] = []
    for source_index, descriptor in enumerate(source_descriptors):
        for target_index in descriptor_tree.query_ball_point(descriptor, tolerance):
            candidate_pairs.append((source_index, int(target_index)))
    if not candidate_pairs:
        return None

    generator = np.random.default_rng(20260711)
    if len(candidate_pairs) > settings.max_triangle_hypotheses:
        selected = generator.choice(len(candidate_pairs), settings.max_triangle_hypotheses, replace=False)
        candidate_pairs = [candidate_pairs[int(index)] for index in selected]
    else:
        generator.shuffle(candidate_pairs)

    target_tree = KDTree(target_vectors)
    chord_tolerance = 2.0 * math.sin(math.radians(settings.rotation_inlier_tolerance_deg) * 0.5)
    best_rotation: np.ndarray | None = None
    best_score = -1
    best_error = math.inf
    for source_triangle_index, target_triangle_index in candidate_pairs:
        source_indices = source_triangles[source_triangle_index]
        target_indices = target_triangles[target_triangle_index]
        source_triangle = source_vectors[np.asarray(source_indices)]
        target_triangle = target_vectors[np.asarray(target_indices)]
        permutation = _best_triangle_permutation(source_triangle, target_triangle)
        rotation = _rotation_from_pairs(source_triangle, target_triangle[np.asarray(permutation)])
        predicted = source_vectors @ rotation.T
        distances, _indices = target_tree.query(predicted)
        inliers = distances <= chord_tolerance
        score = int(np.count_nonzero(inliers))
        error = float(np.mean(distances[inliers])) if score else math.inf
        if score > best_score or (score == best_score and error < best_error):
            best_rotation = rotation
            best_score = score
            best_error = error
    if best_rotation is None or best_score < settings.min_initial_rotation_inliers:
        return None

    predicted = source_vectors @ best_rotation.T
    distances, indices = target_tree.query(predicted)
    inliers = distances <= chord_tolerance
    if np.count_nonzero(inliers) >= 3:
        best_rotation = _rotation_from_pairs(source_vectors[inliers], target_vectors[indices[inliers]])
    return SphericalRotationTransform(best_rotation, image_shape, focal_px)


def _unique_nearest_matches(
    transform: PointTransform,
    source: StarCatalog,
    target: StarCatalog,
    max_distance_px: float,
) -> list[PixelMatch]:
    """按预测位置进行一对一最近邻匹配，避免多颗星抢占同一目标。"""

    predicted = transform.predict(source.points)
    finite = np.all(np.isfinite(predicted), axis=1)
    target_tree = KDTree(target.points)
    distances = np.full(len(source.points), np.inf, dtype=np.float64)
    indices = np.full(len(source.points), -1, dtype=np.int64)
    if np.any(finite):
        query_distances, query_indices = target_tree.query(
            predicted[finite],
            distance_upper_bound=max_distance_px,
        )
        distances[finite] = query_distances
        indices[finite] = query_indices
    candidates = [
        PixelMatch(int(a_index), int(b_index), float(distance))
        for a_index, (b_index, distance) in enumerate(zip(indices, distances))
        if b_index < len(target.points) and distance <= max_distance_px
    ]
    candidates.sort(key=lambda match: match.distance_px)
    claimed_targets: set[int] = set()
    unique: list[PixelMatch] = []
    for match in candidates:
        if match.b_index not in claimed_targets:
            unique.append(match)
            claimed_targets.add(match.b_index)
    return unique


def _fit_homography_from_star_matches(
    source: StarCatalog,
    target: StarCatalog,
    matches: list[PixelMatch],
    settings: AdjacentStarAlignmentConfig,
) -> HomographyTransform | None:
    """用星点候选匹配拟合 A→B 单应性，并仅保留 RANSAC 内点。"""

    if len(matches) < 4:
        return None
    a_points = np.asarray([source.points[match.a_index] for match in matches], dtype=np.float64)
    b_points = np.asarray([target.points[match.b_index] for match in matches], dtype=np.float64)
    matrix, inlier_mask = cv2.findHomography(
        a_points,
        b_points,
        cv2.RANSAC,
        settings.homography_match_distance_px,
        maxIters=settings.homography_ransac_max_iterations,
        confidence=settings.homography_ransac_confidence,
    )
    if matrix is None or inlier_mask is None or int(inlier_mask.sum()) < 4:
        return None
    return HomographyTransform(matrix.astype(np.float64))


def _star_correspondences(
    image_a: np.ndarray,
    image_b: np.ndarray,
    focal_px: float,
    settings: AdjacentStarAlignmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """复用 StarAlign 的双初值与 RANSAC 精配准得到重合星点。"""

    star_settings = settings or AdjacentStarAlignmentConfig()
    source = _detect_stars(image_a, star_settings)
    target = _detect_stars(image_b, star_settings)
    image_shape = _luminance(image_a).shape
    candidates: list[PointTransform] = []
    count = min(star_settings.max_alignment_stars, len(source.points), len(target.points))
    try:
        transform, _footprint = astroalign.find_transform(
            source.points[:count],
            target.points[:count],
            max_control_points=count,
        )
        candidates.append(SimilarityTransform(transform))
    except (astroalign.MaxIterError, ValueError, TypeError, RuntimeError):
        pass
    try:
        spherical = _estimate_spherical_rotation(source, target, image_shape, focal_px, star_settings)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        spherical = None
    if spherical is not None:
        candidates.append(spherical)
    if not candidates:
        raise RuntimeError("Astroalign 和球面三角形星点匹配均未得到初始变换。")

    best_matches: list[PixelMatch] = []
    for candidate in candidates:
        initial_matches = _unique_nearest_matches(
            candidate,
            source,
            target,
            star_settings.initial_match_distance_px,
        )
        homography = _fit_homography_from_star_matches(source, target, initial_matches, star_settings)
        if homography is None:
            continue
        matches = _unique_nearest_matches(
            homography,
            source,
            target,
            star_settings.final_match_distance_px,
        )
        if len(matches) > len(best_matches):
            best_matches = matches
    if len(best_matches) < star_settings.min_match_count:
        raise RuntimeError(
            f"仅找到 {len(best_matches)} 对可靠重合星点，少于要求的 {star_settings.min_match_count} 对。"
        )
    a_points = np.asarray([source.points[match.a_index] for match in best_matches], dtype=np.float64)
    b_points = np.asarray([target.points[match.b_index] for match in best_matches], dtype=np.float64)
    distances = np.asarray([match.distance_px for match in best_matches], dtype=np.float64)
    return a_points, b_points, float(np.sqrt(np.mean(distances * distances)))


def _landscape_correspondences(
    image_a: np.ndarray,
    image_b: np.ndarray,
    settings: AdjacentLandscapeAlignmentConfig | None = None,
    *,
    max_correspondences: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """复用 StarAlign 的 SIFT、FLANN、ratio test 与 RANSAC 地景配准流程。"""

    landscape_settings = settings or AdjacentLandscapeAlignmentConfig()
    correspondence_limit = max_correspondences or AdjacentAlignmentConfig().max_correspondences
    feature_a = _to_feature_image(image_a, landscape_settings)
    feature_b = _to_feature_image(image_b, landscape_settings)
    detector = cv2.SIFT_create(
        nfeatures=landscape_settings.sift_max_features,
        contrastThreshold=landscape_settings.sift_contrast_threshold,
        edgeThreshold=landscape_settings.sift_edge_threshold,
        sigma=landscape_settings.sift_sigma,
    )
    keypoints_a, descriptors_a = detector.detectAndCompute(feature_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(feature_b, None)
    if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
        raise RuntimeError("SIFT 未检测到可用地景特征。")
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": landscape_settings.flann_trees},
        {"checks": landscape_settings.flann_checks},
    )
    nearest_pairs = matcher.knnMatch(
        np.ascontiguousarray(descriptors_a.astype(np.float32)),
        np.ascontiguousarray(descriptors_b.astype(np.float32)),
        k=2,
    )
    ratio_matches = [
        first
        for pair in nearest_pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < landscape_settings.ratio_test_threshold * second.distance
    ]
    ratio_matches.sort(key=lambda match: match.distance)
    unique_matches = []
    used_targets: set[int] = set()
    for match in ratio_matches:
        if match.trainIdx not in used_targets:
            unique_matches.append(match)
            used_targets.add(match.trainIdx)
    if len(unique_matches) < landscape_settings.min_inlier_matches:
        raise RuntimeError(f"ratio test 后仅有 {len(unique_matches)} 个地景匹配，无法计算配准。")
    a_points = np.float64([keypoints_a[match.queryIdx].pt for match in unique_matches])
    b_points = np.float64([keypoints_b[match.trainIdx].pt for match in unique_matches])
    homography, inlier_mask = cv2.findHomography(
        a_points,
        b_points,
        cv2.RANSAC,
        landscape_settings.ransac_reprojection_threshold_px,
        maxIters=landscape_settings.ransac_max_iterations,
        confidence=landscape_settings.ransac_confidence,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("RANSAC 未能估计有效的地景单应性矩阵。")
    predicted = cv2.perspectiveTransform(a_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    inliers = inlier_mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(inliers)) < landscape_settings.min_inlier_matches:
        raise RuntimeError(
            f"RANSAC 保留的地景内点少于 {landscape_settings.min_inlier_matches} 对。"
        )
    errors = np.linalg.norm(predicted[inliers] - b_points[inliers], axis=1)
    order = np.argsort(errors)[:correspondence_limit]
    return a_points[inliers][order], b_points[inliers][order], float(np.sqrt(np.mean(errors[order] ** 2)))


def _star_focal_length_px(
    frame_model: FrameAstrometricModel,
    settings: AdjacentStarAlignmentConfig,
) -> float:
    """从已导出 Profile 取得可用于球面初配准的保守焦距近似。"""

    profile = frame_model.camera_calibration_profile
    fallback = max(frame_model.image_width_px, frame_model.image_height_px)
    candidate = float(profile.scale_x_px)
    if not math.isfinite(candidate) or candidate < fallback * settings.focal_scale_min_ratio:
        return float(fallback)
    return candidate


def _select_correspondences(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mode: str,
    frame_model: FrameAstrometricModel,
    settings: AdjacentAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """根据用户选定工作模式计算 PixelA↔PixelB 关系。"""

    if mode == ADJACENT_ALIGNMENT_MODE_STARS:
        return _star_correspondences(
            image_a,
            image_b,
            _star_focal_length_px(frame_model, settings.stars),
            settings.stars,
        )
    if mode == ADJACENT_ALIGNMENT_MODE_LANDSCAPE:
        return _landscape_correspondences(
            image_a,
            image_b,
            settings.landscape,
            max_correspondences=settings.max_correspondences,
        )
    raise ValueError(f"不支持的相邻图像对齐模式：{mode}")


def calculate_adjacent_rough_framing(
    model_json_path: str | Path,
    image_b_path: str | Path,
    mode: str,
    settings: AdjacentAlignmentConfig | None = None,
) -> AdjacentFramingResult:
    """由 A 的 model.json 与 A↔B 像素关系快速建立 B 的 Pixel↔ICRS 取景。"""

    if mode not in ADJACENT_ALIGNMENT_MODES:
        raise ValueError(f"不支持的相邻图像对齐模式：{mode}")
    active_settings = settings or load_adjacent_alignment_config()
    json_path = Path(model_json_path).expanduser().resolve()
    b_path = Path(image_b_path).expanduser().resolve()
    frame_model_a, image_a_path = load_adjacent_frame_model(json_path)
    image_a = _load_image(image_a_path)
    image_b = _load_image(b_path)
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    if (width_a, height_a) != (frame_model_a.image_width_px, frame_model_a.image_height_px):
        raise ValueError(
            "相邻图像 A 尺寸与 model.json 不一致："
            f"图像 {width_a} x {height_a} px，模型 {frame_model_a.image_width_px} x {frame_model_a.image_height_px} px。"
        )
    if (width_b, height_b) != (frame_model_a.image_width_px, frame_model_a.image_height_px):
        raise ValueError(
            "当前图像 B 的尺寸必须与相邻图像 model.json 的标定尺寸一致："
            f"当前 {width_b} x {height_b} px，模型 {frame_model_a.image_width_px} x {frame_model_a.image_height_px} px。"
        )

    pixels_a, pixels_b, correspondence_rms = _select_correspondences(
        image_a,
        image_b,
        mode,
        frame_model_a,
        active_settings,
    )
    pixels_a = pixels_a[:active_settings.max_correspondences]
    pixels_b = pixels_b[:active_settings.max_correspondences]
    radec = frame_model_a.pixel_to_sky_points(pixels_a)
    finite = np.all(np.isfinite(radec), axis=1) & np.all(np.isfinite(pixels_b), axis=1)
    radec = radec[finite]
    pixels_b = pixels_b[finite]
    if len(radec) < 4:
        raise RuntimeError("有效 PixelA→ICRS 对应少于 4 对，无法估计当前图像的粗略取景。")

    source_model = fit_source_astrometric_model_with_fixed_profile(
        ra_dec_points=radec,
        pixel_points=pixels_b,
        image_size=(width_b, height_b),
        camera_calibration_profile=frame_model_a.camera_calibration_profile,
        initial_rotation_matrix=frame_model_a.frame_pose.icrs_to_camera,
        profile_source_path=str(json_path),
        solve_mode="adjacent_image_coarse_pose",
    )
    frame_model_b = source_model.to_frame_astrometric_model()
    transform = RoughFramingTransform(
        frame_model=frame_model_b,
        pair_count=int(len(radec)),
        rms_px=float(source_model.rms_px),
        mode=mode,
    )
    return AdjacentFramingResult(
        model_json_path=json_path,
        image_a_path=image_a_path,
        image_b_path=b_path,
        mode=mode,
        correspondence_count=int(len(radec)),
        correspondence_rms_px=float(correspondence_rms),
        source_model=source_model,
        transform=transform,
    )
