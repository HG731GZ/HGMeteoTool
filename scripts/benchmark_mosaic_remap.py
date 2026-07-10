"""测量自由投影拼图 map 构建耗时。

示例：
    conda run -n hgastro python scripts/benchmark_mosaic_remap.py \
        testimages/14mm/A7M3_1214_DSC04149_model.json
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from time import perf_counter

# 允许按文档示例直接执行脚本，而不要求额外设置 PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meteoalign.mosaic.export.geometry import MosaicExportGeometry
from meteoalign.mosaic.export.remap_builder import build_reprojection_map
from meteoalign.mosaic_model_io import _load_mosaic_source_model
from meteoalign.simulator import CameraSettings, RECTILINEAR_LENS_MODEL, ViewSettings


def parse_arguments() -> argparse.Namespace:
    """读取 benchmark 的模型、画布和取景参数。"""

    parser = argparse.ArgumentParser(description="测量自由投影拼图 map 构建耗时")
    parser.add_argument("model_json", type=Path, help="已解算源图的 model.json 路径")
    parser.add_argument("--width", type=int, default=1024, help="目标画布宽度")
    parser.add_argument("--height", type=int, default=768, help="目标画布高度")
    parser.add_argument("--fov", type=float, default=100.0, help="矩形投影水平视场角")
    parser.add_argument("--az", type=float, default=180.0, help="取景中心方位角")
    parser.add_argument("--alt", type=float, default=45.0, help="取景中心高度角")
    parser.add_argument("--roll", type=float, default=0.0, help="取景滚转角")
    parser.add_argument("--block-rows", type=int, default=128, help="每个 map 分块的行数")
    return parser.parse_args()


def main() -> None:
    """构建一次 map 并输出耗时、有效像素数和内存占用。"""

    args = parse_arguments()
    source = _load_mosaic_source_model(args.model_json.expanduser().resolve())
    width = max(1, int(args.width))
    height = max(1, int(args.height))
    fov_deg = max(1.0, min(160.0, float(args.fov)))
    sensor_width_mm = 36.0
    focal_length_mm = sensor_width_mm / (2.0 * math.tan(math.radians(fov_deg * 0.5)))
    camera = CameraSettings(
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_width_mm * height / width,
        image_width_px=width,
        image_height_px=height,
        focal_length_mm=focal_length_mm,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=fov_deg,
    )
    geometry = MosaicExportGeometry(width, height, 0, 0, width, height)
    started_at = perf_counter()
    reprojection_map = build_reprojection_map(
        source_model=source.model,
        camera=camera,
        view=ViewSettings(center_az_deg=args.az, center_alt_deg=args.alt, roll_deg=args.roll),
        observer=source.observer,
        geometry=geometry,
        block_rows=max(1, int(args.block_rows)),
    )
    elapsed_seconds = perf_counter() - started_at
    map_bytes = reprojection_map.map_x.nbytes + reprojection_map.map_y.nbytes + reprojection_map.valid_mask.nbytes
    print(f"模型: {source.json_path}")
    print(f"目标尺寸: {width} x {height}")
    print(f"耗时: {elapsed_seconds:.3f} s")
    print(f"有效像素: {int(reprojection_map.valid_mask.sum())}/{reprojection_map.valid_mask.size}")
    print(f"map 内存: {map_bytes / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
