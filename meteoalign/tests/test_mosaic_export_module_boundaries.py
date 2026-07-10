"""拼图导出 geometry 与 target transform 模块的兼容性测试。"""

from __future__ import annotations

from meteoalign.mosaic.export.geometry import (
    MosaicExportGeometry as GeometryModuleGeometry,
    mosaic_export_cropped_geometry as geometry_module_cropped_geometry,
)
from meteoalign.mosaic.export.target_transform import (
    build_target_icrs_to_pixel_transform_payload as target_module_build_payload,
    target_icrs_to_pixel_transform_payload_matches as target_module_payload_matches,
)
from meteoalign.mosaic_export import (
    MosaicExportGeometry,
    build_target_icrs_to_pixel_transform_payload,
    mosaic_export_cropped_geometry,
    target_icrs_to_pixel_transform_payload_matches,
)


def test_mosaic_export_facade_reexports_geometry_module() -> None:
    """旧导出入口与新 geometry 模块必须使用同一套类型和函数。"""

    geometry = geometry_module_cropped_geometry(
        boundary_width_px=100,
        boundary_height_px=80,
        crop={"left_px": 10, "right_px": 15, "top_px": 5, "bottom_px": 20},
    )

    assert MosaicExportGeometry is GeometryModuleGeometry
    assert mosaic_export_cropped_geometry is geometry_module_cropped_geometry
    assert geometry == MosaicExportGeometry(100, 80, 10, 5, 75, 55)


def test_mosaic_export_facade_reexports_target_transform_module() -> None:
    """旧导出入口与新目标变换模块必须保持同一函数对象。"""

    assert build_target_icrs_to_pixel_transform_payload is target_module_build_payload
    assert target_icrs_to_pixel_transform_payload_matches is target_module_payload_matches
