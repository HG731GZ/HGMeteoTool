"""模块归档后的新旧导入路径兼容性测试。"""

from __future__ import annotations

from meteoalign.mosaic.framing import MosaicResolutionEstimate as ArchivedResolutionEstimate
from meteoalign.mosaic.grid_service import build_coverage_cache as archived_build_coverage_cache
from meteoalign.mosaic.model_io import MosaicCoverageCache as ArchivedCoverageCache
from meteoalign.mosaic.overlay_renderer import coverage_altaz as archived_coverage_altaz
from meteoalign.mosaic_framing import MosaicResolutionEstimate
from meteoalign.mosaic_grid_service import build_coverage_cache
from meteoalign.mosaic_model_io import MosaicCoverageCache
from meteoalign.mosaic_overlay_renderer import coverage_altaz
from meteoalign.projection.grid import build_pixel_grid as archived_build_pixel_grid
from meteoalign.projection.view_state import ProjectionViewState as ArchivedProjectionViewState
from meteoalign.projection_grid import build_pixel_grid
from meteoalign.projection_view_state import ProjectionViewState


def test_archived_modules_keep_legacy_imports_as_compatibility_facades() -> None:
    """归档后的旧模块路径应继续指向新模块中的同一实现。"""

    assert MosaicResolutionEstimate is ArchivedResolutionEstimate
    assert MosaicCoverageCache is ArchivedCoverageCache
    assert ProjectionViewState is ArchivedProjectionViewState
    assert build_coverage_cache is archived_build_coverage_cache
    assert coverage_altaz is archived_coverage_altaz
    assert build_pixel_grid is archived_build_pixel_grid
