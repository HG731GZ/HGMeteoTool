from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import tifffile

from meteoalign.mosaic_export import (
    MosaicExportGeometry,
    MosaicExportSourceImage,
    mosaic_export_cropped_geometry,
    write_mosaic_reprojection_tiff,
)
from meteoalign.simulator import (
    CameraSettings,
    ObserverSettings,
    RECTILINEAR_LENS_MODEL,
    ViewSettings,
    _project_altaz_points,
    camera_basis_from_view,
    compute_altaz_from_radec,
)


class _TargetProjectionSourceModel:
    def __init__(self, camera: CameraSettings, view: ViewSettings, observer: ObserverSettings) -> None:
        self.camera = camera
        self.view = view
        self.observer = observer
        self.basis = camera_basis_from_view(view)

    def sky_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        radec = np.asarray(ra_dec_points, dtype=np.float64)
        alt_deg, az_deg = compute_altaz_from_radec(radec[:, 0], radec[:, 1], self.observer)
        x_px, y_px, valid = _project_altaz_points(alt_deg, az_deg, camera=self.camera, basis=self.basis)
        pixels = np.column_stack((x_px, y_px)).astype(np.float64)
        pixels[~valid] = np.nan
        return pixels


def test_mosaic_export_cropped_geometry_subtracts_all_margins() -> None:
    geometry = mosaic_export_cropped_geometry(
        boundary_width_px=6000,
        boundary_height_px=4000,
        crop={
            "left_px": 100,
            "right_px": 120,
            "top_px": 80,
            "bottom_px": 90,
        },
    )

    assert geometry.crop_left_px == 100
    assert geometry.crop_top_px == 80
    assert geometry.output_width_px == 5780
    assert geometry.output_height_px == 3830


def test_mosaic_export_writes_cropped_uncompressed_u16_tiff_with_exif(tmp_path) -> None:  # type: ignore[no-untyped-def]
    width, height = 8, 6
    source_rgb = np.zeros((height, width, 3), dtype=np.uint16)
    for y_value in range(height):
        for x_value in range(width):
            source_rgb[y_value, x_value] = (x_value * 1000, y_value * 1000, 50000)

    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=200.0,
    )
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=27.0,
        image_width_px=width,
        image_height_px=height,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=90.0,
    )
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=45.0, roll_deg=0.0)
    source_model = _TargetProjectionSourceModel(camera, view, observer)
    source_image = MosaicExportSourceImage(
        path=tmp_path / "source.tif",
        rgb_u16=source_rgb,
        exif_tags=((271, "s", 0, "UnitTestCamera", True),),
        icc_profile=None,
    )
    geometry = MosaicExportGeometry(
        boundary_width_px=width,
        boundary_height_px=height,
        crop_left_px=1,
        crop_top_px=1,
        output_width_px=6,
        output_height_px=4,
    )
    output_path = tmp_path / "export.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=source_model,
        source_image=source_image,
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=2,
    )

    with tifffile.TiffFile(output_path) as tiff:
        page = tiff.pages[0]
        exported = page.asarray()
        compression = page.tags["Compression"].value
        make = page.tags[271].value

    assert exported.shape == (4, 6, 3)
    assert exported.dtype == np.uint16
    assert compression == 1
    assert make == "UnitTestCamera"
    assert np.max(np.abs(exported[0, 0].astype(np.int32) - source_rgb[1, 1].astype(np.int32))) < 8
