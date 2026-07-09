from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import tifffile

from meteoalign.coordinates import unit_vectors_to_radec
from meteoalign.mosaic_export import (
    MosaicExportGeometry,
    MosaicExportSourceImage,
    _finalize_forward_inverse_map,
    _icrs_camera_basis_from_view,
    build_target_icrs_to_pixel_transform_payload,
    mosaic_export_cropped_geometry,
    mosaic_reprojection_map_blocks,
    target_image_points_to_icrs_vectors,
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
        self.icrs_basis = _icrs_camera_basis_from_view(view, observer)
        self.image_width_px = camera.image_width_px
        self.image_height_px = camera.image_height_px

    def sky_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        radec = np.asarray(ra_dec_points, dtype=np.float64)
        alt_deg, az_deg = compute_altaz_from_radec(radec[:, 0], radec[:, 1], self.observer)
        x_px, y_px, valid = _project_altaz_points(alt_deg, az_deg, camera=self.camera, basis=self.basis)
        pixels = np.column_stack((x_px, y_px)).astype(np.float64)
        pixels[~valid] = np.nan
        return pixels

    def icrs_vectors_to_pixel_points(self, vectors: np.ndarray) -> np.ndarray:
        radec = unit_vectors_to_radec(vectors)
        return self.sky_to_pixel_points(radec)

    def pixel_to_sky_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        vectors, valid = target_image_points_to_icrs_vectors(
            pixels[:, 0],
            pixels[:, 1],
            camera=self.camera,
            icrs_basis=self.icrs_basis,
        )
        radec = np.full((pixels.shape[0], 2), np.nan, dtype=np.float64)
        if np.any(valid):
            radec[valid] = unit_vectors_to_radec(vectors[valid])
        return radec


class _CountingTargetProjectionSourceModel(_TargetProjectionSourceModel):
    def __init__(self, camera: CameraSettings, view: ViewSettings, observer: ObserverSettings) -> None:
        super().__init__(camera, view, observer)
        self.projected_vector_count = 0
        self.projected_source_pixel_count = 0

    def icrs_vectors_to_pixel_points(self, vectors: np.ndarray) -> np.ndarray:
        self.projected_vector_count += int(np.asarray(vectors).shape[0])
        return super().icrs_vectors_to_pixel_points(vectors)

    def pixel_to_sky_points(self, pixel_points: np.ndarray) -> np.ndarray:
        self.projected_source_pixel_count += int(np.asarray(pixel_points).shape[0])
        return super().pixel_to_sky_points(pixel_points)


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


def test_mosaic_export_map_blocks_bridge_target_pixels_to_source_pixels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    width, height = 8, 6
    source_rgb = np.full((height, width, 3), 1000, dtype=np.uint16)
    source_rgb[2, 2] = (30000, 40000, 50000)
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
    geometry = MosaicExportGeometry(
        boundary_width_px=width,
        boundary_height_px=height,
        crop_left_px=1,
        crop_top_px=1,
        output_width_px=6,
        output_height_px=4,
    )
    map_blocks = list(
        mosaic_reprojection_map_blocks(
            source_model=source_model,
            camera=camera,
            view=view,
            observer=observer,
            geometry=geometry,
            block_rows=2,
        )
    )
    output_path = tmp_path / "mapped_export.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=source_model,
        source_image=MosaicExportSourceImage(
            path=tmp_path / "source.tif",
            rgb_u16=source_rgb,
            exif_tags=(),
            icc_profile=None,
        ),
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=2,
    )

    exported = tifffile.imread(output_path)
    assert len(map_blocks) == 2
    assert abs(float(map_blocks[0]["map_x"][0, 0]) - 1.0) < 1e-3
    assert abs(float(map_blocks[0]["map_y"][0, 0]) - 1.0) < 1e-3
    assert exported.shape == (4, 6, 3)
    assert np.max(np.abs(exported[1, 1].astype(np.int32) - source_rgb[2, 2].astype(np.int32))) < 8


def test_mosaic_export_uses_target_icrs_to_pixel_transform_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    width, height = 8, 6
    source_rgb = np.full((height, width, 3), 1000, dtype=np.uint16)
    source_rgb[2, 2] = (30000, 40000, 50000)
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
    geometry = MosaicExportGeometry(
        boundary_width_px=width,
        boundary_height_px=height,
        crop_left_px=1,
        crop_top_px=1,
        output_width_px=6,
        output_height_px=4,
    )
    target_transform_payload = build_target_icrs_to_pixel_transform_payload(
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
    )
    output_path = tmp_path / "target_transform_export.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=source_model,
        source_image=MosaicExportSourceImage(
            path=tmp_path / "source.tif",
            rgb_u16=source_rgb,
            exif_tags=(),
            icc_profile=None,
        ),
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=2,
        target_icrs_to_pixel_payload=target_transform_payload,
        map_tile_size_px=4,
    )

    exported = tifffile.imread(output_path)
    assert target_transform_payload["type"] == "icrs_to_cropped_output_pixel"
    assert target_transform_payload["output_width_px"] == 6
    assert "blocks" not in target_transform_payload
    assert "camera" in target_transform_payload
    assert "icrs_camera_basis" in target_transform_payload
    assert exported.shape == (4, 6, 3)
    assert np.max(np.abs(exported[1, 1].astype(np.int32) - source_rgb[2, 2].astype(np.int32))) < 8


def test_mosaic_export_fixed_tile_forward_path_projects_fewer_source_points(tmp_path) -> None:  # type: ignore[no-untyped-def]
    width, height = 9, 5
    source_rgb = np.full((height, width, 3), 1000, dtype=np.uint16)
    source_rgb[2, 2] = (30000, 40000, 50000)
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
    source_model = _CountingTargetProjectionSourceModel(camera, view, observer)
    geometry = MosaicExportGeometry(
        boundary_width_px=width,
        boundary_height_px=height,
        crop_left_px=1,
        crop_top_px=1,
        output_width_px=7,
        output_height_px=3,
    )
    target_transform_payload = build_target_icrs_to_pixel_transform_payload(
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
    )
    output_path = tmp_path / "fixed_tile_forward_export.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=source_model,
        source_image=MosaicExportSourceImage(
            path=tmp_path / "source.tif",
            rgb_u16=source_rgb,
            exif_tags=(),
            icc_profile=None,
        ),
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=4,
        target_icrs_to_pixel_payload=target_transform_payload,
        map_tile_size_px=4,
    )

    exported = tifffile.imread(output_path)
    assert source_model.projected_vector_count == 0
    assert source_model.projected_source_pixel_count < width * height
    assert np.max(np.abs(exported[1, 1].astype(np.int32) - source_rgb[2, 2].astype(np.int32))) < 16


def test_mosaic_export_forward_remap_fills_projected_sampling_gaps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source_width, source_height = 8, 6
    target_width, target_height = 16, 12
    source_rgb = np.full((source_height, source_width, 3), (12000, 16000, 20000), dtype=np.uint16)
    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=200.0,
    )
    source_camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=27.0,
        image_width_px=source_width,
        image_height_px=source_height,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=90.0,
    )
    target_camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=27.0,
        image_width_px=target_width,
        image_height_px=target_height,
        focal_length_mm=24.0,
        lens_model=RECTILINEAR_LENS_MODEL,
        fisheye_fov_deg=90.0,
    )
    view = ViewSettings(center_az_deg=180.0, center_alt_deg=45.0, roll_deg=0.0)
    source_model = _CountingTargetProjectionSourceModel(source_camera, view, observer)
    geometry = MosaicExportGeometry(
        boundary_width_px=target_width,
        boundary_height_px=target_height,
        crop_left_px=2,
        crop_top_px=2,
        output_width_px=12,
        output_height_px=8,
    )
    target_transform_payload = build_target_icrs_to_pixel_transform_payload(
        camera=target_camera,
        view=view,
        observer=observer,
        geometry=geometry,
    )
    output_path = tmp_path / "forward_remap_gap_fill.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=source_model,
        source_image=MosaicExportSourceImage(
            path=tmp_path / "source.tif",
            rgb_u16=source_rgb,
            exif_tags=(),
            icc_profile=None,
        ),
        camera=target_camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=4,
        target_icrs_to_pixel_payload=target_transform_payload,
        map_tile_size_px=4,
    )

    exported = tifffile.imread(output_path)
    white_pixels = np.all(exported == 65535, axis=2)
    assert not np.any(white_pixels)
    assert source_model.projected_source_pixel_count < source_width * source_height
    assert source_model.projected_vector_count == 0


def test_mosaic_export_low_weight_map_points_use_exact_reverse_projection() -> None:
    width, height = 5, 5
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
    source_model = _CountingTargetProjectionSourceModel(camera, view, observer)
    geometry = MosaicExportGeometry(
        boundary_width_px=width,
        boundary_height_px=height,
        crop_left_px=0,
        crop_top_px=0,
        output_width_px=width,
        output_height_px=height,
    )
    target_transform_payload = build_target_icrs_to_pixel_transform_payload(
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
    )
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    accum_x = grid_x.copy()
    accum_y = grid_y.copy()
    weights = np.ones((height, width), dtype=np.float32)
    weights[2, 2] = 0.1
    accum_x[2, 2] = 4.0 * weights[2, 2]
    accum_y[2, 2] = 4.0 * weights[2, 2]

    map_x, map_y = _finalize_forward_inverse_map(
        accum_x,
        accum_y,
        weights,
        4,
        source_model=source_model,
        target_icrs_to_pixel_payload=target_transform_payload,
        exact_remap_repair=True,
    )

    assert abs(float(map_x[2, 2]) - 2.0) < 1e-3
    assert abs(float(map_y[2, 2]) - 2.0) < 1e-3
    assert source_model.projected_vector_count == 1
