from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import tifffile

import meteoalign.mosaic_export as mosaic_export_module
from meteoalign.coordinates import unit_vectors_to_radec
from meteoalign.mosaic.export.remap_repair import fill_forward_inverse_map_holes_fast
from meteoalign.mosaic_export import (
    MosaicExportGeometry,
    MosaicExportSourceImage,
    _copy_full_tiff_exif_metadata,
    _finalize_forward_inverse_map,
    _icrs_camera_basis_from_view,
    _render_mosaic_reprojection_block_from_map,
    build_target_icrs_to_pixel_transform_payload,
    load_mosaic_export_source_image,
    mosaic_reprojection_blocks,
    mosaic_export_cropped_geometry,
    mosaic_reprojection_map_blocks,
    restrict_reprojection_map_to_source_regions,
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


class _LinearAstrometricModel:
    """用于验证 Pixel→ICRS→Pixel 桥接的可逆线性模型。"""

    def __init__(self, width: int, height: int, ra_offset_deg: float, dec_offset_deg: float) -> None:
        self.image_width_px = int(width)
        self.image_height_px = int(height)
        self.ra_offset_deg = float(ra_offset_deg)
        self.dec_offset_deg = float(dec_offset_deg)

    def pixel_to_sky_points(self, pixel_points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        return np.column_stack(
            (
                pixels[:, 0] + self.ra_offset_deg,
                pixels[:, 1] + self.dec_offset_deg,
            )
        )

    def sky_to_pixel_points(self, ra_dec_points: np.ndarray) -> np.ndarray:
        radec = np.asarray(ra_dec_points, dtype=np.float64)
        return np.column_stack(
            (
                radec[:, 0] - self.ra_offset_deg,
                radec[:, 1] - self.dec_offset_deg,
            )
        )

    def icrs_vectors_to_pixel_points(self, vectors: np.ndarray) -> np.ndarray:
        return self.sky_to_pixel_points(unit_vectors_to_radec(vectors))


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


def test_mosaic_export_writes_cropped_lzw_rgba_u16_tiff_with_exif(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
        rgb=source_rgb,
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
        extra_samples = page.tags["ExtraSamples"].value
        make = page.tags[271].value

    assert exported.shape == (4, 6, 4)
    assert exported.dtype == np.uint16
    assert compression == 5
    assert tuple(int(value) for value in extra_samples) == (2,)
    assert make == "UnitTestCamera"
    assert exported[0, 0, 3] == 65535
    assert np.max(np.abs(exported[0, 0, :3].astype(np.int32) - source_rgb[1, 1].astype(np.int32))) < 8

    uncompressed_path = tmp_path / "export_uncompressed.tif"
    write_mosaic_reprojection_tiff(
        output_path=uncompressed_path,
        source_model=source_model,
        source_image=source_image,
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
        block_rows=2,
        tiff_lzw_compression=False,
    )
    with tifffile.TiffFile(uncompressed_path) as tiff:
        assert tiff.pages[0].tags["Compression"].value == 1


def test_mosaic_export_keeps_uint8_source_and_alpha_depth(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """8 位原图重投影后仍应写成 8 位 TIFF，透明通道上限应为 255。"""

    width, height = 8, 6
    source_rgb = np.full((height, width, 3), (20, 80, 160), dtype=np.uint8)
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
    source_image = MosaicExportSourceImage(
        path=tmp_path / "source.jpg",
        rgb=source_rgb,
        exif_tags=(),
        icc_profile=None,
    )
    geometry = MosaicExportGeometry(
        boundary_width_px=width,
        boundary_height_px=height,
        crop_left_px=0,
        crop_top_px=0,
        output_width_px=width,
        output_height_px=height,
    )
    output_path = tmp_path / "export_8bit.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=_TargetProjectionSourceModel(camera, view, observer),
        source_image=source_image,
        camera=camera,
        view=view,
        observer=observer,
        geometry=geometry,
    )

    exported = tifffile.imread(output_path)
    assert exported.dtype == np.uint8
    assert exported.shape == (height, width, 4)
    assert np.max(exported[:, :, 3]) == 255


def test_mosaic_source_loader_preserves_uint8_and_uint16_dtype(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """最终导出读取器不得再由 Pillow 把 16 位 RGB TIFF 降成 8 位。"""

    from PIL import Image

    source_8bit_path = tmp_path / "source_8bit.jpg"
    source_16bit_path = tmp_path / "source_16bit.tif"
    Image.fromarray(np.full((4, 5, 3), 120, dtype=np.uint8), mode="RGB").save(source_8bit_path)
    tifffile.imwrite(
        source_16bit_path,
        np.full((4, 5, 3), 40000, dtype=np.uint16),
        photometric="rgb",
    )

    source_8bit = load_mosaic_export_source_image(source_8bit_path)
    source_16bit = load_mosaic_export_source_image(source_16bit_path)

    assert source_8bit.rgb.dtype == np.uint8
    assert source_8bit.bit_depth == 8
    assert source_16bit.rgb.dtype == np.uint16
    assert source_16bit.bit_depth == 16
    assert np.all(source_16bit.rgb == 40000)


def test_mosaic_export_copies_complete_exif_ifd_tree_from_tiff_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """TIFF 导出应保留原图嵌套 EXIF IFD 中的拍摄与厂商信息。"""

    import tifftools
    from PIL import ExifTags, Image

    source_path = tmp_path / "source_with_exif.tif"
    source_temp_path = tmp_path / "source_with_exif_temp.tif"
    output_path = tmp_path / "output.tif"
    tifffile.imwrite(source_path, np.zeros((3, 4, 3), dtype=np.uint16), photometric="rgb")

    source_info = tifftools.read_tiff(str(source_path))
    source_info["ifds"][0]["tags"][34665] = {
        "ifds": [[
            {
                "path_or_fobj": str(source_path),
                "tags": {
                    33434: {"datatype": 5, "data": [1, 30]},
                    36867: {"datatype": 2, "data": "2026:07:11 22:30:00"},
                    37500: {"datatype": 7, "data": b"CameraMakerNote"},
                },
            }
        ]]
    }
    tifftools.write_tiff(source_info, str(source_temp_path))
    source_temp_path.replace(source_path)
    tifffile.imwrite(output_path, np.zeros((2, 3, 4), dtype=np.uint16), photometric="rgb", extrasamples=("unassalpha",))

    _copy_full_tiff_exif_metadata(source_path, output_path)

    with Image.open(output_path) as image:
        exif_ifd = image.getexif().get_ifd(ExifTags.IFD.Exif)
    assert exif_ifd[36867] == "2026:07:11 22:30:00"
    assert float(exif_ifd[33434]) == 1.0 / 30.0
    assert exif_ifd[37500] == b"CameraMakerNote"


def test_mosaic_render_block_marks_invalid_pixels_transparent() -> None:
    source_rgb = np.zeros((2, 2, 3), dtype=np.uint16)
    source_rgb[0, 0] = (1000, 2000, 3000)
    map_x = np.asarray([[0.0, -1.0]], dtype=np.float32)
    map_y = np.asarray([[0.0, -1.0]], dtype=np.float32)

    rendered = _render_mosaic_reprojection_block_from_map(
        source_rgb=source_rgb,
        map_x=map_x,
        map_y=map_y,
    )

    assert rendered.shape == (1, 2, 4)
    assert np.array_equal(rendered[0, 0], np.asarray([1000, 2000, 3000, 65535], dtype=np.uint16))
    assert np.array_equal(rendered[0, 1], np.asarray([0, 0, 0, 0], dtype=np.uint16))


def test_reprojection_region_mask_keeps_only_selected_source_pixels() -> None:
    """流星区域模式应将框外源图采样点变为透明。"""

    map_x = np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    map_y = np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    restricted_x, restricted_y = restrict_reprojection_map_to_source_regions(
        map_x,
        map_y,
        ((1, 0, 3, 1),),
    )

    assert restricted_x.tolist() == [[-1.0, 1.0, 2.0, -1.0]]
    assert restricted_y.tolist() == [[-1.0, 0.0, 0.0, -1.0]]


def test_mosaic_reprojection_only_projects_selected_source_region() -> None:
    """天空模式的流星区域导出不应计算框外源图到 ICRS 的映射。"""

    width, height = 12, 8
    observer = ObserverSettings(
        observation_time_utc=datetime(2025, 12, 14, 18, 11, 45, tzinfo=timezone.utc),
        latitude_deg=25.0,
        longitude_deg=102.0,
        elevation_m=200.0,
    )
    camera = CameraSettings(
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
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
    source_rgb = np.zeros((height, width, 3), dtype=np.uint16)
    source_rgb[2:5, 3:7] = (50000, 1000, 2000)

    blocks = list(
        mosaic_reprojection_blocks(
            source_model=source_model,
            source_rgb=source_rgb,
            camera=camera,
            view=view,
            observer=observer,
            geometry=geometry,
            map_tile_size_px=1,
            source_pixel_regions=((3, 2, 7, 5),),
        )
    )
    exported = np.vstack(blocks)

    assert source_model.projected_source_pixel_count == 12
    assert np.any(exported[:, :, 3] > 0)
    assert not np.any((exported[:, :, 3] > 0) & (exported[:, :, 0] != 50000))


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
            rgb=source_rgb,
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
    assert exported.shape == (4, 6, 4)
    assert exported[1, 1, 3] == 65535
    assert np.max(np.abs(exported[1, 1, :3].astype(np.int32) - source_rgb[2, 2].astype(np.int32))) < 8


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
            rgb=source_rgb,
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
    assert exported.shape == (4, 6, 4)
    assert exported[1, 1, 3] == 65535
    assert np.max(np.abs(exported[1, 1, :3].astype(np.int32) - source_rgb[2, 2].astype(np.int32))) < 8


def test_mosaic_export_uses_base_image_model_as_pixel_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source_width, source_height = 16, 12
    target_width, target_height = 64, 48
    source_rgb = np.zeros((source_height, source_width, 3), dtype=np.uint16)
    for y_value in range(source_height):
        for x_value in range(source_width):
            source_rgb[y_value, x_value] = (
                1000 + x_value * 1000,
                2000 + y_value * 2000,
                30000,
            )

    # 两个模型的赤经、赤纬零点不同，因此源图会在底图画布上向右 20、向下 15 像素。
    source_model = _LinearAstrometricModel(source_width, source_height, 30.0, 20.0)
    base_model = _LinearAstrometricModel(target_width, target_height, 10.0, 5.0)
    geometry = MosaicExportGeometry(
        boundary_width_px=target_width,
        boundary_height_px=target_height,
        crop_left_px=0,
        crop_top_px=0,
        output_width_px=target_width,
        output_height_px=target_height,
    )
    output_path = tmp_path / "base_model_export.tif"

    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=source_model,
        source_image=MosaicExportSourceImage(
            path=tmp_path / "source.tif",
            rgb=source_rgb,
            exif_tags=(),
            icc_profile=None,
        ),
        camera=None,
        view=None,
        observer=None,
        geometry=geometry,
        target_model=base_model,
        map_tile_size_px=4,
    )

    with tifffile.TiffFile(output_path) as tiff:
        exported = tiff.asarray()
        compression = tiff.pages[0].tags["Compression"].value

    assert exported.shape == (target_height, target_width, 4)
    assert exported.dtype == np.uint16
    assert compression == 5
    assert np.array_equal(exported[0, 0], np.asarray([0, 0, 0, 0], dtype=np.uint16))
    assert np.array_equal(exported[15, 20, :3], source_rgb[0, 0])
    assert exported[15, 20, 3] == 65535
    assert np.array_equal(exported[26, 35, :3], source_rgb[11, 15])
    assert exported[26, 35, 3] == 65535
    assert np.all(exported[47, :, 3] == 0)


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
            rgb=source_rgb,
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
    assert exported.shape == (3, 7, 4)
    assert exported[1, 1, 3] == 65535
    assert np.max(np.abs(exported[1, 1, :3].astype(np.int32) - source_rgb[2, 2].astype(np.int32))) < 16


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
            rgb=source_rgb,
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
    assert exported.shape == (8, 12, 4)
    assert np.all(exported[:, :, 3] == 65535)
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


def test_fast_forward_remap_fill_preserves_smooth_affine_coordinates() -> None:
    """单次传播应填满内部空洞，并保持平滑仿射映射的坐标精度。"""

    height, width = 72, 96
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    expected_x = 20.0 + grid_x * 0.8 + grid_y * 0.1
    expected_y = 10.0 - grid_x * 0.05 + grid_y * 0.9
    covered = np.ones((height, width), dtype=bool)
    covered[18:25, 20:29] = False
    covered[36:45, 52:63] = False
    target_mask = np.ones_like(covered)
    map_x = np.where(covered, expected_x, -1.0).astype(np.float32)
    map_y = np.where(covered, expected_y, -1.0).astype(np.float32)
    progress: list[tuple[int, int]] = []

    filled = fill_forward_inverse_map_holes_fast(
        map_x,
        map_y,
        covered,
        target_mask,
        tile_size=4,
        progress_callback=lambda value, maximum: progress.append((value, maximum)),
    )

    missing = ~covered
    errors = np.hypot(map_x[missing] - expected_x[missing], map_y[missing] - expected_y[missing])
    assert filled.all()
    assert float(np.percentile(errors, 99)) < 5.0
    assert float(np.max(errors)) < 5.0
    assert progress[0][0] == 0
    assert progress[-1][0] == progress[-1][1]


def test_mosaic_tiff_writer_consumes_rgba_blocks_without_vstack(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """TIFF 写入应逐块编码，不得重新用 vstack 组装完整 RGBA。"""

    top = np.full((2, 5, 4), (100, 200, 300, 65535), dtype=np.uint16)
    bottom = np.full((2, 5, 4), (400, 500, 600, 65535), dtype=np.uint16)

    def fake_blocks(**_kwargs):  # type: ignore[no-untyped-def]
        yield top
        yield bottom

    def forbidden_vstack(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("流式 TIFF 写入不应调用 np.vstack")

    monkeypatch.setattr(mosaic_export_module, "mosaic_reprojection_blocks", fake_blocks)
    monkeypatch.setattr(mosaic_export_module.np, "vstack", forbidden_vstack)
    output_path = tmp_path / "streamed_blocks.tif"
    write_mosaic_reprojection_tiff(
        output_path=output_path,
        source_model=object(),
        source_image=MosaicExportSourceImage(
            path=tmp_path / "source.tif",
            rgb=np.zeros((2, 2, 3), dtype=np.uint16),
            exif_tags=(),
            icc_profile=None,
        ),
        camera=None,
        view=None,
        observer=None,
        geometry=MosaicExportGeometry(5, 4, 0, 0, 5, 4),
        block_rows=2,
        tiff_lzw_compression=True,
    )

    exported = tifffile.imread(output_path)
    assert np.array_equal(exported[:2], top)
    assert np.array_equal(exported[2:], bottom)
