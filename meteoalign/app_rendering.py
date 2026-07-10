from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np
from PyQt5.QtCore import QItemSelectionModel, QTimer, Qt

from .app_constants import (
    AUTO_MATCH_SEARCH_MAG_LIMIT,
    LENS_MODELS,
    PREVIEW_LONG_SIDE_PX,
    REFERENCE_LABEL_MODE_FIXED_COUNT,
    REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT,
    REFERENCE_LABEL_MODES,
)
from .catalog import project_root
from .image_preview import ImagePreview
from .simulator import (
    CameraSettings,
    FISHEYE_EQUIDISTANT,
    FISHEYE_EQUISOLID,
    HorizontalMilkyWayCatalog,
    HorizontalSolarSystemCatalog,
    HorizontalStarCatalog,
    ObserverSettings,
    ProjectedStarMap,
    ProjectedSolarSystemObject,
    RECTILINEAR_LENS_MODEL,
    ReferenceStar,
    ViewSettings,
    _star_rgb,
    _star_style,
    compute_horizontal_catalog,
    compute_horizontal_milky_way,
    compute_horizontal_solar_system,
    project_horizontal_catalog,
    select_reference_stars,
)


class RenderingMixin:
    """渲染与模拟 Mixin：星图投影、参考星选取、相机参数、渲染调度。"""

    ui: object
    catalog: object
    milky_way_catalog: object
    renderer: object
    scene: object
    star_map_item: object
    real_image_item: object
    real_image_scene: object
    reference_scene: object
    reference_star_map_item: object
    _syncing_camera_dimensions: bool
    _horizontal_cache_key: object
    _horizontal_cache: HorizontalStarCatalog | None
    _milky_way_cache_key: object
    _milky_way_cache: HorizontalMilkyWayCatalog | None
    _solar_system_cache_key: object
    _solar_system_cache: HorizontalSolarSystemCatalog | None
    _last_render_size: object
    _last_reference_render_size: object
    _current_star_map: ProjectedStarMap | None
    _current_reference_star_map: ProjectedStarMap | None
    _current_reference_stars: tuple
    ui_config: object
    render_timer: QTimer
    _manual_reference_star_ids: list
    _imported_reference_star_by_id: dict
    _auto_match_reference_star_ids: list
    _excluded_reference_star_ids: list
    _mask_excluded_reference_star_ids: set
    current_image_preview: ImagePreview | None
    current_sky_mask: np.ndarray | None
    current_sky_masked_image: object

    def _handle_tab_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        QTimer.singleShot(0, self.fit_all_graphics_views)

    def schedule_render(self, *unused, delay_ms: int = 120) -> None:  # type: ignore[no-untyped-def]
        self.render_timer.start(delay_ms)

    def _reference_label_mode(self) -> str:
        index = self.ui.comboBoxReferenceLabelMode.currentIndex()
        if index < 0 or index >= len(REFERENCE_LABEL_MODES):
            return REFERENCE_LABEL_MODE_FIXED_COUNT
        return REFERENCE_LABEL_MODES[index]

    def _update_reference_label_controls(self) -> None:
        is_fixed_count = self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_COUNT
        self.ui.labelReferenceStarCount.setEnabled(is_fixed_count)
        self.ui.spinBoxReferenceStarCount.setEnabled(is_fixed_count)
        self.ui.labelReferenceMagLimit.setEnabled(not is_fixed_count)
        self.ui.doubleSpinBoxReferenceMagLimit.setEnabled(not is_fixed_count)

    def _handle_reference_label_mode_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._handle_reference_label_options_changed()

    def _handle_reference_label_options_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_reference_label_controls()
        self._refresh_reference_stars_from_current_map()
        self.schedule_render()

    def _reference_star_from_star_map_index(
        self,
        star_map: ProjectedStarMap,
        star_index: int,
        output_index: int,
    ) -> ReferenceStar:
        star_id = str(star_map.star_ids[star_index]).strip()
        display_name = str(star_map.display_names[star_index]).strip()
        common_name = str(star_map.common_names[star_index]).strip()
        name = common_name or display_name or star_id
        return ReferenceStar(
            index=output_index,
            star_id=star_id,
            name=name,
            display_name=display_name,
            common_name=common_name,
            ra_deg=float(star_map.ra_deg[star_index]),
            dec_deg=float(star_map.dec_deg[star_index]),
            mag_v=float(star_map.mag_v[star_index]),
            sim_x=float(star_map.x_px[star_index]),
            sim_y=float(star_map.y_px[star_index]),
            alt_deg=float(star_map.alt_deg[star_index]),
            az_deg=float(star_map.az_deg[star_index]),
        )

    def _reference_star_with_index(
        self,
        star: ReferenceStar,
        index: int,
        index_label: str | None = None,
    ) -> ReferenceStar:
        return ReferenceStar(
            index=index,
            star_id=star.star_id,
            name=star.name,
            display_name=star.display_name,
            common_name=star.common_name,
            ra_deg=star.ra_deg,
            dec_deg=star.dec_deg,
            mag_v=star.mag_v,
            sim_x=star.sim_x,
            sim_y=star.sim_y,
            alt_deg=star.alt_deg,
            az_deg=star.az_deg,
            object_type=star.object_type,
            index_label=str(index_label if index_label is not None else (star.index_label or index)),
        )

    def _reference_stars_with_display_labels(self, stars: list[ReferenceStar]) -> tuple[ReferenceStar, ...]:
        labeled_stars: list[ReferenceStar] = []
        regular_index = 1
        auto_index_by_group: dict[str, int] = {}
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        for star in stars:
            star_id = star.star_id.strip()
            if star_id in auto_match_star_ids:
                group_id = self._auto_match_group_id_for_star_id(star_id) or "A"
                auto_index = auto_index_by_group.get(group_id, 1)
                index_label = f"{group_id}{auto_index}"
                auto_index_by_group[group_id] = auto_index + 1
            else:
                index_label = str(regular_index)
                regular_index += 1
            labeled_stars.append(self._reference_star_with_index(star, len(labeled_stars) + 1, index_label))
        return tuple(labeled_stars)

    def _projected_reference_star_lookup(self, star_map: ProjectedStarMap) -> dict[str, ReferenceStar]:
        lookup: dict[str, ReferenceStar] = {}
        for star_index in range(len(star_map)):
            reference_star = self._reference_star_from_star_map_index(star_map, star_index, output_index=0)
            if reference_star.star_id:
                lookup[reference_star.star_id] = reference_star

        return lookup

    def _matched_reference_star_ids_from_table(self) -> list[str]:
        matched_star_ids: list[str] = []
        store = getattr(self, "_star_pair_store", None)
        if store is None:
            return matched_star_ids
        for record in store.snapshot():
            star_id = record.star_id
            if not star_id or star_id in matched_star_ids:
                continue
            matched_star_ids.append(star_id)
        return matched_star_ids

    def _select_current_reference_stars(self, star_map: ProjectedStarMap) -> tuple[ReferenceStar, ...]:
        self._normalize_auto_match_groups()
        if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
            auto_reference_stars = select_reference_stars(
                star_map=star_map,
                max_count=None,
                mag_limit=self.ui.doubleSpinBoxReferenceMagLimit.value(),
            )
        else:
            auto_reference_stars = select_reference_stars(
                star_map=star_map,
                max_count=self.ui.spinBoxReferenceStarCount.value(),
            )

        # 手动点选的参考星以星表编号保存；每次渲染后用当前投影坐标重新生成行。
        ordered_stars: list[ReferenceStar] = []
        seen_star_ids: set[str] = set()
        excluded_star_ids = set(self._excluded_reference_star_ids) | set(
            getattr(self, "_mask_excluded_reference_star_ids", set())
        )
        auto_match_star_ids = set(self._auto_match_reference_star_ids)
        for star in auto_reference_stars:
            star_id = star.star_id.strip()
            if not star_id or star_id in seen_star_ids or star_id in excluded_star_ids or star_id in auto_match_star_ids:
                continue
            seen_star_ids.add(star_id)
            ordered_stars.append(star)

        manual_lookup = self._projected_reference_star_lookup(star_map)
        imported_lookup = getattr(self, "_imported_reference_star_by_id", {})

        def append_lookup_star(
            star_id: str,
            *,
            keep_if_excluded: bool = False,
            keep_if_auto: bool = False,
        ) -> None:
            star_id = star_id.strip()
            if not star_id or star_id in seen_star_ids:
                return
            if not keep_if_auto and star_id in auto_match_star_ids:
                return
            if not keep_if_excluded and star_id in excluded_star_ids:
                return
            lookup_star = manual_lookup.get(star_id)
            if lookup_star is None:
                lookup_star = imported_lookup.get(star_id)
            if lookup_star is None:
                return
            seen_star_ids.add(star_id)
            ordered_stars.append(lookup_star)

        # 已经配对的星必须保留在表格中，否则参考图切换到全星表投影后会丢失拟合锚点。
        for star_id in self._matched_reference_star_ids_from_table():
            append_lookup_star(star_id, keep_if_excluded=True)

        for star_id in self._manual_reference_star_ids:
            if star_id in auto_match_star_ids:
                continue
            append_lookup_star(star_id)

        for star_id in self._auto_match_reference_star_ids:
            append_lookup_star(star_id, keep_if_auto=True)

        return self._reference_stars_with_display_labels(ordered_stars)

    def _reference_selection_star_map(self) -> ProjectedStarMap | None:
        return self._current_reference_star_map or self._current_star_map

    def _refresh_reference_stars_from_current_map(self) -> None:
        star_map = self._reference_selection_star_map()
        if star_map is None:
            return
        reference_stars = self._select_current_reference_stars(star_map)
        self._update_star_pair_table(reference_stars)

    def _row_for_star_id(self, star_id: str) -> int | None:
        for row in range(self.ui.tableWidgetStarPairs.rowCount()):
            if self._star_pair_star_id(row) == star_id:
                return row
        return None

    def _select_star_pair_row_by_id(self, star_id: str) -> int | None:
        row = self._row_for_star_id(star_id)
        if row is not None:
            table = self.ui.tableWidgetStarPairs
            row_index = table.model().index(row, 0)
            table.selectionModel().setCurrentIndex(
                row_index,
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
            )
            table.scrollTo(row_index)
            table.setFocus(Qt.OtherFocusReason)
            return row
        return None

    def _sensor_aspect_ratio(self) -> float:
        sensor_height = max(self.ui.doubleSpinBoxSensorHeight.value(), 1e-6)
        return max(self.ui.doubleSpinBoxSensorWidth.value() / sensor_height, 1e-6)

    def _bounded_image_width(self, value: float) -> int:
        return min(max(int(round(value)), self.ui.spinBoxImageWidth.minimum()), self.ui.spinBoxImageWidth.maximum())

    def _bounded_image_height(self, value: float) -> int:
        return min(max(int(round(value)), self.ui.spinBoxImageHeight.minimum()), self.ui.spinBoxImageHeight.maximum())

    def _set_image_dimensions(self, width_px: int, height_px: int) -> None:
        self._syncing_camera_dimensions = True
        self.ui.spinBoxImageWidth.blockSignals(True)
        self.ui.spinBoxImageHeight.blockSignals(True)
        self.ui.spinBoxImageWidth.setValue(self._bounded_image_width(width_px))
        self.ui.spinBoxImageHeight.setValue(self._bounded_image_height(height_px))
        self.ui.spinBoxImageWidth.blockSignals(False)
        self.ui.spinBoxImageHeight.blockSignals(False)
        self._syncing_camera_dimensions = False

    def _sync_image_size_to_sensor_long_side(self) -> None:
        aspect_ratio = self._sensor_aspect_ratio()
        long_side = max(self.ui.spinBoxImageWidth.value(), self.ui.spinBoxImageHeight.value())
        if aspect_ratio >= 1.0:
            width_px = self._bounded_image_width(long_side)
            height_px = self._bounded_image_height(width_px / aspect_ratio)
        else:
            height_px = self._bounded_image_height(long_side)
            width_px = self._bounded_image_width(height_px * aspect_ratio)
        self._set_image_dimensions(width_px, height_px)

    def _handle_sensor_size_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self._syncing_camera_dimensions:
            return
        self._sync_image_size_to_sensor_long_side()
        self.schedule_render()

    def _handle_image_width_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self._syncing_camera_dimensions:
            return
        width_px = self.ui.spinBoxImageWidth.value()
        height_px = self._bounded_image_height(width_px / self._sensor_aspect_ratio())
        self._set_image_dimensions(width_px, height_px)

    def _handle_image_height_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if self._syncing_camera_dimensions:
            return
        height_px = self.ui.spinBoxImageHeight.value()
        width_px = self._bounded_image_width(height_px * self._sensor_aspect_ratio())
        self._set_image_dimensions(width_px, height_px)

    def _swap_camera_orientation(self) -> None:
        sensor_width = self.ui.doubleSpinBoxSensorWidth.value()
        sensor_height = self.ui.doubleSpinBoxSensorHeight.value()
        self._syncing_camera_dimensions = True
        self.ui.doubleSpinBoxSensorWidth.blockSignals(True)
        self.ui.doubleSpinBoxSensorHeight.blockSignals(True)
        self.ui.doubleSpinBoxSensorWidth.setValue(sensor_height)
        self.ui.doubleSpinBoxSensorHeight.setValue(sensor_width)
        self.ui.doubleSpinBoxSensorWidth.blockSignals(False)
        self.ui.doubleSpinBoxSensorHeight.blockSignals(False)
        self._syncing_camera_dimensions = False
        self._sync_image_size_to_sensor_long_side()
        self.schedule_render()

    def _lens_model(self) -> str:
        index = self.ui.comboBoxLensModel.currentIndex()
        if index < 0 or index >= len(LENS_MODELS):
            return RECTILINEAR_LENS_MODEL
        return LENS_MODELS[index]

    def _update_lens_model_controls(self) -> None:
        lens_model = self._lens_model()
        is_fisheye = lens_model != RECTILINEAR_LENS_MODEL
        locked = bool(getattr(self, "_simulator_controls_locked", False))
        max_fov = 300.0
        self.ui.doubleSpinBoxFisheyeFov.setMaximum(max_fov)
        if self.ui.doubleSpinBoxFisheyeFov.value() > max_fov:
            self.ui.doubleSpinBoxFisheyeFov.setValue(max_fov)
        self.ui.labelFisheyeFov.setEnabled(is_fisheye and not locked)
        self.ui.doubleSpinBoxFisheyeFov.setEnabled(is_fisheye and not locked)

    def _handle_lens_model_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_lens_model_controls()
        self.schedule_render()

    def _observer_settings(self) -> ObserverSettings:
        local_dt = self.ui.dateTimeEditObservation.dateTime().toPyDateTime()
        offset = timezone(timedelta(hours=self.ui.doubleSpinBoxUtcOffset.value()))
        aware_dt = local_dt.replace(tzinfo=offset)
        return ObserverSettings(
            observation_time_utc=aware_dt.astimezone(timezone.utc),
            latitude_deg=self.ui.doubleSpinBoxLatitude.value(),
            longitude_deg=self.ui.doubleSpinBoxLongitude.value(),
            elevation_m=self.ui.doubleSpinBoxElevation.value(),
        )

    def _camera_settings_for_image_size(self, image_width_px: int, image_height_px: int) -> CameraSettings:
        return CameraSettings(
            sensor_width_mm=self.ui.doubleSpinBoxSensorWidth.value(),
            sensor_height_mm=self.ui.doubleSpinBoxSensorHeight.value(),
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            focal_length_mm=self.ui.doubleSpinBoxFocalLength.value(),
            lens_model=self._lens_model(),
            fisheye_fov_deg=self.ui.doubleSpinBoxFisheyeFov.value(),
        )

    def _output_camera_settings(self) -> CameraSettings:
        return self._camera_settings_for_image_size(
            image_width_px=self.ui.spinBoxImageWidth.value(),
            image_height_px=self.ui.spinBoxImageHeight.value(),
        )

    def _preview_image_size(self) -> tuple[int, int]:
        aspect_ratio = self._sensor_aspect_ratio()
        if aspect_ratio >= 1.0:
            width_px = PREVIEW_LONG_SIDE_PX
            height_px = max(128, int(round(PREVIEW_LONG_SIDE_PX / aspect_ratio)))
        else:
            height_px = PREVIEW_LONG_SIDE_PX
            width_px = max(128, int(round(PREVIEW_LONG_SIDE_PX * aspect_ratio)))
        return width_px, height_px

    def _preview_camera_settings(self) -> CameraSettings:
        image_width_px, image_height_px = self._preview_image_size()
        return self._camera_settings_for_image_size(image_width_px=image_width_px, image_height_px=image_height_px)

    def _render_element_scale(self, camera: CameraSettings) -> float:
        return max(camera.image_width_px, camera.image_height_px) / float(PREVIEW_LONG_SIDE_PX)

    def _aligned_star_element_scale(self, target_size: tuple[int, int]) -> float:
        long_side = max(float(target_size[0]), float(target_size[1]), 1.0)
        base_scale = long_side / float(PREVIEW_LONG_SIDE_PX)
        # 对齐到真实图像后，场景尺寸通常远大于预览图；这里用配置倍率补偿 fitInView 后的视觉缩小。
        return max(0.75, min(12.0, base_scale * self.ui_config.aligned_reference_scale_multiplier))

    def _view_settings(self) -> ViewSettings:
        return ViewSettings(
            center_az_deg=self.ui.doubleSpinBoxAz.value(),
            center_alt_deg=self.ui.doubleSpinBoxAlt.value(),
            roll_deg=self.ui.doubleSpinBoxRoll.value(),
        )

    def _get_horizontal_catalog(self, observer: ObserverSettings, mag_limit: float) -> HorizontalStarCatalog:
        cache_key = (
            int(observer.observation_time_utc.timestamp()),
            round(observer.latitude_deg, 8),
            round(observer.longitude_deg, 8),
            round(observer.elevation_m, 3),
            round(mag_limit, 3),
        )
        if self._horizontal_cache_key != cache_key or self._horizontal_cache is None:
            self._horizontal_cache = compute_horizontal_catalog(
                catalog=self.catalog,
                observer=observer,
                visible_mag_limit=mag_limit,
            )
            self._horizontal_cache_key = cache_key
        return self._horizontal_cache

    def _get_horizontal_milky_way(self, observer: ObserverSettings) -> HorizontalMilkyWayCatalog:
        cache_key = (
            int(observer.observation_time_utc.timestamp()),
            round(observer.latitude_deg, 8),
            round(observer.longitude_deg, 8),
            round(observer.elevation_m, 3),
        )
        if self._milky_way_cache_key != cache_key or self._milky_way_cache is None:
            self._milky_way_cache = compute_horizontal_milky_way(
                milky_way=self.milky_way_catalog,
                observer=observer,
            )
            self._milky_way_cache_key = cache_key
        return self._milky_way_cache

    def _get_horizontal_solar_system(self, observer: ObserverSettings) -> HorizontalSolarSystemCatalog:
        cache_key = (
            int(observer.observation_time_utc.timestamp()),
            round(observer.latitude_deg, 8),
            round(observer.longitude_deg, 8),
            round(observer.elevation_m, 3),
        )
        if self._solar_system_cache_key != cache_key or self._solar_system_cache is None:
            self._solar_system_cache = compute_horizontal_solar_system(observer)
            self._solar_system_cache_key = cache_key
        return self._solar_system_cache

    def _build_projected_star_map(
        self,
        camera: CameraSettings | None = None,
        visible_mag_limit: float | None = None,
    ) -> tuple[ObserverSettings, CameraSettings, ViewSettings, float, ProjectedStarMap]:
        observer = self._observer_settings()
        camera = camera or self._preview_camera_settings()
        view = self._view_settings()
        mag_limit = self.ui.doubleSpinBoxMagLimit.value() if visible_mag_limit is None else float(visible_mag_limit)
        horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
        horizontal_milky_way = self._get_horizontal_milky_way(observer)
        horizontal_solar_system = self._get_horizontal_solar_system(observer)
        star_map = project_horizontal_catalog(
            horizontal_catalog=horizontal_catalog,
            camera=camera,
            view=view,
            visible_mag_limit=mag_limit,
            horizontal_milky_way=horizontal_milky_way,
            horizontal_solar_system=horizontal_solar_system,
            star_color_mag_limit=self.ui_config.star_color_mag_limit,
        )
        return observer, camera, view, mag_limit, star_map

    def _reference_catalog_mag_limit(self, extra_mag_limit: float | None = None) -> float:
        mag_limit = float(self.ui.doubleSpinBoxMagLimit.value())
        if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
            mag_limit = max(mag_limit, float(self.ui.doubleSpinBoxReferenceMagLimit.value()))
        if self._auto_match_reference_star_ids:
            mag_limit = max(mag_limit, float(AUTO_MATCH_SEARCH_MAG_LIMIT))
        if extra_mag_limit is not None:
            mag_limit = max(mag_limit, float(extra_mag_limit))
        return mag_limit

    def _aligned_solar_system_objects(
        self,
        horizontal_solar_system: HorizontalSolarSystemCatalog,
        transform: object,
        target_size: tuple[int, int],
    ) -> tuple[ProjectedSolarSystemObject, ...]:
        if len(horizontal_solar_system) == 0:
            return ()

        points = transform.transform_radec_points(
            np.column_stack((horizontal_solar_system.ra_deg, horizontal_solar_system.dec_deg))
        )
        width_px, height_px = target_size
        inside = (
            np.all(np.isfinite(points), axis=1)
            & (points[:, 0] >= 0.0)
            & (points[:, 0] <= width_px - 1)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= height_px - 1)
        )

        objects: list[ProjectedSolarSystemObject] = []
        for index in np.flatnonzero(inside):
            color = (
                int(horizontal_solar_system.color_rgb[index, 0]),
                int(horizontal_solar_system.color_rgb[index, 1]),
                int(horizontal_solar_system.color_rgb[index, 2]),
            )
            objects.append(
                ProjectedSolarSystemObject(
                    object_id=str(horizontal_solar_system.object_ids[index]),
                    display_name=str(horizontal_solar_system.display_names[index]),
                    kernel_name=str(horizontal_solar_system.kernel_names[index]),
                    ra_deg=float(horizontal_solar_system.ra_deg[index]),
                    dec_deg=float(horizontal_solar_system.dec_deg[index]),
                    mag_v=float(horizontal_solar_system.mag_v[index]),
                    sim_x=float(points[index, 0]),
                    sim_y=float(points[index, 1]),
                    alt_deg=float(horizontal_solar_system.alt_deg[index]),
                    az_deg=float(horizontal_solar_system.az_deg[index]),
                    radius_px=float(horizontal_solar_system.radius_px[index]),
                    color_rgb=color,
                    alpha=255,
                    above_horizon=bool(horizontal_solar_system.alt_deg[index] >= 0.0),
                    reference_allowed=bool(horizontal_solar_system.reference_allowed[index]),
                )
            )
        return tuple(objects)

    def _build_aligned_reference_star_map(
        self,
        transform: object,
        target_size: tuple[int, int],
        visible_mag_limit: float | None = None,
    ) -> ProjectedStarMap:
        width_px = max(1, int(target_size[0]))
        height_px = max(1, int(target_size[1]))
        observer = self._observer_settings()
        mag_limit = self._reference_catalog_mag_limit(visible_mag_limit)
        horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
        horizontal_solar_system = self._get_horizontal_solar_system(observer)

        if len(horizontal_catalog) > 0:
            points = transform.transform_radec_points(
                np.column_stack((horizontal_catalog.ra_deg, horizontal_catalog.dec_deg))
            )
            inside = (
                np.all(np.isfinite(points), axis=1)
                & (points[:, 0] >= 0.0)
                & (points[:, 0] <= width_px - 1)
                & (points[:, 1] >= 0.0)
                & (points[:, 1] <= height_px - 1)
            )
        else:
            points = np.empty((0, 2), dtype=np.float64)
            inside = np.asarray([], dtype=bool)

        radius, intensity = _star_style(horizontal_catalog.mag_v[inside], mag_limit)
        star_rgb = _star_rgb(
            mag_v=horizontal_catalog.mag_v[inside],
            star_color_mag_limit=self.ui_config.star_color_mag_limit,
            intensity=intensity,
            color_index_bv=horizontal_catalog.color_index_bv[inside],
            spectral_type=horizontal_catalog.spectral_type[inside],
        )
        star_count = int(np.count_nonzero(inside))
        alpha = np.full(star_count, 255, dtype=np.uint8)
        solar_system_objects = self._aligned_solar_system_objects(
            horizontal_solar_system,
            transform=transform,
            target_size=(width_px, height_px),
        )

        # 参考星图进入配准模式后只表达“当前模型投到真实图像窗口里的星”，不再附带地平线遮罩或模拟视野网格。
        return ProjectedStarMap(
            width=width_px,
            height=height_px,
            source_name=horizontal_catalog.source_name,
            x_px=points[inside, 0].astype(np.float64),
            y_px=points[inside, 1].astype(np.float64),
            radius_px=radius,
            intensity=intensity,
            alpha=alpha,
            above_horizon=horizontal_catalog.alt_deg[inside] >= 0.0,
            star_ids=horizontal_catalog.star_ids[inside],
            display_names=horizontal_catalog.display_names[inside],
            common_names=horizontal_catalog.common_names[inside],
            ra_deg=horizontal_catalog.ra_deg[inside],
            dec_deg=horizontal_catalog.dec_deg[inside],
            alt_deg=horizontal_catalog.alt_deg[inside].astype(np.float64),
            az_deg=horizontal_catalog.az_deg[inside].astype(np.float64),
            mag_v=horizontal_catalog.mag_v[inside],
            color_index_bv=horizontal_catalog.color_index_bv[inside],
            spectral_type=horizontal_catalog.spectral_type[inside],
            star_rgb=star_rgb,
            grid_lines=(),
            direction_labels=(),
            catalog_count=len(horizontal_catalog),
            lens_model=str(getattr(transform, "lens_model", self._lens_model())),
            sky_circle_radius_px=None,
            horizon_shadow_rects=(),
            milky_way_polygons=(),
            solar_system_objects=solar_system_objects,
        )

    def _display_star_map(self, star_map: ProjectedStarMap, reference_stars: tuple[ReferenceStar, ...]) -> None:
        render_size = (star_map.width, star_map.height)
        should_fit = self._last_render_size != render_size or self.star_map_item.boundingRect().isEmpty()
        self.star_map_item.setPos(0, 0)
        self.star_map_item.set_star_map(
            star_map,
            reference_stars=reference_stars,
            element_scale=1.0,
            draw_common_names=False,
            number_reference_stars=False,
        )
        self.scene.setSceneRect(0, 0, star_map.width, star_map.height)
        self._last_render_size = render_size
        if should_fit:
            self.fit_star_map()

    def _fit_reference_map_if_display_changed(self, display_key: tuple[object, ...]) -> None:
        should_fit = self._last_reference_render_size != display_key or self.reference_star_map_item.boundingRect().isEmpty()
        self._last_reference_render_size = display_key
        if should_fit:
            self.fit_reference_map()

    def _display_real_image_preview(self, preview: ImagePreview) -> None:
        self._clear_star_pair_annotations()
        self.real_image_item.setPos(0, 0)
        self.real_image_item.set_image(self._real_image_for_current_mask_preview())
        self.real_image_scene.setSceneRect(0, 0, preview.image.width(), preview.image.height())
        self._restore_star_pair_annotations_from_table()
        self._update_reference_alignment_transform()
        self.fit_real_image()

    def render_now(self) -> None:
        try:
            _observer, _camera, view, _mag_limit, star_map = self._build_projected_star_map()
            self._current_star_map = star_map
            self._current_reference_star_map = star_map
            reference_stars = self._select_current_reference_stars(star_map)
            self._display_star_map(star_map, reference_stars)
            self._update_star_pair_table(reference_stars)
            if self._reference_label_mode() == REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT:
                reference_mode_text = f"标注星等 <= {self.ui.doubleSpinBoxReferenceMagLimit.value():.1f} mag"
            else:
                reference_mode_text = f"标注星数 {self.ui.spinBoxReferenceStarCount.value()} 颗"
            manual_count = len(
                [
                    star_id
                    for star_id in self._manual_reference_star_ids
                    if any(star.star_id == star_id for star in reference_stars)
                ]
            )
            if manual_count:
                reference_mode_text = f"{reference_mode_text}，手动 {manual_count} 颗"
            self.ui.statusbar.showMessage(
                "星表: {catalog_count}  视野内: {visible_count}  地平线上: {above_count}  "
                "银河面: {mw_count}  太阳系: {solar_count}  参考星: {reference_count} ({reference_mode})  "
                "镜头: {lens_name}  Az: {az:.2f} deg  Alt: {alt:.2f} deg".format(
                    catalog_count=star_map.catalog_count,
                    visible_count=len(star_map),
                    above_count=star_map.above_horizon_count,
                    mw_count=len(star_map.milky_way_polygons),
                    solar_count=len(star_map.solar_system_objects),
                    reference_count=len(reference_stars),
                    reference_mode=reference_mode_text,
                    lens_name=self.ui.comboBoxLensModel.currentText(),
                    az=view.center_az_deg,
                    alt=view.center_alt_deg,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 界面层需要把可恢复输入错误显示出来。
            self.ui.statusbar.showMessage(f"渲染失败: {exc}")
