from __future__ import annotations

from .mosaic_common import *  # noqa: F401, F403
from .mosaic_model_io import (
    MosaicCoverageCache,
    MosaicModelFitData,
    MosaicSourceModel,
    MosaicSourceTextureCache,
    _expanded_polygon_points,
    _load_mosaic_source_model,
)


class MosaicProjectionMixin:
    """自由投影拼图预览 Mixin。"""

    ui: object
    renderer: object
    catalog: object
    milky_way_catalog: object
    ui_config: object

    def _init_mosaic_projection_page(self) -> None:
        if not hasattr(self.ui, "mosaicProjectionView"):
            return
        self.mosaic_scene = QGraphicsScene(self)
        self.mosaic_image_item = GraphicsImageItem()
        self.mosaic_scene.addItem(self.mosaic_image_item)
        self.ui.mosaicProjectionView.setScene(self.mosaic_scene)
        self.ui.mosaicProjectionView.installEventFilter(self)
        self.ui.mosaicProjectionView.viewport().installEventFilter(self)
        self.ui.mosaicProjectionView.viewport().setMouseTracking(True)

        self.mosaic_render_timer = QTimer(self)
        self.mosaic_render_timer.setSingleShot(True)
        self.mosaic_render_timer.timeout.connect(self.render_mosaic_projection_now)
        self._mosaic_sky_preview_renderer = SkyPreviewRenderService(self.renderer)
        self._mosaic_texture_renderer = ProjectedTextureRenderer()
        self._mosaic_source_model: MosaicSourceModel | None = None
        self._mosaic_coverage_cache: MosaicCoverageCache | None = None
        self._mosaic_source_texture_cache: MosaicSourceTextureCache | None = None
        self._mosaic_last_drag_pos: QPoint | None = None
        self._mosaic_center_az_deg = 0.0
        self._mosaic_center_alt_deg = 20.0
        self._mosaic_roll_deg = 0.0
        self._init_mosaic_projection_defaults()

    def _init_mosaic_projection_defaults(self) -> None:
        if not hasattr(self.ui, "comboBoxMosaicProjection"):
            return
        self.ui.comboBoxMosaicProjection.setCurrentIndex(0)
        self.ui.doubleSpinBoxMosaicFov.setValue(120.0)
        if hasattr(self.ui, "doubleSpinBoxMosaicAz"):
            self.ui.doubleSpinBoxMosaicAz.setValue(self._mosaic_center_az_deg)
        if hasattr(self.ui, "doubleSpinBoxMosaicAlt"):
            self.ui.doubleSpinBoxMosaicAlt.setValue(self._mosaic_center_alt_deg)
        if hasattr(self.ui, "doubleSpinBoxMosaicRoll"):
            self.ui.doubleSpinBoxMosaicRoll.setValue(self._mosaic_roll_deg)
        self.ui.doubleSpinBoxMosaicMagLimit.setValue(6.5)
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            self.ui.comboBoxMosaicOverlayMode.setCurrentIndex(0)
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            self.ui.doubleSpinBoxMosaicOverlayOpacity.setValue(100.0)
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            self.ui.doubleSpinBoxMosaicCoverageOpacity.setValue(100.0)
        if hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            self.ui.spinBoxMosaicGridPrecision.setValue(MOSAIC_COVERAGE_GRID_LONG_SIDE)
        self._init_mosaic_observer_controls()
        self._update_mosaic_projection_controls()
        self._update_mosaic_grid_precision_tooltip()
        self._set_mosaic_grid_controls_enabled(False)
        self._update_mosaic_model_labels()
        self._update_mosaic_view_label()

    def _init_mosaic_observer_controls(self) -> None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return
        self.ui.dateTimeEditMosaicObservation.setDateTime(QDateTime.currentDateTime())
        utc_offset = datetime.now().astimezone().utcoffset()
        if utc_offset is None:
            utc_offset = timedelta(hours=8)
        self.ui.doubleSpinBoxMosaicUtcOffset.setValue(utc_offset.total_seconds() / 3600.0)
        self.ui.doubleSpinBoxMosaicLatitude.setValue(0.0)
        self.ui.doubleSpinBoxMosaicLongitude.setValue(0.0)
        self.ui.doubleSpinBoxMosaicElevation.setValue(0.0)
        self._set_mosaic_observer_controls_enabled(False)

    def _connect_mosaic_projection_inputs(self) -> None:
        if not hasattr(self.ui, "pushButtonImportMosaicModel"):
            return
        self.ui.pushButtonImportMosaicModel.clicked.connect(self.import_mosaic_model_json)
        self.ui.comboBoxMosaicProjection.currentIndexChanged.connect(self._handle_mosaic_projection_changed)
        self.ui.doubleSpinBoxMosaicFov.valueChanged.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "doubleSpinBoxMosaicAz"):
            self.ui.doubleSpinBoxMosaicAz.valueChanged.connect(self._handle_mosaic_view_controls_changed)
        if hasattr(self.ui, "doubleSpinBoxMosaicAlt"):
            self.ui.doubleSpinBoxMosaicAlt.valueChanged.connect(self._handle_mosaic_view_controls_changed)
        if hasattr(self.ui, "doubleSpinBoxMosaicRoll"):
            self.ui.doubleSpinBoxMosaicRoll.valueChanged.connect(self._handle_mosaic_view_controls_changed)
        self.ui.doubleSpinBoxMosaicMagLimit.valueChanged.connect(self.schedule_mosaic_render)
        self.ui.checkBoxMosaicShowGrid.toggled.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "checkBoxMosaicShowCoverage"):
            self.ui.checkBoxMosaicShowCoverage.toggled.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            self.ui.comboBoxMosaicOverlayMode.currentIndexChanged.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            self.ui.doubleSpinBoxMosaicOverlayOpacity.valueChanged.connect(self.schedule_mosaic_render)
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            self.ui.doubleSpinBoxMosaicCoverageOpacity.valueChanged.connect(self.schedule_mosaic_render)
        if hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            self.ui.spinBoxMosaicGridPrecision.valueChanged.connect(self._update_mosaic_grid_precision_tooltip)
        if hasattr(self.ui, "pushButtonResolveMosaicGrid"):
            self.ui.pushButtonResolveMosaicGrid.clicked.connect(self.resolve_mosaic_grid_again)
        self.ui.pushButtonResetMosaicView.clicked.connect(self.reset_mosaic_projection_view)
        if hasattr(self.ui, "dateTimeEditMosaicObservation"):
            self.ui.dateTimeEditMosaicObservation.dateTimeChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicUtcOffset.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicLatitude.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicLongitude.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.doubleSpinBoxMosaicElevation.valueChanged.connect(self._handle_mosaic_observer_changed)
            self.ui.pushButtonResetMosaicObserver.clicked.connect(self.reset_mosaic_observer_from_json)
        self.ui.labelMosaicModelPath.installEventFilter(self)
        self.ui.labelMosaicSourceImage.installEventFilter(self)
        self.ui.labelMosaicModelInfo.installEventFilter(self)
        self.ui.labelMosaicViewInfo.installEventFilter(self)

    def _handle_mosaic_projection_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self._update_mosaic_projection_controls()
        self.schedule_mosaic_render()

    def _handle_mosaic_view_controls_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self.ui, "doubleSpinBoxMosaicAz"):
            self._mosaic_center_az_deg = float(self.ui.doubleSpinBoxMosaicAz.value()) % 360.0
        if hasattr(self.ui, "doubleSpinBoxMosaicAlt"):
            self._mosaic_center_alt_deg = max(-90.0, min(90.0, float(self.ui.doubleSpinBoxMosaicAlt.value())))
        if hasattr(self.ui, "doubleSpinBoxMosaicRoll"):
            self._mosaic_roll_deg = float(self.ui.doubleSpinBoxMosaicRoll.value())
            while self._mosaic_roll_deg > 180.0:
                self._mosaic_roll_deg -= 360.0
            while self._mosaic_roll_deg < -180.0:
                self._mosaic_roll_deg += 360.0
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _handle_mosaic_observer_changed(self, *unused) -> None:  # type: ignore[no-untyped-def]
        self.schedule_mosaic_render(delay_ms=120)

    def _update_mosaic_projection_controls(self) -> None:
        if not hasattr(self.ui, "doubleSpinBoxMosaicFov"):
            return
        projection_model = self._mosaic_projection_model()
        if projection_model == SKY_MATCHING_MODEL_RECTILINEAR:
            maximum_fov = 160.0
        elif projection_model in (SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT, SKY_MATCHING_MODEL_FISHEYE_EQUISOLID):
            maximum_fov = 300.0
        else:
            maximum_fov = 360.0
        was_blocked = self.ui.doubleSpinBoxMosaicFov.blockSignals(True)
        self.ui.doubleSpinBoxMosaicFov.setMaximum(maximum_fov)
        if self.ui.doubleSpinBoxMosaicFov.value() > maximum_fov:
            self.ui.doubleSpinBoxMosaicFov.setValue(maximum_fov)
        self.ui.doubleSpinBoxMosaicFov.blockSignals(was_blocked)

    def _mosaic_projection_model(self) -> str:
        if not hasattr(self.ui, "comboBoxMosaicProjection"):
            return SKY_MATCHING_MODEL_RECTILINEAR
        index = self.ui.comboBoxMosaicProjection.currentIndex()
        if index < 0 or index >= len(MOSAIC_PROJECTION_MODELS):
            return SKY_MATCHING_MODEL_RECTILINEAR
        return MOSAIC_PROJECTION_MODELS[index]

    def import_mosaic_model_json(self) -> None:
        default_dir = project_root() / "outputs"
        if self._mosaic_source_model is not None:
            default_dir = self._mosaic_source_model.json_path.parent
        elif not default_dir.exists():
            default_dir = project_root()
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入源图模型 JSON",
            str(default_dir),
            SOURCE_MODEL_JSON_FILTER,
        )
        if not file_path:
            return
        self.load_mosaic_model_json(file_path)

    def load_mosaic_model_json(self, file_path: str | Path) -> None:
        json_path = Path(file_path).expanduser()
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            source_model = _load_mosaic_source_model(json_path)
            coverage_cache = self._build_mosaic_coverage_cache(source_model.model)
        except Exception as exc:  # noqa: BLE001 - 导入入口需要把 JSON 和模型错误直接反馈到界面。
            QMessageBox.critical(self, "导入源图模型失败", str(exc))
            self.ui.statusbar.showMessage(f"导入源图模型失败: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._mosaic_source_model = source_model
        self._mosaic_coverage_cache = coverage_cache
        self._mosaic_source_texture_cache = None
        self._set_mosaic_observer_controls_from_source_model(source_model)
        self._set_mosaic_observer_controls_enabled(True)
        self._set_mosaic_grid_controls_enabled(True)
        self._update_mosaic_grid_precision_tooltip()
        self._reset_mosaic_center_from_model()
        self._update_mosaic_model_labels()
        self.schedule_mosaic_render(delay_ms=0)
        self.ui.statusbar.showMessage(f"已导入源图模型: {source_model.json_path}")

    def _bounded_mosaic_utc_offset(self, offset_hours: float) -> float:
        if not hasattr(self.ui, "doubleSpinBoxMosaicUtcOffset"):
            return float(offset_hours)
        return max(
            float(self.ui.doubleSpinBoxMosaicUtcOffset.minimum()),
            min(float(self.ui.doubleSpinBoxMosaicUtcOffset.maximum()), float(offset_hours)),
        )

    def _mosaic_qdatetime_for_observer(self, observer: ObserverSettings, utc_offset_hours: float) -> QDateTime:
        local_timezone = timezone(timedelta(hours=self._bounded_mosaic_utc_offset(utc_offset_hours)))
        local_time = observer.observation_time_utc.astimezone(local_timezone)
        qt_time = QDateTime.fromString(local_time.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss")
        if qt_time.isValid():
            return qt_time
        return QDateTime.currentDateTime()

    def _set_mosaic_observer_controls_enabled(self, enabled: bool) -> None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return
        for control_name in (
            "dateTimeEditMosaicObservation",
            "doubleSpinBoxMosaicUtcOffset",
            "doubleSpinBoxMosaicLatitude",
            "doubleSpinBoxMosaicLongitude",
            "doubleSpinBoxMosaicElevation",
            "pushButtonResetMosaicObserver",
        ):
            getattr(self.ui, control_name).setEnabled(enabled)

    def _set_mosaic_observer_controls_from_source_model(self, source_model: MosaicSourceModel) -> None:
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return
        controls = (
            self.ui.dateTimeEditMosaicObservation,
            self.ui.doubleSpinBoxMosaicUtcOffset,
            self.ui.doubleSpinBoxMosaicLatitude,
            self.ui.doubleSpinBoxMosaicLongitude,
            self.ui.doubleSpinBoxMosaicElevation,
        )
        previous_blocks = [control.blockSignals(True) for control in controls]
        try:
            utc_offset_hours = self._bounded_mosaic_utc_offset(source_model.utc_offset_hours)
            self.ui.dateTimeEditMosaicObservation.setDateTime(
                self._mosaic_qdatetime_for_observer(source_model.observer, utc_offset_hours)
            )
            self.ui.doubleSpinBoxMosaicUtcOffset.setValue(utc_offset_hours)
            self.ui.doubleSpinBoxMosaicLatitude.setValue(float(source_model.observer.latitude_deg))
            self.ui.doubleSpinBoxMosaicLongitude.setValue(float(source_model.observer.longitude_deg))
            self.ui.doubleSpinBoxMosaicElevation.setValue(float(source_model.observer.elevation_m))
        finally:
            for control, was_blocked in zip(controls, previous_blocks, strict=True):
                control.blockSignals(was_blocked)

    def reset_mosaic_observer_from_json(self) -> None:
        source_model = self._mosaic_source_model
        if source_model is None:
            return
        self._set_mosaic_observer_controls_from_source_model(source_model)
        self.schedule_mosaic_render(delay_ms=0)

    def _mosaic_observer_from_controls(self) -> ObserverSettings | None:
        if self._mosaic_source_model is None:
            return None
        if not hasattr(self.ui, "dateTimeEditMosaicObservation"):
            return self._mosaic_source_model.observer
        local_dt = self.ui.dateTimeEditMosaicObservation.dateTime().toPyDateTime()
        offset = timezone(timedelta(hours=float(self.ui.doubleSpinBoxMosaicUtcOffset.value())))
        aware_dt = local_dt.replace(tzinfo=offset)
        return ObserverSettings(
            observation_time_utc=aware_dt.astimezone(timezone.utc),
            latitude_deg=float(self.ui.doubleSpinBoxMosaicLatitude.value()),
            longitude_deg=float(self.ui.doubleSpinBoxMosaicLongitude.value()),
            elevation_m=float(self.ui.doubleSpinBoxMosaicElevation.value()),
        )

    def _mosaic_grid_precision_value(self) -> int:
        if not hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            return MOSAIC_COVERAGE_GRID_LONG_SIDE
        return max(
            MOSAIC_GRID_MIN_PRECISION,
            min(MOSAIC_GRID_MAX_PRECISION, int(self.ui.spinBoxMosaicGridPrecision.value())),
        )

    def _mosaic_grid_shape_for_size(self, width: int, height: int) -> tuple[int, int]:
        precision = self._mosaic_grid_precision_value()
        return grid_shape_for_long_side(width, height, precision, min_minor_cells=3)

    def _update_mosaic_grid_precision_tooltip(self, *unused) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            return
        source_model = self._mosaic_source_model
        if source_model is None:
            precision = self._mosaic_grid_precision_value()
            text = f"下一次导入模型将使用长边 {precision} 点的贴图网格。"
        else:
            rows, columns = self._mosaic_grid_shape_for_size(
                source_model.image_width_px,
                source_model.image_height_px,
            )
            text = f"下一次重新求解将使用 {columns} x {rows} 个源图网格点。"
        self.ui.spinBoxMosaicGridPrecision.setToolTip(text)
        if hasattr(self.ui, "pushButtonResolveMosaicGrid"):
            self.ui.pushButtonResolveMosaicGrid.setToolTip(text)

    def _set_mosaic_grid_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self.ui, "spinBoxMosaicGridPrecision"):
            self.ui.spinBoxMosaicGridPrecision.setEnabled(True)
        if hasattr(self.ui, "pushButtonResolveMosaicGrid"):
            self.ui.pushButtonResolveMosaicGrid.setEnabled(bool(enabled))

    def _build_mosaic_coverage_cache(self, model: FrameAstrometricModel) -> MosaicCoverageCache:
        width = max(1, int(model.image_width_px))
        height = max(1, int(model.image_height_px))
        rows, columns = self._mosaic_grid_shape_for_size(width, height)
        sky_grid = build_pixel_radec_grid(model, width, height, rows, columns)
        return MosaicCoverageCache(
            grid_rows=rows,
            grid_columns=columns,
            grid_x_px=sky_grid.pixel_grid.x_px,
            grid_y_px=sky_grid.pixel_grid.y_px,
            ra_deg=sky_grid.first_deg,
            dec_deg=sky_grid.second_deg,
            valid=sky_grid.valid,
        )

    def resolve_mosaic_grid_again(self) -> None:
        source_model = self._mosaic_source_model
        if source_model is None:
            self.ui.statusbar.showMessage("尚未导入源图模型，无法重新求解自由投影网格。")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._mosaic_coverage_cache = self._build_mosaic_coverage_cache(source_model.model)
        except Exception as exc:  # noqa: BLE001 - 手动重算入口需要把模型错误直接反馈到界面。
            QMessageBox.critical(self, "重新求解自由投影网格失败", str(exc))
            self.ui.statusbar.showMessage(f"重新求解自由投影网格失败: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._update_mosaic_grid_precision_tooltip()
        self.schedule_mosaic_render(delay_ms=0)
        cache = self._mosaic_coverage_cache
        if cache is not None:
            self.ui.statusbar.showMessage(f"已重新求解自由投影网格: {cache.grid_columns} x {cache.grid_rows}")

    def _mosaic_coverage_altaz(
        self,
        cache: MosaicCoverageCache,
        observer: ObserverSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return radec_grid_to_altaz(cache.ra_deg, cache.dec_deg, cache.valid, observer)

    def _reset_mosaic_center_from_model(self) -> None:
        source_model = self._mosaic_source_model
        effective_model, cache, _observer = self._effective_mosaic_model_and_coverage()
        if source_model is None or effective_model is None or cache is None:
            self._mosaic_center_az_deg = 0.0
            self._mosaic_center_alt_deg = 20.0
            self._mosaic_roll_deg = 0.0
            self._update_mosaic_view_label()
            return

        center_point = np.asarray(
            [[source_model.image_width_px * 0.5, source_model.image_height_px * 0.5]],
            dtype=np.float64,
        )
        center_radec = effective_model.pixel_to_sky_points(center_point)
        if np.all(np.isfinite(center_radec)):
            alt_deg, az_deg = compute_altaz_from_radec(center_radec[:, 0], center_radec[:, 1], _observer)
            self._mosaic_center_alt_deg = float(alt_deg[0])
            self._mosaic_center_az_deg = float(az_deg[0]) % 360.0
        else:
            cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, _observer)
            valid_cache = cache_valid & np.isfinite(cache_alt) & np.isfinite(cache_az)
            if np.any(valid_cache):
                vectors = local_vectors_from_altaz(cache_alt[valid_cache], cache_az[valid_cache])
                mean_vector = np.mean(vectors, axis=0)
                norm = float(np.linalg.norm(mean_vector))
                if norm > 1e-12:
                    mean_vector = mean_vector / norm
                    self._mosaic_center_alt_deg = float(np.rad2deg(np.arcsin(np.clip(mean_vector[2], -1.0, 1.0))))
                    self._mosaic_center_az_deg = float(np.rad2deg(np.arctan2(mean_vector[0], mean_vector[1]))) % 360.0
        self._mosaic_roll_deg = 0.0
        self._set_mosaic_fov_from_coverage(cache)
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()

    def _set_mosaic_fov_from_coverage(self, cache: MosaicCoverageCache | None = None) -> None:
        if cache is None:
            cache = self._mosaic_coverage_cache
        if cache is None:
            return
        observer = self._mosaic_observer_from_controls()
        if observer is None:
            source_model = self._mosaic_source_model
            observer = None if source_model is None else source_model.observer
        if observer is None:
            return
        cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, observer)
        valid = cache_valid & np.isfinite(cache_alt) & np.isfinite(cache_az)
        if not np.any(valid):
            return
        center_vector = local_vectors_from_altaz(
            np.asarray([self._mosaic_center_alt_deg], dtype=np.float64),
            np.asarray([self._mosaic_center_az_deg], dtype=np.float64),
        )[0]
        vectors = local_vectors_from_altaz(cache_alt[valid], cache_az[valid])
        dots = np.sum(vectors * center_vector[None, :], axis=1)
        angles_deg = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
        if angles_deg.size == 0:
            return
        suggested_fov = float(np.percentile(angles_deg, 98.0) * 2.35)
        suggested_fov = max(25.0, min(self.ui.doubleSpinBoxMosaicFov.maximum(), suggested_fov))
        was_blocked = self.ui.doubleSpinBoxMosaicFov.blockSignals(True)
        self.ui.doubleSpinBoxMosaicFov.setValue(suggested_fov)
        self.ui.doubleSpinBoxMosaicFov.blockSignals(was_blocked)

    def reset_mosaic_projection_view(self) -> None:
        self._reset_mosaic_center_from_model()
        self.schedule_mosaic_render(delay_ms=0)

    def schedule_mosaic_render(self, *unused, delay_ms: int = 40) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(self, "mosaic_render_timer"):
            return
        self.mosaic_render_timer.start(delay_ms)

    def _mosaic_render_size(self) -> tuple[int, int]:
        size = self.ui.mosaicProjectionView.viewport().size()
        return (
            max(MOSAIC_RENDER_MIN_SIZE_PX, int(size.width())),
            max(MOSAIC_RENDER_MIN_SIZE_PX, int(size.height())),
        )

    def _mosaic_camera_for_render(self, width: int, height: int) -> CameraSettings:
        projection_model = self._mosaic_projection_model()
        fov_deg = max(1.0, float(self.ui.doubleSpinBoxMosaicFov.value()))
        sensor_width_mm = 36.0
        sensor_height_mm = sensor_width_mm * height / max(float(width), 1.0)
        focal_length_mm = 24.0
        if projection_model == SKY_MATCHING_MODEL_RECTILINEAR:
            fov_deg = min(fov_deg, 160.0)
            focal_length_mm = sensor_width_mm / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
        return CameraSettings(
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            image_width_px=width,
            image_height_px=height,
            focal_length_mm=max(focal_length_mm, 0.05),
            lens_model=projection_model,
            fisheye_fov_deg=fov_deg,
        )

    def _mosaic_view_settings(self, camera: CameraSettings | None = None) -> ViewSettings:
        if camera is None:
            width, height = self._mosaic_render_size()
            camera = self._mosaic_camera_for_render(width, height)
        state = ProjectionViewState.from_camera_and_view(
            camera,
            ViewSettings(
                center_az_deg=self._mosaic_center_az_deg,
                center_alt_deg=self._mosaic_center_alt_deg,
                roll_deg=self._mosaic_roll_deg,
            ),
        )
        return state.to_view_settings()

    def _effective_mosaic_model_and_coverage(
        self,
    ) -> tuple[FrameAstrometricModel | None, MosaicCoverageCache | None, ObserverSettings | None]:
        source_model = self._mosaic_source_model
        if source_model is None:
            return None, None, None
        observer = self._mosaic_observer_from_controls() or source_model.observer
        return source_model.model, self._mosaic_coverage_cache, observer

    def _mosaic_overlay_mode(self) -> str:
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            index = self.ui.comboBoxMosaicOverlayMode.currentIndex()
            if 0 <= index < len(MOSAIC_OVERLAY_MODES):
                return MOSAIC_OVERLAY_MODES[index]
        return MOSAIC_OVERLAY_MODE_COVERAGE

    def _mosaic_overlay_opacity(self) -> float:
        if hasattr(self.ui, "doubleSpinBoxMosaicOverlayOpacity"):
            value = self.ui.doubleSpinBoxMosaicOverlayOpacity.value()
        elif hasattr(self.ui, "doubleSpinBoxMosaicCoverageOpacity"):
            value = self.ui.doubleSpinBoxMosaicCoverageOpacity.value()
        else:
            value = 100.0
        return max(0.0, min(1.0, float(value) / 100.0))

    def _mosaic_overlay_enabled(self) -> bool:
        if hasattr(self.ui, "comboBoxMosaicOverlayMode"):
            return True
        if hasattr(self.ui, "checkBoxMosaicShowCoverage"):
            return bool(self.ui.checkBoxMosaicShowCoverage.isChecked())
        return True

    def _set_mosaic_view_controls_from_state(self) -> None:
        control_values = (
            ("doubleSpinBoxMosaicAz", self._mosaic_center_az_deg % 360.0),
            ("doubleSpinBoxMosaicAlt", self._mosaic_center_alt_deg),
            ("doubleSpinBoxMosaicRoll", self._mosaic_roll_deg),
        )
        for control_name, value in control_values:
            if not hasattr(self.ui, control_name):
                continue
            control = getattr(self.ui, control_name)
            was_blocked = control.blockSignals(True)
            control.setValue(float(value))
            control.blockSignals(was_blocked)

    def render_mosaic_projection_now(self) -> None:
        if not hasattr(self.ui, "mosaicProjectionView"):
            return
        width, height = self._mosaic_render_size()
        source_model = self._mosaic_source_model
        effective_model, coverage_cache, observer = self._effective_mosaic_model_and_coverage()
        if source_model is None or effective_model is None or coverage_cache is None or observer is None:
            image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
            image.fill(QColor(0, 0, 0))
            self.mosaic_image_item.set_image(image)
            self.mosaic_scene.setSceneRect(0.0, 0.0, float(width), float(height))
            self.ui.mosaicProjectionView.resetTransform()
            return

        try:
            camera = self._mosaic_camera_for_render(width, height)
            view = self._mosaic_view_settings(camera)
            mag_limit = float(self.ui.doubleSpinBoxMosaicMagLimit.value())
            horizontal_catalog = self._get_horizontal_catalog(observer, mag_limit)
            horizontal_milky_way = self._get_horizontal_milky_way(observer)
            horizontal_solar_system = self._get_horizontal_solar_system(observer)
            image = self._mosaic_sky_preview_renderer.render(
                scene=SkySceneData(
                    horizontal_catalog=horizontal_catalog,
                    horizontal_milky_way=horizontal_milky_way,
                    horizontal_solar_system=horizontal_solar_system,
                ),
                camera=camera,
                view=view,
                visible_mag_limit=mag_limit,
                style=SkyPreviewStyle(
                    draw_common_names=False,
                    number_reference_stars=False,
                    draw_background=True,
                    draw_horizon_shadow=True,
                    draw_grid=self.ui.checkBoxMosaicShowGrid.isChecked(),
                    draw_solar_system_labels=True,
                    draw_direction_labels=self.ui.checkBoxMosaicShowGrid.isChecked(),
                ),
            )
            if self._mosaic_overlay_enabled():
                if self._mosaic_overlay_mode() == MOSAIC_OVERLAY_MODE_SOURCE_IMAGE:
                    self._paint_mosaic_source_image(image, camera, view, coverage_cache, observer, source_model)
                else:
                    self._paint_mosaic_coverage(image, camera, view, coverage_cache, observer)
            self.mosaic_image_item.set_image(image)
            self.mosaic_scene.setSceneRect(0.0, 0.0, float(width), float(height))
            self.ui.mosaicProjectionView.resetTransform()
            self._update_mosaic_view_label()
        except Exception as exc:  # noqa: BLE001 - 预览渲染要把模型和投影错误反馈给用户。
            self.ui.statusbar.showMessage(f"自由投影预览渲染失败: {exc}")

    def _paint_mosaic_coverage(
        self,
        image: QImage,
        camera: CameraSettings,
        view: ViewSettings,
        cache: MosaicCoverageCache | None,
        observer: ObserverSettings,
    ) -> None:
        if cache is None:
            return
        cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, observer)
        screen_grid = project_altaz_grid_to_screen(
            cache_alt,
            cache_az,
            camera=camera,
            view=view,
            valid=cache_valid,
        )
        screen_x = screen_grid.x_px
        screen_y = screen_grid.y_px
        valid = screen_grid.valid
        screen_longitudes = screen_grid.screen_longitudes_rad
        opacity = self._mosaic_overlay_opacity()
        fill_color = QColor(205, 205, 205, int(round(255.0 * opacity)))
        edge_color = QColor(245, 245, 245, int(round(220.0 * min(1.0, opacity + 0.25))))
        max_cell_bbox_area = max(256.0, image.width() * image.height() * 0.35)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill_color)
        for row in range(cache.grid_rows - 1):
            for column in range(cache.grid_columns - 1):
                if not bool(
                    valid[row, column]
                    and valid[row, column + 1]
                    and valid[row + 1, column + 1]
                    and valid[row + 1, column]
                ):
                    continue
                quad = grid_cell_quad(screen_x, screen_y, row, column)
                xs = quad[:, 0]
                ys = quad[:, 1]
                if screen_longitudes is not None and cell_crosses_angle_break(screen_longitudes, row, column):
                    continue
                bbox_area = max(float(np.max(xs) - np.min(xs)), 0.0) * max(float(np.max(ys) - np.min(ys)), 0.0)
                if bbox_area <= 0.05 or bbox_area > max_cell_bbox_area:
                    continue
                polygon = QPolygonF(_expanded_polygon_points(xs, ys, 0.75))
                painter.drawPolygon(polygon)

        painter.setPen(edge_color)
        painter.setBrush(Qt.NoBrush)
        self._paint_mosaic_coverage_boundary(painter, screen_x, screen_y, valid)
        painter.end()

    def _mosaic_source_texture(self, source_model: MosaicSourceModel) -> MosaicSourceTextureCache | None:
        image_path = source_model.source_image_path
        if image_path is None:
            self.ui.statusbar.showMessage("源图模型 JSON 未记录真实图像路径，无法显示原图。")
            return None
        try:
            resolved_path = image_path.expanduser().resolve()
        except OSError:
            resolved_path = image_path.expanduser()
        cache = self._mosaic_source_texture_cache
        if cache is not None and cache.source_image_path == resolved_path:
            return cache
        if not resolved_path.exists():
            self.ui.statusbar.showMessage(f"源图不存在，无法显示原图: {resolved_path}")
            return None
        try:
            preview = load_image_preview(resolved_path, max_long_side_px=MOSAIC_SOURCE_TEXTURE_LONG_SIDE_PX)
            source_rgb = qimage_to_rgb_array(preview.image)
        except Exception as exc:  # noqa: BLE001 - 预览渲染入口只在状态栏提示缺图或解码错误。
            self.ui.statusbar.showMessage(f"读取源图失败，无法显示原图: {exc}")
            return None
        texture_cache = MosaicSourceTextureCache(
            source_image_path=resolved_path,
            source_rgb=source_rgb.astype(np.uint8),
            source_scale_x=source_rgb.shape[1] / max(float(source_model.image_width_px), 1.0),
            source_scale_y=source_rgb.shape[0] / max(float(source_model.image_height_px), 1.0),
            source_width_px=int(source_model.image_width_px),
            source_height_px=int(source_model.image_height_px),
        )
        self._mosaic_source_texture_cache = texture_cache
        return texture_cache

    def _paint_mosaic_source_image(
        self,
        image: QImage,
        camera: CameraSettings,
        view: ViewSettings,
        cache: MosaicCoverageCache | None,
        observer: ObserverSettings,
        source_model: MosaicSourceModel,
    ) -> None:
        if cache is None:
            return
        texture = self._mosaic_source_texture(source_model)
        if texture is None:
            return
        cache_alt, cache_az, cache_valid = self._mosaic_coverage_altaz(cache, observer)
        self._mosaic_texture_renderer.paint_on_qimage(
            image,
            camera=camera,
            view=view,
            source_rgb=texture.source_rgb,
            source_grid_x_px=cache.grid_x_px,
            source_grid_y_px=cache.grid_y_px,
            source_scale_x=texture.source_scale_x,
            source_scale_y=texture.source_scale_y,
            alt_deg=cache_alt,
            az_deg=cache_az,
            valid_points=cache_valid,
            opacity=self._mosaic_overlay_opacity(),
        )

    def _paint_mosaic_coverage_boundary(
        self,
        painter: QPainter,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        edge_indices = [
            [(0, column) for column in range(screen_x.shape[1])],
            [(row, screen_x.shape[1] - 1) for row in range(screen_x.shape[0])],
            [(screen_x.shape[0] - 1, column) for column in range(screen_x.shape[1] - 1, -1, -1)],
            [(row, 0) for row in range(screen_x.shape[0] - 1, -1, -1)],
        ]
        for edge in edge_indices:
            current = QPolygonF()
            previous_x: float | None = None
            previous_y: float | None = None
            for row, column in edge:
                if not bool(valid[row, column]):
                    if len(current) >= 2:
                        painter.drawPolyline(current)
                    current = QPolygonF()
                    previous_x = None
                    previous_y = None
                    continue
                x_value = float(screen_x[row, column])
                y_value = float(screen_y[row, column])
                if previous_x is not None and math.hypot(x_value - previous_x, y_value - previous_y) > 240.0:
                    if len(current) >= 2:
                        painter.drawPolyline(current)
                    current = QPolygonF()
                current.append(QPointF(x_value, y_value))
                previous_x = x_value
                previous_y = y_value
            if len(current) >= 2:
                painter.drawPolyline(current)

    def _handle_mosaic_event_filter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if not hasattr(self.ui, "mosaicProjectionView"):
            return False
        if watched in (
            getattr(self.ui, "labelMosaicModelPath", None),
            getattr(self.ui, "labelMosaicSourceImage", None),
            getattr(self.ui, "labelMosaicModelInfo", None),
            getattr(self.ui, "labelMosaicViewInfo", None),
        ):
            if event.type() in (QEvent.Resize, QEvent.Show):
                QTimer.singleShot(0, lambda label=watched: self._refresh_elided_label(label))
            return False
        if watched in (self.ui.mosaicProjectionView, self.ui.mosaicProjectionView.viewport()):
            if event.type() == QEvent.NativeGesture and self._handle_mosaic_native_fov_zoom(event):
                return True
        if watched is not self.ui.mosaicProjectionView.viewport():
            return False
        if event.type() == QEvent.Resize:
            self.schedule_mosaic_render()
            return False
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._mosaic_last_drag_pos = event.pos()
            self.ui.mosaicProjectionView.viewport().setCursor(Qt.ClosedHandCursor)
            return True
        if event.type() == QEvent.MouseMove and self._mosaic_last_drag_pos is not None:
            dx = event.pos().x() - self._mosaic_last_drag_pos.x()
            dy = event.pos().y() - self._mosaic_last_drag_pos.y()
            self._mosaic_last_drag_pos = event.pos()
            if event.modifiers() & Qt.ControlModifier:
                self._apply_mosaic_roll_drag_delta(dx)
            else:
                self._apply_mosaic_center_drag_delta(dx, dy)
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self._mosaic_last_drag_pos = None
            self.ui.mosaicProjectionView.viewport().unsetCursor()
            self.render_mosaic_projection_now()
            return True
        if event.type() == QEvent.Wheel:
            if not self._mosaic_wheel_zoom_enabled():
                return False
            self._apply_mosaic_fov_wheel(event.angleDelta().y())
            return True
        return False

    def _apply_mosaic_center_drag_delta(self, dx: int, dy: int) -> None:
        width, height = self._mosaic_render_size()
        camera = self._mosaic_camera_for_render(width, height)
        self._mosaic_center_az_deg, self._mosaic_center_alt_deg = sky_center_after_drag(
            center_az_deg=self._mosaic_center_az_deg,
            center_alt_deg=self._mosaic_center_alt_deg,
            dx_px=dx,
            dy_px=dy,
            horizontal_fov_deg=horizontal_fov_deg(camera),
            vertical_fov_deg=vertical_fov_deg(camera),
            viewport_width_px=width,
            viewport_height_px=height,
            min_degrees_per_pixel=0.005,
        )
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _apply_mosaic_roll_drag_delta(self, dx: int) -> None:
        self._mosaic_roll_deg = roll_after_drag(self._mosaic_roll_deg, dx, drag_sign=-1.0)
        self._set_mosaic_view_controls_from_state()
        self._update_mosaic_view_label()
        self.schedule_mosaic_render(delay_ms=10)

    def _apply_mosaic_fov_wheel(self, wheel_delta: int) -> None:
        zoom_factor = wheel_zoom_factor(wheel_delta, MOSAIC_ZOOM_FACTOR)
        if zoom_factor is None:
            return
        self._apply_mosaic_fov_zoom_factor(zoom_factor)
        self.render_mosaic_projection_now()

    def _apply_mosaic_fov_zoom_factor(self, zoom_factor: float) -> None:
        if not np.isfinite(zoom_factor) or zoom_factor <= 0.0 or abs(zoom_factor - 1.0) <= 1e-4:
            return
        current = float(self.ui.doubleSpinBoxMosaicFov.value())
        target = clamp_fov(
            current / zoom_factor,
            self.ui.doubleSpinBoxMosaicFov.minimum(),
            self.ui.doubleSpinBoxMosaicFov.maximum(),
        )
        self.ui.doubleSpinBoxMosaicFov.setValue(target)

    def _mosaic_wheel_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "wheel_zoom_enabled", True))

    def _mosaic_touchpad_pinch_zoom_enabled(self) -> bool:
        return bool(getattr(self.ui_config, "touchpad_pinch_zoom_enabled", True))

    def _mosaic_zoom_policy(self) -> ViewZoomPolicy:
        maximum = self.ui.doubleSpinBoxMosaicFov.maximum() if hasattr(self.ui, "doubleSpinBoxMosaicFov") else 360.0
        minimum = self.ui.doubleSpinBoxMosaicFov.minimum() if hasattr(self.ui, "doubleSpinBoxMosaicFov") else 1.0
        return ViewZoomPolicy(
            wheel_enabled=self._mosaic_wheel_zoom_enabled(),
            pinch_enabled=self._mosaic_touchpad_pinch_zoom_enabled(),
            min_fov=float(minimum),
            max_fov=float(maximum),
        )

    def _handle_mosaic_native_fov_zoom(self, event) -> bool:  # type: ignore[no-untyped-def]
        zoom_factor = native_gesture_zoom_factor(event, self._mosaic_zoom_policy())
        if zoom_factor is None:
            return False
        self._apply_mosaic_fov_zoom_factor(zoom_factor)
        self.render_mosaic_projection_now()
        return True

    def _update_mosaic_model_labels(self) -> None:
        if not hasattr(self.ui, "labelMosaicModelPath"):
            return
        source_model = self._mosaic_source_model
        if source_model is None:
            self._set_elided_label_text(self.ui.labelMosaicModelPath, "未导入")
            self._set_elided_label_text(self.ui.labelMosaicSourceImage, "未导入")
            self._set_elided_label_text(self.ui.labelMosaicModelInfo, "-")
            return
        self._set_elided_label_text(
            self.ui.labelMosaicModelPath,
            source_model.json_path.name,
            str(source_model.json_path),
        )
        source_tooltip = str(source_model.source_image_path) if source_model.source_image_path is not None else ""
        self._set_elided_label_text(self.ui.labelMosaicSourceImage, source_model.source_image_text, source_tooltip)
        rms_text = f"{source_model.rms_px:.2f}px" if np.isfinite(source_model.rms_px) else "-"
        info_text = (
            f"{source_model.image_width_px} x {source_model.image_height_px} px  "
            f"星点 {source_model.pair_count}  RMS {rms_text}"
        )
        self._set_elided_label_text(self.ui.labelMosaicModelInfo, info_text)

    def _update_mosaic_view_label(self) -> None:
        if not hasattr(self.ui, "labelMosaicViewInfo"):
            return
        projection_name = SKY_KNOWN_PROJECTION_DISPLAY_NAMES.get(self._mosaic_projection_model(), self._mosaic_projection_model())
        text = (
            f"{projection_name}  Az {self._mosaic_center_az_deg:.2f} deg  "
            f"Alt {self._mosaic_center_alt_deg:.2f} deg  Roll {self._mosaic_roll_deg:.2f} deg"
        )
        self._set_elided_label_text(self.ui.labelMosaicViewInfo, text)

__all__ = [name for name in globals() if not name.startswith("__")]
