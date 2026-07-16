from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterator

from PyQt5.QtCore import QObject, pyqtSignal

from .simulator import ObserverSettings
from .star_pair_model import (
    CONSTRAINT_ANCHOR,
    CONSTRAINT_SOFT,
    PAIR_ORIGIN_AUTO_MATCH,
    PAIR_ORIGIN_MANUAL,
    PsfFit,
    StarPairRecord,
    star_pair_records_from_payloads,
)


class StarPairStore(QObject):
    """星点匹配数据的统一存储，作为整个应用的唯一权威数据源。

    所有星点匹配的增删改查、约束设置和分组管理都通过本 Store 进行。
    UI 层（表格、标注）只负责显示，不再直接持有业务数据。

    信号
    ----
    records_changed :
        任何记录增删改后发出，UI 可监听此信号刷新显示。
    record_updated(str) :
        单条记录更新时发出，参数为 star_id。
    store_cleared :
        清空全部记录时发出。
    """

    records_changed = pyqtSignal()
    record_updated = pyqtSignal(str)
    store_cleared = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._records: dict[str, StarPairRecord] = {}

    # ------------------------------------------------------------------
    # 基本 CRUD
    # ------------------------------------------------------------------

    def add(self, record: StarPairRecord) -> None:
        """添加或替换一条星点匹配记录，以 star_id 为键。"""
        star_id = record.star_id
        if not star_id:
            return
        self._records[star_id] = record
        self.records_changed.emit()
        self.record_updated.emit(star_id)

    def remove(self, star_id: str) -> bool:
        """删除一条记录，返回是否确实删除了数据。"""
        if star_id in self._records:
            del self._records[star_id]
            self.records_changed.emit()
            return True
        return False

    def get(self, star_id: str) -> StarPairRecord | None:
        """按 star_id 获取记录。"""
        return self._records.get(star_id)

    def clear(self) -> None:
        """清空全部记录。"""
        self._records.clear()
        self.records_changed.emit()
        self.store_cleared.emit()

    def __contains__(self, star_id: str) -> bool:
        return star_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[StarPairRecord]:
        return iter(self._records.values())

    @property
    def star_ids(self) -> list[str]:
        """返回当前所有记录的 star_id 列表。"""
        return list(self._records.keys())

    # ------------------------------------------------------------------
    # 字段级更新
    # ------------------------------------------------------------------

    def _replace_record(
        self,
        star_id: str,
        *,
        emit_records_changed: bool = True,
        **changes: object,
    ) -> StarPairRecord | None:
        """用 dataclass replace 统一更新单条记录并发出信号。"""

        record = self._records.get(star_id)
        if record is None:
            return None
        new_record = replace(record, **changes)
        self._records[star_id] = new_record
        if emit_records_changed:
            self.records_changed.emit()
        self.record_updated.emit(star_id)
        return new_record

    def update_position(
        self,
        star_id: str,
        image_x: float,
        image_y: float,
        psf: PsfFit | None = None,
    ) -> StarPairRecord | None:
        """更新星点的图像坐标和可选的 PSF 拟合结果。

        返回更新后的记录；若 star_id 不存在则返回 None。
        """
        return self._replace_record(
            star_id,
            image_x_px=float(image_x),
            image_y_px=float(image_y),
            psf=psf,
        )

    def update_psf(self, star_id: str, psf: PsfFit) -> StarPairRecord | None:
        """仅更新 PSF 拟合结果。"""
        return self._replace_record(star_id, psf=psf)

    def set_constraint(
        self,
        star_id: str,
        mode: str,
        weight: float = 1.0,
    ) -> StarPairRecord | None:
        """设置拟合约束模式和权重。"""
        if mode not in (CONSTRAINT_ANCHOR, CONSTRAINT_SOFT):
            mode = CONSTRAINT_ANCHOR
        if mode == CONSTRAINT_SOFT:
            weight = max(0.01, min(1.0, float(weight)))
        else:
            weight = 1.0
        return self._replace_record(
            star_id,
            fit_constraint_mode=mode,
            fit_weight=float(weight),
        )

    def set_group(
        self,
        star_id: str,
        group_id: str | None,
        group_name: str | None = None,
    ) -> StarPairRecord | None:
        """设置自动匹配分组。"""
        return self._replace_record(
            star_id,
            group_id=group_id,
            group_name=group_name,
        )

    def set_enabled(self, star_id: str, enabled: bool) -> StarPairRecord | None:
        """启用或禁用某条记录。"""
        return self._replace_record(star_id, enabled=bool(enabled))

    def set_residual(
        self,
        star_id: str,
        dx: float | None,
        dy: float | None,
        distance: float | None,
    ) -> StarPairRecord | None:
        """设置残差信息。"""
        return self._replace_record(
            star_id,
            emit_records_changed=False,
            residual_dx_px=dx,
            residual_dy_px=dy,
            residual_px=distance,
        )

    def set_pair_origin(self, star_id: str, origin: str) -> StarPairRecord | None:
        """设置匹配来源（手动/自动匹配）。"""
        if origin not in (PAIR_ORIGIN_MANUAL, PAIR_ORIGIN_AUTO_MATCH):
            origin = PAIR_ORIGIN_MANUAL
        return self._replace_record(star_id, pair_origin=origin)

    # ------------------------------------------------------------------
    # 批量操作
    # ------------------------------------------------------------------

    def add_records(self, records: list[StarPairRecord]) -> int:
        """批量添加记录，返回实际添加数量。"""
        count = 0
        for record in records:
            star_id = record.star_id
            if not star_id:
                continue
            self._records[star_id] = record
            count += 1
        if count > 0:
            self.records_changed.emit()
        return count

    def restore_from_payloads(
        self,
        payloads: list[object],
        observer: ObserverSettings | None = None,
    ) -> list[StarPairRecord]:
        """从 JSON payload 列表恢复记录并存入 Store。返回恢复的记录列表。"""
        records = star_pair_records_from_payloads(payloads, observer=observer)
        self.add_records(records)
        return records

    def remove_records(self, star_ids: set[str]) -> int:
        """批量删除记录，返回实际删除数量。"""
        count = 0
        for star_id in star_ids:
            if star_id in self._records:
                del self._records[star_id]
                count += 1
        if count > 0:
            self.records_changed.emit()
        return count

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def snapshot(self) -> list[StarPairRecord]:
        """返回当前所有记录的快照列表，供导出和对齐使用。"""
        return list(self._records.values())

    def valid_fit_records(self) -> list[StarPairRecord]:
        """返回所有可用于拟合的有效记录。"""
        return [record for record in self._records.values() if record.is_valid_for_fit()]

    def positions(self) -> dict[str, tuple[float, float]]:
        """返回 star_id -> (image_x, image_y) 的映射。"""
        return {
            star_id: record.position
            for star_id, record in self._records.items()
            if all(math.isfinite(v) for v in record.position)
        }

    def auto_match_records(self) -> list[StarPairRecord]:
        """返回所有自动匹配来源的记录。"""
        return [r for r in self._records.values() if r.is_auto_match]

    def manual_records(self) -> list[StarPairRecord]:
        """返回所有手动匹配来源的记录。"""
        return [r for r in self._records.values() if not r.is_auto_match]

    def records_to_json_payloads(self) -> list[dict[str, object]]:
        """将所有记录序列化为 JSON payload 列表。"""
        return [record.to_json_payload() for record in self._records.values()]

    def records_by_group(self, group_id: str) -> list[StarPairRecord]:
        """返回指定分组的记录。"""
        return [r for r in self._records.values() if r.group_id == group_id]

    def matched_count(self) -> int:
        """返回已有有效图像位置的记录数。"""
        return sum(
            1
            for r in self._records.values()
            if all(math.isfinite(v) for v in (r.image_x_px, r.image_y_px))
        )

__all__ = ["StarPairStore"]
