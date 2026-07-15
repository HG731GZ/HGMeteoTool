from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QHeaderView, QTableWidgetItem, QWidget

from ..meteor_showers import METEOR_SHOWER_SPECS
from ..ui.ui_meteor_shower_selection_dialog import Ui_MeteorShowerSelectionDialog


class MeteorShowerSelectionDialog(QDialog):
    """显示全年流星雨，并返回用户勾选的项目编号。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        selected_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.ui = Ui_MeteorShowerSelectionDialog()
        self.ui.setupUi(self)
        self._selected_ids = tuple(selected_ids)
        self._populate_table()
        self.ui.pushButtonSelectAllMeteorShowers.clicked.connect(lambda: self._set_all_checked(True))
        self.ui.pushButtonClearMeteorShowers.clicked.connect(lambda: self._set_all_checked(False))

    def _populate_table(self) -> None:
        specs = METEOR_SHOWER_SPECS
        table = self.ui.tableWidgetMeteorShowers
        table.setRowCount(len(specs))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for column in range(2, 6):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        selected = set(self._selected_ids)
        for row, spec in enumerate(specs):
            check_item = QTableWidgetItem()
            check_item.setData(Qt.UserRole, spec.shower_id)
            check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            check_item.setCheckState(Qt.Checked if spec.shower_id in selected else Qt.Unchecked)
            table.setItem(row, 0, check_item)
            activity_text = (
                f"{spec.activity_start[0]:02d}-{spec.activity_start[1]:02d} ～ "
                f"{spec.activity_end[0]:02d}-{spec.activity_end[1]:02d}"
            )
            radiant_text = f"RA {spec.radiant_ra_deg:g}° / Dec {spec.radiant_dec_deg:+g}°"
            zhr_text = str(spec.zhr) if spec.zhr is not None else "可变"
            values = (spec.display_name, activity_text, zhr_text, radiant_text, spec.color_hex)
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if column == 3 and spec.zhr is None:
                    item.setToolTip("IMO 将峰值 ZHR 标为可变；没有固定数值时本版本不生成流星。")
                if column == 5:
                    item.setBackground(QColor(spec.color_hex))
                    item.setForeground(QColor(20, 20, 20))
                    item.setToolTip(spec.color_note)
                table.setItem(row, column, item)
        self.ui.labelActiveDate.setText(f"全年流星雨：{len(specs)} 个，不根据日期隐藏项目。")

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.ui.tableWidgetMeteorShowers.rowCount()):
            self.ui.tableWidgetMeteorShowers.item(row, 0).setCheckState(state)

    def selected_ids(self) -> tuple[str, ...]:
        result: list[str] = []
        for row in range(self.ui.tableWidgetMeteorShowers.rowCount()):
            item = self.ui.tableWidgetMeteorShowers.item(row, 0)
            if item.checkState() == Qt.Checked:
                result.append(str(item.data(Qt.UserRole)))
        return tuple(result)


__all__ = ["MeteorShowerSelectionDialog"]
