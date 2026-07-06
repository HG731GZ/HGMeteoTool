from __future__ import annotations

from PyQt5.QtCore import Qt

from .alignment import (
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_POLYNOMIAL,
    SKY_MATCHING_MODEL_RECTILINEAR,
)
from .simulator import FISHEYE_EQUISOLID, RECTILINEAR_LENS_MODEL

# ---------------------------------------------------------------------------
# 镜头与配准模型选项
# ---------------------------------------------------------------------------

LENS_MODELS = (
    RECTILINEAR_LENS_MODEL,
    FISHEYE_EQUISOLID,
)
SKY_ALIGNMENT_MODELS = (
    SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
    SKY_MATCHING_MODEL_RECTILINEAR,
    SKY_MATCHING_MODEL_FISHEYE_EQUIDISTANT,
    SKY_MATCHING_MODEL_FISHEYE_EQUISOLID,
    SKY_MATCHING_MODEL_MERCATOR,
    SKY_MATCHING_MODEL_CYLINDRICAL_EQUIDISTANT,
)
SKY_ALIGNMENT_MODEL_ALIASES = {
    SKY_MATCHING_MODEL_POLYNOMIAL: SKY_MATCHING_MODEL_ANCHOR_INTERPOLATION,
}

# ---------------------------------------------------------------------------
# 参考星标注模式
# ---------------------------------------------------------------------------

REFERENCE_LABEL_MODE_FIXED_COUNT = "fixed_count"
REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT = "fixed_mag_limit"
REFERENCE_LABEL_MODES = (
    REFERENCE_LABEL_MODE_FIXED_COUNT,
    REFERENCE_LABEL_MODE_FIXED_MAG_LIMIT,
)

# ---------------------------------------------------------------------------
# 预览与视图缩放
# ---------------------------------------------------------------------------

PREVIEW_LONG_SIDE_PX = 1920
REAL_IMAGE_MAX_ZOOM_SCALE = 2.0
IMAGE_VIEW_ZOOM_IN_FACTOR = 1.25
IMAGE_VIEW_ZOOM_OUT_FACTOR = 0.8
TOUCHPAD_ZOOM_SENSITIVITY = 1.6
TOUCHPAD_ZOOM_MIN_FACTOR = 0.55
TOUCHPAD_ZOOM_MAX_FACTOR = 1.8
STAR_PICK_TOUCHPAD_STEPS_PER_ZOOM_UNIT = 12.0
STAR_PICK_CIRCLE_STEP_PX = 10

# ---------------------------------------------------------------------------
# PSF 与自动配对
# ---------------------------------------------------------------------------

MIN_PSF_RADIUS_PX = 4
AUTO_PAIR_MAX_SEARCH_RADIUS_PX = 120
AUTO_PAIR_RMS_RADIUS_SCALE = 3.0
REFERENCE_STAR_PICK_SCREEN_RADIUS_PX = 32

# ---------------------------------------------------------------------------
# 星对表格列索引与角色
# ---------------------------------------------------------------------------

STAR_PAIR_INDEX_COLUMN = 0
STAR_PAIR_NAME_COLUMN = 1
STAR_PAIR_POSITION_COLUMN = 2
STAR_PAIR_RESIDUAL_COLUMN = 3
STAR_PAIR_RESIDUAL_WIDTH_SAMPLE = "999.99"
STAR_PAIR_SORT_KEY_INDEX = "index"
STAR_PAIR_SORT_KEY_RESIDUAL = "residual"
STAR_PAIR_SORTABLE_COLUMNS = {
    STAR_PAIR_INDEX_COLUMN: STAR_PAIR_SORT_KEY_INDEX,
    STAR_PAIR_RESIDUAL_COLUMN: STAR_PAIR_SORT_KEY_RESIDUAL,
}
STAR_PAIR_ROW_TYPE_ROLE = Qt.UserRole + 1
STAR_PAIR_FIT_ROLE = Qt.UserRole + 2
STAR_PAIR_CONSTRAINT_MODE_ROLE = Qt.UserRole + 3
STAR_PAIR_FIT_WEIGHT_ROLE = Qt.UserRole + 4
STAR_PAIR_POSITION_ROLE = Qt.UserRole + 5
STAR_PAIR_AUTO_GROUP_ROLE = Qt.UserRole + 6
STAR_PAIR_ROW_TYPE_MANUAL = "manual"
STAR_PAIR_ROW_TYPE_MANUAL_GROUP = "manual_group"
STAR_PAIR_ROW_TYPE_AUTO_GROUP = "auto_match_group"
STAR_PAIR_ROW_TYPE_AUTO_MATCH = "auto_match"
STAR_PAIR_MANUAL_GROUP_LABEL = "手动匹配"

# ---------------------------------------------------------------------------
# JSON 会话格式
# ---------------------------------------------------------------------------

STAR_PAIR_SESSION_FORMAT = "meteoalign_star_pair_session"
STAR_PAIR_SESSION_VERSION = 1
STAR_PAIR_SESSION_JSON_FILTER = "MeteoAlign 星点配对 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"
SOURCE_MODEL_JSON_FILTER = "MeteoAlign 源图映射 JSON (*.json);;JSON 文件 (*.json);;所有文件 (*)"

# ---------------------------------------------------------------------------
# 配准状态与残差
# ---------------------------------------------------------------------------

ALIGNMENT_STATUS_MAX_CHARS = 68
RESIDUAL_WARNING_MIN_PX = 25.0
RESIDUAL_SEVERE_MIN_PX = 50.0
RESIDUAL_SEVERE_RMS_SCALE = 2.0

# ---------------------------------------------------------------------------
# 星点渲染
# ---------------------------------------------------------------------------

STAR_RADIUS_ZOOM_EXPONENT = 0.32
STAR_RADIUS_MIN_ZOOM_SCALE = 0.48

# ---------------------------------------------------------------------------
# 自动匹配参数
# ---------------------------------------------------------------------------

AUTO_MATCH_SEARCH_MAG_LIMIT = 8.0
AUTO_MATCH_CONSTRAINT_ANCHOR = "anchor"
AUTO_MATCH_CONSTRAINT_SOFT = "soft"
AUTO_MATCH_CONSTRAINT_MODES = (
    AUTO_MATCH_CONSTRAINT_ANCHOR,
    AUTO_MATCH_CONSTRAINT_SOFT,
)
AUTO_MATCH_DEFAULT_SOFT_WEIGHT = 0.3
AUTO_MATCH_MIN_AMPLITUDE = 2.0
AUTO_MATCH_DUPLICATE_MIN_DISTANCE_PX = 4.0
AUTO_MATCH_ANNOTATION_LIMIT = 250

# ---------------------------------------------------------------------------
# 星点标注
# ---------------------------------------------------------------------------

STAR_ANNOTATION_PSF_SIGMA_SCALE = 3.0
STAR_ANNOTATION_MIN_RADIUS_PX = 5.0
STAR_ANNOTATION_FALLBACK_RADIUS_PX = 8.0
STAR_ANNOTATION_MAX_RADIUS_PX = 80.0

# ---------------------------------------------------------------------------
# 聚焦与双击
# ---------------------------------------------------------------------------

STAR_PAIR_FOCUS_MIN_MATCHED_COUNT = 4
STAR_PAIR_FOCUS_ZOOM_FIT_SCALE = 8.0
STAR_PAIR_FOCUS_MARKER_RADIUS_PX = 24.0
