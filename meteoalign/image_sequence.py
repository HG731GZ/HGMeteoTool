from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

from .image_preview import SUPPORTED_IMAGE_SUFFIXES


EXIF_IFD_POINTER = 34665
EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME_DIGITIZED = 36868
EXIF_SUBSEC_TIME_ORIGINAL = 37521
EXIF_SUBSEC_TIME_DIGITIZED = 37522
EXIF_OFFSET_TIME_ORIGINAL = 36881
EXIF_OFFSET_TIME_DIGITIZED = 36882
TIFF_XML_PACKET = 700


@dataclass(frozen=True)
class ImageSequenceItem:
    """序列图像条目，capture_datetime 保留 EXIF 原始时区语义。"""

    path: Path
    capture_datetime: datetime
    capture_datetime_utc: datetime | None
    capture_time_source: str


@dataclass(frozen=True)
class RejectedSequenceImage:
    """导入序列时被跳过的图像及原因。"""

    path: Path
    reason: str


_DATETIME_RE = re.compile(
    r"^\s*(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$"
)
_OFFSET_RE = re.compile(r"^\s*([+\-])(\d{2})(?::?(\d{2}))?\s*$")
_XMP_ATTR_RE = re.compile(
    r"(?:exif:DateTimeOriginal|exif:DateTimeDigitized|xmp:CreateDate|photoshop:DateCreated)"
    r"\s*=\s*([\"'])(.*?)\1",
    re.DOTALL,
)
_XMP_ELEMENT_RE = re.compile(
    r"<(?:exif:DateTimeOriginal|exif:DateTimeDigitized|xmp:CreateDate|photoshop:DateCreated)"
    r"\b[^>]*>(.*?)</(?:exif:DateTimeOriginal|exif:DateTimeDigitized|xmp:CreateDate|photoshop:DateCreated)>",
    re.DOTALL,
)


def _exif_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00").strip()
    return str(value).strip("\x00").strip()


def _parse_exif_datetime(value: object, subsecond: object | None = None) -> datetime:
    text = _exif_text(value)
    match = _DATETIME_RE.match(text)
    if match is None:
        raise ValueError(f"EXIF 时间格式无法识别：{text!r}")
    year, month, day, hour, minute, second, fractional = match.groups()
    microsecond = 0
    if subsecond is not None:
        fractional = _exif_text(subsecond)
    if fractional:
        digits = "".join(ch for ch in fractional if ch.isdigit())
        if digits:
            microsecond = int((digits + "000000")[:6])
    return datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(second),
        microsecond,
    )


def _parse_xmp_datetime(value: object) -> datetime:
    text = _exif_text(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return _parse_exif_datetime(text)


def _parse_exif_offset(value: object | None) -> timezone | None:
    if value is None:
        return None
    match = _OFFSET_RE.match(_exif_text(value))
    if match is None:
        return None
    sign_text, hours_text, minutes_text = match.groups()
    sign = -1 if sign_text == "-" else 1
    hours = int(hours_text)
    minutes = int(minutes_text or "0")
    if hours > 23 or minutes > 59:
        return None
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _exif_ifd_groups(exif: object) -> list[object]:
    groups: list[object] = [exif]
    get_ifd = getattr(exif, "get_ifd", None)
    if callable(get_ifd):
        try:
            exif_ifd = get_ifd(EXIF_IFD_POINTER)
        except Exception:  # noqa: BLE001 - 部分 TIFF 的 IFD 指针不规范，忽略后继续尝试 XMP。
            exif_ifd = None
        if exif_ifd:
            groups.insert(0, exif_ifd)
    return groups


def _group_get(group: object, tag: int) -> object | None:
    get_value = getattr(group, "get", None)
    if not callable(get_value):
        return None
    return get_value(tag)


def _capture_time_from_exif_groups(groups: list[object]) -> tuple[datetime, str] | None:
    # DateTimeOriginal 才是真正的拍摄时间；DateTime 是文件/软件修改时间，不能拿来兜底。
    candidates = (
        (EXIF_DATETIME_ORIGINAL, EXIF_SUBSEC_TIME_ORIGINAL, EXIF_OFFSET_TIME_ORIGINAL, "EXIF DateTimeOriginal"),
        (EXIF_DATETIME_DIGITIZED, EXIF_SUBSEC_TIME_DIGITIZED, EXIF_OFFSET_TIME_DIGITIZED, "EXIF DateTimeDigitized"),
    )
    for group in groups:
        for datetime_tag, subsec_tag, offset_tag, source_name in candidates:
            raw_datetime = _group_get(group, datetime_tag)
            if raw_datetime is None:
                continue
            parsed = _parse_exif_datetime(raw_datetime, _group_get(group, subsec_tag))
            offset = _parse_exif_offset(_group_get(group, offset_tag))
            if offset is not None:
                return parsed.replace(tzinfo=offset), f"{source_name}+OffsetTime"
            return parsed, source_name
    return None


def _xmp_text_from_payload(payload: object) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="ignore").strip("\x00")
    return str(payload).strip("\x00")


def _xmp_capture_datetime_values(xmp_text: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    try:
        root = ElementTree.fromstring(xmp_text.encode("utf-8"))
    except ElementTree.ParseError:
        root = None
    if root is not None:
        wanted_names = {
            "{http://ns.adobe.com/exif/1.0/}DateTimeOriginal": "XMP exif:DateTimeOriginal",
            "{http://ns.adobe.com/exif/1.0/}DateTimeDigitized": "XMP exif:DateTimeDigitized",
            "{http://ns.adobe.com/xap/1.0/}CreateDate": "XMP xmp:CreateDate",
            "{http://ns.adobe.com/photoshop/1.0/}DateCreated": "XMP photoshop:DateCreated",
        }
        for element in root.iter():
            for attr_name, source_name in wanted_names.items():
                raw_value = element.attrib.get(attr_name)
                if raw_value:
                    values.append((source_name, raw_value))
            source_name = wanted_names.get(element.tag)
            if source_name and element.text:
                values.append((source_name, element.text))
    if not values:
        for match in _XMP_ATTR_RE.finditer(xmp_text):
            source_token = match.group(0).split("=", 1)[0].strip()
            values.append((f"XMP {source_token}", match.group(2)))
        for match in _XMP_ELEMENT_RE.finditer(xmp_text):
            tag_name = match.group(0).split(">", 1)[0].lstrip("<").split()[0]
            values.append((f"XMP {tag_name}", match.group(1)))
    return values


def _capture_time_from_xmp_payloads(payloads: list[object]) -> tuple[datetime, str] | None:
    priority = (
        "DateTimeOriginal",
        "DateTimeDigitized",
        "CreateDate",
        "DateCreated",
    )
    collected: list[tuple[int, str, datetime]] = []
    for payload in payloads:
        xmp_text = _xmp_text_from_payload(payload)
        if not xmp_text.strip():
            continue
        for source_name, raw_value in _xmp_capture_datetime_values(xmp_text):
            try:
                parsed = _parse_xmp_datetime(raw_value)
            except ValueError:
                continue
            rank = min(
                (index for index, token in enumerate(priority) if token in source_name),
                default=len(priority),
            )
            collected.append((rank, source_name, parsed))
    if not collected:
        return None
    _rank, source_name, parsed = min(collected, key=lambda item: item[0])
    return parsed, source_name


def _sequence_item_from_datetime(path: Path, parsed: datetime, source_name: str) -> ImageSequenceItem:
    if parsed.tzinfo is None:
        return ImageSequenceItem(
            path=path.resolve(),
            capture_datetime=parsed,
            capture_datetime_utc=None,
            capture_time_source=source_name,
        )
    aware = parsed.astimezone(parsed.tzinfo)
    return ImageSequenceItem(
        path=path.resolve(),
        capture_datetime=aware,
        capture_datetime_utc=aware.astimezone(timezone.utc),
        capture_time_source=source_name,
    )


def read_image_capture_time(path: str | Path) -> ImageSequenceItem:
    """从图像 EXIF 中读取拍摄时间，不退回到文件系统时间。"""

    image_path = Path(path).expanduser()
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("当前只支持 TIFF、JPG 与 PNG 图像。")
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在：{image_path}")

    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            exif_result = None
            if exif:
                exif_result = _capture_time_from_exif_groups(_exif_ifd_groups(exif))
            xmp_payloads = []
            if exif and exif.get(TIFF_XML_PACKET) is not None:
                xmp_payloads.append(exif.get(TIFF_XML_PACKET))
            for info_key in ("XMLPacket", "xmp", "XMP"):
                if image.info.get(info_key) is not None:
                    xmp_payloads.append(image.info[info_key])
    except UnidentifiedImageError as exc:
        raise ValueError("无法识别图像文件。") from exc
    except OSError as exc:
        raise ValueError(f"无法读取图像 EXIF：{exc}") from exc

    if exif_result is not None:
        parsed, source_name = exif_result
        return _sequence_item_from_datetime(image_path, parsed, source_name)

    xmp_result = _capture_time_from_xmp_payloads(xmp_payloads)
    if xmp_result is not None:
        parsed, source_name = xmp_result
        return _sequence_item_from_datetime(image_path, parsed, source_name)

    raise ValueError("EXIF/XMP 中没有可用的原始拍摄时间字段。")


def sequence_sort_key(item: ImageSequenceItem) -> datetime:
    if item.capture_datetime_utc is not None:
        return item.capture_datetime_utc.replace(tzinfo=None)
    return item.capture_datetime.replace(tzinfo=None)


def sequence_item_time_delta_seconds(item: ImageSequenceItem, first_item: ImageSequenceItem) -> float:
    if item.capture_datetime_utc is not None and first_item.capture_datetime_utc is not None:
        return (item.capture_datetime_utc - first_item.capture_datetime_utc).total_seconds()
    return (item.capture_datetime.replace(tzinfo=None) - first_item.capture_datetime.replace(tzinfo=None)).total_seconds()


def sequence_item_local_datetime(item: ImageSequenceItem, utc_offset_hours: float) -> datetime:
    """转换为界面 QDateTimeEdit 使用的本地无时区时间。"""

    if item.capture_datetime_utc is None:
        return item.capture_datetime.replace(tzinfo=None)
    offset = timezone(timedelta(hours=float(utc_offset_hours)))
    return item.capture_datetime_utc.astimezone(offset).replace(tzinfo=None)


def sequence_item_observation_time_utc(item: ImageSequenceItem, utc_offset_hours: float) -> datetime:
    """转换为星空计算使用的 UTC 时间。"""

    if item.capture_datetime_utc is not None:
        return item.capture_datetime_utc
    offset = timezone(timedelta(hours=float(utc_offset_hours)))
    return item.capture_datetime.replace(tzinfo=offset).astimezone(timezone.utc)


def collect_image_sequence(paths: list[str] | tuple[str, ...]) -> tuple[list[ImageSequenceItem], list[RejectedSequenceImage]]:
    """批量读取序列 EXIF 时间，并按拍摄时间排序。"""

    items: list[ImageSequenceItem] = []
    rejected: list[RejectedSequenceImage] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        image_path = Path(raw_path).expanduser()
        try:
            resolved_path = image_path.resolve()
        except OSError:
            resolved_path = image_path
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        try:
            items.append(read_image_capture_time(resolved_path))
        except Exception as exc:  # noqa: BLE001 - 逐文件汇总给界面层统一提示。
            rejected.append(RejectedSequenceImage(path=resolved_path, reason=str(exc)))

    items.sort(key=sequence_sort_key)
    return items, rejected
