"""流星自动检测的配置、引擎定位与协议数据。"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .preference_manager import ensure_preference_file, update_preference_values
from .runtime_paths import frozen_app_sibling_dir, frozen_resource_roots, is_frozen_app, source_project_root


METEOR_DETECTION_PREFERENCE_PREFIX = "meteor_detection_"
METEOR_DETECTION_PROVIDERS = ("auto", "cpu", "dml", "cuda", "coreml")


@dataclass(frozen=True)
class MeteorDetectionOptions:
    """传给 MetDet worker 的用户可调参数。"""

    engine_path: str = ""
    model_path: str = ""
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    multiscale: int = 2
    partition: int = 2
    provider: str = "auto"
    box_expansion_ratio: float = 0.10

    def worker_options(self) -> dict[str, object]:
        """返回 worker 协议中的 options 对象。"""

        options: dict[str, object] = {
            "confidence_threshold": float(self.confidence_threshold),
            "nms_threshold": float(self.nms_threshold),
            "multiscale": int(self.multiscale),
            "partition": int(self.partition),
            "provider": self.provider,
            "box_expansion_ratio": float(self.box_expansion_ratio),
            "overwrite": True,
        }
        model_path = self.model_path.strip()
        if model_path:
            options["model_path"] = str(Path(model_path).expanduser().resolve())
        return options


@dataclass(frozen=True)
class MeteorWorkerInvocation:
    """QProcess 启动 worker 所需的程序、参数与工作目录。"""

    program: str
    arguments: tuple[str, ...]
    working_directory: str
    resolved_path: Path


def load_meteor_detection_options(path: str | Path | None = None) -> MeteorDetectionOptions:
    """从 preference.json 读取检测参数，并把异常值限制到安全范围。"""

    values = ensure_preference_file(path)

    def text_value(name: str, default: str = "") -> str:
        value = values.get(METEOR_DETECTION_PREFERENCE_PREFIX + name, default)
        return str(value).strip() if value is not None else default

    def float_value(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(values.get(METEOR_DETECTION_PREFERENCE_PREFIX + name, default))
        except (TypeError, ValueError):
            return default
        return min(max(value, minimum), maximum)

    def int_value(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(values.get(METEOR_DETECTION_PREFERENCE_PREFIX + name, default))
        except (TypeError, ValueError):
            return default
        return min(max(value, minimum), maximum)

    provider = text_value("provider", "auto").lower()
    if provider not in METEOR_DETECTION_PROVIDERS:
        provider = "auto"
    return MeteorDetectionOptions(
        engine_path=text_value("engine_path"),
        model_path=text_value("model_path"),
        confidence_threshold=float_value("confidence_threshold", 0.25, 0.0, 1.0),
        nms_threshold=float_value("nms_threshold", 0.45, 0.0, 1.0),
        multiscale=int_value("multiscale", 2, 0, 8),
        partition=int_value("partition", 2, 2, 12),
        provider=provider,
        box_expansion_ratio=float_value("box_expansion_ratio", 0.10, 0.0, 5.0),
    )


def save_meteor_detection_options(
    options: MeteorDetectionOptions,
    path: str | Path | None = None,
) -> bool:
    """持久化检测选项。"""

    updates = {
        METEOR_DETECTION_PREFERENCE_PREFIX + key: value
        for key, value in asdict(options).items()
    }
    return update_preference_values(updates, path=path)


def resolve_meteor_worker_invocation(engine_path: str = "") -> MeteorWorkerInvocation:
    """定位源码或 onedir 打包的 worker，并生成跨平台启动参数。"""

    configured_path = engine_path.strip()
    if configured_path:
        candidates = (Path(configured_path).expanduser(),)
    else:
        candidates = _automatic_engine_candidates()

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.absolute()
        invocation = _invocation_for_candidate(candidate)
        if invocation is not None:
            return invocation

    if configured_path:
        raise FileNotFoundError(f"无法在指定位置找到 metdet_worker：{configured_path}")
    raise FileNotFoundError("未找到 metdet_worker；请在“选项”中指定引擎文件或完整 onedir 目录。")


def _automatic_engine_candidates() -> tuple[Path, ...]:
    """返回开发环境与冻结应用中可能的 worker 位置。"""

    candidates: list[Path] = []
    if is_frozen_app():
        for root in frozen_resource_roots():
            candidates.extend((root / "metdet_worker", root))
        sibling = frozen_app_sibling_dir()
        candidates.extend((sibling / "metdet_worker", sibling))
    else:
        project_root = source_project_root()
        candidates.extend(
            (
                project_root / "metdet_worker",
                project_root.parent / "MetDetPy" / "dist" / "metdet_worker",
                project_root.parent / "MetDetPy",
            )
        )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _invocation_for_candidate(candidate: Path) -> MeteorWorkerInvocation | None:
    """把单个文件或目录候选转换为启动信息。"""

    if candidate.is_file():
        return _invocation_for_file(candidate)
    if not candidate.is_dir():
        return None

    executable_names = ("metdet_worker.exe", "metdet_worker") if sys.platform == "win32" else (
        "metdet_worker",
        "metdet_worker.exe",
    )
    for name in executable_names:
        executable = candidate / name
        if executable.is_file():
            return _invocation_for_file(executable)
    source_worker = candidate / "metdet_worker.py"
    if source_worker.is_file():
        return _invocation_for_file(source_worker)
    nested_directory = candidate / "metdet_worker"
    if nested_directory.is_dir() and nested_directory != candidate:
        return _invocation_for_candidate(nested_directory)
    return None


def _invocation_for_file(worker_path: Path) -> MeteorWorkerInvocation | None:
    suffix = worker_path.suffix.lower()
    if suffix == ".py":
        if is_frozen_app():
            return None
        return MeteorWorkerInvocation(
            program=sys.executable,
            arguments=("-u", str(worker_path)),
            working_directory=str(worker_path.parent),
            resolved_path=worker_path,
        )
    if suffix not in {"", ".exe"}:
        return None
    return MeteorWorkerInvocation(
        program=str(worker_path),
        arguments=(),
        working_directory=str(worker_path.parent),
        resolved_path=worker_path,
    )


__all__ = [
    "METEOR_DETECTION_PROVIDERS",
    "MeteorDetectionOptions",
    "MeteorWorkerInvocation",
    "load_meteor_detection_options",
    "resolve_meteor_worker_invocation",
    "save_meteor_detection_options",
]
