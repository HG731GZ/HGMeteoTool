"""用 AST 守护模块依赖边界，避免重构时重新引入反向依赖。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ROOT = PACKAGE_ROOT / "application"
CORE_PACKAGE_NAMES = (
    "projection",
    "astro",
    "alignment",
    "astrometry",
    "calibration",
    "photometric",
)
FORBIDDEN_CORE_IMPORT_PREFIXES = ("PyQt5", "meteoalign.application.app_")


@dataclass(frozen=True)
class ImportOccurrence:
    """一条 import 的来源、目标和源码位置。"""

    source_path: Path
    source_module: str
    target_module: str
    line_number: int

    def format_location(self) -> str:
        """生成适合 pytest 失败信息的源码位置。"""

        return f"{self.source_path.relative_to(PACKAGE_ROOT.parent)}:{self.line_number} -> {self.target_module}"


def _module_name_for_path(path: Path) -> str:
    """将包内 Python 文件路径转换为完整模块名。"""

    relative_path = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = ("meteoalign", *relative_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_from_import_base(source_module: str, node: ast.ImportFrom) -> str:
    """解析相对 import 的模块前缀，不加载任何业务模块。"""

    if node.level == 0:
        return node.module or ""

    package_parts = source_module.rsplit(".", 1)[0].split(".")
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _import_occurrences(path: Path) -> list[ImportOccurrence]:
    """提取一个文件的静态 import，不执行其中的代码。"""

    source_module = _module_name_for_path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    occurrences: list[ImportOccurrence] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            occurrences.extend(
                ImportOccurrence(path, source_module, alias.name, node.lineno)
                for alias in node.names
            )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        base_module = _resolve_from_import_base(source_module, node)
        if node.module:
            occurrences.append(ImportOccurrence(path, source_module, base_module, node.lineno))
            continue

        # ``from . import module`` 的目标位于别名中，需要逐个补全。
        occurrences.extend(
            ImportOccurrence(path, source_module, f"{base_module}.{alias.name}", node.lineno)
            for alias in node.names
        )
    return occurrences


def _package_python_files(package_name: str) -> list[Path]:
    """返回存在的核心包中的 Python 文件；尚未拆出的包不参与检查。"""

    package_path = PACKAGE_ROOT / package_name
    if not package_path.is_dir():
        return []
    return sorted(package_path.rglob("*.py"))


def _is_forbidden_core_import(target_module: str) -> bool:
    """判断目标模块是否违反核心层依赖规则。"""

    return any(
        target_module == prefix or target_module.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_CORE_IMPORT_PREFIXES
    )


def test_numerical_core_packages_do_not_import_qt_or_app_modules() -> None:
    """数值核心包不得向上依赖 Qt 界面或 app_* Mixin。"""

    violations: list[ImportOccurrence] = []
    for package_name in CORE_PACKAGE_NAMES:
        for path in _package_python_files(package_name):
            violations.extend(
                occurrence
                for occurrence in _import_occurrences(path)
                if _is_forbidden_core_import(occurrence.target_module)
            )

    assert not violations, "核心层存在禁止依赖：\n" + "\n".join(
        violation.format_location() for violation in violations
    )


def test_sequence_matching_does_not_import_other_app_modules() -> None:
    """序列匹配逻辑不得再向其他 app_* Mixin 或 UI 常量模块取依赖。"""

    path = APPLICATION_ROOT / "app_sequence_matching.py"
    violations = [
        occurrence
        for occurrence in _import_occurrences(path)
        if occurrence.target_module.startswith("meteoalign.application.app_")
    ]

    assert not violations, "序列匹配模块存在 application/app_* 依赖：\n" + "\n".join(
        violation.format_location() for violation in violations
    )
