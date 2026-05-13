from __future__ import annotations

import ast
import re
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1] / "src" / "sfc"
FORBIDDEN_IMPORT = "legacy_abaqus_prototype"
FORBIDDEN_TEXT_PATTERNS = {
    "legacy_abaqus_prototype": re.compile(r"legacy_abaqus_prototype", re.IGNORECASE),
    "Abaqus": re.compile(r"abaqus", re.IGNORECASE),
    "GLOBALM.mat": re.compile(r"GLOBALM\.mat", re.IGNORECASE),
    "GLOBALK.mat": re.compile(r"GLOBALK\.mat", re.IGNORECASE),
    "Force*.mat": re.compile(r"Force\d*\.mat|Force\*\.mat", re.IGNORECASE),
    ".odb": re.compile(r"\.odb", re.IGNORECASE),
    ".xlsx": re.compile(r"\.xlsx", re.IGNORECASE),
    "scipy.io.loadmat": re.compile(r"\bloadmat\b|scipy\.io", re.IGNORECASE),
    "pandas.read_excel": re.compile(r"\bread_excel\b", re.IGNORECASE),
    "abaqus_workspace": re.compile(r"abaqus_workspace", re.IGNORECASE),
    "absolute Windows path": re.compile(r"[A-Za-z]:[\\/]"),
}


def _imported_module_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


def test_core_modules_do_not_import_legacy_abaqus_prototype() -> None:
    offenders: list[tuple[Path, str]] = []

    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module_name in _imported_module_names(tree):
            if module_name == FORBIDDEN_IMPORT or module_name.startswith(
                FORBIDDEN_IMPORT + "."
            ):
                offenders.append((path.relative_to(CORE_ROOT), module_name))

    assert offenders == []


def test_core_modules_do_not_reference_forbidden_legacy_inputs() -> None:
    offenders: list[tuple[Path, str]] = []

    for path in sorted(CORE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_TEXT_PATTERNS.items():
            if pattern.search(text):
                offenders.append((path.relative_to(CORE_ROOT), label))

    assert offenders == []
