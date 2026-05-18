import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_generate():
    module_path = ROOT / "validation" / "plot_sdf_method_schematic.py"
    spec = importlib.util.spec_from_file_location("plot_sdf_method_schematic", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate


def test_sdf_method_schematic_outputs(tmp_path: Path) -> None:
    generate = _load_generate()
    outputs = generate(tmp_path)

    assert outputs["png"].is_file()
    assert outputs["pdf"].is_file()
    assert outputs["png"].stat().st_size > 10_000
    assert outputs["pdf"].stat().st_size > 1_000
