from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("T2Z_NATIVE_SOURCE_ROOT", ROOT.parent / "TTT4dynamics")).expanduser().absolute()
LIGHT_CASE = {
    "episode_index": 0, "seed": 20260725,
    "control_colors": ["blue", "red"], "initial_lamp_on": True,
    "press_sequence": ["red", "blue", "red", "red", "blue", "blue", "blue", "red"],
    "button_positions_xy_m": {"red": [-0.12, -0.055], "blue": [-0.12, 0.055]},
    "event_frame": 71, "chunk_start": 55,
}


def config_for(task):
    return json.loads((PACKAGE / "tasks" / task / "config.json").read_text())


def python_for(task):
    supplied = os.environ.get(f"T2Z_NATIVE_PYTHON_{task.upper()}") or os.environ.get("T2Z_NATIVE_PYTHON")
    if supplied:
        return Path(supplied).expanduser().absolute()
    repository = "robomme_benchmark" if task == "light_switch" else "TTT4dynamics"
    root = SOURCE.parent / repository if task == "light_switch" else SOURCE
    return root / ".venv/bin/python"


def resources_for(task):
    return {
        "simulator_python": str(python_for(task)), "source_root": str(SOURCE),
        "worker_timeout": 600, "method": "standard", "seed": 123,
        "support_actions": config_for(task)["support_actions"], "save_predictions": False,
        "tracking": {},
    }
