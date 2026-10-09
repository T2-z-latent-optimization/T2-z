"""Real physics tests; no checkpoint or predicted-oracle selection is used."""
import hashlib
import importlib
import json

import numpy as np
import pytest

from ..core.artifacts import write_json
from ..core.cases import prepare_cases
from .. import TASKS
from ..core.rpc import WorkerClient
from ..core.runner import Runner
from ..core.selection import load_rules
from ...config import load_evaluation_config
from .fixtures import ROOT, SOURCE, config_for, python_for, resources_for


CASES = [
    ("friction", {"friction_mu": 0.002}, 0),
    ("friction", {"friction_mu": 0.1}, 4),
    ("friction", {"friction_mu": 0.2}, 9),
    ("gravity", {"gravity_mps2": 0.25}, 0),
    ("gravity", {"gravity_mps2": 9.81}, 5),
    ("gravity", {"gravity_mps2": 50.0}, 9),
    ("mass_collision", {"target_mass_kg": 0.01}, 0),
    ("mass_collision", {"target_mass_kg": 0.25}, 4),
    ("mass_collision", {"target_mass_kg": 2.0}, 8),
    ("mass_balance", {"ratio_index": 0}, 0),
    ("mass_balance", {"ratio_index": 9}, 7),
    ("mass_balance", {"ratio_index": 29}, 14),
    ("mass_friction", {"pair_index": 0}, 0),
    ("mass_friction", {"pair_index": 50}, 4),
    ("mass_friction", {"pair_index": 99}, 8),
    *[("multi_background", {"background_index": index, "friction_mu": 0.05}, 4) for index in range(5)],
    ("light_switch", {"causal_class": "both", "initial_lamp_on": True, "decision_button": "red"}, 0),
    ("light_switch", {"causal_class": "both", "initial_lamp_on": True, "decision_button": "blue"}, 1),
    ("light_switch", {"causal_class": "neither", "initial_lamp_on": False, "decision_button": "blue"}, 1),
]


@pytest.mark.parametrize("task,overrides,action_index", CASES, ids=[f"{task}_{number:02d}" for number, (task, _, _) in enumerate(CASES)])
def test_native_protocol(task, overrides, action_index, native_output):
    config = config_for(task)
    case = {**config["default_case"], **overrides}
    client = WorkerClient(python_for(task), "simulator", {"task": task, "config": config, "case": case, "source_root": str(SOURCE)}, native_output / "worker.log", timeout=600)
    try:
        paused = client.call("prepare")
        candidates = client.call("candidates")
        assert all(item["model_actions"].shape == (len(paused["native_frame_indices"]), 7) for item in candidates)
        assert client.call("fingerprint") == paused["fingerprint"]
        result = client.call("execute", action_index)
        np.testing.assert_array_equal(result["views"][0], paused["views"])
        assert result["views"].shape == (len(paused["native_frame_indices"]), 2, 224, 224, 3)
        assert np.isfinite(result["actions"]).all() and np.isfinite(result["native_actions"]).all()
        assert len(result["native_actions"]) == result["native_control_iterations"]
        assert len(result["native_control_frame_indices"]) == result["native_control_iterations"]
        assert len(result["recorded_actions"]) == len(result["recorded_frame_indices"])
        controls = [row for row in result["native_log"] if not row.get("recorded_without_next_transition")]
        for command, row in zip(result["native_actions"], controls):
            np.testing.assert_allclose(command, row.get("native_action", row.get("native_joint_action")), atol=1e-7)
        expected_controls = 81 if task == "mass_balance" else 60 if task in {"mass_collision", "mass_friction"} else 32 if task == "light_switch" else None
        if expected_controls is not None:
            assert result["native_control_iterations"] == expected_controls
        if task in {"mass_collision", "mass_friction", "mass_balance", "gravity", "light_switch"}:
            assert result["conditioning_max_abs_error"] < 1e-6
        if task == "light_switch":
            assert result["actual_control_timestep"] == pytest.approx(0.05)
            events = result["press_events"]
            assert len(events) == 1
            assert events[0]["button_color"] == case["decision_button"]
            assert 11 <= events[0]["step"] - paused["frame_index"] <= 22
            assert events[0]["lamp_before"] == case["initial_lamp_on"]
            assert events[0]["lamp_after"] == (case["initial_lamp_on"] ^ events[0]["controls_lamp"])
        if task in {"friction", "mass_balance", "multi_background"}:
            assert np.all(result["native_actions"][:, -1] == 1)
            assert np.all(result["actions"][:, -1] == 0)
        if task in {"mass_collision", "mass_friction"}:
            # Original all-zero tail commands include gripper=0. Preserve
            # their recorded 0.5 value rather than inventing a closed hold.
            np.testing.assert_allclose(result["recorded_actions"][:, -1], (1.0 - result["native_actions"][:, -1]) / 2.0)
        np.savez_compressed(native_output / "actions.npz", recorded=result["actions"], planned=result["planned_actions"], native=result["native_actions"], native_control_frame_indices=result["native_control_frame_indices"])
        perception = importlib.import_module(f"wan_video_action.evaluation.sim_action_selection.tasks.{task}.perception")
        tracking = load_evaluation_config(ROOT / config["tracking"]) if isinstance(config["tracking"], str) else config["tracking"]
        state, value = perception.outcome(result["views"], paused["fps"], tracking, load_rules(task))
        write_json(native_output / "result.json", {"task": task, "case": case, "action_index": action_index, "rgb_tracking_outcome": value, "native_control_iterations": result["native_control_iterations"], "conditioning_max_abs_error": result["conditioning_max_abs_error"], "padded_frames": int(result["padded"].sum()), "first_rgb_identical": True, "source_hashes": paused["sources"]})
        for path, digest in paused["sources"].items():
            assert hashlib.sha256(__import__("pathlib").Path(path).read_bytes()).hexdigest() == digest
    finally:
        client.shutdown()


@pytest.mark.parametrize("task", TASKS)
def test_runner_video_selection_then_real_execution(task, native_output):
    """Wiring test with explicit deterministic model double, not neural accuracy."""
    config = config_for(task)
    resources = resources_for(task)
    if isinstance(config["tracking"], str):
        resources["tracking"] = load_evaluation_config(ROOT / config["tracking"])
    runner = Runner(task, config, resources, native_output)
    length = 33 if task == "light_switch" else 61 if task in {"gravity", "mass_collision", "mass_friction"} else 41

    class PredictionDouble:
        def __init__(self):
            self.requests = []

        def call(self, method, argument):
            if method == "freeze_context":
                return {"test_double": True, "not_a_checkpoint_accuracy_test": True}
            assert set(argument) == {"views", "actions", "seed"}
            self.requests.append(argument)
            frames = np.repeat(argument["views"], length, axis=0)
            # Synthetic tracked red target motion, derived only from candidate
            # amplitude. No real candidate outcomes are available to this double.
            center_x = int(80 + 70 * float(np.max(argument["actions"][:, 0])))
            frames[1:, 0, 110:130, center_x - 5:center_x + 5] = [255, 0, 0]
            if task == "mass_friction":
                # Exercise a real threshold crossing rather than asking the
                # unchanged selector to execute an unreachable prediction.
                center_y = int(100 + 80 * float(np.max(argument["actions"][:, 0])))
                frames[1:, 0] = 0
                frames[1:, 0, center_y - 5:center_y + 5, 105:115] = [255, 0, 0]
            if task == "light_switch":
                frames[1:, 0] = [240, 220, 60]
            return frames

    runner.model = PredictionDouble()
    case = prepare_cases(task, [config["default_case"]])[0]
    result = runner.run_decision(case, 0, runner.rules["targets"][0]["name"])
    expected_candidates = 15 if task == "mass_balance" else 2 if task == "light_switch" else 9 if task in {"mass_collision", "mass_friction"} else 10
    assert len(runner.model.requests) == expected_candidates
    directory = native_output / ("decision_00000_" + runner.rules["targets"][0]["name"])
    with np.load(directory / "current_observation.npz") as current:
        native_current = current["native_views"].copy()
    for index, request in enumerate(runner.model.requests):
        observation = directory / f"candidate_{index:02d}_current_observation.npz"
        if task not in {"light_switch", "mass_friction"}:
            observation = directory / "current_observation.npz"
        with np.load(observation) as captured:
            np.testing.assert_array_equal(request["views"], captured["views"])
        if task == "mass_friction":
            audit = json.loads((directory / f"candidate_{index:02d}_capture.json").read_text())
            assert audit["native_prefix_frame_count"] == 3
            assert audit["native_current_rgb_exact"]
            assert audit["native_contact_check_passed"]
            assert not audit["postcollision_observations_used"]
            assert not audit["main_simulator_advanced_for_encoding"]
            assert audit["maximum_used_transition_action_error"] < 1e-6
            with np.load(directory / f"candidate_{index:02d}_encoding_prefix.npz") as prefix:
                assert len(prefix["raw"]) == 3
                np.testing.assert_array_equal(prefix["raw"][:1], native_current)
                np.testing.assert_array_equal(prefix["native_frame_indices"], audit["native_prefix_frame_indices"])
            transitions = json.loads((directory / f"candidate_{index:02d}_prefix_native_log.json").read_text())
            assert len(transitions) == 2 and not any(row["block_block_contact"] for row in transitions)
        elif task != "light_switch":
            np.testing.assert_array_equal(request["views"], runner.model.requests[0]["views"])
    assert result["selected_action_index"] is not None
    expected_controls = 81 if task == "mass_balance" else 60 if task in {"mass_collision", "mass_friction"} else 32 if task == "light_switch" else None
    if expected_controls is not None:
        assert result["native_control_iterations"] == expected_controls
    else:
        assert result["native_control_iterations"] > 0
    assert result["simulator_frozen_during_prediction"]
    assert list(native_output.glob("decision_*/executed.mp4"))
    write_json(native_output / "wiring_test_scope.json", {"native_physics": "real_original_collector", "model": "explicit_deterministic_test_double", "neural_checkpoint_accuracy_validated": False})
