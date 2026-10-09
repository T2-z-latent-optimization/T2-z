"""Standalone child entrypoint. stdout is exclusively a framed private pipe."""
from __future__ import annotations

import contextlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from wan_video_action.evaluation.sim_action_selection.core.rpc import read_message, write_message


def configure_model_runtime(configuration):
    if not configuration.get("overrides", {}).get("deterministic_algorithms", False):
        return None
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    instance = None
    # Native libraries occasionally write directly to fd 1, bypassing Python's
    # redirect_stdout. Reserve a duplicate for RPC and redirect fd 1 itself.
    outgoing = os.fdopen(os.dup(outgoing.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), 1)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            while True:
                request = read_message(incoming)
                method, argument = request["method"], request.get("argument")
                try:
                    if method == "initialize":
                        if instance is not None:
                            raise RuntimeError("Worker already initialized")
                        runtime = None
                        if sys.argv[1] == "model":
                            runtime = configure_model_runtime(argument)
                            if runtime is not None:
                                print(f"model_runtime={runtime!r}", file=sys.stderr, flush=True)
                            from wan_video_action.evaluation.sim_action_selection.core.model import WanPredictor
                            instance = WanPredictor(argument)
                        else:
                            plugin = importlib.import_module(
                                "wan_video_action.evaluation.sim_action_selection.tasks."
                                + argument["task"] + ".task"
                            )
                            if argument["case"].get("candidate_initializations"):
                                from wan_video_action.evaluation.sim_action_selection.core.candidate_scenes import CandidateScenes
                                isolation = argument.get("isolate_candidate_scenes")
                                if isolation is None:
                                    isolation = getattr(plugin, "isolate_candidate_scenes", False)
                                if isolation:
                                    from wan_video_action.evaluation.sim_action_selection.core.rpc import RemoteSimulator
                                    log_root = argument.get("candidate_worker_log_dir")
                                    log_root = Path(log_root) if log_root else Path(tempfile.mkdtemp(prefix="t2z-candidate-scenes-"))
                                    log_root.mkdir(parents=True, exist_ok=True)
                                    scene_count = [0]
                                    def create_scene(case, initialization=argument, directory=log_root, counter=scene_count):
                                        number = counter[0]
                                        counter[0] += 1
                                        return RemoteSimulator(
                                            sys.executable, {**initialization, "case": case},
                                            directory / f"scene_{number:03d}.log",
                                        )
                                else:
                                    def create_scene(case, initialization=argument):
                                        return plugin.create_simulator({**initialization, "case": case})
                                instance = CandidateScenes(create_scene, argument["case"])
                            else:
                                instance = plugin.create_simulator(argument)
                        result = {"ready": True}
                        if runtime is not None:
                            result["model_runtime"] = runtime
                    else:
                        if instance is None:
                            raise RuntimeError("Initialize the worker first")
                        operation = getattr(instance, method)
                        result = operation() if argument is None else operation(argument)
                    write_message(outgoing, {"ok": True, "result": result})
                    if method == "close":
                        instance = None
                        break
                except Exception:
                    write_message(outgoing, {"ok": False, "error": traceback.format_exc()})
        except EOFError:
            pass
        finally:
            if instance is not None:
                instance.close()
            outgoing.close()


if __name__ == "__main__":
    main()
