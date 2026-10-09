__version__ = "0.1.0"

__all__ = [
    "build_wan_video_action_pipeline",
    "WanVideoActionEncoder",
    "WanVideoUnit_ActionEmbedder",
    "model_fn_wan_video_action",
    "LoadCobotAction",
    "merge_yaml_and_args",
    "prepare_runtime_config",
    "add_general_config",
]


def __getattr__(name):
    if name in {
        "build_wan_video_action_pipeline",
        "WanVideoUnit_ActionEmbedder",
        "model_fn_wan_video_action",
    }:
        from .pipelines.wan_video_action import (
            WanVideoUnit_ActionEmbedder,
            build_wan_video_action_pipeline,
            model_fn_wan_video_action,
        )

        return {
            "build_wan_video_action_pipeline": build_wan_video_action_pipeline,
            "WanVideoUnit_ActionEmbedder": WanVideoUnit_ActionEmbedder,
            "model_fn_wan_video_action": model_fn_wan_video_action,
        }[name]
    if name == "WanVideoActionEncoder":
        from .models import WanVideoActionEncoder

        return WanVideoActionEncoder
    if name == "LoadCobotAction":
        from .data import LoadCobotAction

        return LoadCobotAction
    if name in {"merge_yaml_and_args", "prepare_runtime_config", "add_general_config"}:
        from .parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config

        return {
            "merge_yaml_and_args": merge_yaml_and_args,
            "prepare_runtime_config": prepare_runtime_config,
            "add_general_config": add_general_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
