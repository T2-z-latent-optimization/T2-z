from __future__ import annotations

import numpy as np

from ...core.types import Candidate


def _plan_segment(simulator, pose, start_qpos):
    planner = simulator.planner
    target = np.r_[pose.p, pose.q]
    # Original policy: three calls, each screw call tries twice, then three RRT
    # attempts. All calls below are pure kinematic planning: no env.step.
    for _ in range(6):
        result = planner.planner.plan_screw(
            target, start_qpos.copy(), time_step=simulator.env.unwrapped.control_timestep,
            use_point_cloud=planner.use_point_cloud,
        )
        if result["status"] == "Success":
            return np.asarray(result["position"], dtype=np.float64)
    for _ in range(3):
        result = planner.planner.plan_qpos_to_pose(
            target, start_qpos.copy(), time_step=simulator.env.unwrapped.control_timestep,
            use_point_cloud=planner.use_point_cloud, rrt_range=0.0,
            planning_time=1, planner_name="RRTstar", wrt_world=True,
        )
        if result["status"] == "Success":
            return np.asarray(result["position"], dtype=np.float64)
    raise RuntimeError("Original screw/RRT planning policy could not construct a button candidate")


def build_candidates(simulator):
    import sapien

    original = simulator.original
    bank = []
    simulator.native_bank = {}
    robot_qpos = original.base._vec(simulator.env.unwrapped.agent.robot.get_qpos()).astype(np.float64)
    previous = simulator.transform.previous_euler_xyz
    for index, color in enumerate(("red", "blue")):
        qpos = robot_qpos.copy()
        position = original.base._vec(simulator.env.unwrapped.buttons[color].pose.p)[:3].astype(np.float64)
        ready = position.copy()
        ready[2] = 0.15
        commands = []
        waypoints = [ready, position, ready]
        if color == simulator.reference_color and simulator.remaining_path is not None:
            remainder = simulator.remaining_path
            commands.extend(np.r_[row[:7], -1.0] for row in remainder)
            if len(remainder):
                qpos[:7] = remainder[-1, :7]
            if simulator.pending_target_z is not None and simulator.pending_target_z < 0.10:
                waypoints = [ready]
            else:
                waypoints = [position, ready]
        for waypoint in waypoints:
            pose = sapien.Pose(p=waypoint, q=[0.0, 1.0, 0.0, 0.0])
            path = _plan_segment(simulator, pose, qpos)
            if not len(path):
                continue
            commands.extend(np.r_[row[:7], -1.0] for row in path)
            qpos[:7] = path[-1, :7]
        commands = np.asarray(commands, dtype=np.float64)
        required = simulator.chunk.length - 1
        if len(commands) < required:
            raise ValueError(
                f"Button {color} has only {len(commands)} future native commands for a {required}-transition chunk; "
                "no extra hold steps are invented. Adjust the motion-template decision position."
            )
        commands = commands[:required]
        transform = original.AbsoluteEEFActionTransform(simulator.env, simulator.planner)
        transform.previous_euler_xyz = None if previous is None else previous.copy()
        raw = [simulator.paused_frame["action"].copy()]
        raw.extend(transform(command) for command in commands)
        simulator.native_bank[index] = commands
        bank.append(Candidate(index, float(index), {"button_color": color}, np.stack(raw), "planned_joint_waypoints_original_pd_joint_pos"))
    return bank
