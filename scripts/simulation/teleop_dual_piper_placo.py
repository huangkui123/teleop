#!/usr/bin/env python3
"""Kinematic twin of the dual-PiPER hardware teleoperation pipeline.

This entry point intentionally has no SSH or ROS command transport.  It uses
the same XR mapping, Placo IK provider, PiPER joint conversion, command limits,
Grip dead-man behavior, and Trigger gripper behavior as the hardware script,
then displays the resulting joint state in MeshCat.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from placo_utils.visualization import frame_viz, robot_viz

# Support direct execution from a source checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xrobotoolkit_teleop.hardware.interface.piper_ssh import (
    PIPER_SIDES,
    PiperCommandLimiter,
    model_positions_to_piper_command,
    piper_feedback_to_model_positions,
)
from xrobotoolkit_teleop.headless.piper import (
    DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
    DualPiperJointTargetProvider,
    create_dual_piper_joint_target_provider,
)


# This is the same folded configuration currently used on the AgileX robot,
# rounded to make the standalone simulation deterministic and symmetric.
DEFAULT_INITIAL_ARM = (0.0, 0.0, -0.5, 0.0, 1.07, 0.0, 0.08)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the exact local XR-to-IK-to-joint pipeline used by "
            "teleop_dual_piper_ssh.py without connecting to the robot."
        )
    )
    parser.add_argument("--control-rate", type=float, default=50.0, help="simulation rate in Hz")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="XR translation scale")
    parser.add_argument(
        "--position-only",
        action="store_true",
        help=(
            "match hardware translation-only mode while holding the wrist "
            "orientation captured when Grip is pressed"
        ),
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=0.6,
        help="same command slew limit as hardware, in rad/s",
    )
    parser.add_argument(
        "--max-gripper-speed",
        type=float,
        default=0.04,
        help="same gripper slew limit as hardware, in m/s",
    )
    parser.add_argument(
        "--max-joint-tracking-error",
        type=float,
        default=0.20,
        help="same maximum joint setpoint lead as hardware, in rad",
    )
    parser.add_argument(
        "--max-gripper-tracking-error",
        type=float,
        default=0.02,
        help="same maximum gripper setpoint lead as hardware, in m",
    )
    parser.add_argument(
        "--release-timeout",
        type=float,
        default=30.0,
        help="time allowed for both Grip controls to be released at startup",
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically open the MeshCat page",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="exit after this many seconds; zero runs until Ctrl+C",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    positive = {
        "--control-rate": args.control_rate,
        "--scale-factor": args.scale_factor,
        "--max-joint-speed": args.max_joint_speed,
        "--max-gripper-speed": args.max_gripper_speed,
        "--max-joint-tracking-error": args.max_joint_tracking_error,
        "--max-gripper-tracking-error": args.max_gripper_tracking_error,
        "--release-timeout": args.release_timeout,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
    if not math.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("--duration must be a non-negative finite value")


def wait_for_grips_released(provider: DualPiperJointTargetProvider, timeout: float) -> None:
    print("Release both controller Grip buttons to establish the simulation reference...")
    deadline = time.monotonic() + timeout
    released_since: float | None = None
    while True:
        now = time.monotonic()
        grip_values = (
            float(provider.xr_client.get_key_value_by_name("left_grip")),
            float(provider.xr_client.get_key_value_by_name("right_grip")),
        )
        if max(grip_values) < 0.2:
            if released_since is None:
                released_since = now
            elif now - released_since >= 0.5:
                print("Both Grip buttons are released. Simulation reference accepted.")
                return
        else:
            released_since = None
        if now >= deadline:
            raise RuntimeError("both Grip buttons must be released before simulation starts")
        time.sleep(0.02)


def model_feedback(simulated_state: Mapping[str, Sequence[float]]) -> dict[str, float]:
    positions: dict[str, float] = {}
    for side in PIPER_SIDES:
        positions.update(piper_feedback_to_model_positions(side, simulated_state[side]))
    return positions


def set_provider_state_for_display(
    provider: DualPiperJointTargetProvider,
    simulated_state: Mapping[str, Sequence[float]],
) -> None:
    for joint_name, position in model_feedback(simulated_state).items():
        offset = provider.placo_robot.get_joint_offset(joint_name)
        provider.placo_robot.state.q[offset] = float(position)
    provider.placo_robot.update_kinematics()


def task_target_transform(
    provider: DualPiperJointTargetProvider, arm_name: str
) -> np.ndarray:
    if provider.effector_control_mode[arm_name] == "position":
        transform = np.eye(4)
        transform[:3, 3] = provider.effector_task[arm_name].target_world
        return transform
    return provider.effector_task[arm_name].T_world_frame.copy()


def update_visualization(
    provider: DualPiperJointTargetProvider,
    visualization: Any,
    active: Mapping[str, bool],
) -> None:
    visualization.display(provider.placo_robot.state.q)
    frame_viz("sim_world", np.eye(4), opacity=0.65, scale=0.7)
    for side in PIPER_SIDES:
        arm_name = f"{side}_arm"
        link_name = provider.manipulator_config[arm_name]["link_name"]
        actual_transform = provider.placo_robot.get_T_world_frame(link_name)
        target_transform = (
            task_target_transform(provider, arm_name)
            if active[side]
            else actual_transform
        )
        frame_viz(
            f"sim_{side}_actual", actual_transform, opacity=1.0, scale=0.55
        )
        frame_viz(
            f"sim_{side}_target", target_transform, opacity=0.35, scale=0.8
        )


def print_diagnostics(
    provider: DualPiperJointTargetProvider,
    simulated_state: Mapping[str, Sequence[float]],
    active: Mapping[str, bool],
) -> None:
    print("Simulation state:")
    for side in PIPER_SIDES:
        arm_name = f"{side}_arm"
        link_name = provider.manipulator_config[arm_name]["link_name"]
        actual_transform = provider.placo_robot.get_T_world_frame(link_name)
        actual_xyz = actual_transform[:3, 3]
        target_xyz = (
            task_target_transform(provider, arm_name)[:3, 3]
            if active[side]
            else actual_xyz
        )
        q_text = ", ".join(f"{value:+.2f}" for value in simulated_state[side][:6])
        print(
            f"  {side:5s} active={str(active[side]):5s} "
            f"actual_xyz={np.round(actual_xyz, 3).tolist()} "
            f"target_xyz={np.round(target_xyz, 3).tolist()} q=[{q_text}]"
        )


def run_simulation(args: argparse.Namespace) -> int:
    validate_arguments(args)
    manipulator_config = copy.deepcopy(DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG)
    if args.position_only:
        for config in manipulator_config.values():
            config["control_mode"] = "position_fixed_orientation"

    provider = create_dual_piper_joint_target_provider(
        scale_factor=args.scale_factor,
        control_rate_hz=args.control_rate,
        manipulator_config=manipulator_config,
    )
    simulated_state: dict[str, tuple[float, ...]] = {
        side: tuple(DEFAULT_INITIAL_ARM) for side in PIPER_SIDES
    }
    limiter = PiperCommandLimiter(
        max_joint_speed=args.max_joint_speed,
        max_gripper_speed=args.max_gripper_speed,
        max_joint_tracking_error=args.max_joint_tracking_error,
        max_gripper_tracking_error=args.max_gripper_tracking_error,
    )
    engaged = {side: False for side in PIPER_SIDES}
    previous_active = {side: False for side in PIPER_SIDES}

    try:
        for side in PIPER_SIDES:
            limiter.reset(side, simulated_state[side])
        set_provider_state_for_display(provider, simulated_state)
        visualization = robot_viz(provider.placo_robot, "dual_piper_sim")
        meshcat_url = visualization.viewer.url()
        print(f"MeshCat: {meshcat_url}")
        if args.open_browser:
            webbrowser.open(meshcat_url)

        wait_for_grips_released(provider, args.release_timeout)
        control_mode = "POSITION ONLY" if args.position_only else "FULL POSE"
        print(f"Dual-PiPER kinematic simulation started ({control_mode}).")
        print("Hold left/right Grip to move that arm; Trigger controls its gripper.")
        print("XR axis mapping: controller +X -> robot -Y, +Y -> +Z, +Z -> -X.")
        print("MeshCat frames: opaque=actual end effector, translucent=IK target.")

        period = 1.0 / args.control_rate
        started_at = time.monotonic()
        previous_loop_time = started_at
        next_tick = started_at
        next_report = started_at

        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - started_at >= args.duration:
                return 0

            command = provider.update(model_feedback(simulated_state))
            active = {
                side: bool(command.active.get(f"{side}_arm", False))
                for side in PIPER_SIDES
            }
            dt = max(0.0, now - previous_loop_time)
            previous_loop_time = now

            for side in PIPER_SIDES:
                if active[side]:
                    if not previous_active[side]:
                        limiter.reset(side, simulated_state[side])
                        print(f"{side} arm active")
                    engaged[side] = True
                    desired = model_positions_to_piper_command(side, command.positions)
                    simulated_state[side] = limiter.limit(
                        side, desired, simulated_state[side], dt
                    )
                elif engaged[side]:
                    if previous_active[side]:
                        print(f"{side} arm released; holding simulated position")
                    limiter.reset(side, simulated_state[side])
                previous_active[side] = active[side]

            set_provider_state_for_display(provider, simulated_state)
            update_visualization(provider, visualization, active)

            if now >= next_report:
                print_diagnostics(provider, simulated_state, active)
                next_report = now + 2.0

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        return 130
    finally:
        provider.close()


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        return run_simulation(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
