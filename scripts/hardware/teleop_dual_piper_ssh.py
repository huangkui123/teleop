#!/usr/bin/env python3
"""XR teleoperation for dual AgileX PiPER arms through a persistent SSH link."""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Any

# Allow this hardware entry point to run directly from a source checkout even
# when the package has not yet been installed into the active environment.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xrobotoolkit_teleop.hardware.interface.piper_ssh import (
    PIPER_SIDES,
    PiperCommandLimiter,
    PiperSshBridge,
    PiperSshError,
    RemotePiperState,
    model_positions_to_piper_command,
    piper_feedback_to_model_positions,
    piper_status_issues,
)


DEFAULT_REMOTE_SETUP = (
    "/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teleoperate two PiPER arms whose ROS 1 driver runs on an SSH host. "
            "The default is a dry run; physical commands require --execute."
        )
    )
    parser.add_argument("--host", default="agilex", help="SSH host or ~/.ssh/config alias")
    parser.add_argument(
        "--remote-setup",
        default=DEFAULT_REMOTE_SETUP,
        help="catkin setup.bash on the robot",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check SSH, ROS topics, feedback, and arm faults, then exit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow physical JointState commands (without this flag, run dry only)",
    )
    parser.add_argument("--control-rate", type=float, default=50.0, help="control rate in Hz")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="XR translation scale")
    parser.add_argument(
        "--position-only",
        action="store_true",
        help=(
            "control XYZ only while holding the wrist orientation captured "
            "when Grip is pressed"
        ),
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=0.6,
        help="command slew limit in rad/s",
    )
    parser.add_argument(
        "--max-gripper-speed",
        type=float,
        default=0.04,
        help="command slew limit for gripper opening in m/s",
    )
    parser.add_argument(
        "--max-joint-tracking-error",
        type=float,
        default=0.20,
        help="maximum joint setpoint lead over feedback in rad",
    )
    parser.add_argument(
        "--max-gripper-tracking-error",
        type=float,
        default=0.02,
        help="maximum gripper setpoint lead over feedback in m",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.4,
        help="stop if SSH feedback is older than this many seconds",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=0.25,
        help="remote bridge stops republishing after this command gap in seconds",
    )
    parser.add_argument(
        "--connect-timeout", type=float, default=8.0, help="SSH connection timeout in seconds"
    )
    parser.add_argument(
        "--release-timeout",
        type=float,
        default=30.0,
        help="time allowed for both grip controls to be released at startup",
    )
    parser.add_argument(
        "--left-feedback-topic", default="/puppet/joint_left", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--right-feedback-topic", default="/puppet/joint_right", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--left-status-topic", default="/puppet/arm_status_left", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--right-status-topic", default="/puppet/arm_status_right", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--left-command-topic", default="/master/joint_left", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--right-command-topic", default="/master/joint_right", help=argparse.SUPPRESS
    )
    return parser


def create_bridge(args: argparse.Namespace) -> PiperSshBridge:
    return PiperSshBridge(
        host=args.host,
        remote_setup=args.remote_setup,
        left_feedback_topic=args.left_feedback_topic,
        right_feedback_topic=args.right_feedback_topic,
        left_status_topic=args.left_status_topic,
        right_status_topic=args.right_status_topic,
        left_command_topic=args.left_command_topic,
        right_command_topic=args.right_command_topic,
        feedback_rate_hz=args.control_rate,
        watchdog_timeout=args.watchdog_timeout,
        state_timeout=args.feedback_timeout,
        connect_timeout=args.connect_timeout,
        execute=args.execute and not args.check,
    )


def preflight_issues(
    bridge: PiperSshBridge, state: RemotePiperState, *, require_execute: bool
) -> list[str]:
    info = bridge.ready_info
    issues: list[str] = []
    for side in PIPER_SIDES:
        issues.extend(piper_status_issues(side, state.statuses.get(side)))

    conflicts = info.get("existing_command_publishers") or {}
    if conflicts:
        issues.append(f"existing command publisher(s): {conflicts}")
    connections = info.get("command_subscribers") or {}
    for side in PIPER_SIDES:
        if int(connections.get(side, 0)) < 1:
            issues.append(f"{side}: no subscriber on the command topic")
    if require_execute:
        if not bool(info.get("execute_allowed", False)):
            issues.append("remote bridge did not start in execute mode")
    return issues


def print_preflight(
    bridge: PiperSshBridge,
    state: RemotePiperState,
    bridge_description: str | None = None,
) -> None:
    info = bridge.ready_info
    description = bridge_description or f"SSH/ROS bridge ready on {bridge.host}"
    print(f"{description}: {info.get('node', '<unknown node>')}")
    for side in PIPER_SIDES:
        joints = ", ".join(f"{value:+.3f}" for value in state.positions[side][:6])
        gripper = state.positions[side][6]
        status = state.statuses.get(side) or {}
        print(
            f"  {side:5s}: joints=[{joints}], gripper={gripper:.4f} m, "
            f"arm_status={int(status.get('arm_status', -1))}, "
            f"ctrl_mode={int(status.get('ctrl_mode', -1))}, "
            f"err_code={int(status.get('err_code', 0))}"
        )
    connections = info.get("command_subscribers") or {}
    print(
        "  command subscribers: "
        + ", ".join(f"{side}={int(connections.get(side, 0))}" for side in PIPER_SIDES)
    )


def model_feedback(state: RemotePiperState) -> dict[str, float]:
    positions: dict[str, float] = {}
    for side in PIPER_SIDES:
        positions.update(piper_feedback_to_model_positions(side, state.positions[side]))
    return positions


def wait_for_grips_released(provider: Any, timeout: float) -> None:
    print("Release both controller grip buttons to establish a safe reference...")
    deadline = time.monotonic() + timeout
    released_since: float | None = None
    while True:
        now = time.monotonic()
        values = (
            float(provider.xr_client.get_key_value_by_name("left_grip")),
            float(provider.xr_client.get_key_value_by_name("right_grip")),
        )
        if max(values) < 0.2:
            if released_since is None:
                released_since = now
            elif now - released_since >= 0.5:
                print("Both grip buttons are released. Reference accepted.")
                return
        else:
            released_since = None
        if now >= deadline:
            raise PiperSshError(
                "both grip buttons must be released before teleoperation can start"
            )
        time.sleep(0.02)


def hold_measured_positions(
    bridge: PiperSshBridge,
    engaged: dict[str, bool],
    feedback_timeout: float,
) -> None:
    """Replace outstanding moving setpoints with recent measured positions."""

    if not bridge.execute or not any(engaged.values()):
        return
    try:
        for _ in range(3):
            state = bridge.latest_state(max_age=feedback_timeout)
            targets = {
                side: state.positions[side] if engaged[side] else None
                for side in PIPER_SIDES
            }
            bridge.send_targets(
                left=targets["left"],
                right=targets["right"],
                active={side: False for side in PIPER_SIDES},
            )
            time.sleep(0.02)
    except (OSError, PiperSshError, ValueError):
        # The remote watchdog still stops command publication if the link is gone.
        pass


def run_teleoperation(
    args: argparse.Namespace,
    *,
    bridge: PiperSshBridge | None = None,
    bridge_description: str | None = None,
) -> int:
    if not math.isfinite(args.control_rate) or args.control_rate <= 0.0:
        raise ValueError("--control-rate must be positive")
    if not math.isfinite(args.feedback_timeout) or args.feedback_timeout <= 0.0:
        raise ValueError("--feedback-timeout must be positive")
    if not math.isfinite(args.release_timeout) or args.release_timeout <= 0.0:
        raise ValueError("--release-timeout must be positive")
    if not math.isfinite(args.scale_factor) or args.scale_factor <= 0.0:
        raise ValueError("--scale-factor must be positive")

    if bridge is None:
        bridge = create_bridge(args)
    provider = None
    engaged = {side: False for side in PIPER_SIDES}
    try:
        state = bridge.start()
        print_preflight(bridge, state, bridge_description)
        issues = preflight_issues(
            bridge, state, require_execute=args.execute and not args.check
        )
        if issues:
            for issue in issues:
                print(f"PRE-FLIGHT ERROR: {issue}", file=sys.stderr)
            return 2
        if args.check:
            print("Pre-flight check passed; no motion command was sent.")
            return 0

        # Importing the IK/XR stack is intentionally delayed so --check works on
        # a maintenance shell that only has Python and OpenSSH installed.
        try:
            from xrobotoolkit_teleop.headless.piper import (
                DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
                create_dual_piper_joint_target_provider,
            )
        except ImportError as exc:
            raise PiperSshError(
                f"XR/IK Python runtime failed in {sys.executable}: {exc}. "
                "Use native CPython 3.10 or newer (not GraalPython). On the "
                "AgileX computer, run "
                "`bash scripts/hardware/setup_piper_agilex_env.sh`, then "
                "`conda activate xrobotoolkit-native`."
            ) from exc

        manipulator_config = copy.deepcopy(DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG)
        if args.position_only:
            for config in manipulator_config.values():
                config["control_mode"] = "position_fixed_orientation"

        provider = create_dual_piper_joint_target_provider(
            scale_factor=args.scale_factor,
            control_rate_hz=args.control_rate,
            manipulator_config=manipulator_config,
        )
        wait_for_grips_released(provider, args.release_timeout)

        limiter = PiperCommandLimiter(
            max_joint_speed=args.max_joint_speed,
            max_gripper_speed=args.max_gripper_speed,
            max_joint_tracking_error=args.max_joint_tracking_error,
            max_gripper_tracking_error=args.max_gripper_tracking_error,
        )
        for side in PIPER_SIDES:
            limiter.reset(side, state.positions[side])

        mode = "LIVE" if args.execute else "DRY RUN"
        print(f"Dual-PiPER teleoperation started ({mode}). Hold a grip to move that arm; Ctrl+C stops.")
        previous_active = {side: False for side in PIPER_SIDES}
        period = 1.0 / args.control_rate
        previous_loop_time = time.monotonic()
        next_tick = previous_loop_time
        next_report = previous_loop_time

        while True:
            now = time.monotonic()
            state = bridge.latest_state(max_age=args.feedback_timeout)
            issues = [
                issue
                for side in PIPER_SIDES
                for issue in piper_status_issues(side, state.statuses.get(side))
            ]
            if issues:
                raise PiperSshError("arm fault detected: " + "; ".join(issues))

            command = provider.update(model_feedback(state))
            active = {
                side: bool(command.active.get(f"{side}_arm", False))
                for side in PIPER_SIDES
            }
            dt = max(0.0, now - previous_loop_time)
            previous_loop_time = now
            targets: dict[str, tuple[float, ...] | None] = {
                side: None for side in PIPER_SIDES
            }

            for side in PIPER_SIDES:
                if active[side]:
                    if not previous_active[side]:
                        limiter.reset(side, state.positions[side])
                        print(f"{side} arm active")
                    engaged[side] = True
                    desired = model_positions_to_piper_command(side, command.positions)
                    targets[side] = limiter.limit(
                        side, desired, state.positions[side], dt
                    )
                elif engaged[side]:
                    # A released grip is a dead-man stop: replace the prior goal
                    # with measured feedback and keep holding it.
                    if previous_active[side]:
                        print(f"{side} arm released; holding measured position")
                    limiter.reset(side, state.positions[side])
                    targets[side] = state.positions[side]
                previous_active[side] = active[side]

            if args.execute and any(target is not None for target in targets.values()):
                bridge.send_targets(
                    left=targets["left"],
                    right=targets["right"],
                    active=active,
                )

            if now >= next_report:
                active_names = [side for side in PIPER_SIDES if active[side]]
                print(
                    f"[{mode}] feedback_age={state.age:.3f}s, "
                    f"active={','.join(active_names) if active_names else 'none'}"
                )
                next_report = now + 2.0

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    finally:
        hold_measured_positions(bridge, engaged, args.feedback_timeout)
        if provider is not None:
            provider.close()
        bridge.close()


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        return run_teleoperation(args)
    except KeyboardInterrupt:
        print("\nTeleoperation stopped by user.")
        return 130
    except (PiperSshError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
