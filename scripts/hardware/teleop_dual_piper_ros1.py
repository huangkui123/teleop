#!/usr/bin/env python3
"""Run dual-PiPER XR teleoperation directly on the AgileX ROS 1 computer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_ROS_ENV_REEXEC_MARKER = "_XROBOT_PIPER_ROS_ENV_SANITIZED"
_ROS_ENV_NOTICE_MARKER = "_XROBOT_PIPER_ROS_ENV_NOTICE"


def _reexec_without_inherited_ros_python_paths() -> None:
    """Keep ROS Noetic's Python 3.8 ABI out of the CPython 3.10 IK process."""

    if os.environ.get(_ROS_ENV_REEXEC_MARKER) == "1":
        if os.environ.get(_ROS_ENV_NOTICE_MARKER) == "1":
            print(
                "Inherited ROS Python/library paths were isolated from the "
                "XR/IK process."
            )
        return

    inherited_paths = os.pathsep.join(
        os.environ.get(name, "") for name in ("PYTHONPATH", "LD_LIBRARY_PATH")
    )
    ros_path_markers = ("/opt/ros/", "/devel/lib", "/install/lib")
    if not any(marker in inherited_paths for marker in ros_path_markers):
        return

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    clean_env[_ROS_ENV_REEXEC_MARKER] = "1"
    clean_env[_ROS_ENV_NOTICE_MARKER] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        clean_env,
    )


_reexec_without_inherited_ros_python_paths()

# Support execution from a source checkout without installing the package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.hardware.teleop_dual_piper_ssh import run_teleoperation
from xrobotoolkit_teleop.hardware.interface.piper_ros1 import (
    DEFAULT_PIPER_SETUP,
    DEFAULT_ROS_PYTHON,
    DEFAULT_ROS_SETUP,
    PiperRos1Bridge,
)
from xrobotoolkit_teleop.hardware.interface.piper_ssh import PiperSshError


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teleoperate two PiPER arms from the AgileX computer through its "
            "local ROS 1 master. The default is a dry run; physical commands "
            "require --execute."
        )
    )
    parser.add_argument(
        "--ros-setup",
        default=DEFAULT_ROS_SETUP,
        help="ROS Noetic setup.bash used by the local ROS subprocess",
    )
    parser.add_argument(
        "--piper-setup",
        default=DEFAULT_PIPER_SETUP,
        help="PiPER catkin workspace setup.bash",
    )
    parser.add_argument(
        "--ros-python",
        default=DEFAULT_ROS_PYTHON,
        help="Python interpreter containing rospy and piper_msgs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check local ROS topics, feedback, arm status, and conflicts, then exit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow physical JointState commands (without this flag, run dry only)",
    )
    parser.add_argument(
        "--control-rate", type=float, default=50.0, help="control rate in Hz"
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1.0, help="XR translation scale"
    )
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
        help="gripper command slew limit in m/s",
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
        help="stop when local ROS feedback is older than this many seconds",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=0.25,
        help="stop publishing after this control-command gap in seconds",
    )
    parser.add_argument(
        "--release-timeout",
        type=float,
        default=30.0,
        help="time allowed for both Grip controls to be released at startup",
    )
    parser.add_argument(
        "--left-feedback-topic",
        default="/puppet/joint_left",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--right-feedback-topic",
        default="/puppet/joint_right",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--left-status-topic",
        default="/puppet/arm_status_left",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--right-status-topic",
        default="/puppet/arm_status_right",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--left-command-topic",
        default="/master/joint_left",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--right-command-topic",
        default="/master/joint_right",
        help=argparse.SUPPRESS,
    )
    return parser


def create_bridge(args: argparse.Namespace) -> PiperRos1Bridge:
    return PiperRos1Bridge(
        ros_setup=args.ros_setup,
        piper_setup=args.piper_setup,
        ros_python=args.ros_python,
        left_feedback_topic=args.left_feedback_topic,
        right_feedback_topic=args.right_feedback_topic,
        left_status_topic=args.left_status_topic,
        right_status_topic=args.right_status_topic,
        left_command_topic=args.left_command_topic,
        right_command_topic=args.right_command_topic,
        feedback_rate_hz=args.control_rate,
        watchdog_timeout=args.watchdog_timeout,
        state_timeout=args.feedback_timeout,
        execute=args.execute and not args.check,
    )


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        bridge = create_bridge(args)
        return run_teleoperation(
            args,
            bridge=bridge,
            bridge_description="Local ROS 1 PiPER bridge ready",
        )
    except KeyboardInterrupt:
        print("\nTeleoperation stopped by user.")
        return 130
    except (PiperSshError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
