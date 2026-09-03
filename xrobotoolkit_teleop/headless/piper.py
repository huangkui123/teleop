"""Headless dual-PiPER teleoperation provider.

The provider keeps XR input, Placo IK, and gripper trigger handling inside the
teleop package while exposing only joint targets to downstream simulators.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


DEFAULT_DUAL_PIPER_URDF_PATH = os.path.join(
    ASSET_PATH, "agilex/piper/dual_piper.urdf"
)

DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG: dict[str, dict[str, Any]] = {
    "right_arm": {
        "link_name": "right_link6",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_joint7", "right_joint8"],
            "open_pos": [0.035, -0.035],
            "close_pos": [0.0, 0.0],
        },
    },
    "left_arm": {
        "link_name": "left_link6",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": ["left_joint7", "left_joint8"],
            "open_pos": [0.035, -0.035],
            "close_pos": [0.0, 0.0],
        },
    },
}


@dataclass(frozen=True)
class JointTargetCommand:
    """Joint targets produced by a headless teleoperation update."""

    positions: dict[str, float]
    active: dict[str, bool]
    timestamp: float = field(default_factory=time.time)


class DualPiperJointTargetProvider(BaseTeleopController):
    """Pull-based dual-PiPER XR/Placo teleoperation provider."""

    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
        manipulator_config: dict[str, dict[str, Any]]
        | None = None,
        floating_base: bool = False,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = 1.0,
        q_init: np.ndarray | None = None,
        dt: float = 1.0 / 60.0,
    ) -> None:
        self._current_positions: dict[str, float] = {}
        self._target_positions: dict[str, float] = {}
        self._initialized_target_joints: set[str] = set()
        self._arm_joint_names: dict[str, list[str]] = {}
        self._all_output_joint_names: list[str] = []
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config
            if manipulator_config is not None
            else DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
            floating_base=floating_base,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=q_init,
            dt=dt,
        )
        self.sync_end_effector_poses_to_placo_tasks()

    def _placo_setup(self) -> None:
        super()._placo_setup()
        self.solver.enable_joint_limits(True)
        self.solver.enable_velocity_limits(True)
        self._arm_joint_names = {}
        self._all_output_joint_names = []
        for arm_name, config in self.manipulator_config.items():
            prefix = _joint_prefix_from_link(config["link_name"])
            arm_joints = [f"{prefix}joint{i}" for i in range(1, 7)]
            self._arm_joint_names[arm_name] = arm_joints
            self._all_output_joint_names.extend(arm_joints)
            gripper_config = config.get("gripper_config")
            if gripper_config is not None:
                self._all_output_joint_names.extend(gripper_config["joint_names"])
        self._all_output_joint_names = list(dict.fromkeys(self._all_output_joint_names))
        self._target_positions = {}

    def _robot_setup(self) -> None:
        pass

    def _update_robot_state(self) -> None:
        for joint_name in self._all_output_joint_names:
            joint_pos = self._current_positions[joint_name]
            _set_placo_joint_position(self.placo_robot, joint_name, joint_pos)
        self.placo_robot.update_kinematics()

    def _send_command(self) -> None:
        targets: dict[str, float] = {}
        for arm_name, arm_joints in self._arm_joint_names.items():
            is_active = bool(self.active.get(arm_name, False))
            for joint_name in arm_joints:
                if is_active:
                    targets[joint_name] = _get_placo_joint_position(
                        self.placo_robot, joint_name
                    )
                else:
                    targets[joint_name] = self._target_positions.get(
                        joint_name, self._current_positions[joint_name]
                    )

        for gripper_name, gripper_target in self.gripper_pos_target.items():
            for joint_name, joint_pos in gripper_target.items():
                targets[joint_name] = float(joint_pos)

        self._target_positions.update(targets)

    def _get_link_pose(self, link_name: str):
        transform = self.placo_robot.get_T_world_frame(link_name)
        link_xyz = transform[:3, 3].copy()
        link_quat = tf.quaternion_from_matrix(transform)
        return link_xyz, link_quat

    def update(
        self, current_positions: Mapping[str, float]
    ) -> JointTargetCommand:
        """Update teleoperation once and return target joint positions by name."""
        provided = {
            name: float(current_positions[name])
            for name in self._all_output_joint_names
            if name in current_positions
        }
        missing = [
            name for name in self._all_output_joint_names if name not in provided
        ]
        if missing:
            raise ValueError(
                "current_positions is missing PiPER joints: " + ", ".join(missing)
            )
        nonfinite = [name for name, value in provided.items() if not np.isfinite(value)]
        if nonfinite:
            raise ValueError(
                "current joint positions must be finite: " + ", ".join(nonfinite)
            )
        self._current_positions.update(provided)
        self._initialize_targets_from_current_positions()
        self._update_ik()
        self._update_gripper_target()
        self._send_command()
        return JointTargetCommand(
            positions={
                name: float(self._target_positions[name])
                for name in self._all_output_joint_names
                if name in self._target_positions
            },
            active={name: bool(value) for name, value in self.active.items()},
        )

    def start(self) -> None:
        """No-op for API symmetry with process/thread based providers."""
        return None

    def _initialize_targets_from_current_positions(self) -> None:
        for joint_name in self._all_output_joint_names:
            if joint_name in self._initialized_target_joints:
                continue
            if joint_name not in self._current_positions:
                continue
            self._target_positions[joint_name] = self._current_positions[joint_name]
            self._initialized_target_joints.add(joint_name)

    def close(self) -> None:
        self._stop_event.set()
        close = getattr(self.xr_client, "close", None)
        if close is not None:
            close()

    def run(self) -> None:
        raise RuntimeError(
            "DualPiperJointTargetProvider is headless; call update() from a host loop."
        )


def create_dual_piper_joint_target_provider(
    urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
    scale_factor: float = 1.0,
    control_rate_hz: float = 60.0,
    manipulator_config: dict[str, dict[str, Any]] | None = None,
) -> DualPiperJointTargetProvider:
    """Create the default dual-PiPER joint-target provider."""
    if not np.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
        raise ValueError("control_rate_hz must be a positive finite value")
    return DualPiperJointTargetProvider(
        robot_urdf_path=urdf_path,
        manipulator_config=manipulator_config,
        scale_factor=scale_factor,
        dt=1.0 / control_rate_hz,
    )


def _joint_prefix_from_link(link_name: str) -> str:
    if not link_name.endswith("link6"):
        raise ValueError(
            f"expected a PiPER link6 end-effector name, got {link_name!r}"
        )
    return link_name[: -len("link6")]


def _get_placo_joint_position(robot, joint_name: str) -> float:
    return float(robot.state.q[robot.get_joint_offset(joint_name)])


def _set_placo_joint_position(robot, joint_name: str, joint_pos: float) -> None:
    robot.state.q[robot.get_joint_offset(joint_name)] = float(joint_pos)
