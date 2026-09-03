"""Local ROS 1 transport for dual AgileX PiPER arms.

The teleoperation process may run in a modern Conda environment while ROS
Noetic and ``piper_msgs`` belong to Ubuntu's system Python.  This transport
therefore starts the small ROS bridge helper with the ROS Python interpreter
on the same computer and exchanges JSON lines through local pipes.  No SSH or
network forwarding is involved.
"""

from __future__ import annotations

import shlex

from xrobotoolkit_teleop.hardware.interface.piper_ssh import (
    PiperSshBridge,
    PiperSshError,
    RemotePiperState,
    _REMOTE_BRIDGE_PATH,
)


DEFAULT_ROS_SETUP = "/opt/ros/noetic/setup.bash"
DEFAULT_PIPER_SETUP = (
    "/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
)
DEFAULT_ROS_PYTHON = "/usr/bin/python3"


class PiperRos1Bridge(PiperSshBridge):
    """Connect to the ROS 1 PiPER driver on the current computer."""

    def __init__(
        self,
        ros_setup: str = DEFAULT_ROS_SETUP,
        piper_setup: str = DEFAULT_PIPER_SETUP,
        ros_python: str = DEFAULT_ROS_PYTHON,
        left_feedback_topic: str = "/puppet/joint_left",
        right_feedback_topic: str = "/puppet/joint_right",
        left_status_topic: str = "/puppet/arm_status_left",
        right_status_topic: str = "/puppet/arm_status_right",
        left_command_topic: str = "/master/joint_left",
        right_command_topic: str = "/master/joint_right",
        feedback_rate_hz: float = 50.0,
        watchdog_timeout: float = 0.25,
        state_timeout: float = 0.4,
        execute: bool = False,
    ) -> None:
        if not ros_python:
            raise ValueError("ros_python must be a non-empty executable path")
        if not ros_setup:
            raise ValueError("ros_setup must be a non-empty setup.bash path")

        super().__init__(
            host="local ROS master",
            remote_setup="",
            left_feedback_topic=left_feedback_topic,
            right_feedback_topic=right_feedback_topic,
            left_status_topic=left_status_topic,
            right_status_topic=right_status_topic,
            left_command_topic=left_command_topic,
            right_command_topic=right_command_topic,
            feedback_rate_hz=feedback_rate_hz,
            watchdog_timeout=watchdog_timeout,
            state_timeout=state_timeout,
            connect_timeout=1.0,
            execute=execute,
            bridge_node_name="xrobotoolkit_piper_ros1_bridge",
        )
        self.ros_setup = ros_setup
        self.piper_setup = piper_setup
        self.ros_python = ros_python

    def start(self, timeout: float = 12.0) -> RemotePiperState:
        """Start the local ROS helper and wait for both arm feedback streams."""

        if not _REMOTE_BRIDGE_PATH.is_file():
            raise PiperSshError(
                f"ROS bridge source not found: {_REMOTE_BRIDGE_PATH}"
            )

        # Catkin setup files can reference unset variables, so nounset must not
        # be enabled here.  Every path and environment value remains shell-quoted.
        lines = ["set -e", f"source {shlex.quote(self.ros_setup)}"]
        if self.piper_setup:
            lines.append(f"source {shlex.quote(self.piper_setup)}")
        for name, value in self._bridge_environment().items():
            lines.append(f"export {name}={shlex.quote(value)}")
        lines.append(
            "exec "
            + shlex.quote(self.ros_python)
            + " -u "
            + shlex.quote(str(_REMOTE_BRIDGE_PATH))
        )

        return self._start_process(
            ["/bin/bash", "-lc", "\n".join(lines)],
            timeout=timeout,
            transport_label="local ROS 1",
        )
