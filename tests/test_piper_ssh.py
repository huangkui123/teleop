import math
import time
import unittest
from unittest.mock import patch

from xrobotoolkit_teleop.hardware.interface.piper_ros1 import PiperRos1Bridge
from xrobotoolkit_teleop.hardware.interface.piper_ssh import (
    PiperCommandLimiter,
    RemotePiperState,
    model_positions_to_piper_command,
    piper_feedback_to_model_positions,
    piper_status_issues,
)


class PiperJointMappingTest(unittest.TestCase):
    def test_feedback_and_command_round_trip(self):
        feedback = (0.1, 1.2, -1.0, 0.3, -0.4, 0.5, 0.08)

        model_positions = piper_feedback_to_model_positions("left", feedback)
        command = model_positions_to_piper_command("left", model_positions)

        self.assertEqual(command[:6], feedback[:6])
        self.assertAlmostEqual(command[6], feedback[6])
        self.assertAlmostEqual(model_positions["left_joint7"], 0.035)
        self.assertAlmostEqual(model_positions["left_joint8"], -0.035)

    def test_rejects_missing_or_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "7 values"):
            piper_feedback_to_model_positions("left", [0.0] * 6)
        with self.assertRaisesRegex(ValueError, "finite"):
            piper_feedback_to_model_positions(
                "right", [0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 0.0]
            )
        with self.assertRaisesRegex(ValueError, "missing right target joints"):
            model_positions_to_piper_command("right", {})


class PiperCommandLimiterTest(unittest.TestCase):
    def test_limits_rate_and_caps_delayed_dt(self):
        limiter = PiperCommandLimiter(
            max_joint_speed=0.6,
            max_gripper_speed=0.04,
            max_joint_tracking_error=1.0,
            max_gripper_tracking_error=1.0,
        )
        feedback = (0.0,) * 7
        limiter.reset("left", feedback)

        result = limiter.limit(
            "left", (1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.08), feedback, 10.0
        )

        self.assertAlmostEqual(result[0], 0.06)
        self.assertAlmostEqual(result[2], -0.06)
        self.assertAlmostEqual(result[6], 0.004)

    def test_limits_setpoint_lead_over_feedback(self):
        limiter = PiperCommandLimiter(
            max_joint_speed=100.0,
            max_gripper_speed=1.0,
            max_joint_tracking_error=0.2,
            max_gripper_tracking_error=0.01,
        )
        feedback = (0.0,) * 7
        limiter.reset("right", feedback)

        result = limiter.limit(
            "right", (1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.08), feedback, 0.1
        )

        self.assertAlmostEqual(result[0], 0.2)
        self.assertAlmostEqual(result[2], -0.2)
        self.assertAlmostEqual(result[6], 0.01)


class PiperStatusTest(unittest.TestCase):
    def test_healthy_and_fault_status(self):
        self.assertEqual(
            piper_status_issues(
                "left",
                {"arm_status": 0, "ctrl_mode": 1, "teach_status": 0, "err_code": 0},
            ),
            [],
        )
        issues = piper_status_issues(
            "right",
            {
                "arm_status": 7,
                "ctrl_mode": 1,
                "teach_status": 0,
                "err_code": 4,
                "joint_2_angle_limit": True,
                "communication_status_joint_5": True,
            },
        )
        self.assertIn("right: arm_status=7 (collision)", issues)
        self.assertIn("right: err_code=4", issues)
        self.assertIn("right: joint 2 angle-limit fault", issues)
        self.assertIn("right: joint 5 communication fault", issues)

    def test_state_age_includes_upstream_ros_age(self):
        state = RemotePiperState(
            positions={"left": (0.0,) * 7, "right": (0.0,) * 7},
            statuses={"left": {"err_code": 0}, "right": {"err_code": 0}},
            source_timestamp=time.time(),
            received_monotonic=time.monotonic(),
            sequence=1,
            feedback_age={"left": 0.75, "right": 0.01},
            status_age={"left": 0.01, "right": 0.02},
        )

        self.assertGreaterEqual(state.age, 0.75)


class PiperRos1BridgeTest(unittest.TestCase):
    def test_local_bridge_starts_ros_python_without_ssh(self):
        bridge = PiperRos1Bridge(
            ros_setup="/opt/ros/noetic/setup.bash",
            piper_setup="/tmp/piper/devel/setup.bash",
            ros_python="/usr/bin/python3",
            execute=True,
        )
        expected_state = object()

        with patch.object(
            bridge, "_start_process", return_value=expected_state
        ) as start_process:
            state = bridge.start(timeout=3.0)

        self.assertIs(state, expected_state)
        command = start_process.call_args.args[0]
        self.assertEqual(command[:2], ["/bin/bash", "-lc"])
        self.assertNotIn("ssh", command)
        shell_script = command[2]
        self.assertIn("source /opt/ros/noetic/setup.bash", shell_script)
        self.assertIn("source /tmp/piper/devel/setup.bash", shell_script)
        self.assertIn("exec /usr/bin/python3 -u", shell_script)
        self.assertIn("export XRT_ALLOW_EXECUTE=1", shell_script)
        self.assertIn(
            "export XRT_BRIDGE_NODE_NAME=xrobotoolkit_piper_ros1_bridge",
            shell_script,
        )
        self.assertEqual(start_process.call_args.kwargs["timeout"], 3.0)
        self.assertEqual(
            start_process.call_args.kwargs["transport_label"], "local ROS 1"
        )


if __name__ == "__main__":
    unittest.main()
