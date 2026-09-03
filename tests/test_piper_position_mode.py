import copy
import unittest
from unittest.mock import patch

import numpy as np

from xrobotoolkit_teleop.hardware.interface.piper_ssh import (
    piper_feedback_to_model_positions,
)
from xrobotoolkit_teleop.headless.piper import (
    DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
    create_dual_piper_joint_target_provider,
)


class FakeXrClient:
    instance = None

    def __init__(self):
        self.keys = {
            "left_grip": 0.0,
            "right_grip": 1.0,
            "left_trigger": 0.0,
            "right_trigger": 0.0,
        }
        self.poses = {
            "left_controller": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "right_controller": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        FakeXrClient.instance = self

    def get_key_value_by_name(self, name):
        return self.keys[name]

    def get_pose_by_name(self, name):
        return self.poses[name]

    def get_motion_tracker_data(self):
        return {}

    def close(self):
        return None


class PiperPositionModeTest(unittest.TestCase):
    def test_translation_mode_ignores_controller_rotation(self):
        config = copy.deepcopy(DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG)
        for arm_config in config.values():
            arm_config["control_mode"] = "position_fixed_orientation"

        with patch(
            "xrobotoolkit_teleop.common.base_teleop_controller.XrClient",
            FakeXrClient,
        ):
            provider = create_dual_piper_joint_target_provider(
                manipulator_config=config,
                control_rate_hz=50.0,
            )

        positions = {}
        initial_arm = (0.0, 0.0, -0.5, 0.0, 1.07, 0.0, 0.08)
        for side in ("left", "right"):
            positions.update(
                piper_feedback_to_model_positions(side, initial_arm)
            )

        try:
            provider.update(positions)
            initial_target = provider.effector_task[
                "right_arm"
            ].T_world_frame.copy()

            # Controller +X/+Y/+Z maps to robot -Y/+Z/-X.  Include a 90-degree
            # controller rotation, which translation-only mode must ignore.
            FakeXrClient.instance.poses["right_controller"] = [
                0.1,
                0.2,
                -0.15,
                0.0,
                0.0,
                np.sqrt(0.5),
                np.sqrt(0.5),
            ]
            provider.update(positions)
            moved_target = provider.effector_task[
                "right_arm"
            ].T_world_frame.copy()

            np.testing.assert_allclose(
                moved_target[:3, :3], initial_target[:3, :3], atol=1e-10
            )
            np.testing.assert_allclose(
                moved_target[:3, 3] - initial_target[:3, 3],
                [0.15, -0.1, 0.2],
                atol=1e-10,
            )
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
