import argparse
import asyncio
import os
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from xrobotoolkit_teleop.integrations.robodojo.adapter import (
    BufferedXrInput,
    RoboDojo6DoFMapper,
    RoboDojoTeleopPolicy,
)
from xrobotoolkit_teleop.integrations.robodojo.preview import build_preview_mosaic
from xrobotoolkit_teleop.integrations.robodojo.remote_transport import (
    JPEG_FIELD,
    apply_render_preset,
    compress_observation,
    install_decimated_observation_adapter,
    parse_stream_cameras,
)
from xrobotoolkit_teleop.integrations.robodojo.runner import (
    build_isaac_client_command,
    build_parser,
    build_remote_isaac_client_command,
    build_server_command,
)
from xrobotoolkit_teleop.integrations.robodojo.server import add_xpolicylab_source


class FakeXrClient:
    def __init__(self):
        self.poses = {
            "left_controller": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "right_controller": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        self.keys = {
            "left_grip": 0.0,
            "right_grip": 0.0,
            "left_trigger": 0.0,
            "right_trigger": 0.0,
        }
        self.closed = False

    def get_pose_by_name(self, name):
        return self.poses[name]

    def get_key_value_by_name(self, name):
        return self.keys[name]

    def close(self):
        self.closed = True


class FakePreview:
    def __init__(self):
        self.observation = None
        self.closed = False

    def update(self, current_observation):
        self.observation = current_observation

    def close(self):
        self.closed = True


class FakeCameraManager:
    def __init__(self):
        self.num_cams = 3
        self.camera_names = [
            ["cam_head", "cam_left_wrist", "cam_right_wrist"],
        ]


class FakeCaptureManager:
    def __init__(self):
        self.step_calls = []

    def step(self, env_ids, cam_ids):
        self.step_calls.append(tuple(cam_ids))
        return [
            {
                "rgb": [
                    {
                        "data": np.full(
                            (4, 6, 4),
                            cam_id + 1,
                            dtype=np.uint8,
                        )
                    }
                    for _env_idx in env_ids
                ]
            }
            for cam_id in cam_ids
        ]


class FakeObsManager:
    ANNOTATORS_TO_COLLECT = {"rgb": "color"}

    def __init__(self, camera_manager, capture_manager):
        self.collect_freq = 25
        self.collect_shape = True
        self.collect_approximate_depth = False
        self.collect_depth = False
        self.collect_intrinsic_matrix = False
        self.collect_extrinsic_matrix = False
        self.camera_manager = camera_manager
        self.capture_manager = capture_manager

    def get_obs(self, env_idx_list):
        result = {}
        for env_idx in env_idx_list:
            result[env_idx] = {
                "vision": {},
                "state": {"counter": np.array([env_idx])},
            }
        return result


class FakeEvalEnv:
    def __init__(self):
        self.num_envs = 1
        self.camera_manager = FakeCameraManager()
        self.capture_manager = FakeCaptureManager()
        self.obs_manager = FakeObsManager(
            self.camera_manager,
            self.capture_manager,
        )
        self.render_count = 0
        self.stream_events = []
        self.reset_count = 0
        self.physx_monitor_enabled = False

    def reset(self, *args, **kwargs):
        self.reset_count += 1

    def render(self):
        self.render_count += 1

    def _stream_vision(self, env_idx, frame):
        self.stream_events.append(
            (
                env_idx,
                self.obs_manager.collect_freq,
                tuple(frame["vision"]),
            )
        )

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        self.render_count += 1
        if env_idx_list is None:
            env_idx_list = [0]
        data = self.obs_manager.get_obs(env_idx_list)
        observations = []
        for env_idx in env_idx_list:
            self._stream_vision(env_idx, data[env_idx])
            current = deepcopy(data[env_idx])
            current["env_idx"] = env_idx
            observations.append(current)
        return observations


def observation(left_pose=None, right_pose=None):
    if left_pose is None:
        left_pose = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    if right_pose is None:
        right_pose = [-0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    return {
        "state": {
            "left_ee_pose": np.asarray(
                left_pose,
                dtype=np.float64,
            ),
            "right_ee_pose": np.asarray(
                right_pose,
                dtype=np.float64,
            ),
            "left_ee_joint_state": np.array([1.0]),
            "right_ee_joint_state": np.array([1.0]),
        }
    }


class RoboDojo6DoFMapperTest(unittest.TestCase):
    def setUp(self):
        self.xr = FakeXrClient()
        self.mapper = RoboDojo6DoFMapper(
            xr_client=self.xr,
            scale_factor=1.0,
        )

    def tearDown(self):
        self.mapper.close()

    def test_inactive_arms_hold_observed_pose_and_map_triggers(self):
        self.xr.keys["left_trigger"] = 0.25
        self.xr.keys["right_trigger"] = 1.0

        action = self.mapper.infer(observation())[0]

        np.testing.assert_allclose(
            action["left_ee_pose"],
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(action["left_ee_joint_state"], [0.75])
        np.testing.assert_allclose(action["right_ee_joint_state"], [0.0])

    def test_grip_latches_without_a_jump_then_tracks_full_6dof(self):
        self.xr.keys["right_grip"] = 1.0

        anchored = self.mapper.infer(observation())[0]["right_ee_pose"]
        np.testing.assert_allclose(
            anchored,
            [-0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        )

        # Raw controller +X/+Y/-Z maps to RoboDojo -Y/+Z/+X. A +90 degree
        # rotation around raw +Z maps to a -90 degree rotation around world X.
        half_sqrt = np.sqrt(0.5)
        self.xr.poses["right_controller"] = np.array(
            [0.1, 0.2, -0.15, 0.0, 0.0, half_sqrt, half_sqrt]
        )
        moved = self.mapper.infer(observation())[0]["right_ee_pose"]

        np.testing.assert_allclose(
            moved[:3],
            [0.05, 0.1, 0.5],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            moved[3:],
            [half_sqrt, -half_sqrt, 0.0, 0.0],
            atol=1.0e-6,
        )

    def test_left_rotation_uses_grip_latched_tool_frame(self):
        xr = FakeXrClient()
        mapper = RoboDojo6DoFMapper(
            xr_client=xr,
            rotation_frame="tool",
            headset_to_world=np.eye(3),
        )
        half_sqrt = np.sqrt(0.5)
        left_pose = [
            0.1,
            0.2,
            0.3,
            half_sqrt,
            0.0,
            0.0,
            half_sqrt,
        ]

        # The controller begins at +90 degrees around Y while the end effector
        # begins at +90 degrees around Z. Grip must align their local frames
        # without requiring those two initial world orientations to match.
        xr.poses["left_controller"] = np.array(
            [0.0, 0.0, 0.0, 0.0, half_sqrt, 0.0, half_sqrt]
        )
        xr.keys["left_grip"] = 1.0
        anchored = mapper.infer(observation(left_pose=left_pose))[0]["left_ee_pose"]
        np.testing.assert_allclose(anchored, left_pose, atol=1.0e-6)

        # Rotate +90 degrees around the controller's local X axis. The same
        # local-X delta must be right-multiplied onto the latched EE pose.
        xr.poses["left_controller"] = np.array([0.0, 0.0, 0.0, 0.5, 0.5, -0.5, 0.5])
        moved = mapper.infer(observation(left_pose=left_pose))[0]["left_ee_pose"]
        np.testing.assert_allclose(
            moved[3:],
            [0.5, 0.5, 0.5, 0.5],
            atol=1.0e-6,
        )
        mapper.close()

    def test_release_and_reengage_reanchors_at_observed_pose(self):
        self.xr.keys["left_grip"] = 1.0
        self.mapper.infer(observation())
        self.xr.poses["left_controller"][:3] = [0.2, 0.0, 0.0]
        moved = self.mapper.infer(observation())[0]["left_ee_pose"].copy()

        self.xr.keys["left_grip"] = 0.0
        released = self.mapper.infer(observation(left_pose=moved))[0]["left_ee_pose"]
        np.testing.assert_allclose(released, moved)

        self.xr.poses["left_controller"][:3] = [0.8, 0.4, -0.3]
        self.xr.keys["left_grip"] = 1.0
        reanchored = self.mapper.infer(observation(left_pose=moved))[0]["left_ee_pose"]
        np.testing.assert_allclose(reanchored, moved)

    def test_invalid_controller_pose_holds_current_pose(self):
        self.xr.keys["left_grip"] = 1.0
        self.xr.poses["left_controller"][3:] = 0.0

        action = self.mapper.infer(observation())[0]

        np.testing.assert_allclose(
            action["left_ee_pose"],
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        )

    def test_missing_state_pose_is_rejected(self):
        obs = observation()
        del obs["state"]["right_ee_pose"]

        with self.assertRaisesRegex(ValueError, "right_ee_pose"):
            self.mapper.infer(obs)

    def test_async_policy_facade_returns_one_action_chunk(self):
        preview = FakePreview()
        policy = RoboDojoTeleopPolicy(
            mapper=self.mapper,
            preview=preview,
        )
        current_observation = observation()

        result = asyncio.run(policy.infer(current_observation))

        self.assertEqual(len(result), 1)
        self.assertIs(preview.observation, current_observation)
        self.assertEqual(
            set(result[0]),
            {
                "left_ee_pose",
                "left_ee_joint_state",
                "right_ee_pose",
                "right_ee_joint_state",
            },
        )
        policy.close()
        self.assertTrue(preview.closed)


class BufferedXrInputTest(unittest.TestCase):
    def test_sampling_replaces_one_complete_cached_snapshot(self):
        source = FakeXrClient()
        buffered = BufferedXrInput(source)
        buffered.sample(raise_on_error=True)

        source.poses["left_controller"][0] = 0.5
        source.keys["left_trigger"] = 0.75
        self.assertEqual(buffered.get_pose_by_name("left_controller")[0], 0.0)
        self.assertEqual(buffered.get_key_value_by_name("left_trigger"), 0.0)

        buffered.sample(raise_on_error=True)
        self.assertEqual(buffered.get_pose_by_name("left_controller")[0], 0.5)
        self.assertEqual(buffered.get_key_value_by_name("left_trigger"), 0.75)
        buffered.close()
        self.assertTrue(source.closed)


class RoboDojoPreviewTest(unittest.TestCase):
    def test_builds_head_and_wrist_camera_mosaic(self):
        current_observation = observation()
        current_observation["vision"] = {
            "cam_head": {
                "color": np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8),
            },
            "cam_left_wrist": {
                "color": np.full((480, 640, 3), [0, 255, 0], dtype=np.uint8),
            },
            "cam_right_wrist": {
                "color": np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8),
            },
        }

        mosaic = build_preview_mosaic(current_observation)

        self.assertIsNotNone(mosaic)
        self.assertEqual(mosaic.shape, (480, 960, 3))
        np.testing.assert_array_equal(mosaic[470, 10], [0, 0, 255])
        np.testing.assert_array_equal(mosaic[230, 950], [0, 255, 0])
        np.testing.assert_array_equal(mosaic[470, 950], [255, 0, 0])

    def test_missing_camera_observations_return_no_frame(self):
        self.assertIsNone(build_preview_mosaic(observation()))

    def test_head_only_transport_uses_the_full_preview_window(self):
        current_observation = observation()
        current_observation["vision"] = {
            "cam_head": {
                "color": np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8),
            },
        }

        preview = build_preview_mosaic(current_observation)

        self.assertIsNotNone(preview)
        self.assertEqual(preview.shape, (480, 640, 3))

    def test_jpeg_transport_round_trip_is_smaller_and_previewable(self):
        current_observation = observation()
        x = np.linspace(0, 255, 640, dtype=np.uint8)
        gradient = np.broadcast_to(x[None, :, None], (480, 640, 3)).copy()
        current_observation["vision"] = {
            camera_name: {"color": np.roll(gradient, index * 40, axis=1)}
            for index, camera_name in enumerate(
                ("cam_head", "cam_left_wrist", "cam_right_wrist")
            )
        }

        compressed, raw_bytes, jpeg_bytes = compress_observation(
            current_observation,
            jpeg_quality=75,
        )

        self.assertEqual(raw_bytes, 3 * 480 * 640 * 3)
        self.assertLess(jpeg_bytes, raw_bytes // 5)
        self.assertIn("color", current_observation["vision"]["cam_head"])
        self.assertNotIn("color", compressed["vision"]["cam_head"])
        self.assertIsInstance(
            compressed["vision"]["cam_head"][JPEG_FIELD],
            bytes,
        )
        mosaic = build_preview_mosaic(compressed)
        self.assertIsNotNone(mosaic)
        self.assertEqual(mosaic.shape, (480, 960, 3))


class RoboDojoRemoteRuntimeAdapterTest(unittest.TestCase):
    def test_minimal_render_preset_disables_rtx_quality_features(self):
        config = {"sim": {}}
        camera_templates = [
            {"resolution": (640, 480), "focal_length": 10.0},
            {"resolution": (640, 480), "focal_length": 13.0},
        ]

        resolution = apply_render_preset(
            config,
            "minimal",
            camera_templates=camera_templates,
        )

        self.assertEqual(resolution, (320, 240))
        self.assertEqual(config["sim"]["render"]["rendering_mode"], "performance")
        self.assertEqual(config["sim"]["render"]["antialiasing_mode"], "Off")
        self.assertFalse(config["sim"]["render"]["enable_reflections"])
        self.assertFalse(config["sim"]["render"]["enable_global_illumination"])
        self.assertFalse(config["sim"]["render"]["enable_shadows"])
        self.assertTrue(config["sim"]["render"]["enable_direct_lighting"])
        self.assertEqual(
            [template["resolution"] for template in camera_templates],
            [(320, 240), (320, 240)],
        )

    def test_quality_render_preset_leaves_configuration_untouched(self):
        config = {"sim": {"render": {"rendering_mode": "quality"}}}
        camera_template = {"resolution": (640, 480)}
        original_config = deepcopy(config)
        original_template = deepcopy(camera_template)

        resolution = apply_render_preset(
            config,
            "quality",
            camera_templates=[camera_template],
        )

        self.assertIsNone(resolution)
        self.assertEqual(config, original_config)
        self.assertEqual(camera_template, original_template)

    def test_head_preview_is_fresh_while_wrist_recording_is_decimated(self):
        env = FakeEvalEnv()
        install_decimated_observation_adapter(
            env,
            preview_every=1,
            vision_every=3,
            stream_cameras=("cam_head",),
        )

        observations = [env.get_obs_batch()[0] for _ in range(5)]

        self.assertEqual(env.render_count, 5)
        self.assertEqual(
            [tuple(item["vision"]) for item in observations],
            [
                ("cam_head",),
                ("cam_head",),
                ("cam_head",),
                ("cam_head",),
                ("cam_head",),
            ],
        )
        self.assertEqual(len(env.stream_events), 9)
        self.assertEqual(
            [event[2] for event in env.stream_events].count(("cam_head",)),
            5,
        )
        head_event = next(
            event for event in env.stream_events if event[2] == ("cam_head",)
        )
        wrist_event = next(
            event for event in env.stream_events if event[2] == ("cam_left_wrist",)
        )
        self.assertEqual(head_event[1], 25)
        self.assertAlmostEqual(wrist_event[1], 25 / 3)
        self.assertIsNotNone(env.obs_manager.camera_manager)
        self.assertIsNotNone(env.obs_manager.capture_manager)

        env.reset()
        self.assertEqual(tuple(env.get_obs_batch()[0]["vision"]), ("cam_head",))
        self.assertEqual(env.render_count, 6)

    def test_preview_decimation_uses_state_only_intermediate_observations(self):
        env = FakeEvalEnv()
        install_decimated_observation_adapter(
            env,
            preview_every=3,
            vision_every=3,
            stream_cameras=("cam_head",),
        )

        observations = [env.get_obs_batch()[0] for _ in range(5)]

        self.assertEqual(env.render_count, 2)
        self.assertEqual(
            [tuple(item["vision"]) for item in observations],
            [
                ("cam_head",),
                (),
                (),
                ("cam_head",),
                (),
            ],
        )

    def test_stream_camera_parser_supports_one_camera_or_all(self):
        self.assertEqual(parse_stream_cameras("cam_head"), ("cam_head",))
        self.assertEqual(
            parse_stream_cameras("cam_head,cam_left_wrist"),
            ("cam_head", "cam_left_wrist"),
        )
        self.assertIsNone(parse_stream_cameras("all"))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            parse_stream_cameras("all,cam_head")


class RoboDojoRunnerTest(unittest.TestCase):
    def _args(self, *, headless=False):
        return argparse.Namespace(
            robodojo_root=Path("/tmp/RoboDojo"),
            task="stack_bowls",
            env_cfg="arx_x5",
            env_gpu=0,
            policy_host="127.0.0.1",
            seed=0,
            eval_env="RoboDojo",
            headless=headless,
            image_codec="jpeg",
            jpeg_quality=75,
            preview_every=1,
            vision_every=3,
            stream_cameras="cam_head",
            render_preset="minimal",
            rotation_frame="tool",
            xr_sample_hz=60.0,
            preview_fps=10.0,
        )

    def test_gui_is_default_and_headless_is_explicit(self):
        gui_command = build_isaac_client_command(
            self._args(),
            19000,
            Path("/tmp/conda.sh"),
        )
        headless_command = build_isaac_client_command(
            self._args(headless=True),
            19000,
            Path("/tmp/conda.sh"),
        )

        self.assertNotIn("--headless", gui_command)
        self.assertIn("--headless", headless_command)
        self.assertIn("--enable_cameras", gui_command)
        self.assertIn("demo_policy", gui_command)

    def test_remote_defaults_use_huangkui_deployment(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.remote_robodojo_root, "/home/huangkui/RoboDojo")
        self.assertEqual(args.remote_conda_base, "/home/huangkui/miniconda3")
        self.assertEqual(args.vision_every, 3)
        self.assertEqual(args.preview_every, 1)
        self.assertEqual(args.stream_cameras, "cam_head")
        self.assertEqual(args.render_preset, "minimal")
        self.assertEqual(args.rotation_frame, "tool")
        self.assertEqual(args.xr_sample_hz, 60.0)

    def test_preview_is_forwarded_only_to_local_policy_server(self):
        args = build_parser().parse_args(["--preview", "--mock-xr"])

        command = build_server_command(args, 19000)

        self.assertIn("--preview", command)
        self.assertIn("--mock-xr", command)
        self.assertEqual(
            command[command.index("--rotation-frame") + 1],
            "tool",
        )

    def test_remote_command_uses_reverse_tunnel_and_external_workdir(self):
        args = self._args(headless=True)
        args.ssh_host = "cscg-g41"
        args.ssh_connect_timeout = 15
        args.remote_robodojo_root = "/home/huangkui/RoboDojo"
        args.remote_conda_base = "/home/huangkui/miniconda3"
        args.remote_workdir = "/home/huangkui/robodojo-teleop-runs"
        args.accept_eula = True

        command = build_remote_isaac_client_command(
            args,
            local_port=19000,
            remote_port=29000,
            run_id="teleop-test",
        )
        remote_command = command[-1]

        self.assertEqual(command[0], "ssh")
        self.assertIn("-tt", command)
        self.assertIn("cscg-g41", command)
        self.assertEqual(
            command[command.index("-R") + 1],
            "127.0.0.1:29000:127.0.0.1:19000",
        )
        self.assertIn(
            "/home/huangkui/RoboDojo/src/eval_client/main.py",
            remote_command,
        )
        self.assertIn("/home/huangkui/robodojo-teleop-runs", remote_command)
        self.assertIn("ws://127.0.0.1:29000", remote_command)
        self.assertIn("--headless", remote_command)
        self.assertIn("--portable-root", remote_command)
        self.assertIn("registryCacheFull", remote_command)
        self.assertIn("XDG_CACHE_HOME", remote_command)
        self.assertIn("PYTHONPYCACHEPREFIX", remote_command)
        self.assertIn("TMPDIR", remote_command)
        self.assertIn("sitecustomize.py", remote_command)
        self.assertIn("ROBODOJO_TELEOP_IMAGE_CODEC", remote_command)
        self.assertIn("ROBODOJO_TELEOP_JPEG_QUALITY", remote_command)
        self.assertIn("ROBODOJO_TELEOP_VISION_EVERY", remote_command)
        self.assertIn("ROBODOJO_TELEOP_PREVIEW_EVERY", remote_command)
        self.assertIn("ROBODOJO_TELEOP_STREAM_CAMERAS", remote_command)
        self.assertIn("ROBODOJO_TELEOP_RENDER_PRESET", remote_command)
        self.assertIn("minimal", remote_command)
        self.assertIn("jpeg", remote_command)
        self.assertIn("75", remote_command)
        self.assertIn("cam_head", remote_command)
        self.assertIn("Assets/Robots/x5/curobo.yml", remote_command)
        self.assertIn("libXt.so.6", remote_command)
        self.assertIn("libGLU.so.1", remote_command)
        self.assertIn("LD_LIBRARY_PATH", remote_command)
        self.assertIn("Refusing to write runtime output", remote_command)
        self.assertNotIn("scp ", remote_command)
        self.assertNotIn("rsync ", remote_command)

    def test_remote_paths_must_be_absolute(self):
        args = self._args(headless=True)
        args.ssh_host = "cscg-g41"
        args.ssh_connect_timeout = 15
        args.remote_robodojo_root = "relative/RoboDojo"
        args.remote_conda_base = "/home/huangkui/miniconda3"
        args.remote_workdir = ""
        args.accept_eula = False

        with self.assertRaisesRegex(ValueError, "absolute remote path"):
            build_remote_isaac_client_command(
                args,
                local_port=19000,
                remote_port=29000,
                run_id="teleop-test",
            )


class RoboDojoProtocolTest(unittest.TestCase):
    def test_xpolicylab_websocket_round_trip(self):
        robodojo_root = Path(os.environ.get("ROBODOJO_ROOT", "/home/arx/RoboDojo"))
        protocol_file = (
            robodojo_root / "XPolicyLab" / "client_server" / "ws" / "model_server.py"
        )
        if not protocol_file.is_file():
            self.skipTest("RoboDojo/XPolicyLab source is unavailable")

        add_xpolicylab_source(robodojo_root)
        from client_server.ws.model_server import (
            PolicyServer,
            PolicyServerConfig,
        )
        from client_server.ws.protocol.client import (
            PolicyEvalClient,
            PolicyEvalClientConfig,
        )

        async def round_trip():
            xr_client = FakeXrClient()
            mapper = RoboDojo6DoFMapper(xr_client=xr_client)
            preview = FakePreview()
            policy = RoboDojoTeleopPolicy(
                mapper=mapper,
                preview=preview,
            )
            server = PolicyServer(
                policy,
                PolicyServerConfig(host="127.0.0.1", port=0),
            )
            await server.start()
            client = PolicyEvalClient(
                PolicyEvalClientConfig(
                    url=server.url,
                    evaluation_id="teleop-adapter-test",
                )
            )
            try:
                await client.connect()
                await client.reset(
                    trial_id="trial-1",
                    action_case_id="stack_bowls_case",
                )
                current_observation = observation()
                current_observation["vision"] = {
                    "cam_head": {
                        "color": np.full(
                            (8, 12, 3),
                            127,
                            dtype=np.uint8,
                        )
                    }
                }
                response = await client.infer(
                    current_observation,
                    trial_id="trial-1",
                    action_case_id="stack_bowls_case",
                )
                actions = response.payload["actions"]
                self.assertEqual(len(actions), 1)
                self.assertIsNotNone(preview.observation)
                np.testing.assert_array_equal(
                    preview.observation["vision"]["cam_head"]["color"],
                    current_observation["vision"]["cam_head"]["color"],
                )
                np.testing.assert_allclose(
                    actions[0]["left_ee_pose"],
                    current_observation["state"]["left_ee_pose"],
                )
            finally:
                await client.close()
                await server.stop()
                policy.close()

        asyncio.run(round_trip())


if __name__ == "__main__":
    unittest.main()
