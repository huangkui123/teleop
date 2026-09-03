"""Runtime-only image compression shim for a remote RoboDojo client.

The launcher copies this module to the external remote work directory as
``sitecustomize.py``.  Python then installs the shim before RoboDojo imports
XPolicyLab, without changing either repository.
"""

from __future__ import annotations

import builtins
import os
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from types import MethodType, ModuleType
from typing import Any

JPEG_FIELD = "color_jpeg"
ENCODING_FIELD = "color_encoding"
_CLIENT_TARGET_MODULES = (
    "client_server.ws.protocol.client",
    "XPolicyLab.client_server.ws.protocol.client",
)
_EVAL_ENV_TARGET_MODULES = ("src.eval_client.eval_env",)
_ORIGINAL_IMPORT = builtins.__import__
_CLIENT_PATCHED = False
_EVAL_ENV_PATCHED = False
_NEED_CLIENT_PATCH = False
_NEED_EVAL_ENV_PATCH = False
_PREVIEW_EVERY = 1
_VISION_EVERY = 1
_STREAM_CAMERAS: tuple[str, ...] | None = None
_RENDER_PRESET = "quality"
_MINIMAL_CAMERA_RESOLUTION = (320, 240)
_MINIMAL_RENDER_OVERRIDES = {
    "rendering_mode": "performance",
    "enable_translucency": False,
    "enable_reflections": False,
    "enable_global_illumination": False,
    "antialiasing_mode": "Off",
    "enable_dlssg": False,
    "enable_dl_denoiser": False,
    "dlss_mode": 0,
    "enable_direct_lighting": True,
    "samples_per_pixel": 1,
    "enable_shadows": False,
    "enable_ambient_occlusion": False,
}


def _rgb_uint8(value: Any, *, camera_name: str):
    import numpy as np

    frame = np.asarray(value)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=2)
        else:
            frame = frame[:, :, :3]
    elif frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame[:3], 0, -1)
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=2)
    else:
        raise ValueError(f"{camera_name} image must be HxW, HxWxC, or CxHxW; " f"got {frame.shape}")

    if np.issubdtype(frame.dtype, np.floating):
        finite = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        frame = np.clip(finite, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def compress_observation(
    observation: Any,
    *,
    jpeg_quality: int,
) -> tuple[Any, int, int]:
    """Replace RGB arrays with JPEG bytes in a shallow observation copy."""

    if not isinstance(observation, Mapping):
        return observation, 0, 0
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        return observation, 0, 0

    import cv2

    compressed_vision: dict[str, Any] = {}
    raw_bytes = 0
    jpeg_bytes = 0
    changed = False
    for camera_name, camera_data in vision.items():
        if not isinstance(camera_data, Mapping) or "color" not in camera_data:
            compressed_vision[camera_name] = camera_data
            continue

        rgb = _rgb_uint8(camera_data["color"], camera_name=str(camera_name))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for camera {camera_name}")

        camera_copy = dict(camera_data)
        camera_copy.pop("color", None)
        payload = encoded.tobytes()
        camera_copy[JPEG_FIELD] = payload
        camera_copy[ENCODING_FIELD] = "jpeg"
        camera_copy.setdefault("shape", tuple(rgb.shape))
        compressed_vision[str(camera_name)] = camera_copy
        raw_bytes += int(rgb.nbytes)
        jpeg_bytes += len(payload)
        changed = True

    if not changed:
        return observation, 0, 0
    compressed = dict(observation)
    compressed["vision"] = compressed_vision
    return compressed, raw_bytes, jpeg_bytes


def parse_stream_cameras(value: str) -> tuple[str, ...] | None:
    """Return selected camera names, or ``None`` when all should be sent."""

    names = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not names:
        raise ValueError("stream camera list cannot be empty")
    if "all" in names:
        if names != ("all",):
            raise ValueError("'all' cannot be combined with individual stream cameras")
        return None
    return names


def apply_render_preset(
    config: Any,
    preset: str,
    *,
    camera_templates: Any = (),
) -> tuple[int, int] | None:
    """Apply a teleop-only render profile before RoboDojo creates the env.

    ``quality`` leaves RoboDojo's configuration untouched. ``minimal`` keeps
    direct lighting for visibility, disables expensive RTX effects and
    anti-aliasing, and renders each camera at one quarter of the original
    640x480 pixel count.
    """

    if preset not in {"minimal", "quality"}:
        raise ValueError(f"unsupported render preset: {preset}")
    if preset == "quality":
        return None

    try:
        sim_config = config["sim"]
    except (KeyError, TypeError) as exc:
        raise ValueError("RoboDojo config has no simulation section") from exc

    render_config = sim_config.get("render")
    if render_config is None:
        sim_config["render"] = {}
        render_config = sim_config["render"]
    for key, value in _MINIMAL_RENDER_OVERRIDES.items():
        render_config[key] = value

    for camera_template in camera_templates:
        camera_template["resolution"] = _MINIMAL_CAMERA_RESOLUTION
    return _MINIMAL_CAMERA_RESOLUTION


def _retain_stream_cameras(
    observations: list[Any],
    stream_cameras: tuple[str, ...] | None,
) -> list[Any]:
    if stream_cameras is None:
        return observations
    selected = set(stream_cameras)
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        vision = observation.get("vision")
        if isinstance(vision, Mapping):
            observation["vision"] = {
                camera_name: camera_data
                for camera_name, camera_data in vision.items()
                if camera_name in selected
            }
    return observations


def _state_only_observations(env: Any, env_idx_list: Any) -> list[Any]:
    """Read robot state without rendering or copying camera buffers to CPU."""

    if getattr(env, "physx_monitor_enabled", False):
        env._check_physx_broken_envs()
    if env_idx_list is None:
        env_idx_list = list(range(env.num_envs))
    else:
        env_idx_list = list(env_idx_list)
    if getattr(env, "physx_monitor_enabled", False):
        env._check_endpose_finite(env_idx_list)

    obs_manager = env.obs_manager
    camera_manager = getattr(obs_manager, "camera_manager", None)
    capture_manager = getattr(obs_manager, "capture_manager", None)
    try:
        obs_manager.camera_manager = None
        obs_manager.capture_manager = None
        data = obs_manager.get_obs(env_idx_list=env_idx_list)
    finally:
        obs_manager.camera_manager = camera_manager
        obs_manager.capture_manager = capture_manager

    observations = []
    for env_idx in env_idx_list:
        env_data = deepcopy(data[env_idx])
        env_data["env_idx"] = env_idx
        observations.append(env_data)
    return observations


def _capture_selected_observations(
    env: Any,
    env_idx_list: Any,
    cam_ids: list[int],
) -> list[Any]:
    """Render once and append only the requested camera buffers to fresh state."""

    import numpy as np

    if env_idx_list is None:
        env_idx_list = list(range(env.num_envs))
    else:
        env_idx_list = list(env_idx_list)

    env.render()
    observations = _state_only_observations(env, env_idx_list)
    observations_by_env = {
        env_idx: observations[index]
        for index, env_idx in enumerate(env_idx_list)
    }
    captured = env.capture_manager.step(
        env_ids=env_idx_list,
        cam_ids=cam_ids,
    )
    obs_manager = env.obs_manager
    annotator_names = obs_manager.ANNOTATORS_TO_COLLECT

    for captured_index, cam_id in enumerate(cam_ids):
        cam_data = captured[captured_index]
        for annotator_name, env_list in cam_data.items():
            collect_name = annotator_names.get(annotator_name)
            if collect_name is None:
                continue
            for list_index, env_idx in enumerate(env_idx_list):
                camera_name = env.camera_manager.camera_names[env_idx][cam_id]
                camera = observations_by_env[env_idx]["vision"].setdefault(
                    camera_name,
                    {},
                )
                value = env_list[list_index]["data"]
                if annotator_name == "rgb":
                    value = value[:, :, :3]
                    if obs_manager.collect_shape:
                        camera["shape"] = value.shape
                camera[collect_name] = value

    for observation in observations:
        for camera in observation["vision"].values():
            if "distance_to_image_plane" not in camera:
                continue
            raw = camera.pop("distance_to_image_plane")
            if hasattr(raw, "ndim") and raw.ndim >= 1 and raw.shape[-1] == 1:
                raw = raw.squeeze(-1)
            if obs_manager.collect_approximate_depth:
                camera["approximate_depth"] = np.clip(
                    raw * 1000,
                    0,
                    65535,
                ).astype(np.uint16)
            if obs_manager.collect_depth:
                camera["depth"] = raw.astype(np.float32)

    if (
        obs_manager.collect_intrinsic_matrix
        or obs_manager.collect_extrinsic_matrix
    ):
        for cam_id in cam_ids:
            for env_idx in env_idx_list:
                camera_name = env.camera_manager.camera_names[env_idx][cam_id]
                camera = observations_by_env[env_idx]["vision"][camera_name]
                if obs_manager.collect_intrinsic_matrix:
                    camera["intrinsic_matrix"] = (
                        env.camera_manager.get_camera_intrinsics(
                            cam_id,
                            env_idx,
                        )
                    )
                if obs_manager.collect_extrinsic_matrix:
                    camera["extrinsic_matrix"] = (
                        env.camera_manager.get_camera_extrinsics(
                            cam_id,
                            env_idx,
                        )
                    )
    return observations


def install_decimated_observation_adapter(
    env: Any,
    *,
    preview_every: int,
    vision_every: int,
    stream_cameras: tuple[str, ...] | None,
) -> Any:
    """Decouple state/control observations from expensive camera observations.

    Cameras sent to the preview are captured every ``preview_every`` control
    observations. Remaining cameras are captured every ``vision_every``
    observations for remote recording. A request with no due camera reads only
    robot state and skips rendering and GPU-to-CPU image copies.
    """

    if preview_every < 1:
        raise ValueError("preview_every must be at least 1")
    if vision_every < 1:
        raise ValueError("vision_every must be at least 1")
    if getattr(env, "_robodojo_teleop_decimated_observations", False):
        return env

    original_reset = env.reset
    original_stream_vision = getattr(env, "_stream_vision", None)
    cadence = {
        "step": 0,
        "capture_batches": 0,
        "preview_frames": 0,
        "started_at": None,
    }

    def reset_cadence(self, *args, **kwargs):
        cadence["step"] = 0
        cadence["capture_batches"] = 0
        cadence["preview_frames"] = 0
        cadence["started_at"] = None
        return original_reset(*args, **kwargs)

    def decimated_get_obs_batch(self, env_idx_list=None, last_frame=False):
        if cadence["started_at"] is None:
            cadence["started_at"] = time.monotonic()

        if env_idx_list is None:
            resolved_env_ids = list(range(self.num_envs))
        else:
            resolved_env_ids = list(env_idx_list)

        all_cam_ids = list(range(self.camera_manager.num_cams))
        if stream_cameras is None:
            preview_cam_ids = all_cam_ids
        else:
            selected = set(stream_cameras)
            preview_cam_ids = [
                cam_id
                for cam_id in all_cam_ids
                if any(
                    self.camera_manager.camera_names[env_idx][cam_id]
                    in selected
                    for env_idx in resolved_env_ids
                )
            ]
        preview_cam_id_set = set(preview_cam_ids)
        record_only_cam_ids = [
            cam_id
            for cam_id in all_cam_ids
            if cam_id not in preview_cam_id_set
        ]

        preview_due = bool(last_frame) or cadence["step"] % preview_every == 0
        record_due = bool(last_frame) or cadence["step"] % vision_every == 0
        due_cam_ids = []
        if preview_due:
            due_cam_ids.extend(preview_cam_ids)
        if record_due:
            due_cam_ids.extend(record_only_cam_ids)
        due_cam_ids = list(dict.fromkeys(due_cam_ids))

        if due_cam_ids:
            observations = _capture_selected_observations(
                self,
                resolved_env_ids,
                due_cam_ids,
            )
            cadence["capture_batches"] += 1
            if preview_due and preview_cam_ids:
                cadence["preview_frames"] += 1
            if callable(original_stream_vision):
                selected = (
                    {
                        self.camera_manager.camera_names[env_idx][cam_id]
                        for env_idx in resolved_env_ids
                        for cam_id in preview_cam_ids
                    }
                    if stream_cameras is not None
                    else None
                )
                for observation in observations:
                    env_idx = observation["env_idx"]
                    for camera_name, camera_data in observation["vision"].items():
                        interval = (
                            preview_every
                            if selected is None or camera_name in selected
                            else vision_every
                        )
                        collect_freq = self.obs_manager.collect_freq
                        try:
                            self.obs_manager.collect_freq = (
                                collect_freq / interval
                            )
                            original_stream_vision(
                                env_idx,
                                {
                                    "vision": {
                                        camera_name: camera_data,
                                    }
                                },
                            )
                        finally:
                            self.obs_manager.collect_freq = collect_freq
        else:
            observations = _state_only_observations(
                self,
                resolved_env_ids,
            )

        if not last_frame:
            cadence["step"] += 1
            if cadence["step"] % 100 == 0:
                elapsed = max(
                    1.0e-9,
                    time.monotonic() - cadence["started_at"],
                )
                print(
                    "[robodojo-teleop] Control loop: "
                    f"{cadence['step'] / elapsed:.2f} Hz, "
                    f"preview frames={cadence['preview_frames']}/"
                    f"{cadence['step']}, "
                    f"camera batches={cadence['capture_batches']}",
                    flush=True,
                )
        return _retain_stream_cameras(observations, stream_cameras)

    env.reset = MethodType(reset_cadence, env)
    env.get_obs_batch = MethodType(decimated_get_obs_batch, env)
    env._robodojo_teleop_decimated_observations = True
    camera_label = "all" if stream_cameras is None else ",".join(stream_cameras)
    print(
        "[robodojo-teleop] Decoupled control/video active: "
        f"preview every {preview_every} control step(s), "
        f"other cameras every {vision_every} control step(s), "
        f"network cameras={camera_label}",
        flush=True,
    )
    return env


def _patch_policy_client(module: ModuleType, *, jpeg_quality: int) -> None:
    global _CLIENT_PATCHED
    if _CLIENT_PATCHED:
        return
    client_class = getattr(module, "PolicyEvalClient", None)
    if client_class is None or getattr(
        client_class,
        "_robodojo_teleop_jpeg_patch",
        False,
    ):
        return

    original_infer = client_class.infer
    first_report = True
    failure_reported = False

    async def compressed_infer(self, observation, *args, **kwargs):
        nonlocal first_report, failure_reported
        try:
            payload, raw_bytes, jpeg_bytes = compress_observation(
                observation,
                jpeg_quality=jpeg_quality,
            )
            if first_report and raw_bytes:
                ratio = raw_bytes / max(1, jpeg_bytes)
                print(
                    "[robodojo-teleop] JPEG transport active: "
                    f"{raw_bytes / 1024:.1f} KiB -> "
                    f"{jpeg_bytes / 1024:.1f} KiB "
                    f"({ratio:.1f}x, quality={jpeg_quality})",
                    flush=True,
                )
                first_report = False
        except Exception as exc:  # noqa: BLE001 - transport must fail open
            payload = observation
            if not failure_reported:
                print(
                    "[robodojo-teleop] WARNING: JPEG transport failed; " f"falling back to raw RGB: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                failure_reported = True
        return await original_infer(self, payload, *args, **kwargs)

    client_class.infer = compressed_infer
    client_class._robodojo_teleop_jpeg_patch = True
    _CLIENT_PATCHED = True


def _patch_eval_env(module: ModuleType) -> None:
    global _EVAL_ENV_PATCHED
    if _EVAL_ENV_PATCHED:
        return
    original_create_eval_env = getattr(module, "create_eval_env", None)
    if original_create_eval_env is None or getattr(
        original_create_eval_env,
        "_robodojo_teleop_decimation_patch",
        False,
    ):
        return

    def create_eval_env_with_decimated_vision(*args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is None:
            raise RuntimeError("RoboDojo create_eval_env did not receive a config")

        from env.camera_manager import camera_manager as camera_manager_module

        resolution = apply_render_preset(
            config,
            _RENDER_PRESET,
            camera_templates=camera_manager_module.REAL_MAP.values(),
        )
        if resolution is None:
            print(
                "[robodojo-teleop] Render preset: quality "
                "(RoboDojo defaults unchanged)",
                flush=True,
            )
        else:
            print(
                "[robodojo-teleop] Render preset: minimal "
                f"({resolution[0]}x{resolution[1]}, performance mode, "
                "RTX effects and anti-aliasing disabled)",
                flush=True,
            )

        env = original_create_eval_env(*args, **kwargs)
        return install_decimated_observation_adapter(
            env,
            preview_every=_PREVIEW_EVERY,
            vision_every=_VISION_EVERY,
            stream_cameras=_STREAM_CAMERAS,
        )

    create_eval_env_with_decimated_vision._robodojo_teleop_decimation_patch = True
    module.create_eval_env = create_eval_env_with_decimated_vision
    _EVAL_ENV_PATCHED = True


def _restore_import_hook_if_ready() -> None:
    client_ready = not _NEED_CLIENT_PATCH or _CLIENT_PATCHED
    eval_env_ready = not _NEED_EVAL_ENV_PATCH or _EVAL_ENV_PATCHED
    if client_ready and eval_env_ready and builtins.__import__ is _transport_import:
        builtins.__import__ = _ORIGINAL_IMPORT


def _transport_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if _NEED_CLIENT_PATCH:
        for module_name in _CLIENT_TARGET_MODULES:
            target = sys.modules.get(module_name)
            if target is not None:
                _patch_policy_client(
                    target,
                    jpeg_quality=int(os.environ.get("ROBODOJO_TELEOP_JPEG_QUALITY", "75")),
                )
    if _NEED_EVAL_ENV_PATCH:
        for module_name in _EVAL_ENV_TARGET_MODULES:
            target = sys.modules.get(module_name)
            if target is not None:
                _patch_eval_env(target)
    _restore_import_hook_if_ready()
    return module


def install_from_environment() -> None:
    """Install runtime-only transport and control/video decoupling hooks."""

    global _NEED_CLIENT_PATCH
    global _NEED_EVAL_ENV_PATCH
    global _PREVIEW_EVERY
    global _RENDER_PRESET
    global _STREAM_CAMERAS
    global _VISION_EVERY

    codec = os.environ.get("ROBODOJO_TELEOP_IMAGE_CODEC", "raw")
    if codec not in {"raw", "jpeg"}:
        raise RuntimeError("ROBODOJO_TELEOP_IMAGE_CODEC must be 'raw' or 'jpeg'")
    quality = int(os.environ.get("ROBODOJO_TELEOP_JPEG_QUALITY", "75"))
    if not 1 <= quality <= 100:
        raise RuntimeError("ROBODOJO_TELEOP_JPEG_QUALITY must be between 1 and 100")

    _PREVIEW_EVERY = int(
        os.environ.get("ROBODOJO_TELEOP_PREVIEW_EVERY", "1")
    )
    if _PREVIEW_EVERY < 1:
        raise RuntimeError("ROBODOJO_TELEOP_PREVIEW_EVERY must be at least 1")
    _VISION_EVERY = int(os.environ.get("ROBODOJO_TELEOP_VISION_EVERY", "1"))
    if _VISION_EVERY < 1:
        raise RuntimeError("ROBODOJO_TELEOP_VISION_EVERY must be at least 1")
    try:
        _STREAM_CAMERAS = parse_stream_cameras(
            os.environ.get("ROBODOJO_TELEOP_STREAM_CAMERAS", "all")
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid ROBODOJO_TELEOP_STREAM_CAMERAS: {exc}") from exc
    _RENDER_PRESET = os.environ.get(
        "ROBODOJO_TELEOP_RENDER_PRESET",
        "quality",
    )
    if _RENDER_PRESET not in {"minimal", "quality"}:
        raise RuntimeError(
            "ROBODOJO_TELEOP_RENDER_PRESET must be 'minimal' or 'quality'"
        )

    _NEED_CLIENT_PATCH = codec == "jpeg"
    _NEED_EVAL_ENV_PATCH = (
        _PREVIEW_EVERY > 1
        or _VISION_EVERY > 1
        or _STREAM_CAMERAS is not None
        or _RENDER_PRESET != "quality"
    )
    if not _NEED_CLIENT_PATCH and not _NEED_EVAL_ENV_PATCH:
        return

    if _NEED_CLIENT_PATCH:
        for module_name in _CLIENT_TARGET_MODULES:
            target = sys.modules.get(module_name)
            if target is not None:
                _patch_policy_client(target, jpeg_quality=quality)
    if _NEED_EVAL_ENV_PATCH:
        for module_name in _EVAL_ENV_TARGET_MODULES:
            target = sys.modules.get(module_name)
            if target is not None:
                _patch_eval_env(target)
    _restore_import_hook_if_ready()
    if (
        (_NEED_CLIENT_PATCH and not _CLIENT_PATCHED)
        or (_NEED_EVAL_ENV_PATCH and not _EVAL_ENV_PATCHED)
    ):
        builtins.__import__ = _transport_import


if __name__ == "sitecustomize":
    install_from_environment()
