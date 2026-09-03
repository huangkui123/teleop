"""Map XR controller motion to RoboDojo dual-arm end-effector actions.

RoboDojo runs inside Isaac Sim's Python environment while XRoboToolkit uses a
separate Python runtime.  This module deliberately has no Isaac Sim imports:
it turns observations received over XPolicyLab's WebSocket protocol into
RoboDojo-compatible actions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

LOGGER = logging.getLogger(__name__)
ARM_SIDES = ("left", "right")
XR_POSE_NAMES = ("left_controller", "right_controller")
XR_KEY_NAMES = ("left_grip", "right_grip", "left_trigger", "right_trigger")
ROTATION_FRAMES = ("tool", "world")
_QUAT_EPS = 1.0e-8


class XrInput(Protocol):
    """Small portion of :class:`XrClient` used by the adapter."""

    def get_pose_by_name(self, name: str) -> np.ndarray: ...

    def get_key_value_by_name(self, name: str) -> float: ...

    def close(self) -> None: ...


class ObservationPreview(Protocol):
    """Optional consumer for images included in RoboDojo observations."""

    def update(self, observation: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class BufferedXrInput:
    """Poll an XR input source independently and expose its latest snapshot.

    The policy request rate is determined by the remote simulator.  Keeping a
    separately sampled snapshot prevents slow camera requests from determining
    when controller poses are read.  ``sample`` is intentionally synchronous so
    the server can call the native XR SDK only from its event-loop thread.
    """

    def __init__(self, source: XrInput) -> None:
        self.source = source
        self._poses: dict[str, np.ndarray] = {}
        self._keys: dict[str, float] = {}
        self._sample_error_reported = False
        self._closed = False

    def sample(self, *, raise_on_error: bool = False) -> bool:
        """Atomically replace the cache with one complete controller snapshot."""

        try:
            poses: dict[str, np.ndarray] = {}
            for name in XR_POSE_NAMES:
                pose = np.asarray(
                    self.source.get_pose_by_name(name),
                    dtype=np.float64,
                )
                if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                    raise ValueError(f"{name} pose must be a finite shape-(7,) array")
                poses[name] = pose.copy()

            keys: dict[str, float] = {}
            for name in XR_KEY_NAMES:
                value = float(self.source.get_key_value_by_name(name))
                if not np.isfinite(value):
                    raise ValueError(f"{name} must be finite")
                keys[name] = value
        except Exception as exc:
            if raise_on_error:
                raise
            if not self._sample_error_reported:
                LOGGER.warning(
                    "XR sampling failed; retaining the last complete controller "
                    "snapshot: %s",
                    exc,
                )
                self._sample_error_reported = True
            return False

        self._poses = poses
        self._keys = keys
        if self._sample_error_reported:
            LOGGER.info("XR controller sampling recovered")
        self._sample_error_reported = False
        return True

    def get_pose_by_name(self, name: str) -> np.ndarray:
        try:
            return self._poses[name].copy()
        except KeyError as exc:
            raise RuntimeError(
                "XR snapshot is not initialized; call sample() first"
            ) from exc

    def get_key_value_by_name(self, name: str) -> float:
        try:
            return self._keys[name]
        except KeyError as exc:
            raise RuntimeError(
                "XR snapshot is not initialized; call sample() first"
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.source.close()


@dataclass
class _ArmReference:
    """Controller and robot poses latched when Grip becomes active."""

    active: bool = False
    controller_position: np.ndarray | None = None
    controller_orientation: np.ndarray | None = None
    end_effector_position: np.ndarray | None = None
    end_effector_orientation: np.ndarray | None = None

    def clear(self) -> None:
        self.active = False
        self.controller_position = None
        self.controller_orientation = None
        self.end_effector_position = None
        self.end_effector_orientation = None


def _normalize_quaternion(quaternion: Any, *, label: str) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"{label} must have shape (4,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain finite values")
    norm = float(np.linalg.norm(value))
    if norm <= _QUAT_EPS:
        raise ValueError(f"{label} has zero length")
    return value / norm


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[1:] *= -1.0
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in ``[w, x, y, z]`` order."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to ``[w, x, y, z]``."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _normalize_quaternion(quaternion, label="basis quaternion")


def _validate_basis(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"headset_to_world must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("headset_to_world must contain finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-6):
        raise ValueError("headset_to_world must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-6):
        raise ValueError("headset_to_world must be a proper rotation")
    return matrix


def _bounded_input(value: Any, *, label: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    return float(np.clip(scalar, 0.0, 1.0))


class RoboDojo6DoFMapper:
    """Pull XR input and produce one dual-X5 RoboDojo action.

    Each arm uses its Grip button as a dead-man switch.  Pressing Grip latches
    the current controller and simulated end-effector poses; subsequent
    controller translation and rotation are applied relative to those poses.
    Releasing Grip immediately tracks the observed pose and clears the latch,
    so pressing it again cannot cause a jump.
    """

    def __init__(
        self,
        xr_client: XrInput | None = None,
        *,
        scale_factor: float = 1.0,
        grip_threshold: float = 0.9,
        rotation_frame: str = "tool",
        headset_to_world: Any = R_HEADSET_TO_WORLD,
    ) -> None:
        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("scale_factor must be a positive finite value")
        if not np.isfinite(grip_threshold) or not 0.0 <= grip_threshold <= 1.0:
            raise ValueError("grip_threshold must be between 0 and 1")
        if rotation_frame not in ROTATION_FRAMES:
            raise ValueError(
                f"rotation_frame must be one of {ROTATION_FRAMES}, got {rotation_frame!r}"
            )

        if xr_client is None:
            # Lazy import keeps unit tests and non-XR tooling independent from
            # the CPython-version-specific XRoboToolkit SDK extension.
            from xrobotoolkit_teleop.common.xr_client import XrClient

            xr_client = XrClient()

        self.xr_client = xr_client
        self.scale_factor = float(scale_factor)
        self.grip_threshold = float(grip_threshold)
        self.rotation_frame = rotation_frame
        self.headset_to_world = _validate_basis(headset_to_world)
        self._basis_quaternion = _quaternion_from_matrix(self.headset_to_world)
        self._basis_quaternion_inverse = _quaternion_conjugate(self._basis_quaternion)
        self._references = {side: _ArmReference() for side in ARM_SIDES}
        self._invalid_tracking_reported = {side: False for side in ARM_SIDES}

    def reset(self) -> None:
        for reference in self._references.values():
            reference.clear()
        for side in ARM_SIDES:
            self._invalid_tracking_reported[side] = False

    def close(self) -> None:
        close = getattr(self.xr_client, "close", None)
        if callable(close):
            close()

    def infer(self, observation: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
        """Return a one-step action chunk expected by ``demo_policy.deploy``."""

        if not isinstance(observation, Mapping):
            raise TypeError("RoboDojo observation must be a mapping")
        state = observation.get("state")
        if not isinstance(state, Mapping):
            raise TypeError("RoboDojo observation is missing a state mapping")

        action: dict[str, np.ndarray] = {}
        for side in ARM_SIDES:
            pose_key = f"{side}_ee_pose"
            current_pose = self._read_end_effector_pose(state, pose_key)
            grip = _bounded_input(
                self.xr_client.get_key_value_by_name(f"{side}_grip"),
                label=f"{side}_grip",
            )
            trigger = _bounded_input(
                self.xr_client.get_key_value_by_name(f"{side}_trigger"),
                label=f"{side}_trigger",
            )

            target_pose = self._target_pose(
                side,
                current_pose,
                enabled=grip > self.grip_threshold,
            )
            action[pose_key] = target_pose.astype(np.float32)

            # RoboDojo's normalized X5 gripper convention is 1=open, 0=closed.
            action[f"{side}_ee_joint_state"] = np.array(
                [1.0 - trigger],
                dtype=np.float32,
            )

        return [action]

    def _read_end_effector_pose(
        self,
        state: Mapping[str, Any],
        key: str,
    ) -> np.ndarray:
        if key not in state:
            raise ValueError(f"RoboDojo state is missing {key}")
        pose = np.asarray(state[key], dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(
                f"RoboDojo state {key} must have shape (7,), got {pose.shape}"
            )
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"RoboDojo state {key} must contain finite values")
        result = pose.copy()
        result[3:] = _normalize_quaternion(result[3:], label=f"{key} quaternion")
        return result

    def _controller_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        raw_pose = np.asarray(
            self.xr_client.get_pose_by_name(f"{side}_controller"),
            dtype=np.float64,
        )
        if raw_pose.shape != (7,):
            raise ValueError(
                f"{side}_controller pose must have shape (7,), got {raw_pose.shape}"
            )
        if not np.all(np.isfinite(raw_pose)):
            raise ValueError(f"{side}_controller pose must contain finite values")

        position = self.headset_to_world @ raw_pose[:3]
        raw_orientation = _normalize_quaternion(
            np.array(
                [
                    raw_pose[6],
                    raw_pose[3],
                    raw_pose[4],
                    raw_pose[5],
                ]
            ),
            label=f"{side}_controller quaternion",
        )
        orientation = _quaternion_multiply(
            _quaternion_multiply(self._basis_quaternion, raw_orientation),
            self._basis_quaternion_inverse,
        )
        return position, _normalize_quaternion(
            orientation,
            label=f"{side}_controller world quaternion",
        )

    def _target_pose(
        self,
        side: str,
        current_pose: np.ndarray,
        *,
        enabled: bool,
    ) -> np.ndarray:
        reference = self._references[side]
        if not enabled:
            if reference.active:
                LOGGER.info("%s arm teleoperation released", side)
            reference.clear()
            self._invalid_tracking_reported[side] = False
            return current_pose.copy()

        try:
            controller_position, controller_orientation = self._controller_pose(side)
        except ValueError as exc:
            if not self._invalid_tracking_reported[side]:
                LOGGER.warning("%s; holding the %s arm", exc, side)
                self._invalid_tracking_reported[side] = True
            reference.clear()
            return current_pose.copy()

        self._invalid_tracking_reported[side] = False
        if not reference.active:
            reference.active = True
            reference.controller_position = controller_position.copy()
            reference.controller_orientation = controller_orientation.copy()
            reference.end_effector_position = current_pose[:3].copy()
            reference.end_effector_orientation = current_pose[3:].copy()
            LOGGER.info("%s arm teleoperation engaged", side)
            return current_pose.copy()

        assert reference.controller_position is not None
        assert reference.controller_orientation is not None
        assert reference.end_effector_position is not None
        assert reference.end_effector_orientation is not None

        translation_delta = (
            controller_position - reference.controller_position
        ) * self.scale_factor
        if self.rotation_frame == "tool":
            # Express the hand rotation in the controller frame latched at Grip
            # engagement, then apply it in the end-effector's latched frame.
            # This aligns the controller's local roll/pitch/yaw axes with the
            # tool axes even when their initial world orientations differ.
            rotation_delta = _quaternion_multiply(
                _quaternion_conjugate(reference.controller_orientation),
                controller_orientation,
            )
            target_orientation = _quaternion_multiply(
                reference.end_effector_orientation,
                rotation_delta,
            )
        else:
            # Compatibility mode: copy the controller's world-frame rotation
            # delta directly onto the end effector.
            rotation_delta = _quaternion_multiply(
                controller_orientation,
                _quaternion_conjugate(reference.controller_orientation),
            )
            target_orientation = _quaternion_multiply(
                rotation_delta,
                reference.end_effector_orientation,
            )
        target_orientation = _normalize_quaternion(
            target_orientation,
            label=f"{side} target quaternion",
        )

        # q and -q encode the same rotation.  Choose the representation nearest
        # the current simulated pose to avoid sign flips between commands.
        if float(np.dot(target_orientation, current_pose[3:])) < 0.0:
            target_orientation *= -1.0

        return np.concatenate(
            [
                reference.end_effector_position + translation_delta,
                target_orientation,
            ]
        )


class RoboDojoTeleopPolicy:
    """Async policy facade consumed by XPolicyLab's WebSocket server."""

    def __init__(
        self,
        mapper: RoboDojo6DoFMapper | None = None,
        preview: ObservationPreview | None = None,
        **mapper_kwargs: Any,
    ) -> None:
        self.mapper = mapper or RoboDojo6DoFMapper(**mapper_kwargs)
        self.preview = preview

    async def infer(self, observation: Mapping[str, Any]):
        # Keeping this coroutine on the server event-loop thread avoids calling
        # the native XR SDK from arbitrary executor worker threads.
        if self.preview is not None:
            self.preview.update(observation)
        return self.mapper.infer(observation)

    async def reset(self):
        self.mapper.reset()

    async def prepare_case(self, case_meta=None):
        self.mapper.reset()

    async def on_trial_end(self, result=None):
        self.mapper.reset()

    def close(self) -> None:
        if self.preview is not None:
            self.preview.close()
        self.mapper.close()
