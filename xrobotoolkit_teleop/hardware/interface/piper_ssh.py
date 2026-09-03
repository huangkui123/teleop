"""Persistent SSH transport for a dual PiPER ROS 1 controller.

The local process deliberately does not connect to the ROS master directly.
ROS 1 advertises dynamic XML-RPC/TCPROS ports, which makes a single SSH port
forward insufficient.  Instead, a small bridge is executed on the robot and
JSON lines are carried over one persistent SSH session.
"""

from __future__ import annotations

import base64
import json
import math
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PIPER_SIDES = ("left", "right")
PIPER_ARM_JOINT_COUNT = 6
PIPER_COMMAND_SIZE = 7

# The ROS driver reports one gripper opening in [0, 0.08] metres.  The URDF
# models two fingers, each with 0.035 m travel (0.07 m total opening).
PIPER_GRIPPER_MAX_OPENING = 0.08
MODEL_GRIPPER_MAX_OPENING = 0.07

PIPER_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.0944, 2.0944),
    (0.0, PIPER_GRIPPER_MAX_OPENING),
)

PIPER_ARM_STATUS_NAMES = {
    1: "emergency stop",
    2: "no solution",
    3: "singularity",
    4: "target position exceeds limit",
    5: "joint communication error",
    6: "joint brake not released",
    7: "collision",
    8: "overspeed during teaching",
    9: "joint status error",
    10: "other error",
    11: "teaching record",
    12: "teaching execution",
    13: "teaching paused",
    14: "main controller over-temperature",
    15: "release resistor over-temperature",
}

_REMOTE_BRIDGE_PATH = Path(__file__).with_name("_piper_ros1_ssh_bridge.py")


class PiperSshError(RuntimeError):
    """Raised when the SSH/ROS bridge cannot safely continue."""


@dataclass(frozen=True)
class RemotePiperState:
    """Latest joint and status feedback received from the remote ROS graph."""

    positions: dict[str, tuple[float, ...]]
    statuses: dict[str, dict[str, Any]]
    source_timestamp: float
    received_monotonic: float
    sequence: int
    feedback_age: dict[str, float]
    status_age: dict[str, float]
    last_command_sequence: int | None = None

    @property
    def age(self) -> float:
        transport_age = max(0.0, time.monotonic() - self.received_monotonic)
        return max(
            transport_age,
            *(max(0.0, value) for value in self.feedback_age.values()),
            *(max(0.0, value) for value in self.status_age.values()),
        )


def piper_feedback_to_model_positions(
    side: str, positions: Sequence[float]
) -> dict[str, float]:
    """Convert a PiPER ROS JointState vector into prefixed URDF joints."""

    _validate_side(side)
    values = _finite_vector(positions, PIPER_COMMAND_SIZE, "feedback positions")
    physical_opening = min(max(values[6], 0.0), PIPER_GRIPPER_MAX_OPENING)
    model_half_opening = (
        physical_opening
        / PIPER_GRIPPER_MAX_OPENING
        * MODEL_GRIPPER_MAX_OPENING
        / 2.0
    )

    result = {
        f"{side}_joint{index + 1}": values[index]
        for index in range(PIPER_ARM_JOINT_COUNT)
    }
    result[f"{side}_joint7"] = model_half_opening
    result[f"{side}_joint8"] = -model_half_opening
    return result


def model_positions_to_piper_command(
    side: str, positions: Mapping[str, float]
) -> tuple[float, ...]:
    """Convert prefixed URDF joint targets into the PiPER driver's 7 values."""

    _validate_side(side)
    required = [f"{side}_joint{index}" for index in range(1, 9)]
    missing = [name for name in required if name not in positions]
    if missing:
        raise ValueError(f"missing {side} target joints: {', '.join(missing)}")

    arm = _finite_vector(
        [positions[f"{side}_joint{index}"] for index in range(1, 7)],
        PIPER_ARM_JOINT_COUNT,
        f"{side} arm targets",
    )
    model_opening = float(positions[f"{side}_joint7"]) - float(
        positions[f"{side}_joint8"]
    )
    if not math.isfinite(model_opening):
        raise ValueError(f"{side} gripper target must be finite")
    model_opening = min(max(model_opening, 0.0), MODEL_GRIPPER_MAX_OPENING)
    physical_opening = (
        model_opening
        / MODEL_GRIPPER_MAX_OPENING
        * PIPER_GRIPPER_MAX_OPENING
    )
    return (*arm, physical_opening)


def piper_status_issues(side: str, status: Mapping[str, Any] | None) -> list[str]:
    """Return human-readable safety issues from a PiperStatusMsg snapshot."""

    _validate_side(side)
    if not status:
        return [f"{side}: no arm status feedback"]

    required_fields = ("arm_status", "ctrl_mode", "teach_status", "err_code")
    missing_fields = [name for name in required_fields if name not in status]
    if missing_fields:
        return [f"{side}: status is missing {', '.join(missing_fields)}"]

    issues: list[str] = []
    arm_status = int(status["arm_status"])
    if arm_status:
        description = PIPER_ARM_STATUS_NAMES.get(arm_status, "unknown")
        issues.append(f"{side}: arm_status={arm_status} ({description})")

    ctrl_mode = int(status["ctrl_mode"])
    if ctrl_mode != 1:
        issues.append(f"{side}: ctrl_mode={ctrl_mode}, expected CAN control mode (1)")

    teach_status = int(status["teach_status"])
    if teach_status:
        issues.append(f"{side}: teach_status={teach_status}, expected disabled (0)")

    err_code = int(status["err_code"])
    if err_code:
        issues.append(f"{side}: err_code={err_code}")

    for index in range(1, PIPER_ARM_JOINT_COUNT + 1):
        if bool(status.get(f"joint_{index}_angle_limit", False)):
            issues.append(f"{side}: joint {index} angle-limit fault")
        if bool(status.get(f"communication_status_joint_{index}", False)):
            issues.append(f"{side}: joint {index} communication fault")
    return issues


class PiperCommandLimiter:
    """Clamp joint targets by position, velocity, and tracking error."""

    def __init__(
        self,
        max_joint_speed: float = 0.6,
        max_gripper_speed: float = 0.04,
        max_joint_tracking_error: float = 0.20,
        max_gripper_tracking_error: float = 0.02,
    ) -> None:
        values = {
            "max_joint_speed": max_joint_speed,
            "max_gripper_speed": max_gripper_speed,
            "max_joint_tracking_error": max_joint_tracking_error,
            "max_gripper_tracking_error": max_gripper_tracking_error,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")

        self.max_joint_speed = float(max_joint_speed)
        self.max_gripper_speed = float(max_gripper_speed)
        self.max_joint_tracking_error = float(max_joint_tracking_error)
        self.max_gripper_tracking_error = float(max_gripper_tracking_error)
        self._last_targets: dict[str, tuple[float, ...]] = {}

    def reset(self, side: str, positions: Sequence[float]) -> None:
        """Reset the rate limiter to measured feedback for one arm."""

        _validate_side(side)
        self._last_targets[side] = _finite_vector(
            positions, PIPER_COMMAND_SIZE, f"{side} reset positions"
        )

    def limit(
        self,
        side: str,
        desired: Sequence[float],
        feedback: Sequence[float],
        dt: float,
    ) -> tuple[float, ...]:
        """Return a bounded target and remember it as the next rate reference."""

        _validate_side(side)
        desired_values = _finite_vector(
            desired, PIPER_COMMAND_SIZE, f"{side} desired positions"
        )
        feedback_values = _finite_vector(
            feedback, PIPER_COMMAND_SIZE, f"{side} feedback positions"
        )
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be a non-negative finite value")

        # Capping dt prevents one delayed loop iteration from authorizing a jump.
        bounded_dt = min(float(dt), 0.1)
        previous = self._last_targets.get(side, feedback_values)
        limited: list[float] = []
        for index, ((lower, upper), target, measured, last) in enumerate(
            zip(PIPER_JOINT_LIMITS, desired_values, feedback_values, previous)
        ):
            speed = (
                self.max_joint_speed
                if index < PIPER_ARM_JOINT_COUNT
                else self.max_gripper_speed
            )
            tracking_error = (
                self.max_joint_tracking_error
                if index < PIPER_ARM_JOINT_COUNT
                else self.max_gripper_tracking_error
            )
            target = min(max(target, lower), upper)
            target = min(max(target, last - speed * bounded_dt), last + speed * bounded_dt)
            target = min(max(target, measured - tracking_error), measured + tracking_error)
            limited.append(min(max(target, lower), upper))

        result = tuple(limited)
        self._last_targets[side] = result
        return result


class PiperSshBridge:
    """Run a ROS bridge remotely and exchange feedback/commands over SSH."""

    def __init__(
        self,
        host: str = "agilex",
        remote_setup: str = (
            "/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash"
        ),
        left_feedback_topic: str = "/puppet/joint_left",
        right_feedback_topic: str = "/puppet/joint_right",
        left_status_topic: str = "/puppet/arm_status_left",
        right_status_topic: str = "/puppet/arm_status_right",
        left_command_topic: str = "/master/joint_left",
        right_command_topic: str = "/master/joint_right",
        feedback_rate_hz: float = 50.0,
        watchdog_timeout: float = 0.25,
        state_timeout: float = 0.4,
        connect_timeout: float = 8.0,
        execute: bool = False,
        bridge_node_name: str = "xrobotoolkit_piper_ssh_bridge",
    ) -> None:
        if not host or host.startswith("-"):
            raise ValueError("host must be a non-empty SSH host or alias")
        if not math.isfinite(feedback_rate_hz) or feedback_rate_hz <= 0.0:
            raise ValueError("feedback_rate_hz must be positive")
        if not math.isfinite(watchdog_timeout) or watchdog_timeout <= 0.0:
            raise ValueError("watchdog_timeout must be positive")
        if not math.isfinite(state_timeout) or state_timeout <= 0.0:
            raise ValueError("state_timeout must be positive")
        if not math.isfinite(connect_timeout) or connect_timeout <= 0.0:
            raise ValueError("connect_timeout must be positive")
        if not bridge_node_name:
            raise ValueError("bridge_node_name must be non-empty")

        self.host = host
        self.remote_setup = remote_setup
        self.feedback_rate_hz = float(feedback_rate_hz)
        self.watchdog_timeout = float(watchdog_timeout)
        self.state_timeout = float(state_timeout)
        self.connect_timeout = float(connect_timeout)
        self.execute = bool(execute)
        self.bridge_node_name = bridge_node_name
        self.topics = {
            "left_feedback": left_feedback_topic,
            "right_feedback": right_feedback_topic,
            "left_status": left_status_topic,
            "right_status": right_status_topic,
            "left_command": left_command_topic,
            "right_command": right_command_topic,
        }

        self._process: subprocess.Popen[str] | None = None
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._latest_state: RemotePiperState | None = None
        self._ready_info: dict[str, Any] | None = None
        self._fatal_error: str | None = None
        self._diagnostics: deque[str] = deque(maxlen=100)
        self._command_sequence = 0
        self._reader_threads: list[threading.Thread] = []

    @property
    def ready_info(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._ready_info or {})

    @property
    def diagnostics(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(self._diagnostics)

    def start(self, timeout: float = 12.0) -> RemotePiperState:
        """Start SSH and wait for both arm feedback streams."""

        if not _REMOTE_BRIDGE_PATH.is_file():
            raise PiperSshError(f"remote bridge source not found: {_REMOTE_BRIDGE_PATH}")

        source = _REMOTE_BRIDGE_PATH.read_bytes()
        encoded = base64.b64encode(source).decode("ascii")
        loader = (
            "import base64;"
            f"exec(compile(base64.b64decode({encoded!r}), "
            "'<piper_ros1_ssh_bridge>', 'exec'))"
        )

        environment = self._bridge_environment()
        # Catkin's generated setup scripts reference unset variables, so nounset
        # cannot be enabled while sourcing them.
        remote_lines = ["set -e", "source /opt/ros/noetic/setup.bash"]
        if self.remote_setup:
            remote_lines.append(f"source {shlex.quote(self.remote_setup)}")
        for name, value in environment.items():
            remote_lines.append(f"export {name}={shlex.quote(value)}")
        remote_lines.append(f"exec python3 -u -c {shlex.quote(loader)}")
        remote_command = "\n".join(remote_lines)
        ssh_command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, math.ceil(self.connect_timeout))}",
            self.host,
            f"bash -lc {shlex.quote(remote_command)}",
        ]

        return self._start_process(
            ssh_command,
            timeout=timeout,
            transport_label="SSH",
        )

    def _bridge_environment(self) -> dict[str, str]:
        """Environment consumed by the ROS 1 JSON bridge helper."""

        return {
            "XRT_BRIDGE_NODE_NAME": self.bridge_node_name,
            "XRT_LEFT_FEEDBACK_TOPIC": self.topics["left_feedback"],
            "XRT_RIGHT_FEEDBACK_TOPIC": self.topics["right_feedback"],
            "XRT_LEFT_STATUS_TOPIC": self.topics["left_status"],
            "XRT_RIGHT_STATUS_TOPIC": self.topics["right_status"],
            "XRT_LEFT_COMMAND_TOPIC": self.topics["left_command"],
            "XRT_RIGHT_COMMAND_TOPIC": self.topics["right_command"],
            "XRT_FEEDBACK_RATE_HZ": str(self.feedback_rate_hz),
            "XRT_WATCHDOG_TIMEOUT": str(self.watchdog_timeout),
            "XRT_STATE_TIMEOUT": str(self.state_timeout),
            "XRT_ALLOW_EXECUTE": "1" if self.execute else "0",
        }

    def _start_process(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        transport_label: str,
    ) -> RemotePiperState:
        """Start a bridge transport and wait for dual-arm ROS feedback."""

        if self._process is not None:
            raise PiperSshError("PiPER bridge is already started")

        try:
            self._process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._process = None
            raise PiperSshError(
                f"failed to start {transport_label} bridge: {exc}"
            ) from exc

        thread_prefix = transport_label.lower().replace(" ", "-")
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            name=f"piper-{thread_prefix}-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=f"piper-{thread_prefix}-stderr",
            daemon=True,
        )
        self._reader_threads = [stdout_thread, stderr_thread]
        for thread in self._reader_threads:
            thread.start()

        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._fatal_error:
                    error = self._fatal_error
                    self.close()
                    raise PiperSshError(error)
                if self._ready_info is not None and self._latest_state is not None:
                    return self._latest_state
                process = self._process
                if process is not None and process.poll() is not None:
                    diagnostics = "\n".join(self._diagnostics)
                    self.close()
                    detail = f":\n{diagnostics}" if diagnostics else ""
                    raise PiperSshError(
                        f"{transport_label} bridge exited with code "
                        f"{process.returncode}{detail}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    diagnostics = "\n".join(self._diagnostics)
                    self.close()
                    detail = f":\n{diagnostics}" if diagnostics else ""
                    raise PiperSshError(
                        f"timed out waiting for dual-arm ROS feedback{detail}"
                    )
                self._condition.wait(timeout=min(remaining, 0.2))

    def latest_state(self, max_age: float | None = None) -> RemotePiperState:
        """Return the newest feedback or raise if it is absent/stale."""

        self._ensure_running()
        with self._condition:
            if self._fatal_error:
                raise PiperSshError(self._fatal_error)
            state = self._latest_state
        if state is None:
            raise PiperSshError("no PiPER feedback has been received")
        if max_age is not None and state.age > max_age:
            raise PiperSshError(
                f"PiPER feedback is stale ({state.age:.3f}s > {max_age:.3f}s)"
            )
        return state

    def send_targets(
        self,
        *,
        left: Sequence[float] | None = None,
        right: Sequence[float] | None = None,
        active: Mapping[str, bool] | None = None,
    ) -> int:
        """Send targets for either or both arms; only executable bridges publish."""

        self._ensure_running()
        if not self.execute:
            raise PiperSshError("bridge is in dry-run mode; pass execute=True to send")
        targets: dict[str, list[float]] = {}
        if left is not None:
            targets["left"] = list(
                _finite_vector(left, PIPER_COMMAND_SIZE, "left command")
            )
        if right is not None:
            targets["right"] = list(
                _finite_vector(right, PIPER_COMMAND_SIZE, "right command")
            )
        if not targets:
            raise ValueError("at least one arm target is required")

        self._command_sequence += 1
        message = {
            "type": "command",
            "sequence": self._command_sequence,
            "targets": targets,
            "active": {
                side: bool((active or {}).get(side, False)) for side in PIPER_SIDES
            },
        }
        self._write_message(message)
        return self._command_sequence

    def close(self) -> None:
        """Stop the remote helper.  This does not disable the robot arms."""

        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                self._write_message({"type": "shutdown"}, check_running=False)
            except (OSError, PiperSshError):
                pass
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        self._process = None
        with self._condition:
            self._condition.notify_all()

    def __enter__(self) -> "PiperSshBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _write_message(
        self, message: Mapping[str, Any], *, check_running: bool = True
    ) -> None:
        if check_running:
            self._ensure_running()
        process = self._process
        if process is None or process.stdin is None:
            raise PiperSshError("SSH bridge stdin is unavailable")
        payload = json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n"
        try:
            with self._write_lock:
                process.stdin.write(payload)
                process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise PiperSshError(f"failed to write to SSH bridge: {exc}") from exc

    def _ensure_running(self) -> None:
        process = self._process
        if process is None:
            raise PiperSshError("SSH bridge is not started")
        return_code = process.poll()
        if return_code is not None:
            diagnostics = "\n".join(self.diagnostics)
            detail = f":\n{diagnostics}" if diagnostics else ""
            raise PiperSshError(
                f"SSH bridge exited with code {return_code}{detail}"
            )

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._add_diagnostic(f"remote stdout: {line}")
                continue
            if not isinstance(message, dict):
                self._add_diagnostic(f"unexpected remote message: {line}")
                continue
            message_type = message.get("type")
            with self._condition:
                if message_type == "ready":
                    self._ready_info = message
                elif message_type == "state":
                    try:
                        positions = {
                            side: _finite_vector(
                                message["positions"][side],
                                PIPER_COMMAND_SIZE,
                                f"remote {side} feedback",
                            )
                            for side in PIPER_SIDES
                        }
                        statuses = {
                            side: dict(message.get("statuses", {}).get(side) or {})
                            for side in PIPER_SIDES
                        }
                        self._latest_state = RemotePiperState(
                            positions=positions,
                            statuses=statuses,
                            source_timestamp=float(message.get("timestamp", 0.0)),
                            received_monotonic=time.monotonic(),
                            sequence=int(message.get("sequence", 0)),
                            feedback_age={
                                side: float(message.get("feedback_age", {})[side])
                                for side in PIPER_SIDES
                            },
                            status_age={
                                side: float(message.get("status_age", {})[side])
                                for side in PIPER_SIDES
                            },
                            last_command_sequence=message.get("last_command_sequence"),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        self._diagnostics.append(
                            f"invalid state from remote bridge: {exc}"
                        )
                elif message_type == "error":
                    detail = str(message.get("message", "unknown remote error"))
                    self._diagnostics.append(f"remote error: {detail}")
                    if bool(message.get("fatal", False)):
                        self._fatal_error = detail
                elif message_type not in {"event", "pong"}:
                    self._diagnostics.append(f"unexpected remote message: {line}")
                self._condition.notify_all()
        with self._condition:
            self._condition.notify_all()

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for raw_line in process.stderr:
            line = raw_line.strip()
            if line:
                self._add_diagnostic(f"remote stderr: {line}")

    def _add_diagnostic(self, line: str) -> None:
        with self._condition:
            self._diagnostics.append(line)
            self._condition.notify_all()


def _validate_side(side: str) -> None:
    if side not in PIPER_SIDES:
        raise ValueError(f"side must be one of {PIPER_SIDES}, got {side!r}")


def _finite_vector(
    values: Sequence[float], expected_size: int, description: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{description} must be a numeric sequence")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a numeric sequence") from exc
    if len(result) != expected_size:
        raise ValueError(
            f"{description} must have {expected_size} values, got {len(result)}"
        )
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{description} must contain only finite values")
    return result
