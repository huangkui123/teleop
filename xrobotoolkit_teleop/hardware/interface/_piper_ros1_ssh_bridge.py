"""Remote half of the PiPER SSH bridge.

This file is read as text by :mod:`piper_ssh` and executed on the ROS 1 host.
It must therefore only import modules available in a standard ROS Noetic
installation plus the PiPER message package.
"""

import json
import math
import os
import queue
import sys
import threading
import time

import rosgraph
import rospy
from piper_msgs.msg import PiperStatusMsg
from sensor_msgs.msg import JointState


SIDES = ("left", "right")
EXPECTED_NAMES = ["joint%d" % index for index in range(7)]
STATUS_FIELDS = (
    "ctrl_mode",
    "arm_status",
    "mode_feedback",
    "teach_status",
    "motion_status",
    "trajectory_num",
    "err_code",
    "joint_1_angle_limit",
    "joint_2_angle_limit",
    "joint_3_angle_limit",
    "joint_4_angle_limit",
    "joint_5_angle_limit",
    "joint_6_angle_limit",
    "communication_status_joint_1",
    "communication_status_joint_2",
    "communication_status_joint_3",
    "communication_status_joint_4",
    "communication_status_joint_5",
    "communication_status_joint_6",
)


def env(name, default):
    return os.environ.get(name, default)


TOPICS = {
    "left_feedback": env("XRT_LEFT_FEEDBACK_TOPIC", "/puppet/joint_left"),
    "right_feedback": env("XRT_RIGHT_FEEDBACK_TOPIC", "/puppet/joint_right"),
    "left_status": env("XRT_LEFT_STATUS_TOPIC", "/puppet/arm_status_left"),
    "right_status": env("XRT_RIGHT_STATUS_TOPIC", "/puppet/arm_status_right"),
    "left_command": env("XRT_LEFT_COMMAND_TOPIC", "/master/joint_left"),
    "right_command": env("XRT_RIGHT_COMMAND_TOPIC", "/master/joint_right"),
}
FEEDBACK_RATE_HZ = float(env("XRT_FEEDBACK_RATE_HZ", "50"))
WATCHDOG_TIMEOUT = float(env("XRT_WATCHDOG_TIMEOUT", "0.25"))
STATE_TIMEOUT = float(env("XRT_STATE_TIMEOUT", "0.4"))
ALLOW_EXECUTE = env("XRT_ALLOW_EXECUTE", "0") == "1"
BRIDGE_NODE_NAME = env(
    "XRT_BRIDGE_NODE_NAME", "xrobotoolkit_piper_ssh_bridge"
)

lock = threading.Lock()
feedback = {"left": None, "right": None}
statuses = {"left": None, "right": None}
feedback_times = {"left": 0.0, "right": 0.0}
status_times = {"left": 0.0, "right": 0.0}
incoming = queue.Queue(maxsize=1)
stdin_closed = threading.Event()


def emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def replace_incoming(message):
    while True:
        try:
            incoming.get_nowait()
        except queue.Empty:
            break
    incoming.put_nowait(message)


def stdin_reader():
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except (TypeError, ValueError) as exc:
                replace_incoming(
                    {"type": "invalid", "message": "invalid JSON command: %s" % exc}
                )
                continue
            replace_incoming(message)
    finally:
        stdin_closed.set()


def normalize_joint_state(message):
    if len(message.position) < 7:
        return None
    if message.name and all(name in message.name for name in EXPECTED_NAMES):
        by_name = dict(zip(message.name, message.position))
        values = [float(by_name[name]) for name in EXPECTED_NAMES]
    else:
        values = [float(value) for value in message.position[:7]]
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def feedback_callback(side, message):
    values = normalize_joint_state(message)
    if values is None:
        return
    with lock:
        feedback[side] = values
        feedback_times[side] = time.monotonic()


def status_callback(side, message):
    value = {name: getattr(message, name) for name in STATUS_FIELDS}
    with lock:
        statuses[side] = value
        status_times[side] = time.monotonic()


def publishers_for(master, topic):
    publishers, _, _ = master.getSystemState()
    return list(dict(publishers).get(topic, []))


def external_command_publishers(master, own_name):
    result = {}
    for side in SIDES:
        nodes = publishers_for(master, TOPICS[side + "_command"])
        external = [name for name in nodes if name != own_name]
        if external:
            result[side] = external
    return result


def status_is_healthy(status):
    if status is None:
        return False
    if int(status.get("arm_status", -1)) != 0:
        return False
    if int(status.get("ctrl_mode", -1)) != 1:
        return False
    if int(status.get("teach_status", -1)) != 0:
        return False
    if int(status.get("err_code", 0)) != 0:
        return False
    for index in range(1, 7):
        if status.get("joint_%d_angle_limit" % index, False):
            return False
        if status.get("communication_status_joint_%d" % index, False):
            return False
    return True


def validate_target(side, values):
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError("%s target must contain 7 values" % side)
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("%s target contains a non-finite value" % side)
    return result


def publish_target(publisher, values, sequence):
    message = JointState()
    message.header.seq = sequence
    message.header.stamp = rospy.Time.now()
    message.name = list(EXPECTED_NAMES)
    message.position = values
    message.velocity = [0.0] * 7
    message.effort = [0.0] * 7
    publisher.publish(message)


def main():
    rospy.init_node(
        BRIDGE_NODE_NAME,
        anonymous=True,
        disable_signals=True,
        log_level=rospy.WARN,
    )
    own_name = rospy.get_name()
    master = rosgraph.Master(own_name)
    conflicts_at_start = external_command_publishers(master, own_name)

    rospy.Subscriber(
        TOPICS["left_feedback"],
        JointState,
        lambda message: feedback_callback("left", message),
        queue_size=1,
        tcp_nodelay=True,
    )
    rospy.Subscriber(
        TOPICS["right_feedback"],
        JointState,
        lambda message: feedback_callback("right", message),
        queue_size=1,
        tcp_nodelay=True,
    )
    rospy.Subscriber(
        TOPICS["left_status"],
        PiperStatusMsg,
        lambda message: status_callback("left", message),
        queue_size=1,
        tcp_nodelay=True,
    )
    rospy.Subscriber(
        TOPICS["right_status"],
        PiperStatusMsg,
        lambda message: status_callback("right", message),
        queue_size=1,
        tcp_nodelay=True,
    )
    command_publishers = {
        side: rospy.Publisher(
            TOPICS[side + "_command"],
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        for side in SIDES
    }

    threading.Thread(target=stdin_reader, name="bridge-stdin", daemon=True).start()

    ready_deadline = time.monotonic() + 8.0
    while not rospy.is_shutdown() and time.monotonic() < ready_deadline:
        with lock:
            feedback_ready = all(feedback[side] is not None for side in SIDES)
            status_ready = all(statuses[side] is not None for side in SIDES)
        subscribers_ready = all(
            command_publishers[side].get_num_connections() > 0 for side in SIDES
        )
        if feedback_ready and status_ready and subscribers_ready:
            break
        time.sleep(0.02)

    with lock:
        feedback_ready = all(feedback[side] is not None for side in SIDES)
        status_ready = all(statuses[side] is not None for side in SIDES)
    connections = {
        side: command_publishers[side].get_num_connections() for side in SIDES
    }
    emit(
        {
            "type": "ready",
            "node": own_name,
            "execute_allowed": ALLOW_EXECUTE,
            "feedback_ready": feedback_ready,
            "status_ready": status_ready,
            "command_subscribers": connections,
            "existing_command_publishers": conflicts_at_start,
            "topics": TOPICS,
        }
    )

    rate = rospy.Rate(max(1.0, FEEDBACK_RATE_HZ))
    state_sequence = 0
    last_command_sequence = None
    last_command_time = 0.0
    last_command = None
    watchdog_reported = False
    conflict_latched = bool(conflicts_at_start)
    last_conflict_check = 0.0

    while not rospy.is_shutdown() and not stdin_closed.is_set():
        now = time.monotonic()
        try:
            message = incoming.get_nowait()
        except queue.Empty:
            message = None

        if message is not None:
            message_type = message.get("type") if isinstance(message, dict) else None
            if message_type == "shutdown":
                break
            if message_type == "invalid":
                emit({"type": "error", "fatal": False, "message": message["message"]})
            elif message_type == "command":
                try:
                    targets = {
                        side: validate_target(side, values)
                        for side, values in message.get("targets", {}).items()
                        if side in SIDES
                    }
                    if not targets:
                        raise ValueError("command has no arm targets")
                    last_command = targets
                    last_command_sequence = int(message.get("sequence", 0))
                    last_command_time = now
                    watchdog_reported = False
                except (TypeError, ValueError) as exc:
                    emit({"type": "error", "fatal": False, "message": str(exc)})
            else:
                emit(
                    {
                        "type": "error",
                        "fatal": False,
                        "message": "unsupported input message",
                    }
                )

        if now - last_conflict_check >= 1.0:
            last_conflict_check = now
            conflicts = external_command_publishers(master, own_name)
            if conflicts and not conflict_latched:
                conflict_latched = True
                emit(
                    {
                        "type": "error",
                        "fatal": True,
                        "message": "another command publisher appeared: %s" % conflicts,
                    }
                )

        command_fresh = last_command is not None and now - last_command_time <= WATCHDOG_TIMEOUT
        if ALLOW_EXECUTE and command_fresh and not conflict_latched:
            with lock:
                current_statuses = dict(statuses)
                current_feedback_times = dict(feedback_times)
                current_status_times = dict(status_times)
            for side, values in last_command.items():
                state_is_fresh = (
                    now - current_feedback_times[side] <= STATE_TIMEOUT
                    and now - current_status_times[side] <= STATE_TIMEOUT
                )
                if state_is_fresh and status_is_healthy(current_statuses.get(side)):
                    publish_target(
                        command_publishers[side], values, last_command_sequence or 0
                    )
        elif last_command is not None and not command_fresh and not watchdog_reported:
            watchdog_reported = True
            emit(
                {
                    "type": "event",
                    "event": "command_watchdog",
                    "last_command_sequence": last_command_sequence,
                }
            )

        with lock:
            current_feedback = {
                side: list(feedback[side]) if feedback[side] is not None else None
                for side in SIDES
            }
            current_statuses = {
                side: dict(statuses[side]) if statuses[side] is not None else None
                for side in SIDES
            }
            current_feedback_times = dict(feedback_times)
            current_status_times = dict(status_times)

        if all(current_feedback[side] is not None for side in SIDES):
            state_sequence += 1
            emit(
                {
                    "type": "state",
                    "sequence": state_sequence,
                    "timestamp": time.time(),
                    "positions": current_feedback,
                    "statuses": current_statuses,
                    "feedback_age": {
                        side: max(0.0, now - current_feedback_times[side])
                        for side in SIDES
                    },
                    "status_age": {
                        side: max(0.0, now - current_status_times[side])
                        for side in SIDES
                    },
                    "last_command_sequence": last_command_sequence,
                }
            )
        rate.sleep()

    rospy.signal_shutdown("PiPER bridge closed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({"type": "error", "fatal": True, "message": "%s: %s" % (type(exc).__name__, exc)})
        raise
