"""Launch the teleop server and one RoboDojo Isaac Sim case."""

from __future__ import annotations

import argparse
import base64
import hashlib
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

POLICY_NAME = "demo_policy"
KIT_ARGS = " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
REMOTE_POLICY_HOST = "127.0.0.1"

REMOTE_CLIENT_SCRIPT = r"""
set -euo pipefail

robodojo_root="$1"
conda_sh="$2"
eval_env="$3"
workdir="$4"
run_id="$5"
env_gpu="$6"
accept_eula="$7"
task_name="$8"
env_cfg="$9"
image_codec="${10}"
jpeg_quality="${11}"
preview_every="${12}"
vision_every="${13}"
stream_cameras="${14}"
render_preset="${15}"
transport_sha256="${16}"
transport_payload="${17}"
shift 17

required_files=(
    "${robodojo_root}/src/eval_client/main.py"
    "${robodojo_root}/XPolicyLab/policy/demo_policy/deploy.py"
    "${robodojo_root}/task/RoboDojo/config/${task_name}.yml"
    "${robodojo_root}/env_cfg/${env_cfg}.yml"
    "${conda_sh}"
)
if [[ "${env_cfg}" == "arx_x5" ]]; then
    required_files+=("${robodojo_root}/Assets/Robots/x5/curobo.yml")
fi
for required_file in "${required_files[@]}"; do
    if [[ ! -r "${required_file}" ]]; then
        echo "[robodojo-teleop] Remote file is missing or unreadable: ${required_file}" >&2
        exit 2
    fi
done

if [[ -z "${workdir}" ]]; then
    workdir="${HOME}/robodojo-teleop-runs"
fi
robodojo_root_real="$(readlink -f -- "${robodojo_root}")"
workdir_real="$(readlink -m -- "${workdir}")"
case "${workdir_real}" in
    "${robodojo_root_real}"|"${robodojo_root_real}"/*)
        echo "[robodojo-teleop] Refusing to write runtime output inside the remote RoboDojo source tree: ${workdir_real}" >&2
        exit 2
        ;;
esac
if [[ "${workdir_real}" =~ [[:space:]] ]]; then
    echo "[robodojo-teleop] Remote work directory cannot contain whitespace: ${workdir_real}" >&2
    exit 2
fi

source "${conda_sh}"
conda deactivate >/dev/null 2>&1 || true
conda activate "${eval_env}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

missing_libraries=()
for library in libXt.so.6 libGLU.so.1; do
    if ! python -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "${library}" >/dev/null 2>&1; then
        missing_libraries+=("${library}")
    fi
done
if (( ${#missing_libraries[@]} > 0 )); then
    echo "[robodojo-teleop] Remote Isaac Sim dependencies are unavailable: ${missing_libraries[*]}" >&2
    echo "[robodojo-teleop] Ask the server administrator to provide libxt6 and libglu1-mesa (or equivalent libraries)." >&2
    exit 2
fi

kit_portable_root="${workdir_real}/isaacsim-kit"
transport_dir="${workdir_real}/teleop-adapter/${transport_sha256}"
mkdir -p -- \
    "${kit_portable_root}/exts" \
    "${transport_dir}" \
    "${workdir_real}/xdg-cache" \
    "${workdir_real}/xdg-config" \
    "${workdir_real}/tmp"

transport_file="${transport_dir}/sitecustomize.py"
transport_file_sha=""
if [[ -r "${transport_file}" ]]; then
    transport_file_sha="$(sha256sum "${transport_file}" | awk '{print $1}')"
fi
if [[ "${transport_file_sha}" != "${transport_sha256}" ]]; then
    transport_tmp="$(mktemp "${transport_dir}/.sitecustomize.XXXXXX")"
    trap 'rm -f -- "${transport_tmp}"' EXIT
    printf '%s' "${transport_payload}" | base64 --decode > "${transport_tmp}"
    decoded_sha="$(sha256sum "${transport_tmp}" | awk '{print $1}')"
    if [[ "${decoded_sha}" != "${transport_sha256}" ]]; then
        echo "[robodojo-teleop] Remote transport adapter checksum mismatch" >&2
        exit 2
    fi
    chmod 600 "${transport_tmp}"
    mv -- "${transport_tmp}" "${transport_file}"
    trap - EXIT
fi

cd "${workdir_real}"
export PYTHONPATH="${transport_dir}:${robodojo_root}:${robodojo_root}/XPolicyLab"
export CUDA_VISIBLE_DEVICES="${env_gpu}"
export EVAL_NUM=1
export ROBODOJO_RUN_ID="${run_id}"
export ROBODOJO_TELEOP_IMAGE_CODEC="${image_codec}"
export ROBODOJO_TELEOP_JPEG_QUALITY="${jpeg_quality}"
export ROBODOJO_TELEOP_PREVIEW_EVERY="${preview_every}"
export ROBODOJO_TELEOP_VISION_EVERY="${vision_every}"
export ROBODOJO_TELEOP_STREAM_CAMERAS="${stream_cameras}"
export ROBODOJO_TELEOP_RENDER_PRESET="${render_preset}"
export XDG_CACHE_HOME="${workdir_real}/xdg-cache"
export XDG_CONFIG_HOME="${workdir_real}/xdg-config"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${workdir_real}/python-cache"
export TMPDIR="${workdir_real}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
if [[ "${accept_eula}" == "yes" ]]; then
    export OMNI_KIT_ACCEPT_EULA=YES
fi

echo "[robodojo-teleop] Remote source: ${robodojo_root}"
echo "[robodojo-teleop] Remote runtime output: ${workdir_real}"
echo "[robodojo-teleop] Image transport: ${image_codec} (JPEG quality ${jpeg_quality})"
echo "[robodojo-teleop] Decoupled video: preview every ${preview_every} step(s), other cameras every ${vision_every} step(s), network cameras=${stream_cameras}"
echo "[robodojo-teleop] Render preset: ${render_preset}"
kit_args=" --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --portable-root ${kit_portable_root} --/app/extensions/registryCacheFull=${kit_portable_root}/exts"
exec "$@" --kit_args "${kit_args}"
""".strip()


def _remote_transport_bundle() -> tuple[str, str]:
    source_path = Path(__file__).with_name("remote_transport.py")
    source = source_path.read_bytes()
    if not source:
        raise RuntimeError(f"Remote transport adapter is empty: {source_path}")
    payload = base64.b64encode(source).decode("ascii")
    return payload, hashlib.sha256(source).hexdigest()


def _validated_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    required = (
        root / "src" / "eval_client" / "main.py",
        root / "XPolicyLab" / "policy" / POLICY_NAME / "deploy.py",
        root / "env_cfg" / "arx_x5.yml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The supplied RoboDojo root is incomplete; missing: " + ", ".join(missing)
        )
    return root


def _resolve_conda_sh(explicit_base: str | None = None) -> Path:
    if explicit_base:
        base = Path(explicit_base).expanduser().resolve()
    elif os.environ.get("CONDA_EXE"):
        base = Path(os.environ["CONDA_EXE"]).expanduser().resolve().parent.parent
    else:
        conda = subprocess.run(
            ["conda", "info", "--base"],
            check=True,
            capture_output=True,
            text=True,
        )
        base = Path(conda.stdout.strip()).expanduser().resolve()
    conda_sh = base / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.is_file():
        raise FileNotFoundError(f"Conda initialization script not found: {conda_sh}")
    return conda_sh


def _choose_port(port: int, host: str) -> int:
    if port < 0 or port > 65535:
        raise ValueError("port must be between 0 and 65535")
    if port:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_for_server(
    process: subprocess.Popen,
    host: str,
    port: int,
    timeout: float,
) -> None:
    try:
        from websockets.sync.client import connect
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The RoboDojo bridge requires websockets>=13. Install the "
            "teleop extra with: python -m pip install -e '.[robodojo]'"
        ) from exc

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"teleoperation server exited before accepting connections (rc={return_code})"
            )
        try:
            with connect(
                f"ws://{host}:{port}",
                open_timeout=0.25,
                close_timeout=0.25,
            ):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(
        f"teleoperation server did not open {host}:{port} within {timeout:g}s"
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def build_server_command(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "xrobotoolkit_teleop.integrations.robodojo.server",
        "--robodojo-root",
        str(args.robodojo_root),
        "--host",
        args.bind_host,
        "--port",
        str(port),
        "--scale-factor",
        str(args.scale_factor),
        "--grip-threshold",
        str(args.grip_threshold),
        "--rotation-frame",
        args.rotation_frame,
        "--xr-sample-hz",
        str(args.xr_sample_hz),
        "--preview-fps",
        str(args.preview_fps),
    ]
    if args.mock_xr:
        command.append("--mock-xr")
    if args.preview:
        command.append("--preview")
    return command


def build_isaac_python_command(
    args: argparse.Namespace,
    port: int,
    robodojo_root: str | Path,
    policy_host: str,
    kit_args: str | None = KIT_ARGS,
) -> list[str]:
    main_file = PurePosixPath(str(robodojo_root)) / "src" / "eval_client" / "main.py"
    python_command = [
        "python",
        "-u",
        str(main_file),
        "--task_name",
        args.task,
        "--env_cfg_type",
        args.env_cfg,
        "--num_envs",
        "1",
        "--enable_cameras",
        "--device_id",
        str(args.env_gpu),
        "--policy_name",
        POLICY_NAME,
        "--port",
        str(port),
        "--protocol",
        "ws",
        "--policy_server_url",
        f"ws://{policy_host}:{port}",
        "--additional_info",
        "ckpt_name=teleop,action_type=ee",
        "--seed",
        str(args.seed),
        "--host",
        policy_host,
    ]
    if kit_args is not None:
        python_command.extend(["--kit_args", kit_args])
    if args.headless:
        python_command.append("--headless")
    return python_command


def build_isaac_client_command(
    args: argparse.Namespace,
    port: int,
    conda_sh: Path,
) -> list[str]:
    python_command = build_isaac_python_command(
        args,
        port,
        args.robodojo_root,
        args.policy_host,
    )

    activate_and_exec = (
        'source "$1"; '
        "conda deactivate >/dev/null 2>&1 || true; "
        'conda activate "$2"; '
        "shift 2; "
        'exec "$@"'
    )
    return [
        "/bin/bash",
        "-c",
        activate_and_exec,
        "robodojo-teleop-client",
        str(conda_sh),
        args.eval_env,
        *python_command,
    ]


def _validated_remote_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute remote path: {value}")
    return path


def build_remote_isaac_client_command(
    args: argparse.Namespace,
    local_port: int,
    remote_port: int,
    run_id: str,
) -> list[str]:
    if not args.ssh_host or args.ssh_host.startswith("-"):
        raise ValueError("--ssh-host must name a valid SSH host")

    remote_root = _validated_remote_path(
        args.remote_robodojo_root,
        "--remote-robodojo-root",
    )
    remote_conda_base = _validated_remote_path(
        args.remote_conda_base,
        "--remote-conda-base",
    )
    if args.remote_workdir:
        _validated_remote_path(args.remote_workdir, "--remote-workdir")

    remote_conda_sh = remote_conda_base / "etc" / "profile.d" / "conda.sh"
    python_command = build_isaac_python_command(
        args,
        remote_port,
        remote_root,
        REMOTE_POLICY_HOST,
        kit_args=None,
    )
    transport_payload, transport_sha256 = _remote_transport_bundle()
    remote_argv = [
        "/bin/bash",
        "-c",
        REMOTE_CLIENT_SCRIPT,
        "robodojo-teleop-remote",
        str(remote_root),
        str(remote_conda_sh),
        args.eval_env,
        args.remote_workdir or "",
        run_id,
        str(args.env_gpu),
        "yes" if args.accept_eula else "no",
        args.task,
        args.env_cfg,
        args.image_codec,
        str(args.jpeg_quality),
        str(args.preview_every),
        str(args.vision_every),
        args.stream_cameras,
        args.render_preset,
        transport_sha256,
        transport_payload,
        *python_command,
    ]
    reverse_forward = (
        f"{REMOTE_POLICY_HOST}:{remote_port}:{args.policy_host}:{local_port}"
    )
    return [
        "ssh",
        # A remote PTY gives the Isaac process a controlling terminal. When
        # the local launcher is interrupted or the SSH connection disappears,
        # sshd tears down that terminal and the remote process receives SIGHUP
        # instead of surviving as an orphan.
        "-tt",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ConnectTimeout={args.ssh_connect_timeout}",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        reverse_forward,
        args.ssh_host,
        shlex.join(remote_argv),
    ]


def _client_environment(
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, str]:
    environment = dict(os.environ)
    # Do not leak the Python 3.13 teleop package path into Isaac Sim's
    # Python 3.11 process.
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.robodojo_root),
            str(args.robodojo_root / "XPolicyLab"),
        ]
    )
    environment["CUDA_VISIBLE_DEVICES"] = str(args.env_gpu)
    environment["EVAL_NUM"] = "1"
    environment["ROBODOJO_RUN_ID"] = run_id
    if args.accept_eula:
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return environment


def run(args: argparse.Namespace) -> int:
    args.robodojo_root = _validated_root(args.robodojo_root)
    task_config = (
        args.robodojo_root / "task" / "RoboDojo" / "config" / f"{args.task}.yml"
    )
    env_config = args.robodojo_root / "env_cfg" / f"{args.env_cfg}.yml"
    if not task_config.is_file():
        raise FileNotFoundError(f"RoboDojo task config not found: {task_config}")
    if not env_config.is_file():
        raise FileNotFoundError(f"RoboDojo environment config not found: {env_config}")

    local_port = _choose_port(args.port, args.policy_host)
    local_now = datetime.now(tz=timezone.utc).astimezone()
    run_id = f"teleop_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}"
    server_command = build_server_command(args, local_port)
    client_environment: dict[str, str] | None
    if args.ssh_host:
        remote_port = args.remote_port or local_port
        if remote_port < 1 or remote_port > 65535:
            raise ValueError("--remote-port must be between 1 and 65535")
        client_command = build_remote_isaac_client_command(
            args,
            local_port,
            remote_port,
            run_id,
        )
        client_environment = None
        client_cwd = Path(__file__).resolve().parents[3]
    else:
        conda_sh = _resolve_conda_sh(args.conda_base)
        client_command = build_isaac_client_command(args, local_port, conda_sh)
        client_environment = _client_environment(args, run_id)
        client_cwd = args.robodojo_root

    if args.dry_run:
        print("[robodojo-teleop] server:")
        print(shlex.join(server_command))
        if args.ssh_host:
            print(
                "[robodojo-teleop] Remote Isaac Sim client "
                f"(SSH reverse tunnel {REMOTE_POLICY_HOST}:{remote_port} -> "
                f"{args.policy_host}:{local_port}):"
            )
        else:
            print("[robodojo-teleop] Isaac Sim client:")
        rendered_client_command = shlex.join(client_command)
        if args.ssh_host:
            transport_payload, transport_sha256 = _remote_transport_bundle()
            rendered_client_command = rendered_client_command.replace(
                transport_payload,
                f"<sitecustomize:{transport_sha256[:12]}>",
            )
        print(rendered_client_command)
        if client_environment is not None:
            print(
                "[robodojo-teleop] client env: "
                f"EVAL_NUM=1 ROBODOJO_RUN_ID={client_environment['ROBODOJO_RUN_ID']} "
                f"PYTHONPATH={client_environment['PYTHONPATH']}"
            )
        return 0

    location = f"SSH host {args.ssh_host}" if args.ssh_host else "this machine"
    print(
        f"[robodojo-teleop] Starting task={args.task} with Isaac Sim on "
        f"{location} ({'headless' if args.headless else 'GUI'}), GPU {args.env_gpu}.",
        flush=True,
    )
    if not args.mock_xr:
        print(
            "[robodojo-teleop] XRoboToolkit PC Service and the headset client "
            "must already be running.",
            flush=True,
        )

    server_process: subprocess.Popen | None = None
    client_process: subprocess.Popen | None = None
    try:
        server_process = subprocess.Popen(
            server_command,
            cwd=Path(__file__).resolve().parents[3],
            start_new_session=True,
        )
        _wait_for_server(
            server_process,
            args.policy_host,
            local_port,
            args.connect_timeout,
        )
        if args.ssh_host:
            ready_message = (
                f"[robodojo-teleop] Local server ready at "
                f"ws://{args.policy_host}:{local_port}; launching remote Isaac Sim "
                f"through {args.ssh_host} reverse port {remote_port}."
            )
        else:
            ready_message = (
                f"[robodojo-teleop] Server ready at "
                f"ws://{args.policy_host}:{local_port}; launching Isaac Sim."
            )
        print(ready_message, flush=True)
        client_process = subprocess.Popen(
            client_command,
            cwd=client_cwd,
            env=client_environment,
            start_new_session=True,
        )

        while True:
            client_return_code = client_process.poll()
            if client_return_code is not None:
                return int(client_return_code)
            server_return_code = server_process.poll()
            if server_return_code is not None:
                raise RuntimeError(
                    f"teleoperation server stopped while Isaac Sim was running "
                    f"(rc={server_return_code})"
                )
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[robodojo-teleop] Stopping teleoperation.", flush=True)
        return 130
    finally:
        _stop_process(client_process)
        _stop_process(server_process)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one RoboDojo task under XR 6DoF teleoperation."
    )
    parser.add_argument(
        "--robodojo-root",
        default=os.environ.get("ROBODOJO_ROOT", "/home/arx/RoboDojo"),
    )
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--eval-env", default="RoboDojo")
    parser.add_argument("--env-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="WebSocket port; 0 selects a free local port.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--grip-threshold", type=float, default=0.9)
    parser.add_argument(
        "--rotation-frame",
        choices=("tool", "world"),
        default="tool",
        help=(
            "Rotation reference frame: 'tool' aligns controller-local rotation "
            "axes to the end effector at Grip engagement; 'world' restores the "
            "previous world-frame behavior."
        ),
    )
    parser.add_argument(
        "--xr-sample-hz",
        type=float,
        default=60.0,
        help="Read local XR controller state independently at this rate.",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=10.0,
        help="Maximum local preview-window refresh rate.",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--conda-base")
    parser.add_argument(
        "--ssh-host",
        help=(
            "Run Isaac Sim on this SSH host and reach the local policy server "
            "through a loopback-only reverse tunnel."
        ),
    )
    parser.add_argument(
        "--remote-robodojo-root",
        default="/home/huangkui/RoboDojo",
        help="Absolute RoboDojo source path on the SSH host.",
    )
    parser.add_argument(
        "--remote-conda-base",
        default="/home/huangkui/miniconda3",
        help="Absolute Conda base path on the SSH host.",
    )
    parser.add_argument(
        "--remote-workdir",
        default="",
        help=(
            "Writable runtime/output directory on the SSH host. The default is "
            "$HOME/robodojo-teleop-runs; it must be outside the RoboDojo source tree."
        ),
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=0,
        help="Remote loopback tunnel port; 0 reuses the selected local port.",
    )
    parser.add_argument("--ssh-connect-timeout", type=int, default=15)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the Isaac Sim GUI.",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Set OMNI_KIT_ACCEPT_EULA=YES for the Isaac Sim process.",
    )
    parser.add_argument(
        "--mock-xr",
        action="store_true",
        help="Use stationary synthetic controllers to test the bridge.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show RoboDojo head and wrist cameras in a local desktop window.",
    )
    parser.add_argument(
        "--image-codec",
        choices=("raw", "jpeg"),
        default="jpeg",
        help="Remote camera transport codec; JPEG avoids sending raw RGB over SSH.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG camera quality for remote transport (1-100).",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=1,
        help=(
            "Capture cameras selected by --stream-cameras every N control observations."
        ),
    )
    parser.add_argument(
        "--vision-every",
        type=int,
        default=3,
        help=(
            "Capture non-preview cameras for remote recording every N control "
            "observations."
        ),
    )
    parser.add_argument(
        "--stream-cameras",
        default="cam_head",
        help=(
            "Comma-separated cameras sent to the local preview, or 'all'. "
            "All captured cameras remain available to RoboDojo's remote recorder."
        ),
    )
    parser.add_argument(
        "--render-preset",
        choices=("minimal", "quality"),
        default="minimal",
        help=(
            "Remote RTX render profile. 'minimal' uses performance mode, "
            "320x240 cameras, no reflections/global illumination/shadows, "
            "and no anti-aliasing; 'quality' preserves RoboDojo defaults."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 1 <= args.jpeg_quality <= 100:
            raise ValueError("--jpeg-quality must be between 1 and 100")
        if args.preview_every < 1:
            raise ValueError("--preview-every must be at least 1")
        if args.vision_every < 1:
            raise ValueError("--vision-every must be at least 1")
        if not args.stream_cameras.strip():
            raise ValueError("--stream-cameras cannot be empty")
        stream_camera_names = tuple(
            part.strip() for part in args.stream_cameras.split(",") if part.strip()
        )
        if "all" in stream_camera_names and stream_camera_names != ("all",):
            raise ValueError(
                "--stream-cameras 'all' cannot be combined with camera names"
            )
        if not math.isfinite(args.xr_sample_hz) or args.xr_sample_hz <= 0.0:
            raise ValueError("--xr-sample-hz must be positive and finite")
        if not math.isfinite(args.preview_fps) or args.preview_fps <= 0.0:
            raise ValueError("--preview-fps must be positive and finite")
        return run(args)
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[robodojo-teleop] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
