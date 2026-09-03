"""XPolicyLab-compatible WebSocket server for RoboDojo XR teleoperation."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np

from .adapter import (
    ROTATION_FRAMES,
    BufferedXrInput,
    RoboDojo6DoFMapper,
    RoboDojoTeleopPolicy,
)
from .preview import RoboDojoCameraPreview


class NeutralXrClient:
    """Stationary input source used to validate the bridge without a headset."""

    def __init__(self) -> None:
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

    def get_pose_by_name(self, name: str):
        return self.poses[name]

    def get_key_value_by_name(self, name: str) -> float:
        return self.keys[name]

    def close(self) -> None:
        return None


def add_xpolicylab_source(robodojo_root: str | Path) -> Path:
    root = Path(robodojo_root).expanduser().resolve()
    xpolicylab_root = root / "XPolicyLab"
    protocol_file = xpolicylab_root / "client_server" / "ws" / "model_server.py"
    if not protocol_file.is_file():
        raise FileNotFoundError(
            f"XPolicyLab WebSocket server not found under RoboDojo root: {protocol_file}"
        )
    source_path = str(xpolicylab_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return root


async def serve(
    *,
    robodojo_root: str | Path,
    host: str,
    port: int,
    scale_factor: float,
    grip_threshold: float,
    rotation_frame: str,
    xr_sample_hz: float,
    preview_fps: float,
    mock_xr: bool = False,
    preview: bool = False,
) -> None:
    if not np.isfinite(xr_sample_hz) or xr_sample_hz <= 0.0:
        raise ValueError("xr_sample_hz must be positive and finite")
    if not np.isfinite(preview_fps) or preview_fps <= 0.0:
        raise ValueError("preview_fps must be positive and finite")

    add_xpolicylab_source(robodojo_root)
    try:
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RoboDojo bridge dependencies are incomplete. Install them from "
            "the teleop repository with: python -m pip install -e '.[robodojo]' "
            f"(missing module: {exc.name})"
        ) from exc

    if mock_xr:
        xr_source = NeutralXrClient()
    else:
        from xrobotoolkit_teleop.common.xr_client import XrClient

        xr_source = XrClient()
    xr_client = BufferedXrInput(xr_source)
    xr_client.sample(raise_on_error=True)
    mapper = RoboDojo6DoFMapper(
        xr_client=xr_client,
        scale_factor=scale_factor,
        grip_threshold=grip_threshold,
        rotation_frame=rotation_frame,
    )
    camera_preview = RoboDojoCameraPreview(refresh_hz=preview_fps) if preview else None
    if camera_preview is not None:
        camera_preview.open()
    policy = RoboDojoTeleopPolicy(
        mapper=mapper,
        preview=camera_preview,
    )
    server = PolicyServer(
        policy,
        PolicyServerConfig(host=host, port=port),
    )
    print(
        f"[robodojo-teleop] WebSocket server listening on ws://{host}:{port} "
        f"(XR={'mock' if mock_xr else 'live'} sampled at {xr_sample_hz:g} Hz, "
        f"scale={scale_factor:g}, rotation-frame={rotation_frame})",
        flush=True,
    )
    print(
        "[robodojo-teleop] Hold left/right Grip to move that arm; "
        "Triggers close the grippers.",
        flush=True,
    )

    async def pump_xr() -> None:
        interval = 1.0 / xr_sample_hz
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            xr_client.sample()
            deadline += interval
            await asyncio.sleep(max(0.0, deadline - loop.time()))

    xr_sample_task = asyncio.create_task(pump_xr())
    preview_task = None
    if camera_preview is not None:

        async def pump_preview() -> None:
            while True:
                camera_preview.pump()
                await asyncio.sleep(1.0 / 30.0)

        preview_task = asyncio.create_task(pump_preview())

    try:
        await server.serve_forever()
    finally:
        xr_sample_task.cancel()
        with suppress(asyncio.CancelledError):
            await xr_sample_task
        if preview_task is not None:
            preview_task.cancel()
            with suppress(asyncio.CancelledError):
                await preview_task
        await server.stop()
        policy.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve XR controller commands to a RoboDojo Isaac Sim client."
    )
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--grip-threshold", type=float, default=0.9)
    parser.add_argument(
        "--rotation-frame",
        choices=ROTATION_FRAMES,
        default="tool",
        help=(
            "Apply rotations in the Grip-latched tool frame (default), or use "
            "the legacy world-frame mapping."
        ),
    )
    parser.add_argument(
        "--xr-sample-hz",
        type=float,
        default=60.0,
        help="Sample XR controllers independently at this rate.",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=10.0,
        help="Maximum local preview refresh rate.",
    )
    parser.add_argument(
        "--mock-xr",
        action="store_true",
        help="Use stationary synthetic controllers for bridge testing.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show head and wrist cameras in a local desktop window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    try:
        asyncio.run(
            serve(
                robodojo_root=args.robodojo_root,
                host=args.host,
                port=args.port,
                scale_factor=args.scale_factor,
                grip_threshold=args.grip_threshold,
                rotation_frame=args.rotation_frame,
                xr_sample_hz=args.xr_sample_hz,
                preview_fps=args.preview_fps,
                mock_xr=args.mock_xr,
                preview=args.preview,
            )
        )
    except KeyboardInterrupt:
        print("\n[robodojo-teleop] Server stopped.", flush=True)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[robodojo-teleop] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
