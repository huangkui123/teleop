"""Local camera preview for remote RoboDojo teleoperation."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .remote_transport import JPEG_FIELD

LOGGER = logging.getLogger(__name__)
CAMERAS = (
    ("cam_head", "Head"),
    ("cam_left_wrist", "Left wrist"),
    ("cam_right_wrist", "Right wrist"),
)


def _rgb_frame(value: Any, *, camera_name: str) -> np.ndarray:
    """Normalize a RoboDojo camera observation to contiguous HWC RGB uint8."""

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
        raise ValueError(f"{camera_name} image must be HxW, HxWxC, or CxHxW; got {frame.shape}")

    if np.issubdtype(frame.dtype, np.floating):
        finite = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        frame = np.clip(finite, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _camera_image(
    observation: Mapping[str, Any],
    camera_name: str,
) -> np.ndarray | None:
    vision = observation.get("vision")
    if not isinstance(vision, Mapping):
        return None
    camera = vision.get(camera_name)
    if not isinstance(camera, Mapping):
        return None
    if JPEG_FIELD in camera:
        payload = camera[JPEG_FIELD]
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError(f"{camera_name} {JPEG_FIELD} must contain encoded bytes")
        import cv2

        encoded = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{camera_name} contains an invalid JPEG image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for field in ("color", "rgb", "colors"):
        if field in camera:
            return _rgb_frame(camera[field], camera_name=camera_name)
    return None


def _fit_bgr(cv2, rgb: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if rgb is None:
        return canvas
    source_height, source_width = rgb.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        np.ascontiguousarray(rgb[:, :, ::-1]),
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def build_preview_mosaic(
    observation: Mapping[str, Any],
    *,
    max_width: int = 1280,
) -> np.ndarray | None:
    """Build a BGR mosaic with the head camera and two wrist cameras."""

    if max_width < 320:
        raise ValueError("preview max_width must be at least 320")

    images = {camera_name: _camera_image(observation, camera_name) for camera_name, _ in CAMERAS}
    available = [image for image in images.values() if image is not None]
    if not available:
        return None

    head = images["cam_head"]
    if head is None:
        head = available[0]
    head_height, head_width = head.shape[:2]

    import cv2

    if len(available) == 1:
        canvas = _fit_bgr(cv2, head, head_width, head_height)
        label = next(
            display_name
            for camera_name, display_name in CAMERAS
            if images[camera_name] is not None
        )
        cv2.putText(
            canvas,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if canvas.shape[1] > max_width:
            scale = max_width / canvas.shape[1]
            canvas = cv2.resize(
                canvas,
                (max_width, max(1, round(canvas.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    side_width = max(1, head_width // 2)
    side_height = max(1, head_height // 2)
    canvas = np.zeros((head_height, head_width + side_width, 3), dtype=np.uint8)

    canvas[:, :head_width] = _fit_bgr(
        cv2,
        images["cam_head"],
        head_width,
        head_height,
    )
    canvas[:side_height, head_width:] = _fit_bgr(
        cv2,
        images["cam_left_wrist"],
        side_width,
        side_height,
    )
    canvas[side_height:, head_width:] = _fit_bgr(
        cv2,
        images["cam_right_wrist"],
        side_width,
        head_height - side_height,
    )

    labels = (
        ("Head", 12, 30),
        ("Left wrist", head_width + 12, 30),
        ("Right wrist", head_width + 12, side_height + 30),
    )
    for label, x, y in labels:
        cv2.putText(
            canvas,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if canvas.shape[1] > max_width:
        scale = max_width / canvas.shape[1]
        canvas = cv2.resize(
            canvas,
            (max_width, max(1, round(canvas.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return canvas


class RoboDojoCameraPreview:
    """Display only the latest remote frame without blocking policy replies."""

    def __init__(
        self,
        *,
        window_name: str = "RoboDojo Teleoperation",
        max_width: int = 1280,
        refresh_hz: float = 10.0,
    ) -> None:
        if not np.isfinite(refresh_hz) or refresh_hz <= 0.0:
            raise ValueError("preview refresh_hz must be positive and finite")
        self.window_name = window_name
        self.max_width = max_width
        self.refresh_hz = float(refresh_hz)
        self._tk = None
        self._window = None
        self._image_label = None
        self._photo = None
        self._pending_observation: Mapping[str, Any] | None = None
        self._last_render_at = 0.0
        self._open = False
        self._enabled = True
        self._missing_images_reported = False
        self._render_error_reported = False

    def open(self) -> None:
        if self._open or not self._enabled:
            return
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "--preview needs a graphical session on the local machine " "(DISPLAY or WAYLAND_DISPLAY is not set)"
            )
        try:
            import tkinter as tk
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("--preview requires Tk GUI support in the local Python environment") from exc

        try:
            window = tk.Tk()
            window.title(self.window_name)
            window.configure(background="black")
            window.protocol("WM_DELETE_WINDOW", self.close)
            window.bind("<Escape>", lambda _event: self.close())
            window.bind("<KeyPress-q>", lambda _event: self.close())
            image_label = tk.Label(
                window,
                text="Waiting for RoboDojo camera frames...",
                foreground="white",
                background="black",
                font=("sans-serif", 18),
                width=64,
                height=20,
            )
            image_label.pack(fill="both", expand=True)
            window.update_idletasks()
            window.update()
        except tk.TclError as exc:
            raise RuntimeError(f"cannot open the local preview window: {exc}") from exc

        self._tk = tk
        self._window = window
        self._image_label = image_label
        self._open = True

    def _show(self, bgr: np.ndarray) -> None:
        assert self._tk is not None
        assert self._window is not None
        assert self._image_label is not None
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        height, width = rgb.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes()
        photo = self._tk.PhotoImage(
            master=self._window,
            data=ppm,
            format="PPM",
        )
        self._image_label.configure(
            image=photo,
            text="",
            width=width,
            height=height,
        )
        self._photo = photo

    def update(self, observation: Mapping[str, Any]) -> None:
        """Replace the queued frame and return without decoding it."""

        if not self._enabled:
            return
        vision = observation.get("vision")
        if not isinstance(vision, Mapping):
            return
        has_image = any(
            isinstance(camera, Mapping)
            and any(field in camera for field in (JPEG_FIELD, "color", "rgb", "colors"))
            for camera in vision.values()
        )
        if not has_image:
            return
        self._pending_observation = observation

    def _render_pending(self, now: float) -> None:
        if self._pending_observation is None:
            return
        if now - self._last_render_at < 1.0 / self.refresh_hz:
            return

        observation = self._pending_observation
        self._pending_observation = None
        self._last_render_at = now
        try:
            mosaic = build_preview_mosaic(
                observation,
                max_width=self.max_width,
            )
        except (TypeError, ValueError) as exc:
            if not self._render_error_reported:
                LOGGER.warning("Cannot render RoboDojo camera observation: %s", exc)
                self._render_error_reported = True
            return
        if mosaic is None:
            if not self._missing_images_reported:
                LOGGER.warning(
                    "RoboDojo observation has no cam_head or wrist RGB images; "
                    "the preview will wait for camera frames"
                )
                self._missing_images_reported = True
            return
        self._missing_images_reported = False
        self._render_error_reported = False
        try:
            self._show(mosaic)
        except self._tk.TclError as exc:
            LOGGER.warning("Local RoboDojo preview window failed: %s", exc)
            self.close()

    def pump(self) -> None:
        if not self._open or not self._enabled:
            return
        assert self._tk is not None
        assert self._window is not None
        self._render_pending(time.monotonic())
        if not self._enabled:
            return
        try:
            self._window.update_idletasks()
            self._window.update()
        except self._tk.TclError:
            self.close()

    def close(self) -> None:
        if not self._enabled and not self._open:
            return
        self._enabled = False
        self._open = False
        window = self._window
        self._window = None
        self._image_label = None
        self._photo = None
        self._pending_observation = None
        if window is not None:
            try:
                window.destroy()
            except self._tk.TclError:
                LOGGER.debug("Local RoboDojo preview window was already closed")
