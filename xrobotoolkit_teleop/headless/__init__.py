"""Headless teleoperation providers.

These providers expose teleoperation as a pull API for external simulators:
call ``update(current_positions)`` and receive target joint positions by name.
"""
