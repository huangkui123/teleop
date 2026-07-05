# AgileX PiPER Assets

This directory contains PiPER meshes and a generated dual-arm URDF used by the
solver teleoperation demo.

Source assets:
- Repository: https://github.com/agilexrobotics/piper_ros
- Package: `src/piper_description`
- License declared by package: BSD

`dual_piper.urdf` was generated from `piper_description.urdf` by duplicating the
single-arm model with `left_` and `right_` prefixes and attaching both roots to a
shared `world` link.
