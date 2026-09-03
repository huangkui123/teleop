# RoboDojo Isaac Sim XR teleoperation

This adapter keeps XRoboToolkit and Isaac Sim in separate Python processes:

```text
Pico/XR controllers
  -> XRoboToolkit PC Service
  -> 60 Hz latest-controller-state buffer
  -> xrobotoolkit teleop server (Python 3.13)
  -> XPolicyLab WebSocket protocol
  -> RoboDojo Isaac Sim client (Python 3.11)
```

No RoboDojo source changes are required. The simulator imports the existing
`XPolicyLab/policy/demo_policy/deploy.py` episode loop while this repository
supplies the actions. For remote runs, a content-addressed runtime shim samples
preview and record-only cameras at different rates. The default captures and
sends `cam_head` every control step while capturing the two wrist cameras every
three steps for remote recording. Remote teleoperation also defaults to the
`minimal` render preset: 320x240 cameras, RTX performance mode, direct lighting
only, and no reflections, global illumination, translucency, shadows, ambient
occlusion, or anti-aliasing.

## Install

Start XRoboToolkit PC Service, connect the headset client, and install the
adapter plus its WebSocket dependencies in the XR environment:

```bash
cd /home/arx/Robotics/teleop
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate xrobotoolkit
python -m pip install -e '.[robodojo]'
```

## Run one case

The default task is `stack_bowls`, the default RoboDojo checkout is
`/home/arx/RoboDojo`, and Isaac Sim opens with a GUI:

```bash
python scripts/simulation/teleop_robodojo.py --accept-eula
```

Passing `--accept-eula` sets `OMNI_KIT_ACCEPT_EULA=YES` and means that you
accept the NVIDIA Omniverse EULA. Omit it if acceptance is already persisted or
if you want Isaac Sim to handle acceptance itself.

An explicit invocation is:

```bash
python scripts/simulation/teleop_robodojo.py \
  --robodojo-root /home/arx/RoboDojo \
  --task stack_bowls \
  --env-cfg arx_x5 \
  --eval-env RoboDojo \
  --env-gpu 0 \
  --scale-factor 1.0 \
  --rotation-frame tool \
  --xr-sample-hz 60 \
  --accept-eula
```

Controls:

- Hold left/right **Grip** to move the corresponding arm.
- Translate and rotate the controller for full relative 6DoF control. Rotation
  defaults to the controller/tool-local frame latched when Grip is pressed, so
  the controller and end effector do not need matching initial world
  orientations.
- Squeeze left/right **Trigger** to close the corresponding gripper.
- Release **Grip** to freeze and re-anchor that arm.
- Press `Ctrl+C` in the launch terminal to stop.

Use `--headless` to suppress the Isaac Sim window. For a remote simulator, add
`--preview` to display the head and both wrist observation cameras in one local
desktop window while Isaac Sim remains headless. Use `--mock-xr` to validate the
server/client bridge with stationary synthetic controllers, and `--dry-run` to
print both process commands without starting them.

Remote runs use JPEG observation transport by default. The teleop launcher
installs a content-addressed runtime shim under the external remote work
directory; it does not modify RoboDojo or XPolicyLab. Override the default with
`--jpeg-quality 60`, or use `--image-codec raw` only for protocol comparison.
The low-latency defaults are:

```text
--render-preset minimal
--rotation-frame tool
--preview-every 1 --vision-every 3 --stream-cameras cam_head
--xr-sample-hz 60 --preview-fps 10
```

Every control observation still carries fresh robot state. Only camera
selection/cadence changes: `--preview-every` controls cameras sent over the
network, while `--vision-every` controls the remaining record-only cameras.
Controller sampling and local preview decoding run independently. Set
`--preview-every 2` or higher to introduce fully state-only control observations
between preview frames. Use `--preview-every 1 --vision-every 1
--stream-cameras all --render-preset quality` to restore the original
per-step, three-camera, 640x480 quality configuration. The minimal preset also
changes remote recording resolution to 320x240.
The configured 25 Hz is simulation/data time, not a guarantee of 25 Hz wall-clock
teleoperation.

Use `--rotation-frame world` only to restore the earlier world-frame rotation
mapping for comparison.

RoboDojo writes its normal result videos and result JSON under
`eval_result/RoboDojo/`. This adapter does not add a demonstration-dataset
recorder.

## Remote simulator over SSH

The same launcher can run Isaac Sim on an SSH host and connect it to the local
XR server through a loopback-only reverse tunnel. It does not expose the policy
port publicly or copy teleop code into the remote RoboDojo checkout.

See [robodojo_remote_teleop.md](robodojo_remote_teleop.md) for the configured
`cscg-g41` command, isolated runtime-output layout, output location, and
troubleshooting.
