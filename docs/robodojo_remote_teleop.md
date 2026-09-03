# 使用本地 XR 遥操远程 RoboDojo

本地机器以独立频率读取 XR 手柄并运行 6DoF 策略服务，`cscg-g41` 运行
RoboDojo 和 Isaac Sim。启动器在远端把控制观测与相机观测解耦、把预览相机
压成 JPEG，并使用仅监听回环地址的 SSH 反向隧道连接两端：

```text
本地头显/手柄 --60 Hz采样--> 最新6DoF状态
                                  ^
                                  |
cscg-g41 RoboDojo --状态请求--> 本地 teleop WebSocket 服务
          |
          +--每个控制步采集头部相机并发到本地预览
          `--每3个控制步采集两路腕部相机并在服务器录像
```

远端不需要安装 teleop，也不需要开放防火墙端口。

默认低延迟参数是 `--render-preset minimal --rotation-frame tool --xr-sample-hz 60
--preview-every 1 --vision-every 3 --stream-cameras cam_head --preview-fps 10`。
最低画质预设使用 320×240 相机、RTX `performance` 模式，关闭反射、全局光照、
透明、阴影、环境光遮蔽和抗锯齿，但保留直接光照以保证画面可见。头部相机每步
更新以保证主视图连续，两路腕部相机降低到每 3 步记录一次。如果把
`--preview-every` 调到 2 或 3，没有任何相机到期的控制步就会跳过
render/camera capture 和 JPEG，只读取最新机器人状态并立即返回基于最新手柄
快照的动作。

## 已部署的服务器布局

- SSH 别名：`cscg-g41`（登录用户 `huangkui`）
- RoboDojo：`/home/huangkui/RoboDojo`
- 独立 Miniconda：`/home/huangkui/miniconda3`
- Conda 环境：`RoboDojo`
- 运行输出和 Isaac Kit 缓存：
  `/home/huangkui/robodojo-teleop-runs`
- 本地 teleop：`/home/arx/Robotics/teleop`
- 本地 RoboDojo：`/home/arx/RoboDojo`

这套部署完全位于 `/home/huangkui`，不会读取或写入 `/home/dong`。启动器也会
拒绝把 `--remote-workdir` 设为远端 RoboDojo 源码目录或其子目录，因此一次
遥操运行不会向源码树写结果或 Kit 缓存。

部署版本（2026-07-28）：

- RoboDojo `e0703b03bb1af6075400e9d60dc17a792793960c`
- XPolicyLab `fe71eb54675cef495fea817a637386a4f4529153`
- IsaacLab `afca7b09d60d8beb9c1cb28b43066499940b969b`
- CuRobo `d17b54ce32cba095c0b000c4c58777075d11de0e`
- Assets `1a3c4c334aef294c31d7a0190d8d6dff68df78e0`
- Python 3.11、Isaac Sim 5.1、PyTorch 2.7/cu128

Assets 固定为本机 RoboDojo 已使用的 39 GB 快照；目标 case 的
`stack_bowls.yml` 和 `arx_x5.yml` 与远端源码逐字节一致。
部署时已从 `curobo_tmp.yml` 重新生成机器人配置，其中的绝对资源路径指向
`/home/huangkui/RoboDojo/Assets`，不包含本机 `/home/arx` 路径。

## 首次准备本地 teleop

确认 SSH 无交互登录和远端环境可用：

```bash
ssh cscg-g41 \
  'source /home/huangkui/miniconda3/etc/profile.d/conda.sh &&
   conda run -n RoboDojo python --version'
```

在本地安装 teleop 的 RoboDojo 依赖：

```bash
cd /home/arx/Robotics/teleop
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate xrobotoolkit
python -m pip install -e '.[robodojo]'
```

真实遥操前，启动本地 XRoboToolkit PC Service，并确认头显和两个控制器已经
连接。

## 先做一次静止输入测试

`--mock-xr` 不依赖头显，适合验证 SSH、隧道、Isaac Sim、资源和动作协议：

```bash
cd /home/arx/Robotics/teleop
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate xrobotoolkit

python scripts/simulation/teleop_robodojo.py \
  --robodojo-root /home/arx/RoboDojo \
  --ssh-host cscg-g41 \
  --remote-workdir /home/huangkui/robodojo-teleop-runs \
  --task stack_bowls \
  --env-cfg arx_x5 \
  --env-gpu 0 \
  --headless \
  --preview \
  --image-codec jpeg \
  --jpeg-quality 75 \
  --preview-every 1 \
  --vision-every 3 \
  --stream-cameras cam_head \
  --render-preset minimal \
  --rotation-frame tool \
  --xr-sample-hz 60 \
  --preview-fps 10 \
  --mock-xr \
  --accept-eula
```

这套部署已于 2026-07-28 在服务器 GPU 1 上完整跑完 `800/800` 步静止输入
测试，启动器退出码为 0，并生成 3 路 640×480、801 帧的 H.264 视频和结果
JSON。由于 `--mock-xr` 不会移动机械臂，`success_rate: 0.0` 是预期的任务结果，
不表示部署或遥操链路失败。

当前启动器已经把远端源码和 Conda 默认值设为 `/home/huangkui` 下的部署。
如果希望把路径全部写明，可以额外添加：

```text
--remote-robodojo-root /home/huangkui/RoboDojo
--remote-conda-base /home/huangkui/miniconda3
```

`--accept-eula` 会为远端进程设置 `OMNI_KIT_ACCEPT_EULA=YES`，表示接受
NVIDIA Omniverse EULA。

## 运行真实 6DoF 遥操

`--headless` 只是不在服务器桌面打开 Isaac Sim 编辑器窗口，并不会关闭任务的
三路观测相机。普通 SSH 无法直接看到远端 GUI；即使去掉 `--headless`，窗口
也只会尝试出现在服务器的 `DISPLAY` 上。

本地启动器的 `--preview` 会显示 WebSocket 观测中的相机。低延迟默认值
`--stream-cameras cam_head` 只传输并显示头部相机；使用
`--stream-cameras all` 才会恢复头部和两路腕部相机的组合画面。图像和动作仍走
同一条 SSH 加密隧道，但相机不是每个控制步都发送。

远程启动默认使用 `--image-codec jpeg --jpeg-quality 75`。启动器会把 teleop
侧的传输适配器按内容哈希部署到
`/home/huangkui/robodojo-teleop-runs/teleop-adapter/`，只在 Python 进程运行时
压缩/筛选 WebSocket 观测并调整相机采集节奏，不写入 RoboDojo 或 XPolicyLab
源码。

确认头显已连接后，保留 `--headless --preview` 并去掉 `--mock-xr`：

```bash
cd /home/arx/Robotics/teleop
source /home/arx/miniconda3/etc/profile.d/conda.sh
conda activate xrobotoolkit

python scripts/simulation/teleop_robodojo.py \
  --robodojo-root /home/arx/RoboDojo \
  --ssh-host cscg-g41 \
  --remote-workdir /home/huangkui/robodojo-teleop-runs \
  --task stack_bowls \
  --env-cfg arx_x5 \
  --env-gpu 0 \
  --headless \
  --preview \
  --image-codec jpeg \
  --jpeg-quality 75 \
  --preview-every 1 \
  --vision-every 3 \
  --stream-cameras cam_head \
  --render-preset minimal \
  --rotation-frame tool \
  --xr-sample-hz 60 \
  --preview-fps 10 \
  --accept-eula
```

操作映射：

- 按住左/右 **Grip**：启用对应机械臂，并在当前位置无跳变地锚定。
- 平移、旋转控制器：控制对应末端的完整相对 6DoF 位姿。默认
  `--rotation-frame tool` 会在按下 Grip 时把控制器局部旋转轴与当时的末端局部
  旋转轴对齐，不要求两者的初始世界朝向一致。
- 左/右 **Trigger**：控制对应夹爪闭合程度。
- 松开 **Grip**：保持机械臂位置；再次按下时重新锚定。
- 本地预览窗口按 `q` 或 `Esc`：只关闭画面窗口，仿真继续运行。
- 本地终端按 `Ctrl+C`：停止本地服务、SSH 隧道和远端 Isaac Sim。

如需对比修改前的世界坐标系旋转行为，可显式使用
`--rotation-frame world`。

## 输出位置

评测 JSON、视频、Isaac Kit/XDG 缓存、Python 字节码缓存和 IsaacLab 临时文件
都在远端工作目录，不在源码树或共享 `/tmp`：

```text
/home/huangkui/robodojo-teleop-runs/eval_result/RoboDojo/
```

## 图像传输性能

使用高画质完整仿真产生的三路 640×480 相机帧进行同链路回放，JPEG 质量 75
的结果为：

- RGB 图像：每次观测 2700 KiB
- JPEG 图像：每次观测约 92 KiB，缩小约 29 倍
- SSH WebSocket 平均往返：由约 227 ms 降到约 20 ms
- 图像协议吞吐能力：由约 4.4 Hz 提高到约 50 Hz

因此优化后网络不再限制当前约 4–5 Hz 的仿真速度。这里的 50 Hz 是传输上限，
不是 Isaac Sim 的实际运行频率。

现在默认又增加了一层控制/视频解耦：

- 每个控制步都发送机器人状态并取回最新 6DoF 动作。
- 本地 XR 状态固定以 60 Hz 更新，不再由远端相机请求频率决定采样时刻。
- 每个控制步采集并发送一次 `cam_head`，维持尽可能连续的主视图。
- 两路腕部相机每 3 个控制步采集一次，只在服务器录像，不进入默认网络预览。
- 如果设置 `--preview-every 2` 或更大，没有相机到期的中间控制步会完全跳过
  render、相机回读和 JPEG。
- 本地预览只保留最新待显示帧，JPEG 解码和窗口绘制不阻塞动作响应。
- 远端每 100 个控制步打印一次实测控制循环 Hz 和新相机帧比例。

`arx_x5` 的语义控制频率为 25 Hz，因此默认头部录像为 25 FPS，两路腕部录像为
8.33 FPS。各录像 writer 会使用对应采样率，避免回放速度错误。这仍不承诺墙钟
控制能达到 25 Hz；实际值取决于双臂 IK、CPU 物理仿真和服务器资源占用。

2026-07-28 在空闲 GPU 2、`stack_bowls`、静止 mock XR 上做的 100 步短测
使用的是修改前的 RoboDojo 高画质配置：

- 默认平衡档
  `--preview-every 1 --vision-every 3 --stream-cameras cam_head`：
  控制 **4.62 Hz**，头部画面随控制步更新，也是约 **4.62 Hz**。
- 控制优先档
  `--preview-every 3 --vision-every 3 --stream-cameras cam_head`：
  控制 **6.26 Hz**，100 步产生 34 个预览帧，约 **2.1 Hz**。

前者是默认值，因为遥操时 4–5 Hz 连续主视图比 2 Hz 画面更实用。实机手柄运动、
任务场景和服务器负载会让结果发生变化。

随后在空闲 GPU 0 上使用当前默认最低画质
`--render-preset minimal --preview-every 1 --vision-every 3` 做了同样的短测：

- 头部图像确认为 320×240，单帧 RGB 由 900 KiB 降为 225 KiB，JPEG 质量 75
  后约 29 KiB。
- 前 100 步控制与头部画面均为 **5.41 Hz**，比上述高画质默认档的 4.62 Hz
  高约 17%。两次测试使用同型号 4090，但不是同一张物理卡，因此该比例用于
  遥操调优参考，不视为严格的渲染 benchmark。

按需要调节：

- 默认主视图：`--preview-every 1`，头部相机随每个控制步更新。
- 更高控制优先级：`--preview-every 2` 或 `3`，代价是主视图更新更稀疏。
- 腕部录像更密：`--vision-every 1`；更省资源：`--vision-every 4` 或 `5`。
- 查看三路画面：`--stream-cameras all`；这会增加 JPEG 和网络开销。
- 只查看某一路腕部相机：例如
  `--stream-cameras cam_left_wrist`。

查看最近生成的结果：

```bash
ssh cscg-g41 \
  'find /home/huangkui/robodojo-teleop-runs/eval_result/RoboDojo \
   -type f \( -name "_result.json" -o -name "*.mp4" \) \
   -printf "%T@ %p\n" 2>/dev/null | sort -nr | head'
```

## 常用选项和排查

- 只打印本地、远端完整命令：添加 `--dry-run`。
- 远端反向端口冲突：添加例如 `--remote-port 19001`。
- 更换任务：修改 `--task`，其名称必须对应远端
  `task/RoboDojo/config/<任务名>.yml`。
- 更换 GPU：修改 `--env-gpu`。该值会同时设置 `CUDA_VISIBLE_DEVICES` 和
  RoboDojo 的 `--device_id`。运行前可用 `ssh cscg-g41 nvidia-smi` 选择空闲
  GPU；文档命令中的 GPU 0 只是示例。
- 图像质量：默认 `--jpeg-quality 75`。网络仍不稳定时可尝试 60；需要无损
  对照诊断时可设置 `--image-codec raw`，但三路原始 RGB 会明显拖慢控制。
- 最低画质是当前遥操默认值：`--render-preset minimal`。它会把远端录像也改为
  320×240；恢复 RoboDojo 原始 640×480 高画质使用
  `--render-preset quality`。
- 恢复修改前的逐步三相机传输行为：
  `--render-preset quality --preview-every 1 --vision-every 1
  --stream-cameras all`。
- 服务器一般使用 `--headless`；去掉它之前需先配置远程桌面和 `DISPLAY`。
- 本地没有图形桌面或通过无 X11 的 SSH 登录本机时，不要添加 `--preview`；
  否则启动器会在远端仿真启动前明确报错。
- 隧道只监听远端 `127.0.0.1`，不要把本地 `--bind-host` 改成公网地址。

检查 CUDA 和关键运行库：

```bash
ssh cscg-g41 '
  source /home/huangkui/miniconda3/etc/profile.d/conda.sh
  conda activate RoboDojo
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  python -c "import ctypes, torch; ctypes.CDLL(\"libXt.so.6\"); ctypes.CDLL(\"libGLU.so.1\"); print(torch.cuda.is_available(), torch.cuda.device_count())"
'
```

检查 SSH 断开后是否有残留的仿真进程：

```bash
ssh cscg-g41 \
  "pgrep -af '/home/huangkui/RoboDojo/src/eval_client/main.py' || true"
```

启动器启用了 `ExitOnForwardFailure`、SSH keepalive 和远端控制终端。隧道建立
失败时不会继续启动 Isaac Sim；本地进程退出或连接断开时，远端进程也会随
控制终端结束。
