# Jetson 棋盘格标定、USB 云台联调与安全验收运行手册

本手册用于 Jetson Orin NX Super、海康 MV-CA013-21UC、8 mm F/2.8 镜头与 USB CDC-ACM 云台主板的实机验收。所有命令默认在 Jetson 上执行。

## 0. 固定硬件与安全约束

- 相机：MV-CA013-21UC，序列号 `00G02809155`。
- 图像：`1280 x 1024`、`BayerRG8`、缓冲区 2。
- 当前相机参数：

```yaml
acquisition_fps: 50
exposure_us: 15000
gain_db: 14.0
```

- 本地显示为 45 FPS，检测为 50 FPS，USB 输出为 50 Hz。
- 棋盘格为 **6 x 9 physical squares**，OpenCV 参数为 **8 x 5 inner corners**，每格 **22 mm**。
- USB 稳定路径为 `/dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00`。
- 纯预测控制最多 **3 frames / 60 ms**；第 4 帧预测必须失效。
- 协议中的 `fire=0` 恒定为零，但 **fire=0 does not turn off the physical laser**。

> **激光危险：**405 nm 激光由硬件上电后常亮，Jetson 软件没有关闭能力。任何标定、静态调试、断线测试或电机受限测试前，都必须物理断开激光供电，或使用固定可靠、完全不透光的机械遮挡，并使用适合 405 nm 的防护眼镜。不要把软件安全包当作 physical laser 急停。

## 1. 更新项目并进入环境

### 1.1 拉取当前功能分支

```bash
cd ~/2025-E-Vision/2025-E-Vision
git fetch origin
git switch feature/classical-white-board-tracking
git pull --ff-only origin feature/classical-white-board-tracking
```

确认分支和提交：

```bash
git branch --show-current
git log -1 --oneline
```

### 1.2 激活 conda 并安装项目

每次 Jetson 重启或新开终端后，先执行：

```bash
cd ~/2025-E-Vision/2025-E-Vision
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 2025-e-vision
python -m pip install -e '.[vision,hardware]'
```

恢复 MVS Python 模块与动态库路径：

```bash
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

快速确认：

```bash
which python
python -V
python -c 'from MvCameraControl_class import MvCamera; print("MVS Python import OK")'
python -c 'import serial; print("pyserial", serial.VERSION)'
```

如需每次激活环境自动设置 MVS 路径，可把两条 `export` 写入单独脚本；验收时仍建议先在当前终端显式执行，避免使用了旧路径却没有发现。

## 2. 排除相机和串口占用

### 2.1 关闭会抢占相机的程序

先退出 MVS Viewer、旧网页服务和旧本地预览。检查：

```bash
ps -ef | grep -E '[m]vviewer|[e]v-camera|[e]v-gimbal|[c]amera_tuning_server|[u]vicorn'
```

若存在旧进程，优先回到对应终端按 `Ctrl+C`，不要同时启动两个相机所有者。

### 2.2 停止 ModemManager

ModemManager 可能探测并短暂占用 `/dev/ttyACM0`。本次启动先停止：

```bash
sudo systemctl stop ModemManager
systemctl is-active ModemManager || true
```

比赛前若确定 Jetson 不需要蜂窝调制解调器，可按现场需求禁用开机启动：

```bash
sudo systemctl disable ModemManager
```

需要恢复时：

```bash
sudo systemctl enable ModemManager
sudo systemctl start ModemManager
```

### 2.3 检查相机和云台 USB

```bash
lsusb
ls -l /dev/serial/by-id/
ls -l /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00
readlink -f /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00
sudo fuser -v /dev/ttyACM0 || true
```

期望稳定路径最终指向 `/dev/ttyACM0` 或另一个 `ttyACM*`。程序配置应始终使用 `/dev/serial/by-id/...`，不要写死 `ttyACM0`。

插拔时可在另一个终端观察：

```bash
sudo dmesg --follow
```

## 3. 棋盘格内参标定

### Stage 0 - physical laser safety

1. 物理断开 405 nm 激光供电，或安装可靠不透光遮挡。
2. 确认云台电机关闭、失能，或机械结构已经可靠约束。
3. 清除镜头前方人员，固定相机与镜头；标定后不能再改变镜头焦距、光圈、相机与镜头装配关系。

### 3.1 检查采集参数

```bash
cd ~/2025-E-Vision/2025-E-Vision
sed -n '1,16p' config/jetson-local.yaml
```

必须确认：

```yaml
camera:
  width: 1280
  height: 1024
  pixel_format: BayerRG8
  acquisition_fps: 50
  exposure_us: 15000
  gain_db: 14.0
```

### 3.2 采集 20-25 images

必须从 Jetson 桌面图形会话运行。检查：

```bash
echo "$DISPLAY"
```

若 Jetson 本地桌面终端为空，可尝试 `export DISPLAY=:0`；SSH 会话即使设置 DISPLAY，也可能没有创建窗口的权限。

启动采集：

```bash
ev-camera-calibration-capture \
  --config config/jetson-local.yaml \
  --serial 00G02809155 \
  --columns 8 \
  --rows 5 \
  --square-mm 22 \
  --output artifacts/calibration/images \
  --width 960 \
  --timeout-ms 100
```

操作：

- `Space`：保存当前合格原始帧。
- `R`：删除本次会话最近保存的一张图。
- `Q`、`Esc` 或关闭窗口：退出。
- 相似姿态会提示警告，但仍允许保存；应主动换姿态，不要只在画面中央平移。

采集要求：

- 保存 **20-25 images**，最终 **at least 15 usable** 且姿态有明显变化。
- 棋盘格应覆盖画面中央、四角、四边，包含近、中、远距离。
- 同时包含绕水平轴、垂直轴的小角度倾斜，不要全部正对相机。
- 每张都必须看到完整的 8 x 5 内角点，不能被裁切、手遮挡或运动模糊。
- 不要改变分辨率，所有输入必须为 `1280 x 1024`。

检查文件：

```bash
find artifacts/calibration/images -maxdepth 1 -type f | sort
find artifacts/calibration/images -maxdepth 1 -type f | wc -l
```

### 3.3 求解内参

```bash
python tools/calibrate_camera.py \
  artifacts/calibration/images \
  --output config/camera_calibration.yaml \
  --columns 8 \
  --rows 5 \
  --square-mm 22 \
  --max-rms 0.5
```

成功输出应包含 `images=... usable_poses=... rejected=... rms_px=...`。验收标准：

- `usable_poses >= 15`；虽然求解器最低可接受 10 张，但正式实机验收不要低于 15 张。
- `rms_px <= 0.5`。
- 若 RMS 超限、有效图不足或输入尺寸混杂，重新采集，不要手工修改结果绕过检查。

### 3.4 核验 camera_calibration.yaml

```bash
ls -lh config/camera_calibration.yaml
sed -n '1,160p' config/camera_calibration.yaml
python - <<'PY'
from pathlib import Path
import yaml

path = Path("config/camera_calibration.yaml")
data = yaml.safe_load(path.read_text(encoding="utf-8"))
print("image_size =", data.get("image_size"))
print("rms_px =", data.get("rms_px"))
print("camera_matrix =", data.get("camera_matrix"))
print("distortion =", data.get("distortion"))
PY
```

必须确认：

- `image_size` 对应 1280 x 1024。
- `rms_px` 是有限数且不大于 0.5。
- `camera_matrix` 为有效 3 x 3 矩阵，焦距为正。
- `distortion` 存在且全部为有限数。

保留标定图和 YAML 备份：

```bash
tar -czf "$HOME/camera-calibration-$(date +%Y%m%d-%H%M%S).tar.gz" \
  config/camera_calibration.yaml artifacts/calibration/images
```

## 4. 软件回归检查

### Stage 1 - software and camera verification

确保相机未被 MVS Viewer 或网页服务占用，然后运行：

```bash
cd ~/2025-E-Vision/2025-E-Vision
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPYCACHEPREFIX="$(mktemp -d)" python -m compileall -q src tests tools
TEST_TMP="$(mktemp -d)"
python -m pytest -q -p no:cacheprovider --basetemp "$TEST_TMP"
```

全部测试通过后再开始 USB 联调。另开终端监控资源和内核 USB 日志：

```bash
tegrastats
```

```bash
sudo dmesg --follow
```

## 5. 启动正式视觉与 USB 输出

### 5.1 本地 OpenCV 窗口模式

该模式不启动网页、FastAPI 或 Uvicorn，只显示靶子绿色框和靶心：

```bash
ev-gimbal-vision \
  --config config/jetson-local.yaml \
  --gimbal-config config/gimbal_usb.yaml \
  --serial 00G02809155 \
  --display \
  --width 640 \
  --display-fps 45 \
  --detection-fps 50
```

### 5.2 无页面后台模式

比赛运行优先使用此模式，内存和图形开销更低：

```bash
ev-gimbal-vision \
  --config config/jetson-local.yaml \
  --gimbal-config config/gimbal_usb.yaml \
  --serial 00G02809155 \
  --detection-fps 50
```

程序启动时应打印相机参数、标定状态、稳定串口路径、50 Hz 输出、预测限制和激光警告。无显示模式会周期打印类似：

```text
camera=Connected acquisition=...fps detection=...fps | usb=up ticks=... valid=... safe=... overruns=... error=none
```

保存联调日志：

```bash
mkdir -p artifacts/logs
ev-gimbal-vision \
  --config config/jetson-local.yaml \
  --gimbal-config config/gimbal_usb.yaml \
  --serial 00G02809155 \
  --detection-fps 50 \
  2>&1 | tee "artifacts/logs/gimbal-vision-$(date +%Y%m%d-%H%M%S).log"
```

如以后创建 systemd 服务，可用以下命令查日志；当前手工运行仍以终端和 `tee` 日志为准：

```bash
journalctl -u ev-gimbal-vision.service -f
```

## 6. 分阶段安全验收

所有阶段必须按下面顺序执行。任一阶段失败，先停机排查，不要跳到后面的电机动作测试。

### Stage 2 - no gimbal and no calibration

目的：验证缺少云台或缺少标定不会阻断相机检测，但 USB 控制必须 fail-closed。

1. 保持激光物理断开/遮挡，云台电机失能。
2. 暂时移走标定文件并断开云台 USB：

```bash
mv config/camera_calibration.yaml config/camera_calibration.yaml.saved
```

3. 启动 `ev-gimbal-vision --display`。
4. 验收：相机、检测和预览继续工作；控制转换器提示标定不可用；所有拟发送控制都应为 **all-zero**，即 yaw、pitch、distance 均为 0 且 `tracking=0`、`fire=0`。
5. 恢复文件：

```bash
mv config/camera_calibration.yaml.saved config/camera_calibration.yaml
```

### Stage 3 - valid calibration with motors disabled

1. 连接云台 USB，但保持电机失能或机械约束。
2. 启动无页面命令。
3. 云台端检查连续帧、CRC、sequence 和安全字段。
4. 无目标、过期结果、相机断开、检测错误时，必须收到 all-zero、`tracking=0`。
5. 有有效真实观测时，才允许 `tracking=1`，yaw/pitch 必须为有限数；distance、fire、target_id、reserved 仍为 0。

### Stage 4 - static target and axis signs

保持电机失能，用固定靶验证方向：

1. 靶心在图像中心：yaw 和 pitch 应接近 0。
2. 将靶放到画面右侧：使用当前 `yaw_sign: 1` 时 yaw 应为正。
3. 将靶放到画面左侧：yaw 应为负。
4. 将靶放到画面下方：使用当前 `pitch_sign: 1` 时 pitch 应为正。
5. 将靶放到画面上方：pitch 应为负。

若实际云台坐标方向相反，只修改 `config/gimbal_usb.yaml` 的 `yaw_sign` 或 `pitch_sign` 为 `-1`，一次只改一个轴并重新验证。不要在检测代码和云台固件两边同时取反。

### Stage 5 - prediction limit

1. 先让完整靶面稳定产生真实观测。
2. 快速遮挡或让靶面短暂出画，制造纯 `PREDICTED`。
3. 只允许第 1–3 个预测结果且源观测年龄不超过 **60 ms**。
4. **fourth predicted frame** 或超过 60 ms 必须立即变为 all-zero、`tracking=0`。
5. 某次预测因超帧数、超时或单轴跳变大于 1.5° 被拒绝后，后续预测保持锁死，直到新的真实观测恢复。
6. 新的有效真实观测恢复时立即允许输出；当前设计不做恢复跳变检查。

### Stage 6 - disconnect and reconnect

1. 电机仍保持失能。
2. 运行中拔掉云台 USB，确认视觉相机和检测继续运行，USB 状态变为 down，程序不退出。
3. 等待至少 1 秒后重新插入，确认稳定 by-id 路径恢复。
4. **first frame after reconnect** 必须是 all-zero、`tracking=0` 的安全包，即使此时靶面有效；下一帧才允许恢复有效控制。
5. sequence 在重连后继续，不因重连重置。

### Stage 7 - controlled motor movement

只有前面全部通过后才允许：

1. 激光仍物理断开或可靠遮挡。
2. 低速、限角、限流，先单轴再双轴；周围不得有人处于机构运动范围。
3. 先把靶放在小偏差位置，确认云台运动能让误差减小而不是放大。
4. 若方向错误，立即急停并修正单个 `yaw_sign`/`pitch_sign`。
5. 确认云台固件在 `tracking=0` 或超过 **more than 100 ms** 没有收到新鲜有效帧时停止跟踪/进入安全状态。
6. 再测试静态靶、缓慢移动车体和短时遮挡；不要一开始就高速运动。

### Stage 8 - shutdown safety

1. 在程序正常运行时按 `Ctrl+C`。
2. USB worker 应先停止，并尝试发送 **5 safe frames**（5 个 all-zero、`tracking=0` 包），然后才关闭相机服务。
3. 云台固件必须在安全包或通信超时后停止跟踪。
4. 软件退出只表示控制输出停止；`fire=0 does not turn off the physical laser`，仍需物理断电或保持可靠遮挡。

## 7. 故障排查

### 7.1 `conda: command not found`

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 2025-e-vision
```

### 7.2 MVS Python 模块或动态库找不到

```bash
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'from MvCameraControl_class import MvCamera; print("MVS OK")'
ls -l /opt/MVS/lib/aarch64/libMvCameraControl.so*
```

### 7.3 相机黑屏、断开或无法打开

```bash
lsusb | grep 2bdf:0001
ps -ef | grep -Ei '[m]vs|[m]vviewer|[e]v-camera|[e]v-gimbal|[u]vicorn'
sudo dmesg --follow
```

关闭第二个相机进程，检查 USB3 线、供电和镜头光圈。不要同时运行网页调参、本地预览和正式云台命令。

### 7.4 没有 `/dev/serial/by-id/...`

```bash
lsusb
ls -l /dev/ttyACM* 2>/dev/null || true
ls -l /dev/serial/by-id/ 2>/dev/null || true
systemctl is-active ModemManager || true
sudo systemctl stop ModemManager
sudo dmesg --follow
```

若 `dmesg` 没有新的 `cdc_acm ... ttyACM*`，优先排查 USB 数据线是否仅供电、Micro-USB 接口固件、接头和供电。不要退回写死 `/dev/ttyACM0` 作为长期配置。

### 7.5 串口权限或被占用

```bash
id
ls -l /dev/ttyACM0
sudo fuser -v /dev/ttyACM0 || true
```

用户应属于 `dialout`。若刚加入组，需要注销登录或重启会话。

### 7.6 标定不可用但画面正常

这是预期的 fail-closed 行为。检查启动日志中的 calibration error，然后核对：

```bash
ls -lh config/camera_calibration.yaml
sed -n '1,160p' config/camera_calibration.yaml
sed -n '1,80p' config/gimbal_usb.yaml
```

重点检查路径、1280 x 1024 尺寸、`rms_px <= 0.5`、矩阵和畸变参数。不要为了让 `tracking=1` 而关闭校验。

### 7.7 USB 一直 down，但视觉正常

这是串口故障隔离的预期表现。检查：

```bash
ls -l /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00
sudo fuser -v /dev/ttyACM0 || true
systemctl is-active ModemManager || true
```

transport 每 1 秒尝试重连；恢复后的第一包仍必须是安全零包。

### 7.8 画面卡顿或负载过高

```bash
tegrastats
ps -ef | grep -E '[e]v-camera|[e]v-gimbal|[u]vicorn|[m]vviewer'
```

比赛时使用无页面后台模式。调试窗口也卡时先降低 `--width`，不要先降低 50 FPS 检测与 USB 安全时序。确认没有浏览器、MVS Viewer 或网页编码服务同时运行。

## 8. 最终验收记录

至少记录以下项目：

- Git commit、Jetson 系统版本、MVS 版本。
- 标定图数量、`usable_poses`、`rejected`、`rms_px`。
- `camera_matrix`、`distortion` 与标定 YAML 备份位置。
- 相机实际 acquisition/detection FPS。
- USB 连接/断开/重连日志，重连首包安全验证。
- 静态中心、上下左右四方向的 yaw/pitch 符号。
- 预测第 1–3 帧有效，第 4 帧或 60 ms 后失效。
- `Ctrl+C` 后 5 个安全帧和云台固件 >100 ms 超时保护。
- 405 nm 激光在全部测试中的物理断开或机械遮挡措施。
## 9. 激光非共轴方位补偿

这一节用于补偿 405 nm 激光发射口与相机光心不重合造成的近距离视差。USB 帧仍保持原来的 26 字节格式；只修正发送给云台的 `yaw` 和 `pitch`，`distance_m`、`fire`、目标编号和保留字段都不改变。

### 9.1 坐标和符号

相机坐标采用 OpenCV 约定：

- `+X`：图像向右。
- `+Y`：图像向下。
- `+Z`：沿相机光轴向前。

`laser_offset_x_mm`、`laser_offset_y_mm`、`laser_offset_z_mm` 表示“激光发射口相对相机光心”的位置，单位为 mm。例如激光口在相机右侧 35 mm、上方 18 mm、前方 12 mm，应填写：

```yaml
laser_offset_x_mm: 35.0
laser_offset_y_mm: -18.0
laser_offset_z_mm: 12.0
```

程序先在相机坐标中计算 XYZ 平移补偿，再应用 `yaw_sign` 和 `pitch_sign`。`laser_yaw_bias_deg` 与 `laser_pitch_bias_deg` 是最终发给云台坐标系的固定角度微调量。

### 9.2 建立 Jetson 本地云台配置

共享的 `config/gimbal_usb.yaml` 保留安全默认值。比赛 Jetson 使用本地文件，避免 USB by-id、机械偏移和微调量被其他电脑覆盖：

```bash
mkdir -p "$HOME/.config/ev-vision"
cp config/gimbal_usb.yaml "$HOME/.config/ev-vision/gimbal_usb.yaml"
nano "$HOME/.config/ev-vision/gimbal_usb.yaml"
```

按 Jetson 实际设备和当前方向填写。当前 USB by-id 以 `ls -l /dev/serial/by-id/` 的输出为准，不要复制已经失效的旧序列号：

```yaml
port: /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_305835803034-if00
baudrate: 115200
output_hz: 50.0
calibration_path: config/camera_calibration.yaml
max_calibration_rms_px: 0.5
max_result_age_ms: 120.0
reconnect_interval_s: 1.0
predicted_control_max_frames: 3
predicted_control_max_age_ms: 60.0
predicted_max_angle_step_deg: 1.5
yaw_sign: -1
pitch_sign: 1

laser_pose_compensation_enabled: true
laser_offset_x_mm: 0.0
laser_offset_y_mm: 0.0
laser_offset_z_mm: 0.0
laser_yaw_bias_deg: 0.0
laser_pitch_bias_deg: 0.0
pose_min_distance_mm: 100.0
pose_max_distance_mm: 10000.0
pose_max_reprojection_error_px: 5.0
```

启用补偿后，只有真实的 `FULL_BOARD` 完整靶面观测会使用四角点做 PnP 距离/视差补偿。圆环、局部靶面和纯预测结果只使用靶心角度加固定 bias；程序不会复用旧距离。PnP 解失败、深度超限或重投影误差过大时，会自动退回靶心角度，不会因为一次姿态解算失败而把有效目标清零。

### 9.3 测量 XYZ 偏移

1. 给相机和激光支架建立不易移动的机械基准面。
2. 尽量量到相机成像光轴中心，而不是相机外壳边缘；无法直接量到时，可用镜头中心作为近似起点。
3. 分别测量激光发射口相对镜头中心的左右、上下、前后距离。
4. 按 9.1 的正负号填写，精确到 1 mm 已足够开始测试。
5. XYZ 是机械尺寸，不要用 bias 去代替；bias 用于安装角度、云台零位及剩余系统误差。

如果暂时无法可靠测量，先保持 XYZ 为 0，只标定 bias；之后再打开 `laser_pose_compensation_enabled` 并逐步加入实测 XYZ。

### 9.4 固定角度 bias 标定

**激光为上电常亮，开始前必须先物理断开激光电源或用可靠的不透光挡板封住发射口。USB 中的 `fire=0` 不能关闭激光。**

1. 限制云台速度、电流和最大转角，准备物理急停。
2. 先将 XYZ 与两个 bias 全设为 0，确认视觉框、靶心和 yaw/pitch 方向正确。
3. 在常用中等距离固定靶子，让云台稳定对准视觉靶心。
4. 在确保无人和无反射物的封闭条件下短时解除遮挡，观察光点相对靶心的偏差。
5. 只调一个轴，每次修改 0.05–0.20°：
   - 光点需要云台向当前“正 yaw”方向修正，就增大 `laser_yaw_bias_deg`；反之减小。
   - 光点需要云台向当前“正 pitch”方向修正，就增大 `laser_pitch_bias_deg`；反之减小。
6. 每次修改后重启进程；bias 不受距离变化影响，应优先在中等距离消除固定角误差。

不要同时修改 `yaw_sign`/`pitch_sign` 和 bias。符号只负责坐标方向，bias 只负责固定零位误差。

### 9.5 近、中、远三距离验收

在近距离、常用距离、远距离各测试一次，并记录：目标距离、视觉 observation source、发送 yaw/pitch、激光落点横纵偏差。

- 三个距离偏差方向和大小几乎相同：继续调固定 bias。
- 中远距离较准、近距离偏差明显：检查 XYZ 测量值和正负号，确认当前观测为 `FULL_BOARD`。
- 完整靶面时补偿有效，局部靶面时退回固定角：这是预期行为，因为局部/预测观测没有可信的实时深度。
- 偏差突然变大或方向翻转：立即遮挡/断开激光，检查 `yaw_sign`、`pitch_sign` 和 XYZ 坐标定义。

建议先把 `pose_max_reprojection_error_px` 保持为 `5.0`。只有在完整靶面四角稳定但经常回退时才逐步放宽，并检查相机标定、靶面尺寸和角点质量，不能为了“总是启用 PnP”无限放宽。

### 9.6 使用最新相机参数启动 Jetson 本地窗口和 USB 输出

比赛相机参数应保存在：

```text
$HOME/.config/ev-vision/competition.yaml
```

当前调好的关键值为：

```yaml
camera:
  exposure_us: 10000
  gain_db: 5.0
  acquisition_fps: 60
  auto_exposure: false
  auto_gain: false
  auto_white_balance: false

detection:
  normalization:
    clahe_clip_limit: 8.5
  white_board:
    min_white_occupancy: 0.5
  rings:
    ratio_tolerance: 0.18
    min_arc_coverage: 0.18
  classical_scoring:
    tracking_threshold: 0.52
    acquisition_threshold: 0.66
```

从 Jetson 桌面终端启动：

```bash
cd ~/2025-E-Vision/2025-E-Vision
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 2025-e-vision

export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ev-gimbal-vision \
  --config "$HOME/.config/ev-vision/competition.yaml" \
  --gimbal-config "$HOME/.config/ev-vision/gimbal_usb.yaml" \
  --serial 00G02809155 \
  --detection-fps 1000 \
  --display \
  --width 512 \
  --display-fps 15
```

若终端提示 `DISPLAY is not set`，说明不是在 Jetson 桌面图形会话中执行；无显示器测试时删除 `--display --width 512 --display-fps 15`。启动摘要必须显示：补偿已启用、XYZ、bias、靶面尺寸、允许的 pose Z 范围和最大重投影误差。终端运行状态会显示最终发送的 `yaw=...deg pitch=...deg`。

### 9.7 激光安全底线

- 405 nm 激光在本项目中为硬件上电常亮，软件和 USB `fire=0` 都不是安全联锁。
- 初次方向、XYZ 和 bias 测试必须物理断电或可靠遮挡。
- 光束保持低于眼睛高度；现场不得有人处在光路或可能反射的方向。
- 移除镜子、玻璃、亮面金属和其他镜面/强反射物。
- 云台使用低速、低电流、小角度限制，并保留可立即切断激光和云台电源的物理急停。
- 检测无效、串口断开或进程退出时，云台控制包会归零，但激光仍然亮；必须由硬件安全措施覆盖这种状态。