# 2025 电赛 E 题视觉瞄准系统设计

- 日期：2026-07-12
- 题目：简易自行瞄准装置（2025 年全国大学生电子设计竞赛 E 题）
- 文档状态：已确认设计
- 负责范围：Jetson 视觉、激光控制、视觉到云台通信接口；不包含底盘巡迹和云台底层驱动

## 1. 项目目标

视觉系统安装在自动寻迹小车上，完成以下功能：

1. 在静止测试中搜索 A4 靶纸并瞄准靶心。
2. 在小车沿 100 cm × 100 cm 方形轨迹运动时，保持 405 nm 激光连续照射靶心。
3. 在小车完成一圈运动期间，使激光在靶面同步绘制一圈半径 6 cm 的圆。
4. 视觉系统输出云台两轴目标角速度，不负责电机、编码器和云台底层闭环。
5. Jetson 只读接收 MSPM0 提供的小车圈进度，不向底盘发送巡迹或电机控制命令。

题目中的主要精度目标是靶心误差和画圆误差不超过 2 cm；软件设计应留出足够裕量，避免把 2 cm 当作正常控制误差目标。

## 2. 已确认硬件

### 2.1 视觉计算平台

- NVIDIA Jetson Orin NX，运行支持的 Super 功耗模式。
- 建议使用 NVMe 存储保存模型、配置、录像和日志。
- 视觉程序需要控制功耗、散热和线程调度，避免高负载降频造成延迟漂移。

### 2.2 相机与镜头

- 相机：海康机器人 MV-CA013-21UC。
- 接口：USB 3.0。
- 相机特性：彩色、全局快门、高帧率工业相机。
- 镜头：8 mm，F/2.8，支持 1/1.8 英寸靶面，C 接口。
- 相机横向安装，默认分辨率 1280 × 1024。
- 相机与激光器刚性安装在二维云台的俯仰框架上，随云台同步运动。

8 mm 镜头在约 50 cm 最近距离下视场偏紧，因此软件必须区分部分靶纸和完整靶纸。只有完整四角稳定可见时，才能进入高精度瞄准和画圆状态。

### 2.3 激光器

- 波长：405 nm。
- Jetson 可以控制激光开关。
- 当前按 TTL 开关接口设计。
- 软件协议保留 0～1000 的功率字段，将来若硬件支持 PWM 可直接扩展。
- Jetson GPIO 只能提供控制信号，不得直接为激光器供电。
- 硬件必须保证默认下拉、上电默认关闭、独立电源开关和异常失效关闭。

### 2.4 云台接口

暂定：

- USB 转串口/UART。
- 921600 baud，8N1，无流控。
- Jetson 以 100 Hz 发送视觉控制帧。
- 云台以 100～200 Hz 返回状态。
- Jetson 发送两轴目标角速度，不发送 PWM、电流或步数。

## 3. 系统边界

### 3.1 Jetson 负责

- 海康 MVS SDK 图像采集。
- 相机畸变校正。
- YOLO 靶纸粗搜索和重新捕获。
- 黑色矩形边框的传统视觉精定位。
- 四条边拟合、四角计算和单应矩阵估计。
- 靶面物理坐标系建立。
- 靶心和半径 6 cm 圆轨迹生成。
- 405 nm 激光实际落点检测。
- 目标运动和系统延迟预测。
- 视觉外环 PD 控制。
- 向云台发送目标角速度。
- 接收云台状态和 MSPM0 圈进度。
- 激光 GPIO/串口控制。
- 日志、录像、调试显示和故障保护。

### 3.2 云台控制器负责

- 两轴电机或舵机驱动。
- 编码器读取。
- 位置环、速度环和必要的电流环。
- 回零、机械限位和软限位。
- 根据 Jetson 角速度指令完成平滑运动。
- 返回角度、角速度、限位、故障和指令确认序号。
- 控制帧超时时自动减速和停止。

### 3.3 MSPM0 底盘控制器负责

- 巡迹。
- 底盘电机控制。
- 圈数和路径状态。
- 向 Jetson 单向提供当前圈进度或关键路径事件。

Jetson 不向 MSPM0 发送电机速度、转向量或巡迹修正指令。

## 4. 总体数据流

```text
海康相机
  ↓ MVS SDK，最新帧覆盖
相机采集层
  ├─ YOLO 靶纸粗搜索
  ├─ 黑框几何精定位
  └─ 405 nm 激光点检测
          ↓
视觉状态机
          ↓
图像坐标 ↔ A4 靶面物理坐标单应性
          ↓
靶心目标 / 半径 6 cm 圆轨迹目标
          ↓
延迟预测 + 前馈 + PD + 死区 + 限速
          ↓
921600 UART，100 Hz 角速度指令
          ↓
二维云台底层控制器
```

旁路输入：

- 云台反馈：角度、角速度、限位、故障、确认序号。
- MSPM0：小车圈进度。
- 激光：Jetson GPIO 或串口控制。

## 5. 软件架构

```text
vision_system/
├── camera/
│   ├── hik_camera
│   └── latest_frame_buffer
├── detection/
│   ├── yolo_board_detector
│   ├── board_geometry_detector
│   ├── red_circle_detector
│   └── laser_spot_detector
├── calibration/
│   ├── camera_calibration
│   ├── laser_calibration
│   └── axis_mapping
├── tracking/
│   ├── board_tracker
│   ├── laser_tracker
│   └── motion_predictor
├── planning/
│   ├── center_target
│   └── circle_trajectory
├── control/
│   ├── visual_servo
│   ├── rate_limiter
│   └── state_machine
├── communication/
│   ├── protocol
│   ├── serial_gimbal
│   ├── mock_gimbal
│   └── chassis_progress
├── laser/
│   ├── laser_interface
│   ├── gpio_laser
│   └── mock_laser
├── logging/
├── config/
├── tests/
└── tools/
```

主要线程：

1. 相机采集线程：只保留最新帧，禁止旧帧堆积。
2. 视觉处理线程：状态机、检测、跟踪、规划和控制。
3. 串口通信线程：100 Hz 发控制帧，接收云台反馈和底盘进度。
4. 日志与调试线程：异步写日志和视频，不阻塞控制链路。

第一版将 YOLO 和传统视觉放在同一视觉处理线程；稳定后再评估 TensorRT 异步化。

## 6. 相机采集与标定

### 6.1 初始相机参数

以下仅为起始值，必须现场实测：

```yaml
camera:
  width: 1280
  height: 1024
  pixel_format: BayerRG8
  acquisition_fps: 120
  exposure_us: 800
  gain_db: 6.0
  auto_exposure: false
  auto_gain: false
  auto_white_balance: false
  buffer_size: 2
```

原则：

- 优先使用短曝光降低运动模糊。
- 自动曝光、增益和白平衡只能在启动调试阶段使用，比赛运行时必须锁定。
- 宁可丢帧，不得处理过期帧。
- 每帧保存相机采集时间戳，而不是仅保存处理完成时间。

### 6.2 相机内参标定

用最终安装的 8 mm 镜头完成棋盘格标定，得到内参矩阵和畸变参数：

```text
K = [fx  0 cx]
    [ 0 fy cy]
    [ 0  0  1]

D = (k1, k2, p1, p2, k3)
```

要求：

- 固定对焦和光圈后再标定。
- 至少采集 20～30 个有效姿态。
- 棋盘格覆盖中心、四角和边缘。
- 包含正视和倾斜姿态。
- 重投影 RMS 建议不超过 0.5 px，目标约 0.3 px。
- 运行前预计算 `initUndistortRectifyMap` 映射表。

## 7. 靶纸检测路线

采用“YOLO 粗定位 + 传统视觉精定位”。

### 7.1 YOLO 粗定位

YOLO 只标注一个类别：

```yaml
0: target_board
```

它只输出靶纸整体检测框，不负责输出四角、靶心或控制量。

训练集：

- 300～500 张。
- 距离覆盖约 0.5～1.8 m。
- 包含正视、斜视、完整靶纸、部分出画、运动模糊和不同照明。
- 只标注整个靶纸框。
- 先训练轻量模型，部署时导出 TensorRT FP16。

推理策略：

- SEARCH：连续或隔帧运行。
- ACQUIRE：每 2～3 帧运行。
- TRACK/CIRCLE：每 10～20 帧做监督检查。
- 连续丢失传统视觉结果后立即恢复连续推理。

建议输入 640 × 512，保持原始宽高比。

### 7.2 黑框传统视觉

在 YOLO ROI 外扩 10%～20% 后：

1. 灰度转换。
2. Otsu 反向阈值与局部自适应阈值组合。
3. 自适应形态学开闭运算。
4. Canny/LSD/霍夫获取边缘或线段候选。
5. 获取凸四边形候选。
6. 根据面积、边框亮度、内部亮度、时间连续性和透视合理性评分。
7. 将边缘点分配给四条边。
8. RANSAC 剔除异常点并最小二乘拟合直线。
9. 相邻直线求交得到浮点四角。

最终精定位不能直接使用 `approxPolyDP` 的整数角点。

### 7.3 四角顺序

四角统一为：

```text
左上、右上、右下、左下
```

排序方法：

- 求四边形中心。
- 按相对中心极角排序。
- 选择左上角作为起点。
- 用叉积检查顺/逆时针方向。

不采用仅基于 `x+y` 的排序方法。

### 7.4 完整和部分靶纸

- PARTIAL：仅检测到部分边、角点或 YOLO 框触碰画面边缘；仅允许低速居中。
- FULL：四条边、四个角和单应性均稳定有效；允许高精度瞄准和画圆。

8 mm 镜头在最近距离下可能裁切上下边框。系统必须先把靶纸居中，不能在不完整几何上直接画圆。

## 8. 靶面坐标和单应性

建立以靶心为原点的物理坐标系：

```text
X 向右为正
Y 向上为正
单位 cm
A4 范围：X ∈ [-10.5, 10.5]
         Y ∈ [-14.85, 14.85]
```

标准角点：

```python
BOARD_CORNERS_CM = [
    [-10.5, +14.85],
    [+10.5, +14.85],
    [+10.5, -14.85],
    [-10.5, -14.85],
]
```

调试用标准矫正图采用 40 px/cm：

- 宽 840 px。
- 高 1188 px。
- 靶心 `(420, 594)`。
- 半径 6 cm 对应 240 px。

使用四角计算图像到标准靶面的单应矩阵，并保存逆矩阵用于将靶面目标点映射回原图。

靶心主估计来自黑框几何中心。红色同心圆只作为辅助校验和小幅修正，因为直径不超过 1 mm 的红色中心点在远距离可能不足一个像素。

## 9. 时序跟踪

采用“预测 + 当前帧重新拟合”：

1. 用 LK 光流或历史速度预测当前四角位置。
2. 在预测区域附近重新搜索黑框边缘。
3. 重新拟合四条直线并求交。
4. 使用轻量自适应低通或 One Euro Filter 抑制静态抖动。

光流只用于预测，不能长期替代几何重新检测，以免累计漂移。

## 10. 激光检测和标定

### 10.1 静态开关差分

静态瞄准允许短暂开关激光：

1. 保存关激光帧。
2. 开启激光并保存开激光帧。
3. 根据黑框或局部特征进行帧间配准。
4. 计算亮度和颜色差分。
5. 在靶面内部筛选新增高亮连通域。
6. 使用亮度加权质心计算光斑中心。

不能预设 405 nm 在相机中一定表现为标准紫色；必须实测 Bayer/BGR/HSV/Lab 各通道响应。

### 10.2 连续点亮检测

发挥部分要求激光连续点亮。此时使用：

- 上一帧位置和运动模型建立局部 ROI。
- 峰值亮度、局部梯度、颜色、面积、圆度和预测距离综合评分。
- 缓慢更新的靶面背景模型分离当前光斑与历史感光痕迹。
- 光斑区域不参与背景更新。
- 云台反馈角速度和控制指令参与下一帧位置预测。

### 10.3 激光参考模型

相机和激光器平行但不严格共轴，因此激光参考点不等于图像中心。建立随距离或靶纸尺度变化的参考模型：

```text
laser_reference_px = f(board_scale, board_pose, yaw, pitch)
```

第一版可按 0.5 m、1.0 m、1.5 m 等距离标定后插值。实际光斑检测失败时短暂退化到参考模型；检测恢复后平滑切回实际落点闭环。

## 11. 目标生成

### 11.1 靶心

靶面目标：

```text
(0, 0) cm
```

### 11.2 半径 6 cm 圆

```text
Xd = 6 cos(theta)
Yd = 6 sin(theta)
```

通过逆单应矩阵映射到原始图像，控制误差为：

```text
期望圆轨迹点 - 实际激光点
```

不能在原始图像中直接画像素圆，否则靶纸斜视时物理轨迹会变成椭圆或畸形曲线。

## 12. 视觉伺服控制

### 12.1 坐标约定

- 图像：`u` 向右，`v` 向下。
- 靶面：`X` 向右，`Y` 向上。
- 云台：yaw 向右为正，pitch 向上为正。
- 方向修正集中在 YAML：`yaw_sign`、`pitch_sign`、`swap_axes`。

### 12.2 像素误差转角度误差

```text
yaw_error   = atan2(u_target - u_laser, fx)
pitch_error = -atan2(v_target - v_laser, fy)
```

控制器内部使用角度误差，避免分辨率或裁剪变化后增益完全失效。

### 12.3 控制律

第一版采用：

```text
轨迹/目标运动前馈 + PD + 微分低通 + 迟滞死区 + 速度限制 + 加速度限制
```

不在第一版使用积分项，避免机械回差、误检和延迟导致积分饱和。

大误差采用较高比例增益和速度上限；进入小误差区后降低增益、增强阻尼并启用迟滞死区。所有增益和限制都放入配置文件，待真实云台完成后整定。

### 12.4 轴间耦合

机械完成后通过分别驱动 yaw 和 pitch，测量目标图像位移，标定 2 × 2 轴映射矩阵。第一版允许以对角矩阵运行，后续加入交叉项补偿相机安装倾斜和轴不正交。

## 13. 延迟和预测

总延迟包含曝光、USB 传输、取帧、视觉处理、串口、云台控制器和机械响应。

视觉端按来源帧时间戳计算数据年龄，并预测执行时刻目标：

```text
predicted_position = measured_position + velocity × estimated_latency
```

控制帧携带 `source_age_us`。云台端可据此拒绝执行过旧的高速指令。

必须记录：

- 相机帧年龄。
- YOLO 和传统视觉耗时。
- 控制计算耗时。
- 串口周期和往返时间。
- 云台报告的控制延迟。

## 14. 状态机

```text
BOOT
  ↓
SAFE
  ↓
SEARCH
  ↓
ACQUIRE
  ↓
CENTER
  ↓
LASER_CALIBRATE
  ↓
AIM
  ├─ TRACK
  └─ CIRCLE

短时丢失 → RECOVER
严重故障 → FAULT
```

### BOOT

初始化配置、相机、标定、模型、串口、日志和激光；激光强制关闭。初始化失败进入 FAULT。

### SAFE

云台目标角速度为零，激光关闭，等待启动信号。

### SEARCH

激光关闭，YOLO 全图搜索，Jetson 生成有限范围蛇形扫描角速度。

### ACQUIRE

在 YOLO ROI 内获取黑框和四角。连续 3～5 帧几何有效后进入 CENTER。

### CENTER

将完整靶纸移入安全视场，避免 8 mm 镜头裁切边框。激光保持关闭。

### LASER_CALIBRATE

用开关差分检测实际光斑并更新激光参考模型。

### AIM

静态快速瞄准靶心；大误差快速收敛，小误差精细控制。

### TRACK

小车运动时激光连续点亮，持续跟踪靶心并进行延迟补偿。

### CIRCLE

根据 MSPM0 圈进度生成圆周相位，以实际激光落点闭环跟踪半径 6 cm 轨迹。

### RECOVER

短时间使用历史单应性和运动预测，降低云台速度。发挥测试期间不能因短时视觉丢失主动关闭连续激光。

### FAULT

云台停止、激光关闭、记录故障，需要明确复位才能退出。

## 15. 画圆与底盘同步

Jetson 只读接收 MSPM0 的当前圈进度：

```cpp
struct ChassisProgress {
    uint8_t  running;
    uint8_t  lap_index;
    uint16_t progress_permille;
    uint32_t elapsed_ms;
};
```

圆相位：

```text
theta = 2π × progress_permille / 1000 + theta0
```

推荐起点为靶心右侧 6 cm，即 `theta0 = 0`。小车到达 100% 圈进度时，圆相位正好完成一圈。

MSPM0 进度可根据路径四段和编码器插值。若不能提供连续进度，最低限度提供起跑、B、C、D 和完成一圈事件，Jetson 在事件间插值。

Jetson 本地相位连续运行，MSPM0 进度缓慢校正相位，避免进度噪声导致目标点跳变。

画圆采用目标速度前馈。可通过相邻相位目标点的数值差分估算图像速度，再转换为云台角速度前馈，PD 只修正残余误差。

## 16. 激光接口

```python
class LaserInterface:
    def off(self) -> None: ...
    def on(self, power: float = 1.0) -> None: ...
    def pulse(self, duration_ms: int) -> None: ...
```

实现：

- `GpioLaser`
- `SerialLaser`
- `MockLaser`

模式：

```text
OFF
PULSE
CONTINUOUS
EMERGENCY
```

TTL 硬件中，功率 0 表示关闭，1～1000 均视为开启。若将来支持 PWM，再映射为占空比。

异常策略：

- 启动、相机掉线、通信故障、线程异常和正常退出时关闭。
- GPIO 初始化失败时禁止进入瞄准状态。
- 硬件必须具备默认下拉或独立看门狗，不能只依赖软件退出回调。

## 17. Jetson 到云台协议

### 17.1 外层帧

```text
Offset  Size  Field
0       1     0xAA
1       1     0x55
2       1     protocol_version
3       1     message_type
4       2     payload_length, little-endian
6       2     sequence, little-endian
8       N     payload
8+N     2     CRC-16/CCITT-FALSE
```

CRC 参数：

- Poly：0x1021
- Init：0xFFFF
- RefIn/RefOut：false
- XorOut：0x0000

消息类型：

```text
0x01 VISION_CONTROL
0x02 GIMBAL_FEEDBACK
0x03 HEARTBEAT
0x04 PARAMETER
0x05 CHASSIS_PROGRESS
0x06 SYSTEM_EVENT
```

### 17.2 视觉控制负载

```cpp
#pragma pack(push, 1)
struct VisionControlPayload {
    uint8_t  mode;
    uint8_t  target_valid;
    uint8_t  laser_mode;
    uint8_t  flags;
    int16_t  yaw_rate_cdeg_s;
    int16_t  pitch_rate_cdeg_s;
    int16_t  error_yaw_mdeg;
    int16_t  error_pitch_mdeg;
    uint16_t board_confidence_permille;
    uint16_t laser_confidence_permille;
    uint32_t source_age_us;
};
#pragma pack(pop)
```

标志位：

```text
bit0 BOARD_FULL_VALID
bit1 HOMOGRAPHY_VALID
bit2 LASER_SPOT_VALID
bit3 USING_PREDICTION
bit4 USING_LASER_MODEL
bit5 CIRCLE_SYNC_VALID
bit6 CAMERA_HEALTHY
bit7 EMERGENCY_STOP
```

### 17.3 云台反馈负载

```cpp
#pragma pack(push, 1)
struct GimbalFeedbackPayload {
    uint8_t  state;
    uint8_t  fault_flags;
    uint16_t ack_sequence;
    int32_t  yaw_angle_mdeg;
    int32_t  pitch_angle_mdeg;
    int16_t  yaw_rate_cdeg_s;
    int16_t  pitch_rate_cdeg_s;
    int16_t  yaw_current_ma;
    int16_t  pitch_current_ma;
    uint16_t supply_mv;
    uint16_t control_latency_us;
    uint32_t controller_time_us;
};
#pragma pack(pop)
```

云台端最低必须返回：状态、故障、确认序号、两轴角度和两轴角速度。

故障位至少包含：yaw/pitch 限位、电机故障、编码器故障、命令超时、过流和急停。

### 17.4 超时规则

云台端：

- 50 ms 无控制帧：开始平滑减速。
- 100 ms 无控制帧：目标角速度归零。
- 500 ms 无控制帧：报告通信故障。

Jetson 端：

- 100 ms 无反馈：禁止继续提高速度。
- 300 ms 无反馈：判定云台通信故障。
- 严重故障：关闭激光并进入 SAFE/FAULT。

## 18. 关键数据结构

```python
@dataclass
class CameraFrame:
    image_bgr: np.ndarray
    frame_id: int
    capture_timestamp_ns: int
    exposure_us: float
    gain_db: float

@dataclass
class TargetGeometry:
    corners_px: np.ndarray
    center_px: tuple[float, float]
    homography_img_to_board: np.ndarray
    homography_board_to_img: np.ndarray
    confidence: float
    timestamp_ns: int

@dataclass
class LaserSpot:
    center_px: tuple[float, float]
    confidence: float
    area_px: float
    peak_intensity: float
    timestamp_ns: int

@dataclass
class VisionObservation:
    timestamp_ns: int
    board_detected: bool
    geometry_valid: bool
    board_confidence: float
    corners_px: np.ndarray | None
    homography: np.ndarray | None
    laser_detected: bool
    laser_confidence: float
    laser_px: tuple[float, float] | None
    target_px: tuple[float, float] | None
    board_reprojection_error_px: float
    frames_since_board_seen: int
    frames_since_laser_seen: int
```

## 19. 配置文件

```yaml
camera:
  serial_number: ""
  width: 1280
  height: 1024
  fps: 120
  exposure_us: 800
  gain_db: 6.0
  pixel_format: BayerRG8

calibration:
  camera_file: config/camera_calibration.yaml
  laser_file: config/laser_calibration.yaml

board_detector:
  yolo_engine: models/target_board_fp16.engine
  confidence: 0.45
  input_width: 640
  input_height: 512
  roi_margin: 0.15

board:
  width_cm: 21.0
  height_cm: 29.7
  rectified_px_per_cm: 40.0

circle:
  radius_cm: 6.0
  start_phase_deg: 0.0
  phase_direction: 1

control:
  command_hz: 100
  yaw_sign: 1
  pitch_sign: 1
  swap_axes: false
  estimated_latency_ms: 25.0
  kp_yaw: null
  kd_yaw: null
  kp_pitch: null
  kd_pitch: null
  max_yaw_rate_deg_s: null
  max_pitch_rate_deg_s: null
  max_yaw_accel_deg_s2: null
  max_pitch_accel_deg_s2: null

serial:
  port: /dev/ttyUSB0
  baudrate: 921600
  feedback_timeout_ms: 300

laser:
  backend: gpio
  active_high: true
  gpio_line: null
```

`null` 项必须在真实云台或 GPIO 信息确定后通过标定填写，程序启动时应对必要参数进行校验。

## 20. 日志与调试

每帧或每个控制周期记录：

- 相机帧号和采集时间。
- 状态机状态。
- YOLO、传统视觉和激光检测耗时。
- 四角、靶心、轨迹目标和光斑坐标。
- 单应性质量和重投影误差。
- 靶纸与光斑置信度。
- 两轴误差和发送角速度。
- 云台角度、角速度、故障和确认序号。
- MSPM0 圈进度。
- 数据年龄和端到端估计延迟。

调试画面叠加：YOLO 框、四角、靶心、6 cm 圆、当前目标点、实际光斑、误差箭头、状态、FPS、延迟和通信状态。

录像和日志必须异步写入，磁盘速度不足时允许丢弃调试帧，不能阻塞控制。

## 21. 故障与安全

以下任一情况进入 SAFE 或 FAULT：

- 相机断开或连续取帧超时。
- 图像时间戳异常或数据年龄过大。
- 串口断开或云台反馈超时。
- 云台报告电机、编码器、过流、限位或急停故障。
- 控制线程超时。
- GPIO 激光控制失败。
- 单应矩阵异常时禁止进入 CIRCLE。

FAULT 的确定行为：

```text
云台角速度命令 = 0
激光 = OFF
记录故障和最近上下文
等待明确复位
```

## 22. 开发与验收顺序

1. PC/Jetson 上完成录像回放框架和 MockGimbal/MockLaser。
2. 接入海康 MVS SDK，验证最新帧采集、时间戳和固定曝光。
3. 完成相机内参标定和畸变校正。
4. 完成纯传统视觉黑框检测和四角精定位。
5. 建立靶面单应性，验证物理坐标误差。
6. 完成 YOLO 数据采集、标注、训练和 TensorRT 部署。
7. 完成 TTL 激光安全控制与静态开关差分检测。
8. 完成连续激光点检测和激光参考模型。
9. 完成串口协议、Mock 云台回环和超时保护。
10. 接入真实云台，整定方向、轴映射、PD、限速和延迟。
11. 完成静态 2 s/4 s 瞄准测试。
12. 完成小车运动靶心跟踪。
13. 接入 MSPM0 圈进度，完成半径 6 cm 同步画圆。
14. 进行不同距离、照明、电量和运动速度的全流程压力测试。

## 23. 验收指标

建议内部目标严于题目要求：

- 相机连续运行无旧帧堆积。
- 控制帧稳定达到 100 Hz。
- 靶纸完整可见时，四角连续稳定输出。
- 单应性异常能够被拒绝，不产生错误画圆指令。
- 静态靶心瞄准在题目时间限制内完成。
- 运动中短时丢失可恢复，云台不发生高速误扫。
- 圆轨迹相位与 MSPM0 圈进度同步误差显著小于 1/4 圈。
- 相机、串口或程序故障时激光可靠关闭，云台可靠停止。
- 所有关键参数均来自配置或标定文件，不在算法中散落不可追踪常数。

## 24. 已冻结决策

1. Jetson Orin NX 为视觉主板。
2. 海康 MV-CA013-21UC 与 8 mm F/2.8 镜头。
3. 相机和激光随二维云台共同运动。
4. 传统视觉负责精定位，YOLO 负责搜索和重新捕获。
5. 激光由 Jetson 控制，按 TTL 开关设计并预留 PWM 功率字段。
6. Jetson 向云台发送目标角速度，不发送底层电机量。
7. UART 921600、100 Hz、CRC-16/CCITT-FALSE。
8. 激光实际落点闭环为主，距离相关参考模型为退化方案。
9. 圆在靶面物理坐标系生成，不在图像中直接画像素圆。
10. Jetson 只读接收 MSPM0 圈进度用于圆相位同步。

## 25. 尚待硬件完成后标定的参数

这些不是未完成的需求，而是必须由实物测试得到的参数：

- 相机内参和畸变。
- 405 nm 在相机各颜色通道中的响应。
- 相机与激光的距离相关参考模型。
- yaw/pitch 正方向和 2 × 2 轴映射矩阵。
- 云台最大速度、最大加速度和机械限位。
- PD 增益、死区和分段阈值。
- 端到端延迟。
- 连续光斑检测阈值和背景更新速率。
- Jetson GPIO 编号或最终串口激光控制方式。
