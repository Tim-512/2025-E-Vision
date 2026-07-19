# Jetson–云台 USB 视觉通信与安全门控设计

**日期：** 2026-07-19
**分支：** `feature/classical-white-board-tracking`

## 1. 目标

在 Jetson Orin NX Super 上增加一条独立于现有项目协议的云台 USB CDC-ACM 输出链路。运行程序从海康 MV-CA013-21UC 获取图像，运行现有传统视觉检测与跟踪，利用棋盘格标定结果将靶心像素坐标转换成相对相机光轴的角度误差，并以 50 Hz 向云台板发送目标帧。

本阶段同时增加 Jetson 本地棋盘格采集工具。系统不使用 YOLO，不使用红色特征判断目标，也不控制激光。

## 2. 已确认的硬件和运行参数

- 视觉主板：Jetson Orin NX Super。
- 相机：Hikrobot MV-CA013-21UC，序列号 `00G02809155`。
- 镜头：8 mm F/2.8，1/1.8 英寸。
- 相机固定格式：1280×1024、BayerRG8、buffer size 2。
- 曝光时间：15000 μs。
- 增益：14 dB。
- 相机采集频率：50 FPS。
- 视觉检测频率：50 FPS。
- Jetson 本地显示频率：45 FPS。
- 云台输出频率：50 Hz。
- 稳定串口路径：`/dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00`。
- 串口库：pyserial；名义配置 115200、8N1、无软硬件流控。CDC-ACM 的波特率不影响 USB 实际传输速率。
- 405 nm 激光由硬件上电常亮，视觉程序始终发送 `fire=0`，不提供激光控制功能。

## 3. 范围和非目标

### 3.1 本阶段实现

1. 低内存、Jetson 本地 OpenCV 棋盘格采集工具。
2. 8×5 内角点、22 mm 方格的相机内参标定和校验。
3. 靶心像素坐标的单点畸变矫正和相对光轴角度换算。
4. 云台专用 A5 5A、CRC16/MODBUS 协议编码器。
5. pyserial CDC-ACM 发送、断线重连和 50 Hz 周期发送。
6. 真实观测和短时纯预测观测的安全门控。
7. 可选 Jetson 本地低延迟检测画面；不启动 FastAPI、Uvicorn 或网页。
8. 协议、角度转换、门控、串口发送和命令行的自动测试。

### 3.2 本阶段不实现

- 不替换或修改现有 `src/ev_vision/protocol.py` 的 AA 55 / CRC16-CCITT-FALSE 协议语义。
- 不接入 YOLO。
- 不通过颜色，尤其不通过红色，判断靶面或靶心。
- 不估算距离；`distance_m` 始终为 `0.0`。
- 不控制激光；`fire` 始终为 `0`。
- 不实现“重新获得真实观测后与预测位置比较并重新确认”的跳变门控。
- 不接收云台姿态反馈；当前协议链路是 Jetson 单向发送目标数据。
- 不对整幅图像执行畸变 remap。

## 4. 棋盘格标定

### 4.1 标定板定义

现有标定板为 6×9 个黑白方格，因此 OpenCV 检测参数为：

- 内角点列数：8。
- 内角点行数：5。
- 方格边长：22.0 mm。
- 标定板固定在平整硬板上。

### 4.2 本地采集命令

增加命令：

```bash
ev-camera-calibration-capture \
  --config config/jetson-local.yaml \
  --serial 00G02809155 \
  --columns 8 \
  --rows 5 \
  --square-mm 22 \
  --output artifacts/calibration/images
```

采集工具直接使用相机原始 1280×1024 图像，预览画面可以缩放，但保存文件不得缩放。它不启动网页服务和目标检测器。

窗口行为：

- 检测到完整 8×5 内角点时绘制角点。
- 显示保存数量、角点检测状态、清晰度和棋盘格覆盖范围提示。
- `Space`：仅在完整角点有效且未贴近图像边缘时保存当前原始图像。
- `R`：删除本次采集的最后一张图片。
- `Q` 或 `Esc`：退出。
- 对与最近保存图片位置、面积和姿态高度相似的画面给出重复姿态提示，避免采集大量等价图片；用户仍需主动按空格保存。

建议采集 20～25 张，至少保证 15 张有效姿态。姿态应覆盖中央、四角、远近、左右倾斜、上下倾斜及少量平面旋转，棋盘格面积约占图像面积的 15%～70%。

### 4.3 标定计算

使用并完善现有命令：

```bash
python tools/calibrate_camera.py \
  artifacts/calibration/images \
  --output config/camera_calibration.yaml \
  --columns 8 \
  --rows 5 \
  --square-mm 22 \
  --max-rms 0.5
```

程序报告输入图片数、有效姿态数和 RMS。验收条件：

- 有效姿态不少于 10，推荐不少于 15。
- 所有有效图片尺寸均为 1280×1024。
- 重投影 RMS 不超过 0.5 px。
- 输出文件通过现有 `Calibration.validate()` 校验。

输出文件为 `config/camera_calibration.yaml`，包含 `image_size`、`camera_matrix`、`distortion` 和 `rms_px`。

标定完成后不得调整镜头对焦环、镜头安装或相机采集分辨率；若调整，必须重新标定。

## 5. 靶心像素到角度误差

运行时只对检测得到的靶心调用 `cv2.undistortPoints()`，不对整幅 1280×1024 图像执行 remap。

输入：

```text
center_px = (u, v)
camera_matrix = K
distortion = D
```

`cv2.undistortPoints()` 返回归一化相机坐标 `(x_n, y_n)`，随后计算：

```python
yaw_raw_deg = degrees(atan(x_n))
pitch_raw_deg = degrees(atan(y_n))
yaw_deg = yaw_sign * yaw_raw_deg
pitch_deg = pitch_sign * pitch_raw_deg
```

原始方向定义：

- 靶心在相机光轴右侧：`yaw_raw_deg > 0`。
- 靶心在相机光轴下方：`pitch_raw_deg > 0`。

`yaw_sign` 和 `pitch_sign` 只能取 `1` 或 `-1`，用于现场匹配云台电机正方向。首轮电机联调必须低速进行；如果云台朝远离目标方向运动，只反转对应 sign，不修改检测或标定算法。

以下情况禁止输出有效角度：

- 标定文件不存在或格式错误。
- 标定尺寸不是 1280×1024。
- RMS 超过 0.5 px。
- 靶心坐标、归一化坐标或角度不是有限数。
- 靶心坐标位于图像范围外。

## 6. 云台 USB 协议

为避免破坏项目现有 AA 55 协议，新增独立的 `ev_vision.gimbal_usb` 包。

### 6.1 帧格式

所有多字节整数和 float 均为小端序。每帧固定 26 字节：

| 偏移 | 长度 | 字段 |
|---:|---:|---|
| 0 | 2 | 帧头 `A5 5A` |
| 2 | 1 | 协议版本 `1` |
| 3 | 1 | 消息类型 `0x01` |
| 4 | 1 | payload 长度 `16` |
| 5 | 1 | flags，固定 `0` |
| 6 | 2 | sequence，逐帧递增，uint16 回绕 |
| 8 | 16 | 目标 payload |
| 24 | 2 | CRC16/MODBUS，低字节在前 |

目标 payload：

```c
float yaw_deg;
float pitch_deg;
float distance_m;
uint8_t tracking;
uint8_t fire;
uint8_t target_id;
uint8_t reserved;
```

CRC 参数：初值 `0xFFFF`，逐位多项式 `0xA001`，覆盖帧头至 payload 最后一个字节，不包含 CRC 自身，CRC 低字节先发送。

### 6.2 字段语义

可靠目标：

```text
yaw_deg    = 目标相对相机光轴的水平角度误差
pitch_deg  = 目标相对相机光轴的垂直角度误差
distance_m = 0.0
tracking   = 1
fire       = 0
target_id  = 0
reserved   = 0
```

无效目标：

```text
yaw_deg    = 0.0
pitch_deg  = 0.0
distance_m = 0.0
tracking   = 0
fire       = 0
target_id  = 0
reserved   = 0
```

## 7. 控制安全门控

### 7.1 真实观测

以下当前图像直接产生的来源可以进入控制门控：

- `FULL_BOARD`
- `CONCENTRIC_ARCS`
- `SINGLE_ARC`
- `WHITE_REGION`
- `FUSED_PARTIAL`

它们还必须满足：

- 上游 `target_valid=true`。
- 跟踪状态为 `TRACKING`。
- 结果年龄不超过 60 ms。
- 中心点存在、有限且位于 1280×1024 图像内。
- 没有相机错误、过期帧、过大位置/尺度/速度或预测过期等失败原因。

满足后立即输出 `tracking=1`；真实观测恢复后不执行额外的“与预测位置比较”和二次确认。

### 7.2 纯预测观测

`observation_source=PREDICTED` 只允许短时参与控制，必须同时满足：

- 跟踪状态为 `PREDICTING`。
- 上游 `target_valid=true`。
- `predicted_frames` 在 1～3 范围内。
- 距离最后一次真实视觉观测的 `source_age_us <= 60000`。
- 当前结果自身年龄不超过 60 ms。
- 预测中心有限且仍位于图像内。
- 相对于上一条已发送的有效控制角度，每个轴的变化绝对值不超过 1.5°。

1.5° 限制只用于纯预测结果。超限时不做裁剪，而是本周期直接输出安全无效帧。真实观测重新出现时可以立即恢复有效输出。

检测内部可以在控制有效期结束后继续利用预测位置限定重新搜索 ROI，但不得继续向云台发送 `tracking=1`。

### 7.3 失效条件

以下任一条件成立时发送安全无效帧：

- 没有目标结果。
- 相机断开或检测线程异常。
- 结果年龄超过 60 ms。
- 第 4 个及后续纯预测结果。
- 距离最后真实观测超过 60 ms。
- 纯预测角度单轴跳变超过 1.5°。
- 中心点或角度非法、越界。
- 标定不可用。

安全无效帧总是将角度、距离、tracking、fire、target_id 和 reserved 清零。

## 8. 运行时结构

### 8.1 模块边界

模块划分固定如下：

```text
src/ev_vision/calibration_capture.py
src/ev_vision/gimbal_usb/__init__.py
src/ev_vision/gimbal_usb/protocol.py
src/ev_vision/gimbal_usb/angles.py
src/ev_vision/gimbal_usb/gate.py
src/ev_vision/gimbal_usb/transport.py
src/ev_vision/gimbal_usb/runtime.py
src/ev_vision/gimbal_usb/cli.py
```

职责：

- `protocol.py`：CRC16/MODBUS、payload 和26字节帧编码；不依赖串口和视觉。
- `angles.py`：加载并验证标定，通过单点去畸变输出角度。
- `gate.py`：将最新检测结果转换为有效或安全无效目标命令。
- `transport.py`：串口打开、写入、错误关闭和重连；不理解视觉结果。
- `runtime.py`：以50 Hz读取最新检测快照、执行门控并发送；不排队旧检测结果。
- `cli.py`：装配相机、检测器、可选本地窗口和USB发送生命周期。
- `calibration_capture.py`：独立的原始标定图采集窗口。

现有 `src/ev_vision/protocol.py` 保持兼容，不复用它的 magic、header 或 CRC。

### 8.2 周期和并发

- 相机线程：最新帧模式，目标50 FPS。
- 检测线程：最新帧模式，目标50 FPS。
- USB发送线程：单调时钟调度，固定50 Hz。
- OpenCV窗口：主线程运行，最多45 FPS，仅显示最新序列匹配的图像和检测结果。
- 不为显示或串口建立无界队列；处理不过来时丢弃旧视觉结果而不是积压。

USB每个周期都发送一帧。没有新的检测结果时，可以短暂重复最近结果，但门控必须重新按当前单调时钟计算结果年龄；达到60 ms后自动发送无效帧。

### 8.3 串口错误和重连

- 启动时打不开串口：检测和本地预览可以继续运行，但记录错误并每1秒重试；不能假装已发送。
- 写入超时、短写或设备断开：立即关闭当前句柄并进入重连。
- 重连后继续进程内 sequence；sequence 按 uint16 回绕。进程重启后从0开始。
- 成功打开串口后，第一帧必须是安全无效帧，之后才允许发送有效目标。
- 正常退出时尽力连续发送5帧安全无效帧后关闭串口。

云台板还应独立实现接收超时保护：超过100 ms未收到有效新帧，或最新帧 `tracking=0` 时，不继续使用旧角度误差驱动云台。该保护用于覆盖Jetson进程崩溃、USB拔出或系统掉电等视觉端无法发送退出帧的情况。

## 9. 配置和命令

使用专门的云台配置段或独立配置对象，避免混淆项目现有 `serial:` 旧协议配置。建议配置内容：

```yaml
gimbal_usb:
  port: /dev/serial/by-id/usb-RoboMaster_Gimbal_Vision_USB_3065356E3034-if00
  baudrate: 115200
  output_hz: 50.0
  reconnect_interval_s: 1.0
  calibration_path: config/camera_calibration.yaml
  max_calibration_rms_px: 0.5
  max_result_age_ms: 60.0
  predicted_control_max_frames: 3
  predicted_control_max_age_ms: 60.0
  predicted_max_angle_step_deg: 1.5
  yaw_sign: 1
  pitch_sign: 1
```

正式运行命令：

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

`--display` 可省略，用于不需要窗口的正式运行。该命令不启动网页、FastAPI或Uvicorn。

## 10. 诊断输出

程序启动时打印：

- 实际相机参数。
- 标定文件、图像尺寸、RMS、fx、fy、cx、cy。
- USB稳定设备路径和连接状态。
- 输出频率和所有门控阈值。
- 激光硬件常亮安全警告。

运行中以限频方式打印或统计：

- 已发送总帧、有效帧、无效帧。
- 当前 sequence。
- 真实观测、纯预测和失效原因计数。
- 串口连接、重连和写错误计数。
- 当前 yaw、pitch、观测来源、结果年龄和真实观测年龄。

不得以每帧打印日志的方式影响50 Hz实时性。

## 11. 测试策略

### 11.1 单元测试

- CRC16/MODBUS已知向量。
- 帧头、版本、类型、长度、flags、sequence、payload和CRC字节序。
- 有效目标和无效目标的26字节编码。
- distance、fire、target_id、reserved固定为0。
- 标定加载、分辨率/RMS拒绝和单点角度换算。
- `yaw_sign`、`pitch_sign`。
- 所有真实观测来源的门控。
- 纯预测第1～3帧且不超过60 ms有效。
- 第4帧、超过60 ms、结果过期、越界和1.5°超限均无效。
- 真实观测恢复时立即有效，不执行重捕跳变检查。
- sequence回绕、短写、断线和重连。
- 标定采集只保存原始分辨率且必须存在完整8×5角点。

### 11.2 集成和实机验收

1. 保持云台电机禁用，重复100帧全零协议测试；CRC和长度错误为0。
2. 使用棋盘格采集20～25张图片并生成RMS≤0.5 px的标定文件。
3. 电机仍禁用，将靶心放到画面右、左、下、上方，在云台调试器中验证角度符号和数值连续。
4. 遮挡靶面，验证纯预测最多3帧/60 ms，之后 `tracking=0` 且角度清零。
5. 拔出USB，验证视觉程序不崩溃并进入重连；重新插入后第一帧无效，随后恢复。
6. 云台低速、激光物理断开或可靠遮挡，验证两个轴均为负反馈；方向错误只修改sign。
7. 最后再以正常速度测试静止、移动、短时遮挡和靶面部分出画场景。

## 12. 安全约束

- 调试阶段405 nm激光必须物理断电或可靠遮挡；软件的 `fire=0` 不会关闭硬件常亮激光。
- 协议和方向验证阶段先禁用云台电机或将机构置于安全状态。
- 视觉端失效时只发送全零无效帧，不保留最后角度。
- 云台板必须以 `tracking` 和接收超时共同作为电机目标更新门控。
- 标定无效时允许本地预览，但不允许有效云台控制。
