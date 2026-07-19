# 2025-E-Vision

2025 年全国大学生电子设计竞赛 E 题“简易自行瞄准装置”的视觉系统工程仓库。

## 当前方案

- 主控：NVIDIA Jetson Orin NX（Super 模式）
- 相机：海康机器人 MV-CA013-21UC（USB3、彩色全局快门）
- 镜头：8 mm、F/2.8、1/1.8 英寸
- 云台通信：USB-UART，921600 baud，8N1，100 Hz 速度指令
- 视觉路线：完整白色 A4 靶面获取 + 灰度同心圆弧/局部白区跟踪 + 最多 3 帧或 150 ms 的有界预测
- 激光：405 nm，电源轨一上电就物理常亮；Jetson 没有激光开关权限，竞赛协议 V2 不含激光控制字段，软件不能让已上电激光变得安全

## 文档

- [传统视觉靶面跟踪设计](docs/superpowers/specs/2026-07-17-classical-white-board-tracking-design.md)
- [传统视觉回放评估](docs/classical-detector-evaluation.md)
- [Jetson 传统视觉部署与实机验收手册](docs/jetson-classical-vision-acceptance.md)
- [视觉与云台通信接口](docs/gimbal-vision-interface.md)
- [相机调参页面实机验收清单](docs/camera-tuning-acceptance.md)

## 范围

本仓库聚焦视觉部分，包括完整白色靶面检测、灰度圆环与局部靶面跟踪、短时预测、闭环瞄准，以及视觉主板与云台/底盘控制器之间的数据接口。云台电机与底盘运动控制不在本仓库实现范围内。

现有协议 V1 的激光字段仅用于兼容旧资料；竞赛使用的 V2 路径不发送激光控制字段，也不能依赖 GPIO、`LaserMode.OFF` 或任何软件指令实现激光安全。

## Jetson 相机验收

先关闭 MVS 图形客户端，避免其独占相机。激光必须通过独立电源隔离、护目镜、可靠光阑/挡光板和现场人员管理保证安全；不要把 Jetson 或本软件当作激光急停。安装 MVS 后设置：

```bash
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

枚举相机：

```bash
python tools/list_cameras.py
```

对序列号 `00G02809155` 先运行 60 秒冒烟测试：

```bash
python tools/camera_smoke_test.py \
  --config config/default.yaml \
  --serial 00G02809155 \
  --duration 60 \
  --timeout-ms 100 \
  --sample artifacts/camera-smoke.jpg
```

短测正常后运行 10 分钟验收，并在另外两个终端观察 `tegrastats` 和 `dmesg --follow`。重点检查输出中的 `disconnect`、`timeout_rate`、`sequence_gap_rate`、`non_monotonic_timestamps` 与 `rss_growth_bytes`。

## 浏览器相机调参与视觉诊断

云台尚未完成时，可以手持相机对已制作的靶板进行调参和传统视觉测试。该页面访问相机、图像诊断、检测、截图和参数配置档，**不提供云台或激光控制**。只要激光电源轨上电，405 nm 激光就会持续发光；调试前优先物理断开激光供电。

### Jetson 安装与本机启动

先关闭 MVS Viewer，避免其独占相机，然后执行：

```bash
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -e '.[vision,tuning]'
python tools/camera_tuning_server.py --config config/default.yaml --host 0.0.0.0 --port 8000
```

在 Jetson 本机打开 `http://127.0.0.1:8000`，或在可信隔离局域网内使用 Jetson 的实际 IP 访问。该开发服务没有登录认证，不得暴露到公网、设置路由器端口转发或在不可信 Wi-Fi 上运行。

完整逐项流程见 [Jetson 传统视觉部署与实机验收手册](docs/jetson-classical-vision-acceptance.md)。

## Jetson 本地低延迟预览

只运行本地 OpenCV 检测窗口、以及网页与本地窗口联合运行的命令见 `docs/runbooks/jetson-local-preview.md`。纯本地模式从 YAML 的 `camera:` 段读取网页实测后保存的曝光、增益和采集帧率，不启动 FastAPI/Uvicorn。405 nm 激光在硬件上电后常亮，软件不能关闭，调试时必须物理断电或可靠遮光。

## Jetson USB 云台与标定运行手册

- [棋盘格标定、USB 云台联调与安全验收](docs/runbooks/jetson-gimbal-vision.md)
