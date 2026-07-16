# 2025-E-Vision

2025 年全国大学生电子设计竞赛 E 题“简易自行瞄准装置”的视觉系统工程仓库。

## 当前方案

- 主控：NVIDIA Jetson Orin NX（Super 模式）
- 相机：海康机器人 MV-CA013-21UC（USB3、彩色全局快门）
- 镜头：8 mm、F/2.8、1/1.8 英寸
- 云台通信：USB-UART，921600 baud，8N1，100 Hz 速度指令
- 视觉路线：YOLO 全局搜索/重捕获 + 黑色 A4 边框传统视觉精定位
- 激光：405 nm，由 Jetson 控制开关

## 文档

- [视觉瞄准系统设计说明](docs/superpowers/specs/2026-07-12-vision-aiming-system-design.md)
- [Jetson 混合靶面检测部署与实机验收手册](docs/hybrid-detector-acceptance.md)

## 范围

本仓库聚焦视觉部分，包括目标检测、几何定位、激光点检测、闭环跟踪、圆轨迹同步，以及视觉主板与云台/底盘控制器之间的数据接口。云台电机与底盘运动控制不在本仓库实现范围内。

## Jetson 相机验收

关闭 MVS 图形客户端，并确保 405 nm 激光断开或保持关闭。安装 MVS 后设置：

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

短测正常后运行 10 分钟验收，并在另外两个终端观察 `tegrastats` 和
`dmesg --follow`。重点检查输出中的 `disconnect`、`timeout_rate`、
`sequence_gap_rate`、`non_monotonic_timestamps` 与 `rss_growth_bytes`。

## 浏览器相机调参与视觉诊断

云台尚未完成时，可以手持相机对已制作的靶板进行调参和传统视觉测试。该页面只访问相机、图像诊断、检测、截图和参数配置档，**不提供云台或激光控制**。开发与验收期间必须让 405 nm 激光保持物理断开并关闭。

### Jetson 安装与本机启动

先关闭 MVS Viewer，避免其独占相机，然后执行：

```bash
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -e '.[vision,tuning]'
ev-camera-tuning --config config/default.yaml --serial 00G02809155
```

默认只监听 Jetson 本机的 `127.0.0.1:8000`。在 Jetson 浏览器中打开：

```text
http://127.0.0.1:8000
```

如命令入口暂时未刷新，也可使用：

```bash
python -m ev_vision.web.camera_tuning_server \
  --config config/default.yaml \
  --serial 00G02809155
```

### 从 Windows 通过可信局域网访问

在 Jetson 查询 IP：

```bash
hostname -I
```

仅当 Jetson 和 Windows 位于可信、隔离的局域网时，显式监听所有网卡：

```bash
ev-camera-tuning \
  --config config/default.yaml \
  --serial 00G02809155 \
  --host 0.0.0.0
```

假设 `hostname -I` 显示 Jetson 地址为 `192.168.1.20`，则在 Windows 浏览器打开：

```text
http://192.168.1.20:8000
```

此开发服务没有登录认证。不得暴露到公网，不得设置路由器端口转发，也不要在不可信 Wi-Fi 上使用 `--host 0.0.0.0`。

完整逐项流程见 [相机调参页面实机验收清单](docs/camera-tuning-acceptance.md)。
