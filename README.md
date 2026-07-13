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
