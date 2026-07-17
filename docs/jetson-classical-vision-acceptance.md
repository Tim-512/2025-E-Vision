# Jetson 传统视觉部署与实机验收

本手册用于 Jetson Orin NX Super、Hikrobot MV-CA013-21UC、8 mm F/2.8 镜头的单次完整实机验收。所有结果必须填写实测值；不得把设计目标当成实测结果。

## 1. Close camera owners and enter the project

只保留一个相机采集进程。关闭 MVS Viewer 后进入项目并设置环境：

```bash
pkill -f MvViewer || true
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

确认 USB3 相机存在，且没有第二个 MVS/MvViewer/调参服务持有相机。

## 2. Compile and run the full suite

```bash
PYTHONPYCACHEPREFIX="$(mktemp -d)" \
python -m compileall -q src tests tools

TEST_TMP="$(mktemp -d)"
python -m pytest \
  -q \
  -p no:cacheprovider \
  --basetemp "$TEST_TMP"
```

记录实际测试数量和耗时，不要只写“通过”。如测试失败，先修复再进行相机验收。

## 3. Start the existing single tuning service

```bash
python tools/camera_tuning_server.py --config config/default.yaml --host 0.0.0.0 --port 8000
```

打开网页并检查 `/api/status`：

- 相机状态为 `Connected`；
- `detection.backend=classical`；
- 没有相机错误；
- `frame_count` 持续增加；
- acquisition FPS 与 detection FPS 为有限实数。

检测器参数只能通过 `PUT /api/detection/config` 原子更新，must not restart or close MVS acquisition。修改参数前后都记录帧计数、相机状态和最后错误，确认采集线程没有关闭或重启。

## 4. Full-board acquisition matrix

把完整靶面依次放在画面中心、上、下、左、右五个位置。每个位置至少获得三个连续确认，并填写：

| 位置 | 连续确认数 | source | state | target_valid | confidence | source age | center error |
|---|---:|---|---|---|---:|---:|---:|
| 中心 |  |  |  |  |  |  |  |
| 上 |  |  |  |  |  |  |  |
| 下 |  |  |  |  |  |  |  |
| 左 |  |  |  |  |  |  |  |
| 右 |  |  |  |  |  |  |  |

进入控制必须由 `FULL_BOARD` 建立；黑胶带只是辅助加分证据，不能作为初次锁定的唯一条件。

## 5. Partial tracking matrix

先用完整靶面确认，再分别让上、下、左、右边缘部分出画，并遮挡一部分白纸。记录被接受的证据、中心误差、连续帧数和退出原因。

证据优先级必须为：

```text
CONCENTRIC_ARCS > FUSED_PARTIAL > SINGLE_ARC > WHITE_REGION > PREDICTED
```

颜色不能决定目标是否成立；不得要求圆环为红色。重点验证靶心周围同心圆弧能够在靶板不完整时继续定位中心。

| 场景 | source | state | target_valid | center error | 备注 |
|---|---|---|---|---:|---|
| 上边出画 |  |  |  |  |  |
| 下边出画 |  |  |  |  |  |
| 左边出画 |  |  |  |  |  |
| 右边出画 |  |  |  |  |  |
| 白纸局部遮挡 |  |  |  |  |  |

## 6. Bounded prediction and loss

在已确认跟踪后移除所有可用证据。预测必须在首先达到的限制处停止：`3 frames` 或 `150 ms`。下一结果必须满足：

```text
state=LOST
target_valid=false
yaw_rate=0
pitch_rate=0
```

`LOST` 时出现局部圆弧或白色区域不能重新获取目标。恢复完整靶面后，必须重新经过多帧 `FULL_BOARD` 确认，不能沿用旧锁定。

## 7. Negative scenes

依次展示以下负样本：白墙、白色书本、显示器、空白 A4、黑底矩形、单圆、不同心圆、错误半径比例圆、饱和激光光斑、历史 ROI 外的圆。每项记录拒绝原因，并确认没有持续有效锁定。

| 负样本 | 拒绝原因 | 最长误锁帧数 | 是否持续锁定 |
|---|---|---:|---|
| 白墙 |  |  |  |
| 白色书本 |  |  |  |
| 显示器 |  |  |  |
| 空白 A4 |  |  |  |
| 黑底矩形 |  |  |  |
| 单圆 |  |  |  |
| 不同心圆 |  |  |  |
| 错误半径比 |  |  |  |
| 饱和激光光斑 |  |  |  |
| 历史 ROI 外圆 |  |  |  |

## 8. Snapshot evidence

分别保存完整跟踪、局部跟踪和被拒绝负样本三组快照。每组必须来自同一个 `source_sequence`，包含原图、overlay、metadata，以及 five classical debug images：

- `normalized-gray.png`
- `white-mask.png`
- `edge-mask.png`
- `ring-arcs.png`
- `candidate-scores.png`

检查 metadata、overlay 和五张调试图的源序号完全一致，禁止混用其他帧的图片。

## 9. Measure, do not invent, performance

在 Jetson 上用实机采集的回放运行：

```bash
python tools/evaluate_classical_detector.py \
  --input artifacts/classical-replay \
  --output artifacts/jetson-classical-metrics.json \
  --config config/default.yaml \
  --fps 20 \
  --save-debug artifacts/jetson-classical-debug
```

把 JSON 中实际的 `measured_detection_fps`、平均/第 95 百分位处理时间、有效率、source/state counts 和 observation age 填入下表。分别比较 10, 15, 20, 30 and 50 ms 曝光。Do not claim the 15 FPS design target unless measured on Jetson.

| 曝光 | measured_detection_fps | mean ms | p95 ms | valid ratio | observation age | 主要 source/state | Jetson 功耗模式 |
|---:|---:|---:|---:|---:|---:|---|---|
| 10 ms |  |  |  |  |  |  |  |
| 15 ms |  |  |  |  |  |  |  |
| 20 ms |  |  |  |  |  |  |  |
| 30 ms |  |  |  |  |  |  |  |
| 50 ms |  |  |  |  |  |  |  |

不得预填或编造 FPS。还应记录 Jetson 型号、系统版本、分辨率、配置提交号和回放文件哈希，以便复现。

## 10. Laser hardware warning

**The laser is physically always on whenever powered; software cannot make it safe.**

**中文强制警告：**405 nm 激光的电源轨一旦上电，激光就持续发射，Jetson 无法关闭激光。必须使用物理电源隔离/急停、适用于 405 nm 的护目镜、可靠光阑或挡光板，并限制无关人员进入光路区域。协议 V2 没有激光控制字段，任何网页、GPIO 或软件状态都不能代替这些硬件和人员安全措施。
