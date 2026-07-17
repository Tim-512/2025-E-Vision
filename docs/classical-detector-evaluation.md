# 传统视觉回放评估器

`tools/evaluate_classical_detector.py` 在不打开海康相机的情况下，对保存的图像、图像目录、视频或 tar 压缩包运行传统视觉检测器。它不会加载 Hikrobot/MVS，也不会导入 Ultralytics，适合在 Windows 开发机和 Jetson 上重复比较同一批输入。

## 输入与命令

支持单张图像、按路径字典序递归读取的图像目录、OpenCV 可读取的视频，以及包含图像的 `.tar`、`.tar.gz`、`.tgz`。压缩包会在临时目录安全解包，绝对路径、父目录跳转和链接成员会被拒绝。

```bash
python tools/evaluate_classical_detector.py \
  --input artifacts/classical-replay \
  --output artifacts/classical-metrics.json \
  --config config/default.yaml \
  --fps 20 \
  --save-debug artifacts/classical-debug
```

```bash
python tools/evaluate_classical_detector.py \
  --input latest-camera-capture.tar.gz \
  --output artifacts/latest-camera-capture-metrics.json \
  --config config/default.yaml \
  --fps 20 \
  --max-frames 300
```

图像和压缩包必须提供正的 `--fps`，它用于生成可重复的帧时间戳。视频在 `--fps` 为正时使用指定值，否则尝试读取容器 FPS。`--max-frames` 必须是正整数。

## 指标说明

- `frames`：实际解码并送入检测器的帧数。
- `valid_frames`：`target_valid=true` 的帧数；有效率为 `valid_frames / frames`。
- `full_board_frames`：观测来源为 `FULL_BOARD` 的帧数。
- `partial_frames`：来源为 `CONCENTRIC_ARCS`、`FUSED_PARTIAL`、`SINGLE_ARC` 或 `WHITE_REGION` 的帧数。
- `predicted_frames`：来源为 `PREDICTED` 的帧数。
- `lost_frames`：跟踪状态为 `LOST` 的帧数。
- `source_counts`：各种观测来源的实测计数。
- `state_counts`：各种跟踪状态的实测计数。
- `max_prediction_streak`：最长连续预测帧数。
- `mean_processing_ms`：只包围 `detector.detect(...)` 的平均处理时间。
- `p95_processing_ms`：同一处理时间的第 95 百分位。
- `measured_detection_fps`：由实测平均处理时间计算的检测吞吐率。

`measured_detection_fps` 会随机器、功耗模式、输入分辨率、场景复杂度和配置变化。必须记录 Jetson Orin NX Super 上实际生成的 JSON，不能复制设计目标或编造 FPS。

## 调试输出

启用 `--save-debug` 后，每帧写入独立的六位序号目录，保存同一源帧对应的 `overlay.png`，以及五种传统视觉调试图：

- `normalized-gray.png`
- `white-mask.png`
- `edge-mask.png`
- `ring-arcs.png`
- `candidate-scores.png`

不要把用户的完整采集压缩包提交到 Git；只提交指标 JSON、必要的少量授权样例或验收记录。
