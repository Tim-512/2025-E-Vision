# Jetson 本地低延迟检测窗口

本说明适用于 Jetson Orin NX Super、海康 MV-CA013-21UC 和项目的传统视觉检测器。本地窗口只显示实时相机画面、绿色靶面轮廓和绿色靶心，不包含网页诊断图、文字、候选评分或直方图。

## 1. 先把网页实测参数写入 YAML

纯本地命令不会读取浏览器内存中的参数，也不会启动网页服务。它只读取 `--config` 指定 YAML 的 `camera:` 段。

在 Jetson 项目目录编辑配置：

```bash
cd ~/2025-E-Vision/2025-E-Vision
nano config/default.yaml
```

将网页最终调好的值写入相应字段。下面仅为格式示例；请以实际网页测得值为准：

```yaml
camera:
  width: 1280
  height: 1024
  pixel_format: BayerRG8
  acquisition_fps: 20
  exposure_us: 50000
  gain_db: 14.0
  auto_exposure: false
  auto_gain: false
  auto_white_balance: false
  buffer_size: 2
```

注意：曝光 `50000 us` 时，理论上不可能真正采集到 `120 FPS`。应把 `acquisition_fps` 写成相机在该曝光下能够稳定工作的值，例如网页实测的 20 FPS，而不是照抄旧默认值。

保存后检查：

```bash
sed -n '1,18p' config/default.yaml
```

## 2. 更新安装入口

每次拉取包含新命令的提交后，在 conda 环境中重新进行 editable 安装：

```bash
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision
python -m pip install -e '.[vision,tuning]'
```

设置海康 MVS Python 和动态库路径：

```bash
export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

关闭 MVS Viewer 和旧服务，避免相机被第二个进程占用：

```bash
ps -ef | grep -E '[e]v-camera-tuning|[e]v-camera-preview|[c]amera_tuning_server|[u]vicorn'
```

如有旧进程，回到启动它的终端按 `Ctrl+C`。不要同时运行两个会打开相机的命令。

## 3. 推荐：只打开 Jetson 本地窗口，不启动网页

必须在 Jetson 本机桌面会话的终端中运行。先检查：

```bash
echo "$DISPLAY"
```

通常会显示 `:0` 或 `:1`。如果本机桌面终端里为空，可以尝试：

```bash
export DISPLAY=:0
```

然后运行：

```bash
ev-camera-preview \
  --config config/default.yaml \
  --serial 00G02809155 \
  --width 640 \
  --display-fps 30 \
  --detection-fps 30
```

该模式只启动：

- 一个海康相机采集会话；
- 传统视觉检测线程；
- Jetson 本地 OpenCV 窗口。

它不会导入或启动 FastAPI、Uvicorn、HTTP/MJPEG 和图像诊断线程。按 `Q`、`Esc` 或终端中的 `Ctrl+C` 退出。

如果尚未重新执行 editable 安装，可以临时使用：

```bash
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" \
python -m ev_vision.preview.cli \
  --config config/default.yaml \
  --width 640 \
  --display-fps 30 \
  --detection-fps 30
```

## 4. 网页和本地窗口同时运行

需要继续调参时才使用联合模式：

```bash
ev-camera-tuning \
  --config config/default.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --local-preview \
  --local-preview-width 640 \
  --local-preview-fps 30
```

网页和本地窗口共用同一个相机与检测服务，不会重复打开相机。网页应用参数后，相机会按原有事务流程重新配置，本地窗口继续显示新参数下的画面。

比赛或只看识别结果时应优先使用第 3 节的纯本地命令，以减少内存、JPEG 编码、HTTP 和浏览器开销。

## 5. 卡顿时的降载顺序

优先按以下顺序降载：

1. 将显示宽度降到 512：`--width 512`；
2. 将显示帧率降到 20：`--display-fps 20`；
3. 将检测帧率降到 20：`--detection-fps 20`；
4. 确认没有浏览器、MVS Viewer 或第二个相机进程；
5. 用 `tegrastats` 检查内存和负载。

示例：

```bash
ev-camera-preview \
  --config config/default.yaml \
  --width 512 \
  --display-fps 20 \
  --detection-fps 20
```

## 6. 常见错误

### `DISPLAY is not set`

说明当前终端没有图形桌面。优先在 Jetson 屏幕上的桌面终端运行。SSH 终端即使设置 `DISPLAY=:0`，也可能因 X 会话权限无法创建窗口；不要使用永久、无鉴权的 `xhost +`。

### 相机打开失败或断连

检查是否有其他进程占用：

```bash
ps -ef | grep -Ei '[m]vs|[m]vviewer|[e]v-camera|[u]vicorn'
lsusb | grep 2bdf:0001
```

### 窗口有画面但没有绿色框

本地窗口只画 `target_valid=true` 且检测序列和画面序列严格匹配的结果。先用网页联合模式检查传统视觉参数；确认后把参数保存回 YAML，再切换纯本地模式。

## 7. 激光安全

405 nm 激光在硬件上电后常亮，Jetson 和当前软件不能关闭它。调试相机和窗口时，必须物理断开激光供电或使用可靠、不透光的机械遮挡，并佩戴适合 405 nm 波段的防护眼镜。软件退出、检测失效或窗口关闭都不代表激光安全。
