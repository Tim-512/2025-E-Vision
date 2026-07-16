# Jetson 混合靶面检测部署与实机验收手册

本文用于在 NVIDIA Jetson Orin NX 上部署并验收“YOLO 全局候选框 + ROI 传统几何精定位 + 时序跟踪”的混合靶面检测链路。验收对象包括海康机器人 MVS 相机、模型后端、浏览器调参页面、检测状态 API 和故障降级行为。

> **验收状态：pending。** 在本手册全部步骤于目标 Jetson、目标相机和目标镜头上执行并归档证据前，硬件验收必须保持 `pending`。Windows 上的单元测试、回放测试或浏览器测试只能证明软件回归，**不能代表 Jetson 硬件验收通过**。

## 1. 安全边界和通过原则

1. 调试和验收全程必须让 **405 nm laser physically disconnected or OFF**（405 nm 激光器物理断开，或由独立硬件确认保持 OFF）。只关闭网页控件、GPIO 软件标志或遮挡光路均不能替代物理断开/独立断电。
2. 本手册不包含激光发射验收。发现激光接通、误亮或状态不确定时，立即停止验收并断电处理。
3. TensorRT engine 与 JetPack、CUDA、TensorRT、GPU 架构和 Ultralytics 版本强相关。`models/target-board.engine` **只能在最终验收用 Jetson 本机由 `models/target-board.pt` 构建**，绝不复制或使用 Windows 生成的 `.engine`，也不得使用另一台 Jetson 上未经本机复验的 engine。
4. 必须先证明 `models/target-board.onnx` fallback 可加载并完成检测，再构建和选择 TensorRT engine。engine 构建或加载失败时，明确回退到已验证的 ONNX，不得用来源不明或未验证的 engine 顶替。
5. 12 项验收必须逐项记录 `pending / pass / fail`。任一安全项、目标有效性项或故障降级项失败，整体验收为 `fail`；未在实机执行则仍为 `pending`。

## 2. Jetson 环境、conda 与 MVS 路径

以下命令在 Jetson 终端执行。若 MVS 安装报告证明路径不同，只修正 `/opt/MVS` 对应路径，并把实际路径写入验收记录。

```bash
cd ~/2025-E-Vision/2025-E-Vision
conda activate 2025-e-vision

export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

printf 'CONDA_PREFIX=%s\n' "$CONDA_PREFIX"
printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
python --version
```

安装项目全部验收依赖，然后做语法编译和完整测试。`TEST_TMP` 使用一次性目录，避免旧缓存污染结果：

```bash
python -m pip install -e ".[dev,vision,tuning,hardware,training]"
python -m compileall -q src tests tools
TEST_TMP="$(mktemp -d)"
python -m pytest -q -p no:cacheprovider --basetemp "$TEST_TMP"
```

保存上述命令的完整输出。完整 pytest 在 Windows 通过不等于 Jetson 通过；本节命令必须在 Jetson 上重新执行。

## 3. 复制模型并先验证 ONNX fallback

从发布机只复制可移植的 `models/target-board.onnx` 和训练检查点 `models/target-board.pt`。示例中的主机名按现场修改：

```bash
# 在发布机执行；不要复制任何 Windows 生成的 .engine
scp models/target-board.onnx models/target-board.pt jetson:~/2025-E-Vision/2025-E-Vision/models/
```

在 Jetson 上确认文件和哈希。若目录已有 engine，先移出活动路径并归档，避免 ONNX 首验被误选；不要删除历史证据：

```bash
cd ~/2025-E-Vision/2025-E-Vision
mkdir -p artifacts/hybrid-acceptance/pre-engine
if test -f models/target-board.engine; then
  mv models/target-board.engine artifacts/hybrid-acceptance/pre-engine/target-board.engine.unverified
fi
sha256sum models/target-board.onnx models/target-board.pt | tee artifacts/hybrid-acceptance/model-hashes-before-engine.txt
```

保持 `config/default.yaml` 中首选路径为 `models/target-board.engine`、fallback 为 `models/target-board.onnx`。在 engine 不存在时启动服务，系统应明确选择 ONNX fallback；不得静默使用其他模型。按第 7 节启动后检查：

```bash
curl -sS http://127.0.0.1:8000/api/detection/status | tee artifacts/hybrid-acceptance/onnx-status.json | python -m json.tool
curl -sS http://127.0.0.1:8000/api/detection/config | tee artifacts/hybrid-acceptance/onnx-config.json | python -m json.tool
```

ONNX 首验通过条件：状态中的模型为可用/就绪，`model_backend` 为 `onnx`，`model_path` 指向 `models/target-board.onnx`，正常靶板能产生黄色模型框和后续绿色几何结果。未满足时停止 engine 构建，先修复 ONNX 或配置问题。

## 4. 仅在 Jetson 本机构建 TensorRT engine

停止当前服务后，在**最终验收用 Jetson 本机**从 `.pt` 构建：

```bash
python tools/export_target_detector.py \
  --model models/target-board.pt \
  --format engine \
  --imgsz 640 \
  --device 0
```

确认导出结果为活动路径，并记录哈希：

```bash
test -f models/target-board.engine
sha256sum models/target-board.pt models/target-board.onnx models/target-board.engine \
  | tee artifacts/hybrid-acceptance/model-hashes-final.txt
```

重新启动后，`/api/detection/status` 应显示 TensorRT/engine 后端和 `models/target-board.engine`。若导出、加载或冒烟检测失败：

- 保留错误日志和失败 engine 的哈希；
- 将失败 engine 移出活动路径；
- 重新启动并确认显式选择已通过首验的 ONNX fallback；
- 在最终记录中把“后端选择”写为 ONNX fallback，并说明原因；
- 绝不从 Windows 拷贝 engine 规避失败。

## 5. 记录软件栈与发布身份

在每次正式验收开始时记录 JetPack、CUDA、TensorRT、Ultralytics、Git 提交和模型 hash。建议统一写入证据目录：

```bash
EVIDENCE="artifacts/hybrid-acceptance/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EVIDENCE"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "conda_prefix=$CONDA_PREFIX"
  echo "python=$(python --version 2>&1)"
  echo "jetpack_release=$(cat /etc/nv_tegra_release 2>/dev/null || true)"
  dpkg-query -W 'nvidia-jetpack' 'nvidia-l4t-core' 2>/dev/null || true
  nvcc --version 2>/dev/null || true
  dpkg-query -W 'libnvinfer*' 2>/dev/null || true
  python -c "import ultralytics; print('ultralytics=' + ultralytics.__version__)"
} | tee "$EVIDENCE/platform.txt"

sha256sum models/target-board.pt models/target-board.onnx models/target-board.engine 2>/dev/null \
  | tee "$EVIDENCE/model-sha256.txt"
cp config/default.yaml "$EVIDENCE/default.yaml"
```

记录中至少要能还原：JetPack/L4T、CUDA、TensorRT、Ultralytics、Python、Git commit、实际模型路径、每个模型的 SHA-256 和生效配置。

## 6. 启动前检查

### 6.1 激光和 MVS 独占检查

在接通相机前完成并拍照记录：

```text
405 nm laser physically disconnected or OFF
```

关闭 MVS Viewer/MVS 图形客户端，确认没有 MVS 进程占用相机：

```bash
pgrep -af 'MVS|MvViewer|MVSViewer' || true
```

若有输出，正常关闭对应图形程序后再检查；不要在 MVS 与本服务之间共享同一相机句柄。

### 6.2 USB 与相机枚举

```bash
lsusb | tee "$EVIDENCE/lsusb.txt"
python tools/list_cameras.py | tee "$EVIDENCE/cameras.txt"
```

必须枚举到海康机器人 MV-CA013-21UC，且目标序列号为 `00G02809155`。枚举不到、序列号不符或设备反复重连时，不得进入检测验收。

### 6.3 检查 8000 端口旧进程

```bash
pgrep -af 'ev-camera-tuning|camera_tuning_server|uvicorn' || true
ss -ltnp | grep ':8000' || true
```

两条检查均不得显示旧的调参服务占用 8000 端口。若发现旧进程，先确认 PID 和启动命令，再用正常终止方式停止并复查；不要直接启动第二个实例覆盖问题。

### 6.4 最终启动门禁

只有以下条件全部满足才可启动：

- 405 nm 激光物理断开或由独立硬件确认 OFF；
- MVS 已关闭，相机只被当前服务使用；
- `tools/list_cameras.py` 枚举到正确型号和序列号；
- 8000 端口无旧 `uvicorn` / `ev-camera-tuning` / 模块进程；
- 模型 SHA-256 与本次发布记录一致；
- 已明确记录选择 Jetson 本机 engine，或选择已验证的 ONNX fallback。

## 7. 启动调参页面

推荐使用安装后的命令入口：

```bash
ev-camera-tuning \
  --config config/default.yaml \
  --serial 00G02809155 \
  --host 0.0.0.0 \
  --port 8000 \
  --output artifacts/camera-tuning \
  --preview-fps 20 \
  --detection-fps 20
```

若命令入口尚未刷新，使用等价模块命令：

```bash
python -m ev_vision.web.camera_tuning_server \
  --config config/default.yaml \
  --serial 00G02809155 \
  --host 0.0.0.0 \
  --port 8000 \
  --output artifacts/camera-tuning \
  --preview-fps 20 \
  --detection-fps 20
```

`0.0.0.0` 仅用于可信、隔离的局域网，不得暴露公网或做路由器端口转发。只在 Jetson 本机验收时可改为 `--host 127.0.0.1`。启动日志保存到证据目录，例如：

```bash
# 需要后台留档时使用；先确保第 6 节检查通过
nohup ev-camera-tuning \
  --config config/default.yaml --serial 00G02809155 \
  --host 0.0.0.0 --port 8000 \
  --output artifacts/camera-tuning --preview-fps 20 --detection-fps 20 \
  >"$EVIDENCE/server.log" 2>&1 &
echo $! | tee "$EVIDENCE/server.pid"
```

## 8. 可复制的状态与进程检查

正式验收期间至少在开始、5 分钟、10 分钟、15 分钟和结束时保存状态：

```bash
curl -sS http://127.0.0.1:8000/api/detection/status | python -m json.tool
curl -sS http://127.0.0.1:8000/api/detection/config | python -m json.tool
curl -sS http://127.0.0.1:8000/api/status | python -m json.tool
ps -o pid,rss,etime,cmd -C python
```

留档示例：

```bash
stamp="$(date +%H%M%S)"
curl -sS http://127.0.0.1:8000/api/detection/status >"$EVIDENCE/detection-status-$stamp.json"
curl -sS http://127.0.0.1:8000/api/detection/config >"$EVIDENCE/detection-config-$stamp.json"
curl -sS http://127.0.0.1:8000/api/status >"$EVIDENCE/service-status-$stamp.json"
ps -o pid,rss,etime,cmd -C python >"$EVIDENCE/process-$stamp.txt"
```

浏览器预览、状态 API 和保存证据时均不得接通 405 nm 激光。

## 9. 12 项可量化实机验收

开始前复制下表到验收记录，初始状态全部填 `pending`。每一项必须包含时间、操作者、后端、状态 JSON、截图/录像或日志路径以及结论。

| # | 验收项 | 可量化通过标准 | 必存证据 | 状态 |
|---|---|---|---|---|
| 1 | 后端选择 | `/api/detection/status` 明确显示 Jetson 本机 `models/target-board.engine`；若 engine 失败，则明确显示已首验的 `models/target-board.onnx` fallback。路径和 SHA-256 与发布记录完全一致。 | status/config JSON、模型哈希、构建或 fallback 原因 | pending |
| 2 | 海康预览与参数控制 | 连续预览 60 s，画面持续更新；修改一次曝光并恢复，`frame_count` 持续增加，运行状态无相机丢失/worker 退出，参数 API 返回实际应用值。 | 60 s 录像、应用前后状态、参数截图 | pending |
| 3 | 黄色模型框 | 清晰真靶板在 30 个采样结果中至少 27 个出现覆盖完整靶板的**黄色模型框**，不得把只出现黄色框视为最终有效目标。 | 含 30 帧计数的录像/截图、candidate/status JSON | pending |
| 4 | 绿色几何结果 | 同一组 30 个采样结果中至少 27 个经几何验收后显示**绿色四边形**和中心点，四角顺序稳定且中心位于靶板内部。 | final-overlay/roi-geometry 图、status JSON | pending |
| 5 | 三帧进入跟踪 | 从非跟踪状态开始，连续 3 帧被接受后进入 `TRACKING`，`confirmation_count >= 3` 且 `target_valid=true`；少于 3 帧不得提前有效。 | 连续逐帧状态或录像 | pending |
| 6 | 移除靶板到丢失 | 移走靶板后，在配置的 `predict_frames + lost_frames`（默认 2+3，即最多 5 个结果周期）内到达 `LOST`；到达 `LOST` 时 `target_valid=false`，旧中心/坐标不得继续作为有效目标发布。 | 移除前后连续状态 JSON、录像 | pending |
| 7 | 矩形干扰物 | 分别用空白 A4、黑框矩形、显示器/桌沿等至少 3 类干扰物，每类持续 60 s；不得出现连续 3 个接受结果维持 `TRACKING`，不得发布持续有效目标。 | 每类 60 s 录像、failure/candidate 状态 | pending |
| 8 | 预测态无效 | 短时遮挡触发 `PREDICTING` 时，采集该状态的全部结果；每一帧都必须 `target_valid=false`，不得向云台语义层发布有效坐标或激光许可。 | `PREDICTING` 连续状态 JSON | pending |
| 9 | 有效结果速率 | 采用足够短、适合预期运动的曝光，稳定 60 s；`detection_fps`/有效结果统计保持约 **15-20 Hz**，60 s 平均值在 15.0–20.0 Hz，且结果年龄不超过配置安全门限。 | 曝光设置、开始/结束状态、60 s 统计 | pending |
| 10 | 20 分钟稳定性 | 连续运行 **20 minutes**：进程存活、相机不丢失、检测 worker 不退出、状态轮询无中断。第 5 分钟热身后每分钟记录 RSS；结束 RSS 相对第 5 分钟增长不超过 100 MiB，且最后 5 个样本不得持续单调增长。 | 20 分钟日志、20 份状态/RSS、`dmesg`/`tegrastats` 片段 | pending |
| 11 | 模型故障安全降级 | 将活动模型临时移出路径或制造一次可恢复的 reload 失败后，reload API 返回错误；预览仍连续更新（10 s 内 `frame_count` 增加且 `preview_fps>0`），模型状态为不可用/错误，`target_valid=false`，旧有效目标被清除。恢复模型并重新加载后另行记录。 | 故障前后 status、reload 响应、10 s 预览录像 | pending |
| 12 | 证据归档完整 | 证据目录包含平台版本、Git commit、配置、PT/ONNX/engine 哈希（按实际后端）、完整 pytest、启动日志、12 项记录、状态 JSON、截图/录像、20 分钟 RSS/状态和异常说明；清单中缺失文件数必须为 0。 | 证据目录清单及压缩包 SHA-256 | pending |

### 模型故障试验的安全操作

只在第 1–10 项证据已保存后执行。先记录活动模型路径，把文件**移动到同一证据目录**而不是删除，然后调用页面“重新加载模型”或 API：

```bash
mkdir -p "$EVIDENCE/model-failure"
mv models/target-board.engine "$EVIDENCE/model-failure/target-board.engine.held"  # 实际后端为 engine 时
curl -sS -X POST http://127.0.0.1:8000/api/detection/model/reload | tee "$EVIDENCE/model-failure/reload-response.json"
sleep 10
curl -sS http://127.0.0.1:8000/api/status | tee "$EVIDENCE/model-failure/service-status.json" | python -m json.tool
curl -sS http://127.0.0.1:8000/api/detection/status | tee "$EVIDENCE/model-failure/detection-status.json" | python -m json.tool
```

若当前正式后端为 ONNX，则移动 `models/target-board.onnx`，不要误动 `.pt` 训练检查点。试验后恢复原路径、核对哈希并重新加载。整个故障试验期间激光仍须物理断开或 OFF。

## 10. 20 分钟稳定性采样

在独立终端保存系统监控：

```bash
tegrastats --interval 1000 >"$EVIDENCE/tegrastats.log" &
echo $! >"$EVIDENCE/tegrastats.pid"
dmesg --follow >"$EVIDENCE/dmesg-follow.log" &
echo $! >"$EVIDENCE/dmesg.pid"

for minute in $(seq 0 20); do
  stamp="$(date +%Y%m%d-%H%M%S)"
  curl -sS http://127.0.0.1:8000/api/status >"$EVIDENCE/status-$minute-$stamp.json" || echo "status_failed" >"$EVIDENCE/status-$minute-$stamp.error"
  curl -sS http://127.0.0.1:8000/api/detection/status >"$EVIDENCE/detection-$minute-$stamp.json" || echo "detection_failed" >"$EVIDENCE/detection-$minute-$stamp.error"
  ps -o pid,rss,etime,cmd -C python >"$EVIDENCE/process-$minute-$stamp.txt"
  test "$minute" -eq 20 || sleep 60
done
```

结束后检查 `server.log`、`dmesg-follow.log` 和状态文件中的 camera loss、USB disconnect、timeout、worker exit、traceback、OOM 与持续 RSS 增长。停止监控进程时使用记录的 PID，并保留全部日志。

## 11. 证据归档与最终签字

```bash
find "$EVIDENCE" -maxdepth 3 -type f -printf '%P\n' | sort | tee "$EVIDENCE/manifest.txt"
tar -C "$(dirname "$EVIDENCE")" -czf "$EVIDENCE.tar.gz" "$(basename "$EVIDENCE")"
sha256sum "$EVIDENCE.tar.gz" | tee "$EVIDENCE.tar.gz.sha256"
```

最终记录必须写明：

- 12 项各自的 `pass/fail`，不得用“基本正常”代替；
- 实际后端是 Jetson 本机 TensorRT engine 还是 ONNX fallback；
- JetPack/CUDA/TensorRT/Ultralytics/Python/Git commit；
- 模型与证据压缩包 SHA-256；
- 所有偏差、失败、重试和选择 fallback 的原因；
- 操作者、复核者和实机执行时间。

只要尚未在实机完成，结论必须保持：

```text
Jetson hybrid detector hardware acceptance: pending
```