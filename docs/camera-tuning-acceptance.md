# Jetson 相机调参页面实机验收清单

适用硬件：

- NVIDIA Jetson Orin NX Super
- Hikrobot MV-CA013-21UC，USB3
- 相机序列号 `00G02809155`
- 8 mm、F/2.8、1/1.8 英寸镜头
- 固定采集格式：1280×1024、BayerRG8、缓冲区数量 2

> 安全前提：云台未完成时只做手持相机测试。405 nm 激光必须物理断开并保持关闭。本页面没有云台、底盘或激光控制接口。相机断开或程序异常时，不得产生任何运动或出光。

## 1. 准备

1. 相机用 USB3 线连接 Jetson，尽量直接连接 USB3 端口。
2. 盖好或断开激光供电，确认激光不会出光。
3. 关闭 MVS Viewer 及其他可能独占相机的程序：

   ```bash
   ps aux | grep -Ei '[m]vs|[m]vviewer'
   ```

   `MvLogServer` 日志服务可以保留；必须关闭正在取流的 Viewer。

4. 进入项目和环境：

   ```bash
   cd ~/2025-E-Vision/2025-E-Vision
   conda activate 2025-e-vision
   export PYTHONPATH="/opt/MVS/Samples/aarch64/Python/MvImport${PYTHONPATH:+:$PYTHONPATH}"
   export LD_LIBRARY_PATH="/opt/MVS/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   python -m pip install -e '.[vision,tuning]'
   ```

5. 枚举相机并确认序列号：

   ```bash
   python tools/list_cameras.py
   ```

   预期能看到 `MV-CA013-21UC` 和 `00G02809155`。

## 2. 本机启动

```bash
ev-camera-tuning \
  --config config/default.yaml \
  --serial 00G02809155
```

在 Jetson 浏览器打开：

```text
http://127.0.0.1:8000
```

预期：

- 页面状态由 Starting 进入 Streaming/Connected 类正常状态；
- 相机型号和序列号正确；
- 页面显示固定格式 `1280 × 1024 · BayerRG8 · Buffer 2`；
- 实时画面连续更新；
- 页面不存在运动或激光操作按钮。

## 3. 参数草稿与显式应用

1. 记录初始曝光、增益、采集帧率和三个自动模式。
2. 修改曝光但不要点击“应用参数”。
3. 确认页面提示“有未应用修改”，实时相机的生效值仍显示旧值。
4. 点击“撤销草稿”，确认输入恢复为当前生效值，相机没有重开。
5. 再次修改曝光，点击“应用参数”。
6. 等待页面显示成功；确认生效值更新，画面亮度变化，采集恢复。
7. 勾选自动曝光/自动增益，确认对应手动输入被禁用；仍需点击“应用参数”才会生效。
8. 点击“载入项目默认值”，确认它只修改草稿；再点击“撤销草稿”可返回当前生效值。

### 非法参数 422 检查

浏览器页面会限制常见范围。另开 Jetson 终端发送一个越界完整请求：

```bash
curl -i -X PUT http://127.0.0.1:8000/api/parameters \
  -H 'Content-Type: application/json' \
  -d '{"exposure_us":0,"gain_db":6,"acquisition_fps":120,"auto_exposure":false,"auto_gain":false,"auto_white_balance":false}'
```

预期 HTTP 状态码为 `422`，相机当前生效参数不变。

## 4. 光照与手持靶板调试

建议每种距离和光照都先保存一组基线：

1. 室内常规光照；
2. 靶板较暗；
3. 靶板有局部高光；
4. 近、中、远三个手持距离；
5. 小角度倾斜和轻微运动。

观察：

- 灰度与 RGB 直方图是否挤在最左或最右；
- 暗部/亮部裁切百分比；
- 中心 ROI 清晰度分数；
- 靶框、角点、中心、十字线和检测文字是否稳定；
- 检测关闭后靶面结果是否停止更新；
- 点击“暂停画面”后，只冻结浏览器显示，采集/诊断计数仍继续。

调焦建议：先关闭自动曝光和自动增益，用中等亮度、较短曝光减少手抖；手动旋转镜头焦环，以中心 ROI 清晰度和靶板边缘锐利程度为参考，再锁紧焦环。

## 5. 配置档

1. 输入安全名称，例如 `indoor-normal`；可填写中文显示名称。
2. 点击“保存当前草稿”。
3. 修改表单后，从列表选择刚才的配置档并点击“载入到草稿”。
4. 确认相机生效值没有改变；只有点击“应用参数”后才改变。
5. 删除配置档时确认浏览器会要求二次确认。
6. 确认项目文件 `config/default.yaml` 没有被页面修改。

## 6. 快照

点击“保存当前完整快照”，然后检查：

```bash
find artifacts/camera-tuning/captures -maxdepth 2 -type f | sort | tail -n 12
```

每个新目录应包含且不覆盖以前的：

```text
original.png
overlay.png
metadata.yaml
```

打开原图与叠加图，确认均为对应时刻画面；检查 YAML 包含相机身份、序列、固定格式、生效参数、运行计数、诊断、检测结果与叠加选项。

## 7. Windows 可信局域网访问

1. Jetson 查询 IP：

   ```bash
   hostname -I
   ```

2. 停止本机服务后重新启动：

   ```bash
   ev-camera-tuning \
     --config config/default.yaml \
     --serial 00G02809155 \
     --host 0.0.0.0
   ```

3. Windows 浏览器访问 `http://<Jetson-IP>:8000`。
4. 验证预览、参数草稿、显式应用、直方图、配置档和快照。
5. 不得做公网映射、端口转发或在不可信网络使用该监听方式。

## 8. 十分钟稳定性

开三个 Jetson 终端：

终端 A：

```bash
ev-camera-tuning --config config/default.yaml --serial 00G02809155
```

终端 B：

```bash
tegrastats
```

终端 C：

```bash
sudo dmesg --follow
```

连续运行至少 10 分钟，期间反复切换叠加、检测、暂停，应用几组有效参数并保存快照。检查：

- 页面仍可操作，没有持续增长的延迟；
- 采集 FPS、预览 FPS、检测 FPS、诊断 FPS 大致稳定；
- timeout/gap 计数没有异常快速增长；
- `tegrastats` 中内存没有持续无界增长；
- `dmesg` 没有 USB 反复重连、带宽或 I/O 错误；
- 慢速或关闭 Windows 浏览器不会阻塞 Jetson 采集。

## 9. 拔插与故障恢复

1. 确认激光仍物理断开且没有任何运动控制连接。
2. 服务运行时拔下相机 USB。
3. 预期页面进入 Disconnected/故障状态并显示错误；页面不能产生运动或出光。
4. 停止服务：`Ctrl+C`。
5. 重新插好相机，确认 `lsusb` 和 `python tools/list_cameras.py` 再次看到序列号。
6. 重新启动服务，确认页面恢复实时画面和正常计数。

当前版本采用明确的“停止服务、重新插接、重新启动”恢复流程；不要在相机仍被其他程序占用时反复启动。

## 10. 验收记录

| 项目 | 观测值 | 通过/不通过 | 备注 |
|---|---|---|---|
| MVS 枚举型号/序列号 |  |  |  |
| 固定格式 1280×1024/BayerRG8/2 |  |  |  |
| 本机实时预览 |  |  |  |
| 草稿不自动应用 |  |  |  |
| 有效参数应用并确认 |  |  |  |
| 非法参数返回 422 |  |  |  |
| 撤销/默认值只修改草稿 |  |  |  |
| 自动模式禁用手动输入 |  |  |  |
| 检测与全部叠加开关 |  |  |  |
| 浏览器暂停不停止采集 |  |  |  |
| 灰度/RGB 直方图与中心 ROI |  |  |  |
| 配置档载入不自动应用 |  |  |  |
| 原图/叠加图/YAML 快照 |  |  |  |
| Windows 可信 LAN 访问 |  |  |  |
| 10 分钟稳定性 |  |  |  |
| 拔相机后故障状态 |  |  |  |
| 无运动、无出光 |  |  |  |
| 重新插接并重启恢复 |  |  |  |
