# 白色靶面获取与灰度圆环跟踪设计

日期：2026-07-17  
状态：已确认，等待实施计划  
分支：`feature/classical-vision`

## 1. 目标

在 Jetson Orin NX Super 与海康 MV-CA013-21UC 相机上，实现不依赖训练数据的比赛靶面检测与跟踪：

1. 以完整白色 A4 靶面作为初次获取和重新获取的主要对象；
2. 黑色边框只作为辅助证据，不作为主要候选生成条件；
3. 不依赖圆环的红色色相，圆环只按灰度边缘、同心关系和固定半径比例识别；
4. 靶面部分离开视野时，融合可见圆弧、局部白色纸面和历史运动模型继续定位靶心；
5. 靶心短暂离开视野时允许有界预测，超时后必须进入 `LOST`；
6. 完整靶面重新出现后重新校正四角、靶心、尺度、透视和速度，消除漂移；
7. 在现有相机调参网页中提供实时叠加、调试视图、在线参数修改和快照导出；
8. 定义视觉与云台之间的最小可靠通信接口。

## 2. 硬件和靶面约束

- 主机：Jetson Orin NX Super；
- 相机：Hikrobot MV-CA013-21UC，USB3；
- 镜头：8 mm、F/2.8、1/1.8 英寸；
- 图像：1280×1024，BayerRG8；
- 靶面：A4 紫外感光纸，四边粘贴约 1.8 cm 宽黑胶带；
- 靶面中心：五个同心圆，理论半径为 20、40、60、80、100 mm；
- 圆环半径比：`1:2:3:4:5`；
- 405 nm 激光由硬件上电常亮，Jetson 不控制激光开关。

圆环在弱光下不一定表现为稳定的红色，因此任何 HSV 红色阈值、红通道优势或颜色分类都不是有效目标成立的必要条件。

## 3. 方案选择

采用“完整白板获取 + 灰度同心圆弧跟踪 + 短时预测”。

未采用的方案：

- 仅检测完整白色矩形：无法处理靶面部分出画；
- 以 CSRT、KCF、模板匹配或长时间光流为主：容易漂移并锁定背景；
- YOLO 主路径：当前时间不足以完成数据采集、标注、训练和比赛环境验证。

通用光流可以在后续作为极短时辅助信息，但不作为建立目标身份的依据。

## 4. 总体数据流

```text
相机帧
  → 灰度转换与局部光照归一化
  → 白色纸面、暗边和梯度基础图
  → 完整白色 A4 候选生成与评分
  → 完整靶面确认并建立历史模型
  → 预测 ROI 内的完整/局部观测
  → 现有跟踪状态机与有界预测
  → 靶心像素和角误差
  → 云台角速度控制帧
```

系统沿用现有状态：

```text
SEARCHING → CONFIRMING → TRACKING → PREDICTING → LOST
```

不建立第二套独立状态机。

## 5. 图像预处理

每帧生成共享的预处理结果：

1. Bayer 转换后的 BGR 图像转灰度；
2. 小核高斯滤波抑制增益噪声，不显著破坏细圆环；
3. CLAHE 增强局部对比度；
4. 使用大尺度模糊或形态学方法估计光照背景；
5. 生成局部亮度归一化图；
6. 生成白色纸面掩码、梯度图、Canny 边缘图和饱和高亮屏蔽图。

预处理必须满足：

- 阴影覆盖一部分纸面时仍能形成较连续的白色候选；
- 黑胶带和细圆环可通过闭运算暂时并入纸面区域；
- 极亮激光光斑及其附近饱和边缘不能被当成圆环证据；
- 核心检测中不使用颜色类别。

## 6. 完整白色靶面获取

### 6.1 候选生成

通过相对亮度、局部阈值、低纹理和形态学闭运算生成纸面候选。阈值基于当前画面的亮度百分位数和局部对比度，不使用固定绝对灰度值作为唯一条件。

对连通区域拟合轮廓、旋转矩形和四边形。候选至少检查：

- 面积占图像比例；
- 凸性；
- 四角顺序和角度；
- 边长、对角线及透视合理性；
- 透视矫正后的 A4 宽高比；
- 四边形内部白色占比；
- 内部纹理复杂度；
- 内部灰度圆环结构；
- 四边附近的暗边支持。

黑边仅增加可信度。黑边缺失、反光或局部模糊不能单独否决一个白色 A4 候选。

### 6.2 候选评分

```text
total_score =
    white_score
  + geometry_score
  + ring_structure_score
  + border_support_score
  + temporal_score
  - jump_penalty
  - texture_penalty
```

`SEARCHING` 中不把历史位置作为硬条件，以允许目标在任意位置重新出现；`TRACKING` 中位置、尺度和速度连续性必须成为严格门限。

普通白墙、白色书本或显示器不能仅凭“白色矩形”完成目标确认。内部圆环/边缘结构和连续多帧一致性是重要身份依据。

## 7. 标准靶面坐标系

完整四角通过后建立靶面坐标系：

```text
宽：210 mm
高：297 mm
原点：纸张左上角
靶心：(105 mm, 148.5 mm)
理论圆环半径：20、40、60、80、100 mm
```

完整靶面状态下，靶心优先由标准 A4 中心经逆透视投影得到，不以单次霍夫圆心作为主结果。

历史靶面模型保存：

- 四角；
- 图像靶心；
- 双向单应矩阵；
- 面积、宽高和投影尺度；
- 理论圆环的当前投影尺度；
- 靶面局部灰度统计；
- 中心速度；
- 最后可信帧序号和采集时间。

## 8. 灰度圆环与圆弧几何

圆环检测使用：

- 灰度梯度和局部亮暗差；
- Canny 边缘；
- 圆弧或椭圆弧拟合；
- 圆弧覆盖角；
- 多圆共同中心；
- `1:2:3:4:5` 半径比例；
- 历史圆心、尺度和运动连续性。

完整靶面可见时，优先在矫正后的靶面平面、理论半径附近搜索。部分靶面可见且无法可靠矫正时，在图像空间拟合圆弧/椭圆弧，并使用历史尺度约束。

观测分级：

- `CONCENTRIC_ARCS`：至少两条满足共同圆心、半径组合和历史尺度的圆弧；
- `SINGLE_ARC`：一条足够长的圆弧，同时满足历史圆心、尺度、运动和附近白色纸面约束；
- `SINGLE_ARC` 只能短时使用，不能长期维持有效跟踪。

算法不要求精确判断每条圆弧对应第几个理论圆，只需找到与理论比例集合兼容的组合。

## 9. 局部白色纸面跟踪

完整四角不可见但已有可信历史模型时，只在预测 ROI 中提取：

- 与历史纸面亮度和纹理接近的局部白色区域；
- 剩余纸张边缘；
- 剩余暗胶带边缘；
- 与理论圆环投影相符的灰度边缘。

该来源记为 `WHITE_REGION`。它不能在 `SEARCHING` 或 `LOST` 中独立建立目标，只能辅助已经确认的目标短时连续跟踪。

圆弧、局部白面和运动模型联合成立时，来源记为 `FUSED_PARTIAL`。

## 10. 跟踪状态和转换

### 10.1 SEARCHING

- 全画面搜索完整或近似完整的白色 A4 四边形；
- 单圆、单圆弧或普通白块不能建立目标；
- 输出 `target_valid=false`。

### 10.2 CONFIRMING

- 默认要求 3 帧中心、面积、角点和结构评分连续；
- 确认期间输出 `target_valid=false`；
- 通过后建立历史模型并进入 `TRACKING`。

### 10.3 TRACKING

观测优先级：

```text
FULL_BOARD
  > CONCENTRIC_ARCS
  > FUSED_PARTIAL
  > SINGLE_ARC
  > WHITE_REGION
```

完整靶面重新出现时立即校正靶心、四角、单应矩阵、圆环尺度、速度和累计漂移。

### 10.4 PREDICTING

没有可靠视觉观测时使用常速度模型：

```text
predicted_center = previous_center + velocity × elapsed_time
```

同时约束最大速度、最大加速度、单帧中心跳变、图像边界、预测帧数和预测时间。

初始限制：

```yaml
predict_max_frames: 3
predict_max_ms: 150
```

帧数和时间任一先超限即停止预测。预测期间 `confidence` 随时间下降，来源为 `PREDICTED`。

### 10.5 LOST

预测超时或证据不充分时：

```text
target_valid = false
state = LOST
yaw_rate = 0
pitch_rate = 0
```

最后可信坐标仅用于日志。云台保持当前位置。视觉扩大搜索范围，并且必须重新发现完整白色 A4、经过 `CONFIRMING` 后才能恢复 `TRACKING`。局部圆弧或白块不能直接从 `LOST` 恢复控制。

## 11. 观测来源

```text
NONE
FULL_BOARD
CONCENTRIC_ARCS
SINGLE_ARC
WHITE_REGION
FUSED_PARTIAL
PREDICTED
```

可靠性顺序：

```text
FULL_BOARD
  > CONCENTRIC_ARCS
  > FUSED_PARTIAL
  > SINGLE_ARC
  > WHITE_REGION
  > PREDICTED
```

## 12. 防误锁硬规则

1. 搜索阶段不能依靠单圆、单圆弧或单白块建立目标；
2. 完整目标必须连续多帧确认；
3. 局部跟踪只能在已有完整靶面历史模型后启用；
4. 圆心、尺度、速度和加速度不能异常跳变；
5. 单圆弧模式有独立的最大持续帧数；
6. 纯预测最长 3 帧且 150 ms；
7. `LOST` 后必须重新确认完整靶面；
8. 不符合 A4 尺度/结构的白色矩形不能接管目标；
9. 没有历史模型和半径关系的背景圆不能接管目标；
10. 激光饱和区域及紧邻边缘不能作为圆环依据；
11. 检测结果年龄超过有效阈值时强制失效；
12. 任何相机、检测线程或严重通信故障均强制 `target_valid=false`。

## 13. 视觉—云台协议 V2

### 13.1 物理层

```text
UART/USB 串口
921600 baud
8N1
无流控
小端
视觉控制帧：100 Hz
云台反馈：100～200 Hz
```

### 13.2 外层帧

保持现有结构：

```text
0..1    0xAA 0x55
2       protocol_version = 2
3       message_type
4..5    payload_length
6..7    sequence
8..     payload
末尾    CRC-16/CCITT-FALSE
```

CRC 参数：Poly `0x1021`、Init `0xFFFF`、不反射、XorOut `0x0000`。

当前阶段实现：

```text
0x01 VISION_CONTROL
0x02 GIMBAL_FEEDBACK
0x03 HEARTBEAT
```

### 13.3 视觉控制负载

```cpp
#pragma pack(push, 1)
struct VisionControlPayloadV2 {
    uint8_t  operating_mode;
    uint8_t  target_valid;
    uint8_t  tracking_state;
    uint8_t  observation_source;
    uint16_t flags;
    uint16_t confidence_permille;
    int16_t  yaw_rate_cdeg_s;
    int16_t  pitch_rate_cdeg_s;
    int16_t  error_yaw_mdeg;
    int16_t  error_pitch_mdeg;
    int16_t  target_x_px;
    int16_t  target_y_px;
    uint32_t source_age_us;
    uint32_t source_frame_sequence;
};
#pragma pack(pop)
```

`target_valid=false` 时角速度必须同时为零。

跟踪状态：

```text
0 SEARCHING
1 CONFIRMING
2 TRACKING
3 PREDICTING
4 LOST
5 FAULT
```

观测来源：

```text
0 NONE
1 FULL_BOARD
2 CONCENTRIC_ARCS
3 SINGLE_ARC
4 WHITE_REGION
5 FUSED_PARTIAL
6 PREDICTED
```

标志位：

```text
bit0  CAMERA_HEALTHY
bit1  FULL_BOARD_VISIBLE
bit2  HOMOGRAPHY_VALID
bit3  MULTIPLE_ARCS_VALID
bit4  SINGLE_ARC_VALID
bit5  WHITE_REGION_VALID
bit6  USING_PREDICTION
bit7  TARGET_NEAR_IMAGE_EDGE
bit8  TARGET_PARTIALLY_OUTSIDE
bit9  OBSERVATION_STALE
bit10 CENTER_JUMP_REJECTED
bit11 SCALE_JUMP_REJECTED
bit12 FEEDBACK_STALE
bit13 GIMBAL_FAULT_RECEIVED
bit14 RESERVED
bit15 EMERGENCY_STOP
```

置信度范围为 0～1000。`source_age_us` 从原始图像采集时间计算，预测时不能重置年龄。`source_frame_sequence` 在重复发送同一观测时保持不变。

角度和速度方向约定：

- 靶心在画面右侧，yaw 误差为正；
- 靶心在画面上侧，pitch 误差为正；
- yaw 正速度使相机向右；
- pitch 正速度使相机向上；
- 电机安装方向差异只通过 `yaw_sign`、`pitch_sign` 配置修正。

激光相关字段从 V2 控制负载移除。旧协议 V1 的 `laser_mode`、`laser_confidence` 和相关标志不参与新控制链路。

### 13.4 云台反馈负载

```cpp
#pragma pack(push, 1)
struct GimbalFeedbackPayloadV2 {
    uint8_t  state;
    uint8_t  fault_flags;
    uint16_t ack_sequence;
    int32_t  yaw_angle_mdeg;
    int32_t  pitch_angle_mdeg;
    int16_t  yaw_rate_cdeg_s;
    int16_t  pitch_rate_cdeg_s;
    uint16_t control_latency_us;
    uint16_t reserved;
    uint32_t controller_time_us;
};
#pragma pack(pop)
```

云台最低返回状态、故障、最近执行序号、两轴角度、两轴实际角速度和单调时间。

云台超时：

- 50 ms 无有效控制帧：平滑减速；
- 100 ms：目标角速度归零并保持；
- 500 ms：设置命令超时故障。

Jetson 反馈超时：

- 100 ms 无反馈：设置 `FEEDBACK_STALE` 并禁止继续提高速度；
- 300 ms 无反馈：控制失效、角速度归零并进入故障/安全恢复。

## 14. 调参网页

复用现有相机调参服务器，不新建第二套服务。检测参数更新只原子替换下一帧使用的检测配置，不关闭相机、不重启 MVS 取流线程。

### 14.1 实时叠加

显示：

- 完整/确认中/拒绝的四边形候选；
- 当前靶心和十字线；
- 通过/未通过的圆弧；
- 局部白面区域；
- 下一帧预测 ROI；
- 状态、来源、有效性、置信度、年龄、中心、速度和预测计数。

### 14.2 调试视图

```text
原图
灰度增强图
白色纸面掩码
边缘图
圆弧拟合图
候选评分图
```

候选评分图应显示各子评分和明确的 `reject_reason`。

### 14.3 现场可调参数

相机：曝光、增益、采集帧率及现有自动项。

白面：

```text
white_percentile
white_local_offset
white_min_area_ratio
white_max_area_ratio
white_min_occupancy
white_max_texture
morph_close_size
```

几何：

```text
acquire_score_threshold
track_score_threshold
min_quad_convexity
max_aspect_error
max_corner_angle_error
min_border_support
```

圆环：

```text
ring_edge_threshold
ring_min_arc_degrees
ring_center_tolerance_px
ring_radius_tolerance_ratio
ring_scale_tolerance_ratio
ring_min_count_for_strong
single_arc_max_frames
```

跟踪：

```text
confirm_frames
roi_margin_ratio
max_center_jump_px
max_scale_jump_ratio
max_velocity_px_s
max_acceleration_px_s2
predict_max_frames
predict_max_ms
lost_reacquire_frames
```

所有配置字段中不出现依赖颜色判断的 `red_*` 参数。

### 14.4 参数档与快照

配置保存到独立的 `config/classical_vision.yaml`，支持保存、加载、另存、导入、导出和恢复默认值。建议参数档：

```text
indoor_static.yaml
indoor_motion.yaml
competition.yaml
```

调试快照包含：

```text
original.png
overlay.png
normalized-gray.png
white-mask.png
edge-mask.png
ring-arcs.png
candidate-scores.png
metadata.yaml
```

元数据保存相机参数、检测配置、状态、来源、置信度、中心、四角、圆弧结果、拒绝原因、阶段耗时、帧序号和采集时间。

## 15. 代码边界

```text
src/ev_vision/detection/
├── classical_board.py
├── image_normalization.py
├── white_board.py
├── ring_geometry.py
├── partial_board.py
├── candidate_scoring.py
└── debug_rendering.py
```

职责：

- `image_normalization.py`：共享预处理；
- `white_board.py`：纸面候选和 A4 几何；
- `ring_geometry.py`：灰度圆弧、共同圆心和半径关系；
- `partial_board.py`：局部纸面、历史模型融合和预测 ROI；
- `candidate_scoring.py`：统一评分和拒绝原因；
- `classical_board.py`：统一检测入口和标准观测输出；
- `debug_rendering.py`：网页调试渲染，不参与核心判断。

继续复用：

```text
src/ev_vision/tracking/board_tracker.py
src/ev_vision/tracking/predictor.py
```

现有 YOLO/混合代码保留，但默认比赛配置切换为：

```yaml
detection:
  backend: classical
```

Ultralytics 不再是比赛运行主路径的必要依赖。

## 16. 错误处理

- 相机断开：立即 `FAULT`、`target_valid=false`、角速度归零；
- 检测线程异常：控制失效，记录错误，取流线程尽可能继续；
- 参数非法：整次更新被拒绝，旧配置继续生效；
- 观测年龄超过 100 ms：设置 `OBSERVATION_STALE` 并失效；
- 相机恢复或线程恢复后：必须重新确认完整靶面，不能沿用旧历史模型。

配置校验至少覆盖阈值顺序、面积范围、预测时限和几何容差的合法范围。

## 17. 测试与验收

### 17.1 合成测试

覆盖：

- 正视、旋转和透视；
- 不均匀光照、阴影、模糊、噪声、过曝和欠曝；
- 四个方向部分出画；
- 两条圆环、一条圆弧和靶心短暂出画；
- 完整靶面重新进入并校正漂移；
- 圆环颜色随机变化，验证不依赖红色。

### 17.2 反例

必须拒绝或不建立有效目标：

- 白墙、白书、显示器；
- 无内部结构的白色 A4；
- 背景黑矩形；
- 单圆或不同圆心的多个圆；
- 半径比例不兼容的同心圆；
- 激光饱和光斑；
- 历史 ROI 外突然出现的圆形干扰物。

### 17.3 状态机

验证完整链路：

```text
FULL_BOARD
→ PARTIAL/ARCS
→ PREDICTED
→ LOST
→ 完整白板重新确认
```

硬断言：

- 单圆弧不能无限维持；
- 预测不能无限维持；
- `LOST`、`FAULT` 和陈旧观测的角速度必须为零；
- `LOST` 后局部证据不能直接恢复有效控制；
- 重新检测完整靶面后历史漂移被重置。

### 17.4 实拍回归

使用现有 `latest-camera-capture.tar.gz`，并补采：完整靶面、四侧出画、靶心刚出画、暗光、运动模糊和复杂背景。

## 18. 性能目标

在当前约 20 FPS 采集条件下：

```text
传统视觉检测：至少 15 FPS
网页预览：10～15 FPS
控制发送：100 Hz
正常有效观测年龄：尽量低于 80 ms
硬失效阈值：100 ms
```

优化顺序：

1. 搜索状态使用缩小图；
2. 跟踪状态只处理预测 ROI；
3. 完整检测与局部跟踪使用不同计算路径；
4. 调试图按需生成；
5. 避免每帧编码全部中间图；
6. 避免 Python 像素循环。

50 ms 曝光会限制帧率并增加运动模糊。实机运动验收需要比较 10、15、20、30、50 ms 曝光，在亮度、清晰度和跟踪稳定性之间选择比赛参数。

## 19. 实施优先级

必须完成：

1. 完整白色 A4 获取；
2. 灰度圆环结构确认；
3. 多帧确认；
4. 局部圆弧/白面跟踪；
5. 3 帧且 150 ms 的有界预测；
6. `LOST` 后控制失效；
7. 网页状态、来源和调试视图；
8. 在线参数与调试快照；
9. 合成和实拍回归测试。

时间允许再做：

- 云台反馈辅助运动补偿；
- 更精细的椭圆弧拟合；
- 极短时光流辅助；
- 自动参数建议。

## 20. 完成标准

本阶段完成需同时满足：

- 默认检测后端为传统视觉；
- 实拍完整靶面能够稳定确认；
- 部分出画时能按证据等级继续跟踪；
- 不依赖圆环颜色；
- 预测严格有界，超时进入 `LOST`；
- `LOST` 后云台控制命令归零；
- 网页可观察和调整关键步骤；
- 所有新增单元、状态机、协议和 API 测试通过；
- Jetson 实机达到可接受帧率和观测年龄；
- 完整靶面重新出现后能够可靠重新获取并纠正漂移。
