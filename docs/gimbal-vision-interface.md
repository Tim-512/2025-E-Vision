# 云台—视觉语义接口边界

本文只定义视觉与云台之间的**传输无关语义**。它不修改当前 `src/ev_vision/protocol.py` 中的既有线协议，也不规定新的字节包格式。

## 1. 云台反馈给视觉的字段

云台侧需要提供以下语义字段：

| 字段 | 含义 |
|---|---|
| `timestamp_ms` | 云台反馈采样时间戳，单位毫秒 |
| `gimbal_ready` | 云台是否完成初始化并允许闭环协同 |
| `yaw_angle_deg` | 当前偏航角，单位度 |
| `pitch_angle_deg` | 当前俯仰角，单位度 |
| `yaw_rate_deg_s` | 当前偏航角速度，单位度每秒 |
| `pitch_rate_deg_s` | 当前俯仰角速度，单位度每秒 |
| `motion_state` | 云台运动/稳定状态 |
| `fault_flags` | 云台故障位集合 |

云台反馈缺失或超时只会阻止系统级激光许可，不会停止相机采集、目标检测或视觉结果更新。

## 2. 视觉提供给云台的字段

视觉侧提供不可变的 `VisionTargetResult`，字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `timestamp_ms` | `int` | 源图像采集时间戳，单位毫秒 |
| `frame_sequence` | `int` | 源帧序号 |
| `target_valid` | `bool` | 当前结果是否允许用于真实目标闭环 |
| `tracking_state` | `str` | 跟踪状态；只有精确值 `TRACKING` 才可能有效 |
| `confidence` | `float` | hybrid 检测的综合置信度 |
| `center_x_px` | `float \| None` | 目标中心图像横坐标 |
| `center_y_px` | `float \| None` | 目标中心图像纵坐标 |
| `offset_x_px` | `float \| None` | 目标中心相对图像中心的横向偏差，右为正 |
| `offset_y_px` | `float \| None` | 目标中心相对图像中心的纵向偏差，下为正 |
| `target_x_mm` | `float \| None` | 靶面坐标系中的目标横坐标，单位毫米 |
| `target_y_mm` | `float \| None` | 靶面坐标系中的目标纵坐标，单位毫米 |
| `corners` | `tuple[tuple[float, float], ...]` | 目标板角点像素坐标；缺失时为空元组 |
| `frame_age_ms` | `float` | 结果生成时相对于源帧时间戳的帧龄，单位毫秒 |
| `laser_permission` | `bool` | 激光许可；本阶段始终为 `false` |

## 3. `target_valid` 规则

`target_valid` 只有在以下条件**全部**满足时才为 `true`：

1. `tracking_state` 精确等于 `TRACKING`；
2. hybrid 结果自身的 `target_valid` 为 `true`；
3. 中心坐标存在；
4. 角点坐标存在且非空；
5. 单应性有效，即 `homography_valid = true`；
6. `target_x_mm` 和 `target_y_mm` 均存在；
7. `frame_age_ms` 未超过配置的最大帧龄。

以下结果一律无效，不得用于真实目标闭环：

- `PREDICTING` 或其他非 `TRACKING` 状态；
- 结果陈旧或超出最大帧龄；
- 多候选歧义；
- 模型不可用或模型错误；
- 相机错误或相机断开；
- 目标位置跳变过大；
- 中心、角点、单应性或靶面坐标缺失。

无效结果仍可携带诊断数据，但接收方必须以 `target_valid` 为最终使用门控，不能继续沿用旧坐标。

## 4. 激光安全边界

`laser_permission` 在本项目阶段**始终为 `false`**，包括目标有效且跟踪稳定的情况。未来由系统级安全控制器结合新鲜的云台反馈、稳定性、瞄准误差、驻留时间、通信健康和全局状态机统一决定激光许可。

开发和调试检测器期间，405 nm 激光必须保持**物理断开或关闭（OFF）**。

## 5. 后续通信任务

具体使用 UART 还是 CAN，以及帧结构、字段打包、字节序、CRC、发送频率、超时和重发策略，将在下一项与云台团队共同完成的通信任务中协定。本语义对象不替代、修改或扩展现有线协议。
