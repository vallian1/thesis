# 第三章原型代码说明

本目录提供了一个**可独立运行**的第三章原型实现：

- `chapter3_tracker.py`：自包含 Python 程序；
- `tracker_output/*.csv`：在 `streamIn1215.txt` 与 `streamIn1216.txt` 上跑出的轨迹结果。

## 为什么不是直接在原 `TwoLaneBeaconAssociationV2_1122` 上打补丁

上传的核心代码依赖大量未提供的工程类，例如：

- `Vehicle`
- `Flow`
- `VehicleState`
- `VehicleStateP`
- `VehicleKalmanFilter`
- `AssociationUtils`
- `BeaconAssociationConfig`

因此当前环境下无法直接编译原工程。为保证“先能跑通并验证第三章方法”，这里给出的是**可直接运行的章节原型版本**，后续可再按模块映射回 Java 工程。

## 原型保留的骨架

原型保留了原始代码的主处理链：

1. 事件流输入；
2. 状态预测；
3. 候选关联；
4. 轨迹管理；
5. 输出轨迹段。

## 原型替换的关键模块

### 1. 固定概率 -> 节点级自适应概率
原代码固定：
- `P_s`
- `P_n`
- `P_w`
- `P_c`

原型改为：
- 节点级检测率后验；
- 节点级多检率后验；
- 节点可靠性 `r_b = p_D^{(b)} (1-p_F^{(b)})`。

### 2. 700ms 硬规则 -> 三状态场景概率判别
原代码通过固定时间阈值判断邻道干扰。
原型改为三状态：
- `I1`：1侧真实，2侧干扰；
- `I2`：2侧真实，1侧干扰；
- `P`：真实双车并行。

并使用多步时序递推而不是单点硬判别。

### 3. 固定代价关联 -> 统一关联评分
原型关联评分融合：
- 运动学残差；
- 节点可靠性；
- 磁峰值相容性；
- 时间长度相容性；
- 场景判别结果。

### 4. 轻中度漏检修复 -> 轨迹段输出接口
原型允许最大 `MAX_SKIP=4` 个观测组跨度参与候选关联；
最终仅保留 `MIN_TRACK_POINTS=4` 的轨迹进入输出。
其余长缺测问题可以作为第四章聚类重构的输入轨迹段。

## 如何运行

```bash
python chapter3_tracker.py streamIn1215.txt streamIn1216.txt -o tracker_output
```

## 输出文件

每个输入文件输出四类结果：

- `*_track_summary.csv`：轨迹摘要；
- `*_track_points.csv`：轨迹点序列；
- `*_sensor_reliability.csv`：节点可靠性估计；
- `*_scene_pairs.csv`：邻道场景判别结果。

另有：
- `run_summary.csv`：整体运行统计。

## 回灌到原 Java 工程的建议

若后续要把原型回灌到原 Java 工程，建议按下面四个方法块替换：

1. `checkBeanconStatus(...)`
   - 保留故障率统计；
   - 增加节点级 Beta 后验参数更新。

2. `adjProbUpdate(...) / adjProbUpdate_missMearsments(...)`
   - 改为显式三状态场景概率向量；
   - 改为前向递推，不再依赖固定 700ms 规则主判决。

3. `associateExistingVehiclesNoLaneLimit(...)`
   - 将原先的拍卖/近邻代价改为统一关联评分。

4. `poeProbabilityUpdate(...) / laneProbabilityUpdate(...)`
   - 将固定 `P_s, P_n, P_w, P_c` 替换为节点级、场景级条件概率。
