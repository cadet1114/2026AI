# AI Emergency Commander

面向灾害救援的可解释决策 Demo。同一条流水线支持两种概率模型：

- `fixed`：专家设计的完整离散贝叶斯网络 CPT。
- `learned`：在相同网络结构上，用 USGS 灾害强度锚定的混合数据学习 CPT。

统一流程：`随机/JSON 场景 -> 贝叶斯推理 -> 风险/优先级 -> 风险感知 A* -> 期望效用 -> 约束分配 -> 状态机执行 -> 动态重规划 -> 结果报告`。

## 本机运行

适配 Apple Silicon Mac（已在 M3 / 16GB 上验证）：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/streamlit run app.py
```

浏览器打开 `http://localhost:8501`，点击一次“随机生成并启动”即可生成可复现随机地图并自动运行算法。左侧实时显示道路、路线和移动单位，右侧同步演示输入校验、贝叶斯推理、风险排序、风险感知 A*、期望效用、全局分配、状态机执行、动态重规划和结果汇总。

运行中可以随时点击“道路坍塌”“火势蔓延”“新增求救”或“无人机情报”。系统默认选择影响最大的目标，也支持高级手动选择；事件会从当前单位位置和任务状态触发重规划。仿真结束后可下载完整 JSON 和 Markdown 报告。

期望效用同时考虑被困概率、生命风险、任务适配度、到达时间、路径风险和单位资源消耗。每个候选方案都会输出六项加权贡献、可行性原因和可复核的中文解释。

## 复现实验

```bash
.venv/bin/emergency-commander download-public \
  --config config/experiment.yaml \
  --output data/public/usgs_earthquakes.csv \
  --metadata artifacts/full_bayesian_experiment/public_data_metadata.json

.venv/bin/emergency-commander run-experiment \
  --config config/experiment.yaml \
  --public-data data/public/usgs_earthquakes.csv \
  --output-dir artifacts/full_bayesian_experiment
```

默认实验使用 2,000 条 USGS 记录锚定 `hazard_intensity`，生成 50,000 条带来源标记的贝叶斯祖先仿真样本并做 5 折交叉验证。USGS 不提供被困人员或道路通行标签，这两类标签明确属于仿真数据，不冒充真实救援真值。

## 已验证结果

| 目标 | 模型 | Brier | Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 被困人员 | 专家 CPT | 0.1771 | 0.7324 | 0.3354 | 0.7415 |
| 被困人员 | 学习 CPT | 0.1769 | 0.7361 | 0.4041 | 0.7405 |
| 道路可通 | 专家 CPT | 0.1544 | 0.7768 | 0.8640 | 0.7740 |
| 道路可通 | 学习 CPT | 0.1532 | 0.7777 | 0.8626 | 0.7766 |

M3 / 16GB 实测：50,000 样本、5 折训练与评估耗时 `92.566s`，Python `tracemalloc` 峰值 `121.632MB`。

## 数据契约

- 输入示例：`examples/scenario_input.json`
- 固定版输出：`examples/decision_output_fixed_v2.json`
- 学习版输出：`examples/decision_output_learned_v2.json`
- 输入/输出 Schema：`schemas/`
- 学习网络：`artifacts/full_bayesian_experiment/learned_network.json`
- 实验报告：`artifacts/full_bayesian_experiment/experiment_report.md`
- 实时仿真内核：`src/emergency_commander/live_simulation.py`
- 随机场景生成器：`src/emergency_commander/random_scenario.py`

入口和出口都会运行 JSON Schema 校验。地面车辆只使用 `roads`，无人机只使用 `air_routes`；候选效用矩阵、单位状态、事件时间线和每一步场景状态都包含在统一输出中。

## 可信边界

完整网络和 CPT 学习是可解释参数学习，不学习任务分配策略、不学习 A* 搜索策略，也不做在线增量学习。更详细的设计和执行记录见 `docs/superpowers/` 与 `HANDOFF.md`。

## 网页验收

1. 点击“随机生成并启动”，确认地图、随机种子和算法轨道出现，且无需第二次点击。
2. 观察仿真时钟、单位位置和算法日志自动变化。
3. 暂停或保持运行，点击任意突发事件；确认事件数增加并进入 `REPLAN`。
4. 恢复运行，确认重规划数增加且时钟继续推进。
5. 进入“最终结果”，确认结束原因、救援人数、事件记录以及 JSON/Markdown 下载按钮存在。

浏览器验收截图位于 `output/playwright/live-simulation-*.png`。
