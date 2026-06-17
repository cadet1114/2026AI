# AI Emergency Commander B 分工 Demo

这是 B 分工的 Streamlit 系统与可视化控制台，负责把灾情场景、概率推理、任务分配、路径规划、动态重规划和救援报告串成一个可演示闭环。

当前主流程按“稳定演示优先”设计：默认使用预设 24×24 俯视像素灾区地图；自然语言灾情变化由千问解析成 `update_json`；系统把更新应用到当前地图后自动重新推理、分配任务并规划路线。Qwen-VL 图片识别保留为可选实验功能，不作为课堂演示的必要依赖。

## 运行方式

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

如果依赖下载很慢，可以先运行兜底 HTML 版：

```powershell
py -3.13 web_demo.py
```

然后打开：

```text
http://127.0.0.1:8502/demo.html
```

## 千问 API

不要把 API Key 写进代码。推荐在当前 PowerShell 里设置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="你的千问API Key"
.\.venv\Scripts\streamlit.exe run app.py
```

也可以在网页左侧“千问 API 设置”里临时粘贴 Key，该方式只保存在本次页面会话中。

当前接入：

- `qwen-max`：默认用于自然语言灾情解析和中文救援报告生成。
- `qwen-vl-max`：默认用于可选图片识别，把标准网格图或真实灾区图片抽象成 24×24 场景 JSON。
- 页面左侧可以切换回 `qwen-plus` / `qwen-vl-plus` 作为备用。

## 文本模型来源

左侧“模型接入设置”可以选择自然语言和报告生成使用哪个文本模型后端：

- `千问 API`：调用 DashScope 兼容接口，适合直接联网演示。
- `本地 Qwen 7B`：调用本机或组员机器上部署的 OpenAI-compatible 接口，例如 `http://127.0.0.1:8000/v1/chat/completions`。

注意：普通 Qwen 7B 只能处理文字，适合“自然语言灾情更新”和“救援报告生成”；图片识别仍然需要 Qwen-VL 或其他视觉模型。如果选择本地 Qwen 7B，页面会额外保留一个可选的 `DASHSCOPE_API_KEY` 输入框，仅用于 Qwen-VL 图片识别。

## 算法引擎来源

左侧“算法引擎设置”可以在两种模式之间切换：

- `B 演示引擎`：使用本项目内置的临时推理、分配和 A* 逻辑，保证课堂演示稳定。
- `A 同学算法适配`：保留 B 端 24×24 区块地图、坍塌、断路、火灾、拥堵、建筑、水域等图层，把当前网格临时转换为 graph，再调用 A 同学仓库里的贝叶斯推理、期望效用任务分配和风险感知 A*。

当前适配层位于 `core/a_engine_adapter.py`。它不会替换前端地图，只把算法输出转回 B 端的概率表、任务表、路线和路线代价；A 引擎原本的无人机直线航线会重新投影到 B 端 24×24 网格，并继续调用 A 的风险感知 A*，避免路线视觉上直接穿过建筑、断路、水域等障碍。如果 A 引擎不可用，页面会提示并回退到 `B 演示引擎`，避免演示中断。

## 推荐演示流程

1. 点击“加载初始灾区场景”。
2. 页面自动完成概率推理、任务分配和路线规划。
3. 在“自然语言灾情更新”里输入变化，例如：`无人机发现 C 区北侧道路可以通行，但 C 区火势扩大，SOS 信号增强。`
4. 点击“应用灾情更新并重新规划”。
5. 查看地图、概率表、任务分配和路线摘要是否刷新。
6. 点击“模拟道路塌方”，展示动态重规划。
7. 点击“生成救援报告”，有 Key 时优先调用千问，失败或无 Key 时使用模板 fallback。

## 场景模式

- 预设场景用于稳定课堂演示，地图、灾区指标和路线变化都可控，适合第一次完整讲解系统流程。
- 随机场景用于证明算法不是写死的。左侧“场景初始化”里可以填写随机种子后点击“随机生成灾区场景”；同一个 seed 会生成同一张 24×24 地图，方便课堂复现。
- 随机场景会先生成道路骨架，确保救援中心、医院和 A/B/C 灾区之间至少部分连通，再叠加建筑、水域、公园、火灾、拥堵、塌方和断路图层。
- 随机场景生成后会自动执行概率推理、任务分配、路线规划和报告模板更新，并经过可达性校验；如果 100 次内无法生成至少两条有效救援路线，则回退到预设 `data/scenario.json`，避免出现完全不可规划的地图。

## 场景 JSON 导出

左侧“场景 JSON 导出”可以下载当前地图 JSON。当前先使用临时格式，结构与 `data/scenario.json` 完全一致，方便算法同学先读取：

- `zones`：A/B/C 区灾情指标。
- `units`：救援车和无人机起点。
- `map`：24×24 地图、救援中心、医院、目标区、道路、建筑、水域、公园、火灾、拥堵、塌方、断路等图层。

页面每次加载、随机生成、图片识别、自然语言更新或道路塌方重规划后，都会自动保存一份当前场景到 `exports/current_scenario.json`。等算法组给出正式 schema 后，只需要把导出函数改成对应字段映射。

## 路径代价模型

路径规划不是只比较路线长度，也会比较地形代价：

- 救援车走道路：单格代价 `1.0`
- 救援车走草地/空地/绿地：单格代价 `1.8`
- 拥堵：救援车额外 `+3.5`，无人机额外 `+0.8`
- 火灾：救援车额外 `+5.0`
- 塌方风险：救援车额外 `+4.0`
- 建筑、水域、断路：救援车不可通行
- 无人机基础代价 `1.0`，可以飞越障碍，但火灾、塌方等风险会增加代价

页面“路线摘要”会同时显示路线长度和路线代价，因此可以解释为什么系统有时宁愿绕远路，也不直接穿越高风险区域。

切换到 `A 同学算法适配` 后，路线表中的“路线代价 / 路径风险 / ETA / A*扩展节点”来自 A 同学的风险感知 A* 输出；地图区块仍然按 B 端像素化俯视图展示。

## update_json 结构

自然语言更新会被转换成下面这种结构，再由 `core/demo_engine.py` 应用到当前场景。当前地图坐标范围是 `x=0..23, y=0..23`：

```json
{
  "updates": [
    {
      "type": "target_update",
      "target": "C",
      "fields": {
        "sos_signal": 0.95,
        "fire": 0.9,
        "road_damage": 0.35
      }
    },
    {
      "type": "remove_blocked_cells",
      "cells": [[5, 4]]
    }
  ]
}
```

支持的更新类型：

- `target_update`
- `add_blocked_cells` / `remove_blocked_cells`
- `add_fire_cells` / `remove_fire_cells`
- `add_congestion_cells` / `remove_congestion_cells`
- `add_collapse_cells` / `remove_collapse_cells`

## 文件说明

- `app.py`：Streamlit 主控台。
- `core/demo_engine.py`：B 版本临时推理、任务分配、A* 路径规划、场景更新逻辑和随机场景生成。
- `core/qwen_client.py`：DashScope/Qwen API 调用、自然语言更新解析、图片识别和报告生成。
- `data/scenario.json`：预设 24×24 灾区场景，包含道路、建筑、水域、公园、火灾、塌方和断路等图层。
- `assets/disaster_grid_input.png`：可选 Qwen-VL 识别用标准网格图。
- `assets/real_disaster_scene_input.png`：可选 Qwen-VL 真实图片抽象测试图。

## 后续对接

- A 同学的正式概率模型、任务分配和路径规划已通过 `core/a_engine_adapter.py` 做了第一版接入；后续如果组员给出正式 schema，只需要调整这个适配层的字段映射。
- C 同学的正式报告模块可以替换 `generate_qwen_report` 或模板 `generate_report`。
- B 分工主要保持 `app.py` 的可视化和交互结构稳定。
