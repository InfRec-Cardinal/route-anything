# Route Anything

对应 2026 年 10 月 24 日论文 **《Route Anything: an attempt for AGI through a simple omni-modal router》** 的实验代码。

论文研究三个问题：逐输入路由有多大收益、学习型路由器能获得多少收益，以及质量与成本之间如何取舍。路由动作可以是语言模型、图像/视频/语音生成器或机器人策略。

[English](README.md) · [论文与代码对应](docs/paper-map.md) · [实验入口](docs/experiments.md) · [数据说明](docs/data.md) · [验证记录](docs/validation.md)

## 本次发布范围

- **全模态实验**：提供可发布的训练、路由和评估实现，具体覆盖范围见 [omnimodal](experiments/omnimodal/README.md)。全模态 outcome matrix 属于商业机密，不公开原始记录、派生表、划分文件、嵌入、媒体或训练检查点，也不上传到 Hugging Face。没有另行获授权的数据，不能复现论文中的全模态实测分数。
- **代码与文本实验**：保留 CodeRouterBench / RouterBench 的离线分析，并补充新版论文的[独立质量下限认证](experiments/quality_floor/README.md)。现有 [Hugging Face 数据集](https://huggingface.co/datasets/Lance1573/route-anything-data) 是旧版代码/文本实验的回放包，不是 25 个 benchmark 的全模态矩阵。
- **历史扩展**：保留旧版解析示例、allocation、latency、R6 等目录，但不将其全部列为新版论文实验。

## 新版论文结果

以下数字来自论文，不能解读为本公开仓库已独立复现了私有数据实验：

| 实验 | 结果与评估范围 |
| --- | --- |
| 路由上限 | 24 个 benchmark 的 19,814 个完整面板测试输入：固定模型 82.60，逐输入 oracle 87.98。完整矩阵含 25 个 benchmark，SciCode 不在此面板中。 |
| Capability-tree router | 单独的 3,444 输入面板：固定模型 79.68，router 79.98。 |
| 机器人路由目标 | outcome regression 成功率 71.55%；winner-label 分类为 42.8%。 |
| 代码价格路由 | 拟合零损失约束：效率 25.5 倍，测试质量下降 0.44 个百分点；独立认证：200 个划分均满足测试下限，效率中位数 19.4 倍。 |

不同实验使用不同匹配面板，不能横跨表格比较固定基线。拟合满足质量下限不等于在新输入上有保证；200 个划分的结果也不等于任意部署条件下的保证。

## 快速运行

建议 Python 3.11 或更新版本：

```bash
git clone https://github.com/InfRec-Cardinal/route-anything.git
cd route-anything
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --outdir runs/analytical
```

最后一条命令无需数据或 API，验证旧版保留的数学示例，生成图表和数值汇总。全模态代码的合成示例只验证实现行为，不生成论文实验结果；GPU 训练依赖和数据接口见对应目录说明。

公开代码/文本回放：

```bash
python scripts/download_data.py
python scripts/run_experiment.py X2
python scripts/run_experiment.py X10
```

RouterBench 需从上游单独取得。数据位置和实验入口见[数据说明](docs/data.md)与[实验指南](docs/experiments.md)。旧脚本会写入各自实验目录，建议在独立副本运行，以保留归档结果。

新采集需显式配置 `ROUTER_API_BASE_URL` 和 `ROUTER_API_KEY`。本次更新不发起付费模型调用，也不向 Hugging Face 上传数据。代码使用 MIT 协议，第三方数据、模型及 benchmark 保留原协议。
