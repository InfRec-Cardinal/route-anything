# Route Anything

论文 **Route Anything: A Theory of Inference Routers** 的实验代码仓库，研究额外调用模型、验证器、路由信号和记忆何时值得付出成本。

## 快速开始

建议使用 Python 3.11 或更新版本：

```bash
git clone https://github.com/InfRec-Cardinal/route-anything.git
cd route-anything
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --outdir runs/analytical
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活环境。上述命令不需要数据或 API 密钥，会生成解析图、坐标表和公式校验结果。

按照[数据说明](docs/data.md)准备 CodeRouterBench 等输入后，可以复现基准实验：

```bash
python scripts/run_experiment.py X2
python scripts/run_experiment.py X10
```

原始实验脚本会在各自目录写入结果和图；需要保留存档结果时，请使用单独的仓库副本。离线分析使用已记录的模型结果；重新采集模型响应则需要自行配置服务和密钥。

## 内容

- `experiments/`：组合覆盖、尝试阈值、理想记忆、延迟、资源分配及补充分析。
- `experiments_x/`：失败后升级、验证器、最优策略、信号价值、记忆与资源敏感性；另含可选的 GPU 步级路由实验。
- `scripts/`：解析复现、数据下载与统一离线入口。
- `docs/`：[实验地图](docs/experiments.md)、[数据来源](docs/data.md)、[复现口径](docs/reproducibility.md)和验证记录。

代码与逐任务数据分别发布，汇总结果随代码保留。实验沿用原始划分和种子，并保留负结果；解析模型、离线回放和新增模型调用的证据范围在各实验说明中分别列出。

代码采用 [MIT](LICENSE) 许可，第三方数据和模型遵循各自的原始许可。
