# IntervalFuzzy-MetaGDM

面向区间模糊偏好的群体共识方法复现仓库。模型以区间中心与半宽表示成员偏好，通过 Transformer 编码最近交互历史，以社会注意力聚合可变规模成员状态，并用共享 SAC Actor-Critic 输出连续协调建议。

本仓库的 `main` 分支保存论文 *A Reward Mechanism for Group Consensus Decision Making With Social Attention Under Interval-Valued Fuzzy Preferences* 所用的核心代码、原始 5000 回合训练配置与冻结检查点，以及补充的领域对比实验代码和数据。新修订的 Overleaf 稿件另行交付；仓库 `paper/` 中的旧稿是归档文件，不代表这次修订稿。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `code/` | 区间环境、上下文编码器、社会注意力、SAC 主模型及 RL 基线 |
| `scripts/` | 规则比较、区间审计、上下文消融、表图生成脚本 |
| `experiments/domain_comparison_20260927/` | 三篇领域论文的受限适配、配对评估入口和独立审计脚本 |
| `configs/` | 论文实验使用的 5000 回合参数快照 |
| `checkpoints/` | 主模型及 N=10/40 的 PPO、DDPG、SAC 检查点 |
| `results/` | 场景级记录、汇总表、配对置信区间与实验审计 |
| `figures/paper/` | 论文实际使用的流程图、18 张正文数据图和 4 张附录数据图 |
| `docs/` | 复现流程、结果血缘及论文实验映射 |
| `paper/` | 既有英文稿件归档；本轮 Overleaf 修订稿不在此目录 |

## 快速开始

```powershell
git clone https://github.com/songyang531/IntervalFuzzy-MetaGDM.git
cd IntervalFuzzy-MetaGDM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd code
python -m py_compile env.py model.py agent.py replay_buffer.py context_utils.py revision_experiments.py diagnostics.py rl_baselines.py
```

在 CPU 上可以完成读取、评估和小规模复核；完整 5000 回合训练建议使用支持 CUDA 的 PyTorch 环境。

## 论文主实验

主模型采用随机种子 `20260507`，在 `N=10` 环境训练 5000 回合，冻结后直接评估 `N=10,40,100`。完整命令、结果文件与论文节次对应关系见：

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/PAPER_EXPERIMENT_MAP.md`](docs/PAPER_EXPERIMENT_MAP.md)
- [`docs/RESULTS_LINEAGE.md`](docs/RESULTS_LINEAGE.md)

核心检查点 SHA-256：

```text
7127330b0a3d1ec408f27b7617317c3ea13b87fba3d6542534f05fa490623d1e
```

## 结果口径

- 成功：在 120 步内达到共识阈值 `0.90`。
- 区间更新：整体平移中心，保持半宽不变。
- 冻结测试：不更新模型、优化器或经验池。
- 配对实验：同一方法单元共享场景种子、初始区间和成员隐藏参数。
- 区间与历史审计使用现有 5000 回合检查点，不进行额外训练。

## 文献启发的领域对比（补充实验）

代码以 `main` 分支为准。[`experiments/domain_comparison_20260927/README.md`](experiments/domain_comparison_20260927/README.md) 说明三篇共识研究在同一任务中的受限适配、与原论文完整算法的区别及复跑命令。入口为 [`run_all.py`](experiments/domain_comparison_20260927/run_all.py)，独立核查为 [`audit.py`](experiments/domain_comparison_20260927/audit.py)；原论文冻结检查点不重新训练。对应的 [`episodes.csv`](results/domain_comparison_20260927_release/episodes.csv)、[`trajectories.csv`](results/domain_comparison_20260927_release/trajectories.csv)、[`summary.csv`](results/domain_comparison_20260927_release/summary.csv) 和 [`audit.json`](results/domain_comparison_20260927_release/audit.json) 均在 `results/domain_comparison_20260927_release/`。这组探索性结果不能称为对三篇论文原生完整模型的直接胜出。

## 证据边界

当前论文主结果来自一个训练随机种子和每单元 50 个冻结场景。仓库保留完整规则压力测试结果，其中隐藏响应异质性显示出明显优势；噪声与反馈缺失场景则暴露了鲁棒性边界。因此，本仓库不支持“所有场景、所有指标均优于基线”的表述。

## 引用

若本仓库对你的研究有帮助，请引用仓库中的 [`CITATION.cff`](CITATION.cff)。

