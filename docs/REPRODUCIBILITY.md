# 复现说明

## 1. 环境

论文实验使用 Python、PyTorch、NumPy、pandas 和 Matplotlib。已记录环境为 PyTorch `2.6.0+cu126`，但代码不依赖特定显卡型号。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd code
```

## 2. 主模型训练与冻结评估

以下命令对应 `configs/train_meta_5000.json`：

```powershell
python revision_experiments.py --mode train_meta `
  --train-agents 10 --agents 10 40 100 `
  --episodes 50 --train-episodes 5000 `
  --max-steps 120 --threshold 0.9 `
  --interval-half-width-range 0.02 0.08 `
  --context-window 10 --num-heads 4 `
  --context-mode learned --aggregation social_attention `
  --lr 0.0003 --gamma 0.95 --tau 0.01 --alpha 0.1 `
  --seed 20260507 --device cuda `
  --checkpoint-tag interval_only_5000
```

训练采用四阶段课程阈值 `0.70/0.80/0.85/0.90`，最大步数依次为正式上限的 `0.75/0.8333/0.9167/1.0`。前 3% 回合用于经验池预热。

已训练权重位于 `checkpoints/`。论文主模型文件名以 `meta_gdm_social_attention_ctxlearned_...` 开头，SHA-256 见 `RESULTS_LINEAGE.md`。

## 3. PPO、DDPG 与普通 SAC

```powershell
python revision_experiments.py --mode rl_baselines `
  --rl-algorithms SAC PPO DDPG `
  --agents 10 40 --episodes 50 --train-episodes 5000 `
  --max-steps 120 --threshold 0.9 `
  --interval-half-width-range 0.02 0.08 `
  --seed 20260507 --device cuda
```

三种基线使用展平群体状态和联合动作，因此 `N=10` 与 `N=40` 分别训练。论文使用的检查点均在 `checkpoints/`，场景级结果和汇总结果位于 `results/main_5000/rl_5000_raw.csv` 与 `rl_5000_summary.csv`。

## 4. 规则方法与隐藏响应异质性

在仓库根目录运行：

```powershell
python scripts/run_stage1_meta_vs_rules.py
python scripts/analyze_stage1_meta_vs_rules.py
```

输出应写入新的本地目录，不要覆盖仓库内的已发布结果。已发布数据位于 `results/rule_comparison/`。比较包含静态同分布、隐藏响应异质性、中途响应切换、噪声与部分反馈缺失、区间分布迁移五类场景。

## 5. 区间模糊度与历史遮蔽

区间审计不训练模型，只加载冻结检查点：

```powershell
python scripts/run_existing_interval_model_zero_training_audit.py
python scripts/plot_existing_interval_zero_training_audit.py
```

已发布的场景级结果为 `results/interval_audit/episodes.csv`，汇总为 `summary.csv`。超大的逐步轨迹文件未纳入 Git；它可由相同脚本、随机种子和检查点重新生成。

隐藏异质性下的真实历史与全零历史配对消融：

```powershell
python scripts/run_hidden_heterogeneity_context_ablation.py
```

结果位于 `results/history_ablation/`。两种条件使用完全相同的环境种子，只改变输入编码器的历史序列。

## 6. 表格与图形

论文表格的发布副本位于 `results/tables/`。正文实际使用的 19 张图片位于 `figures/paper/`；该目录不保留论文未引用的旧组合图或诊断图。

部分制图脚本保存了原工作区目录约定。若在新克隆仓库运行，可通过命令行输出参数或将脚本中的根目录指向仓库根目录。发布 CSV 与 PNG 均可直接用于数值核验。

## 7. 快速核验

```powershell
python -m py_compile code\env.py code\model.py code\agent.py code\replay_buffer.py code\context_utils.py code\revision_experiments.py code\diagnostics.py code\rl_baselines.py
python -c "import pandas as pd; print(pd.read_csv('results/main_5000/main_5000_summary.csv'))"
```

完整训练的耗时依赖 GPU 与成员规模。冻结评估远快于训练。

