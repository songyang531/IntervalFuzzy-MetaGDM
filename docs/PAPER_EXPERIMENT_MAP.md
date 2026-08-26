# 论文实验与仓库文件映射

| 论文内容 | 执行入口 | 结果文件 | 图形 |
| --- | --- | --- | --- |
| 模型训练过程 | `code/revision_experiments.py` | `results/main_5000/training_5000_raw.csv`、`training_5000_summary.csv`、检查点训练历史 | `figures/paper/fig_01_training_curves.png` |
| 静态规则对比 | `code/revision_experiments.py --mode baselines` | `results/main_5000/rules_5000_raw.csv`、`rules_5000_summary.csv`、`results/tables/table02_rule_baselines.csv` | 参考实验图目录中的规则图 |
| PPO/DDPG/SAC 对比 | `code/revision_experiments.py --mode rl_baselines` | `results/main_5000/rl_5000_raw.csv`、`rl_5000_summary.csv`、`results/tables/table03_rl_baselines.csv` | `figures/paper/fig_04_rl_baselines.png` |
| 区间模糊度冻结测试 | `scripts/run_existing_interval_model_zero_training_audit.py` | `results/interval_audit/episodes.csv`、`summary.csv` | `figures/paper/fig02_interval_robustness_positive.png` |
| 跨规模冻结测试 | 同上，加载 N=10 主检查点直接评估 N=40/100 | `results/interval_audit/summary.csv` | `figures/paper/fig02_interval_robustness_positive.png` |
| 隐藏响应异质性 | `scripts/run_stage1_meta_vs_rules.py`、`analyze_stage1_meta_vs_rules.py` | `results/rule_comparison/stage1_raw.csv`、`stage1_summary.csv`、配对效应 CSV | `figures/paper/fig05_hidden_heterogeneity_rules.png` |
| 真实历史/全零历史消融 | `scripts/run_hidden_heterogeneity_context_ablation.py` | `results/history_ablation/context_ablation_raw.csv`、`context_ablation_summary.csv`、`context_ablation_paired_ci.csv` | `figures/paper/fig06_context_real_vs_zero.png`、`fig07_context_paired_benefits.png` |
| 区间信息保持 | `scripts/run_existing_interval_model_zero_training_audit.py` | `results/interval_audit/summary.csv`、`episodes.csv` | `figures/paper/fig05_interval_information_preservation.png` |
| 核心模块与敏感性补充实验 | `code/revision_experiments.py` | `results/main_5000/core_*`、`attention_*`、`latent_*`、`reward_*` | `figures/reference_experiments/` |

论文正文采用的核心检查点为 `checkpoints/meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth`。

