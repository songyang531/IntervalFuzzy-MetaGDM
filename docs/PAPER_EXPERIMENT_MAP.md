# 论文实验与仓库文件映射

| 论文内容 | 执行入口 | 结果文件 | 图形 |
| --- | --- | --- | --- |
| 模型交互与训练流程 | `code/agent.py`、`code/model.py`、`code/env.py` | 模型结构与训练更新逻辑 | `figures/paper/fig01_workflow.png` |
| 5000 回合训练过程 | `code/revision_experiments.py` | 检查点训练历史、`results/main_5000/training_5000_raw.csv` | `figures/paper/fig02a_episode_reward.png` 至 `fig02d_critic_loss.png` |
| PPO、DDPG、SAC 对比 | `code/revision_experiments.py --mode rl_baselines` | `results/main_5000/rl_5000_raw.csv`、`rl_5000_summary.csv`、`results/tables/table03_rl_baselines.csv` | `figures/paper/fig03a_success_rate.png` 至 `fig03d_episode_reward.png` |
| 隐藏响应异质性规则对比 | `scripts/run_stage1_meta_vs_rules.py`、`scripts/analyze_stage1_meta_vs_rules.py` | `results/rule_comparison/stage1_raw.csv`、`stage1_summary.csv` | `figures/paper/fig04a_success_rate.png` 至 `fig04d_adjustment_cost.png` |
| 真实历史与全零历史消融 | `scripts/run_hidden_heterogeneity_context_ablation.py` | `results/history_ablation/context_ablation_raw.csv`、`context_ablation_summary.csv` | `figures/paper/fig05a_success_rate.png` 至 `fig05d_episode_reward.png` |
| 区间信息保持 | `scripts/run_existing_interval_model_zero_training_audit.py` | `results/interval_audit/summary.csv`、`episodes.csv` | `figures/paper/fig06a_center_consensus.png`、`fig06b_hausdorff_consistency.png` |
论文正文采用的核心检查点为 `checkpoints/meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth`。

