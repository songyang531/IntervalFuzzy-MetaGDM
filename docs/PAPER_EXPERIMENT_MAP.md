# 论文实验与仓库文件映射

| 论文内容 | 执行入口 | 结果文件 | 图形 |
| --- | --- | --- | --- |
| 模型交互与训练流程 | `code/agent.py`、`code/model.py`、`code/env.py` | 模型结构与训练更新逻辑 | `figures/paper/fig01_workflow.png`（用户提供，不由脚本生成） |
| 5000 回合训练过程 | `code/revision_experiments.py` | `results/main_5000/training_history_5000.csv` | `figures/paper/fig02a_episode_reward.png` 至 `fig02d_critic_loss.png` |
| PPO、DDPG、SAC 对比 | `code/revision_experiments.py --mode rl_baselines` | `results/main_5000/rl_5000_raw.csv`、`rl_5000_summary.csv`、`results/tables/table03_rl_baselines.csv` | `figures/paper/fig03a_success_rate.png` 至 `fig03d_episode_reward.png` |
| 隐藏响应异质性规则对比 | `scripts/run_stage1_meta_vs_rules.py`、`scripts/analyze_stage1_meta_vs_rules.py` | `results/rule_comparison/stage1_raw.csv`、`stage1_summary.csv` | `figures/paper/fig04a_success_rate.png` 至 `fig04d_adjustment_cost.png` |
| 真实历史与全零历史消融 | `scripts/run_hidden_heterogeneity_context_ablation.py` | `results/history_ablation/context_ablation_raw.csv`、`context_ablation_summary.csv` | `figures/paper/fig05a_success_rate.png` 至 `fig05d_episode_reward.png` |
| 区间信息保持 | `scripts/run_existing_interval_model_zero_training_audit.py` | `results/interval_audit/summary.csv`、`episodes.csv` | `figures/paper/fig06a_center_consensus.png`、`fig06b_hausdorff_consistency.png` |
| 附录过程证据 | `scripts/generate_paper_figures.py` | `results/appendix_process/*.csv` | `figures/paper/figA1a_ordinary_n10.png` 至 `figA1d_persona_acceptance.png` |

除流程图外，论文使用的全部 22 张数据图可统一运行以下命令重绘：

```bash
python scripts/generate_paper_figures.py
```

该脚本从上述 CSV 读取数据，生成独立子图，并在不重采样图像的前提下裁去外部白边。图中不嵌入 `(a)`、`(b)` 等编号；编号和子图标题由论文 LaTeX 源文件统一生成。

论文正文采用的核心检查点为 `checkpoints/meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth`。