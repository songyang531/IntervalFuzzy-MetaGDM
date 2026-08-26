# 结果血缘与证据边界

## 主模型

- 训练入口：`code/revision_experiments.py --mode train_meta`
- 训练种子：`20260507`
- 训练成员数：`N=10`
- 训练回合：`5000`
- 正式阈值：`0.90`
- 单回合最大步数：`120`
- 历史窗口：`10`
- Transformer：`2` 层、`4` 头
- 潜变量维度：`8`
- 聚合：社会注意力
- 优化：SAC，`lr=3e-4, gamma=0.95, tau=0.01, alpha=0.1`

主检查点：

```text
checkpoints/meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth
SHA-256 7127330b0a3d1ec408f27b7617317c3ea13b87fba3d6542534f05fa490623d1e
```

区间审计在运行前后验证该检查点哈希保持不变，模型状态哈希也保持不变，训练或更新调用次数为零。

## 论文数据来源

- 5000 回合训练、规则、RL 基线、消融及敏感性：`results/main_5000/`
- 论文同构表格：`results/tables/`
- 点值极限、窄区间、训练区间与历史审计：`results/interval_audit/`
- 规则压力测试与隐藏响应异质性：`results/rule_comparison/`
- 隐藏异质性下的真实/全零历史配对消融：`results/history_ablation/`

## 限制

1. 当前论文主模型只使用一个训练随机种子；场景层面的配对区间不能替代多训练种子方差。
2. 每个实验单元使用 50 个冻结场景。
3. PPO、DDPG、SAC 的展平网络需要针对不同群体规模分别训练；主模型只在 N=10 训练后直接迁移。
4. 规则压力测试显示，隐藏响应异质性是当前模型的优势场景；执行噪声、观测噪声和 30% 反馈缺失是明确的鲁棒性缺口。
5. 当前方法是上下文条件化的元强化学习结构，没有独立的 MAML 外循环，也不应表述为测试时梯度适应。

