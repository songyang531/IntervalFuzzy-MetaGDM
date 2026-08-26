"""Build paper-ready figures from the merged 5000-episode experiment suite."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tmp" / "reference_interval_only_5000_seed20260507_20260822"
DATA = RUN / "merged_results"
OUT = RUN / "paper_figures"
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
COLORS = ["#2E6FBB", "#E07A2D", "#2A9D8F", "#7B61A8", "#697386", "#C44E52"]


def setup() -> None:
    fm.fontManager.addfont(str(FONT))
    plt.rcParams.update(
        {
            "font.family": "SimHei",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> str:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def framework_figure() -> str:
    """Draw the reference-aligned interval extension and training/evaluation flow."""
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    boxes = [
        (0.25, 2.75, 2.05, 1.15, "区间模糊偏好\n中心 c、半宽 h", "#E8F1FB"),
        (2.75, 2.75, 2.10, 1.15, "5维成员状态\n端点共识/成本", "#EAF6F0"),
        (5.30, 2.75, 2.10, 1.15, "共享 Actor\n标量建议动作", "#FFF1E5"),
        (7.85, 2.75, 2.05, 1.15, "环境响应\n区间整体平移", "#F2ECF8"),
        (10.35, 2.75, 2.35, 1.15, "奖励与下一状态\n共识·成本·公平", "#FCECEC"),
        (2.75, 0.65, 2.10, 1.15, "10轮历史 token\n建议·移动·奖励", "#EEF2F5"),
        (5.30, 0.65, 2.10, 1.15, "Transformer\n潜变量 z", "#EEF2F5"),
        (7.85, 0.65, 2.05, 1.15, "社会注意力\n双 Critic", "#EEF2F5"),
        (10.35, 0.65, 2.35, 1.15, "SAC 更新\n5000回合训练", "#EEF2F5"),
    ]
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#324A5F", linewidth=1.15)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [
        ((2.30, 3.33), (2.75, 3.33)), ((4.85, 3.33), (5.30, 3.33)),
        ((7.40, 3.33), (7.85, 3.33)), ((9.90, 3.33), (10.35, 3.33)),
        ((11.52, 2.75), (11.52, 1.80)), ((10.35, 1.23), (9.90, 1.23)),
        ((7.85, 1.23), (7.40, 1.23)), ((5.30, 1.23), (4.85, 1.23)),
        ((3.80, 1.80), (3.80, 2.75)), ((8.88, 1.80), (6.35, 2.75)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "#324A5F"})
    ax.text(6.6, 4.45, "区间模糊偏好 Meta-GDM：状态—上下文—策略—环境—更新闭环", ha="center", fontsize=13, weight="bold")
    return save(fig, "fig_00_framework.png")


def summary(group: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{group}_5000_summary.csv")


def grouped_metrics(group: str, models: list[str], labels: list[str], name: str, ns=(10, 40, 100)) -> str:
    df = summary(group)
    if group == "rl":
        shared = summary("main")
        shared = shared[shared["Agents"].isin(ns)].copy()
        shared["Model"] = "Meta-GDM"
        df = pd.concat([df, shared], ignore_index=True)
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 4.5))
    x = np.arange(len(models))
    width = 0.22 if len(ns) == 3 else 0.34
    metrics = [
        ("Success", "成功率", lambda v: 100 * v),
        ("Steps", "平均收敛轮数", lambda v: v),
        ("Total Cost", "人均调整成本", None),
        ("Total Reward", "平均回合奖励", lambda v: v),
    ]
    for idx, n in enumerate(ns):
        off = (idx - (len(ns) - 1) / 2) * width
        for ax, (field, title, transform) in zip(axes, metrics):
            vals = []
            for model in models:
                row = df[(df.Model == model) & (df.Agents == n)].iloc[0]
                value = float(row[field])
                if field == "Total Cost":
                    value /= n
                elif transform is not None:
                    value = transform(value)
                vals.append(value)
            ax.bar(x + off, vals, width, label=f"N={n}")
            ax.set_title(title)
            ax.set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("%")
    axes[1].set_ylabel("轮")
    axes[2].set_ylabel("成本/人")
    axes[3].set_ylabel("奖励")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    return save(fig, name)


def training_curves() -> str:
    paths = sorted((RUN / "source" / "revision_results" / "checkpoints").glob("*interval_only_5000_training_history.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one main training history, got {paths}")
    df = pd.read_csv(paths[0])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.0))
    fields = [
        ("Episode Mean Reward", "回合平均奖励"),
        ("Final Consensus", "终态共识度"),
        ("Actor Loss", "Actor损失"),
        ("Critic Loss", "Critic损失"),
    ]
    for ax, (field, title) in zip(axes.flat, fields):
        values = pd.to_numeric(df[field], errors="coerce")
        ax.plot(df.Episode, values, color=COLORS[0], lw=0.45, alpha=0.10)
        ax.plot(df.Episode, values.rolling(100, min_periods=20).mean(), color=COLORS[0], lw=1.7)
        ax.set(title=title, xlabel="训练回合")
    axes[0, 1].set_ylim(-0.03, 1.03)
    return save(fig, "fig_01_training_curves.png")


def main_scale() -> str:
    df = summary("main")
    ns = [10, 40, 100]
    row = {n: df[df.Agents == n].iloc[0] for n in ns}
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 4.3))
    axes[0].plot(ns, [100 * row[n].Success for n in ns], marker="o", lw=2.1, color=COLORS[0])
    axes[1].plot(ns, [row[n].Steps for n in ns], marker="o", lw=2.1, color=COLORS[1])
    axes[2].plot(ns, [row[n]["Total Cost"] / n for n in ns], marker="o", lw=2.1, color=COLORS[2])
    axes[3].plot(ns, [row[n]["Cost Gini"] for n in ns], marker="o", lw=2.1, color=COLORS[3])
    for ax in axes:
        ax.set_xticks(ns)
        ax.set_xlabel("决策者数量 N")
    axes[0].set(title="成功率", ylabel="%", ylim=(-3, 103))
    axes[1].set(title="平均收敛轮数", ylabel="轮")
    axes[2].set(title="人均调整成本", ylabel="成本/人")
    axes[3].set(title="成本基尼系数", ylabel="Gini")
    return save(fig, "fig_02_main_scale.png")


def line_sensitivity(group: str, models: list[str], xvalues: list[float], xlabel: str, name: str) -> str:
    df = summary(group)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    for idx, n in enumerate((10, 40, 100)):
        rows = [df[(df.Model == model) & (df.Agents == n)].iloc[0] for model in models]
        axes[0].plot(xvalues, [100 * r.Success for r in rows], marker="o", color=COLORS[idx], label=f"N={n}")
        axes[1].plot(xvalues, [r.Steps for r in rows], marker="o", color=COLORS[idx], label=f"N={n}")
        axes[2].plot(xvalues, [r["Total Cost"] / n for r in rows], marker="o", color=COLORS[idx], label=f"N={n}")
    axes[0].set(title="成功率", ylabel="%")
    axes[1].set(title="平均收敛轮数", ylabel="轮")
    axes[2].set(title="人均调整成本", ylabel="成本/人")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.legend(frameon=False, fontsize=8)
    return save(fig, name)


def training_sensitivity() -> list[str]:
    return [
        line_sensitivity("training", ["context_window=5", "context_window=10", "context_window=20"], [5, 10, 20], "上下文窗口", "fig_10_context_window.png"),
        line_sensitivity("training", ["attention_heads=1", "attention_heads=2", "attention_heads=4", "attention_heads=8"], [1, 2, 4, 8], "注意力头数", "fig_11_attention_heads.png"),
        line_sensitivity("training", ["lr=0.0001", "lr=0.0003", "lr=0.001"], [1e-4, 3e-4, 1e-3], "学习率", "fig_12_learning_rate.png"),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    outputs = [framework_figure(), training_curves(), main_scale()]
    outputs.append(grouped_metrics("rules", ["Bounded confidence", "Cluster consensus", "Cost-aware heuristic", "Mean consensus", "Minimum Cost Consensus (MCCM)", "Random"], ["有界置信", "聚类", "成本感知", "均值", "MCCM", "随机"], "fig_03_rules.png"))
    outputs.append(grouped_metrics("rl", ["PPO", "DDPG", "SAC", "Meta-GDM"], ["PPO", "DDPG", "SAC", "Meta-GDM"], "fig_04_rl_baselines.png", ns=(10, 40)))
    outputs.append(grouped_metrics("core", ["Full Meta-GDM", "w/o context encoder", "w/o social attention", "w/o both"], ["Full", "去上下文", "去社会注意力", "两者均去除"], "fig_05_core_ablation.png"))
    outputs.append(grouped_metrics("attention", ["social_attention", "standard_mha", "mean_pooling", "no_attention"], ["社会注意力", "标准MHA", "均值池化", "无注意力"], "fig_06_attention_ablation.png"))
    outputs.append(grouped_metrics("context", ["Full context VAE", "Deterministic context", "No context"], ["变分上下文", "确定性上下文", "无上下文"], "fig_07_context_ablation.png"))
    outputs.append(line_sensitivity("latent", ["z=2", "z=4", "z=8", "z=16", "z=32"], [2, 4, 8, 16, 32], "潜变量维度", "fig_08_latent_dim.png"))
    outputs.append(line_sensitivity("reward", ["cost=25", "cost=50", "cost=75", "cost=100"], [25, 50, 75, 100], "成本系数", "fig_09_cost_coefficient.png"))
    outputs.append(line_sensitivity("reward", ["boundary=10", "boundary=20", "boundary=40"], [10, 20, 40], "边界惩罚权重", "fig_09b_boundary_penalty.png"))
    outputs.extend(training_sensitivity())
    (OUT / "figure_manifest.json").write_text(json.dumps({"figures": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "figures": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
