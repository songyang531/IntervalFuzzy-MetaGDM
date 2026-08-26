"""Create auditable figures from the frozen interval checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WIDTH_ORDER = ["crisp_limit", "narrow", "trained_support", "wide_ood"]
WIDTH_LABELS = ["点值极限\n[0,0]", "窄区间\n[.01,.04]", "训练范围\n[.02,.08]", "宽区间OOD\n[.08,.16]"]
COLORS = {10: "#2F6690", 40: "#3A7D44", 100: "#C8553D"}


def style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def width_stress(summary: pd.DataFrame, output: Path):
    data = summary[summary["history_mode"] == "real"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    x = np.arange(len(WIDTH_ORDER))
    for n in (10, 40, 100):
        rows = data[data["Agents"] == n].set_index("width_regime").loc[WIDTH_ORDER]
        axes[0].plot(x, rows["Success_mean"], marker="o", linewidth=2, color=COLORS[n], label=f"N={n}")
        axes[1].plot(x, rows["Success_AUC_mean"], marker="o", linewidth=2, color=COLORS[n], label=f"N={n}")
    for ax, ylabel in zip(axes, ["有效成功率", "归一化成功AUC"]):
        ax.set_xticks(x, WIDTH_LABELS)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    axes[0].set_title("冻结模型的区间宽度压力测试")
    axes[1].set_title("宽度增大显著压低及时成功")
    fig.savefig(output / "fig01_width_stress_success_auc.png", bbox_inches="tight")
    plt.close(fig)


def consensus_decomposition(summary: pd.DataFrame, output: Path):
    data = summary[summary["history_mode"] == "real"].copy()
    metrics = [
        ("Final_center_consensus_mean", "中心共识"),
        ("Final_width_consensus_mean", "宽度上限"),
        ("Final_endpoint_consensus_mean", "端点综合共识"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True, sharey=True)
    x = np.arange(len(WIDTH_ORDER))
    for ax, (metric, title) in zip(axes, metrics):
        for n in (10, 40, 100):
            rows = data[data["Agents"] == n].set_index("width_regime").loc[WIDTH_ORDER]
            ax.plot(x, rows[metric], marker="o", linewidth=2, color=COLORS[n], label=f"N={n}")
        ax.axhline(0.90, color="#B23A48", linestyle="--", linewidth=1.3, label="阈值0.90")
        ax.set_xticks(x, WIDTH_LABELS)
        ax.set_ylim(0.68, 1.01)
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("共识度")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.savefig(output / "fig02_center_width_endpoint_consensus.png", bbox_inches="tight")
    plt.close(fig)


def reachability(summary: pd.DataFrame, output: Path):
    data = summary[summary["history_mode"] == "real"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True, sharey=True)
    x = np.arange(len(WIDTH_ORDER))
    width = 0.34
    for ax, n in zip(axes, (10, 40, 100)):
        rows = data[data["Agents"] == n].set_index("width_regime").loc[WIDTH_ORDER]
        ax.bar(x - width / 2, rows["Initial_width_ceiling_reaches_threshold_mean"], width, color="#8AA399", label="理论可达率")
        ax.bar(x + width / 2, rows["Success_mean"], width, color=COLORS[n], label="实际成功率")
        ax.set_xticks(x, WIDTH_LABELS)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"N={n}")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("比例")
    axes[-1].legend(frameon=False)
    fig.suptitle("固定宽度异质性造成的共识可达性上限", fontsize=13)
    fig.savefig(output / "fig03_width_ceiling_reachability.png", bbox_inches="tight")
    plt.close(fig)


def history_ablation(summary: pd.DataFrame, output: Path):
    data = summary[summary["width_regime"] == "trained_support"].copy()
    modes = ["real", "zero", "shuffled"]
    labels = ["真实历史", "空历史", "成员历史打乱"]
    x = np.arange(3)
    width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)
    for offset, n in enumerate((10, 40, 100)):
        rows = data[data["Agents"] == n].set_index("history_mode").loc[modes]
        location = x + (offset - 1) * width
        axes[0].bar(location, rows["Success_mean"], width, color=COLORS[n], label=f"N={n}")
        axes[1].bar(location, rows["Steps_mean"], width, color=COLORS[n], label=f"N={n}")
    axes[0].set_ylabel("有效成功率")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("上下文历史的成功率贡献")
    axes[1].set_ylabel("平均收敛/封顶轮数")
    axes[1].set_ylim(0, 125)
    axes[1].set_title("打乱成员归属主要降低效率")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    fig.savefig(output / "fig04_history_causal_ablation.png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.input.resolve()
    summary = pd.read_csv(root / "summary.csv")
    style()
    width_stress(summary, root)
    consensus_decomposition(summary, root)
    reachability(summary, root)
    history_ablation(summary, root)
    figures = sorted(root.glob("fig*.png"))
    manifest = {
        "source_summary_sha256": hashlib.sha256((root / "summary.csv").read_bytes()).hexdigest(),
        "figures": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in figures
        },
        "smoothing": "none",
        "interpolation": "none",
    }
    (root / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(root)


if __name__ == "__main__":
    main()
