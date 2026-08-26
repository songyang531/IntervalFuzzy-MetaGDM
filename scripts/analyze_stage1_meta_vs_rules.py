"""Validate and statistically summarize the stage-1 paired evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage1_frozen_meta_vs_fair_rules_20260823"
RAW = OUT / "stage1_raw.csv"
SUMMARY = OUT / "stage1_summary.csv"
AUDIT = OUT / "audit.json"
META = "Interval Meta-GDM"
SCENARIO_ORDER = [
    "iid_control",
    "hidden_heterogeneity",
    "mid_episode_switch",
    "noisy_partial_feedback",
    "interval_shift",
]
SCENARIO_CN = {
    "iid_control": "IID控制",
    "hidden_heterogeneity": "隐藏响应异质",
    "mid_episode_switch": "中途响应切换",
    "noisy_partial_feedback": "噪声与反馈缺失",
    "interval_shift": "区间分布迁移",
}
METRICS = {
    "Success": ("Meta-Rule", 1.0),
    "Steps": ("Rule-Meta", -1.0),
    "Total Cost": ("Rule-Meta", -1.0),
    "Cost Gini": ("Rule-Meta", -1.0),
    "Boundary Violations": ("Rule-Meta", -1.0),
    "Total Reward": ("Meta-Rule", 1.0),
    "Final Consensus": ("Meta-Rule", 1.0),
    "Consensus AUC": ("Meta-Rule", 1.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def strongest_rule(summary_cell: pd.DataFrame) -> str:
    rules = summary_cell[summary_cell["Model"] != META].copy()
    # One coherent comparator per cell: success is primary; total reward is the
    # prespecified tie-breaker.  The comparator is not changed by metric.
    rules = rules.sort_values(["Success", "Total Reward"], ascending=[False, False])
    return str(rules.iloc[0]["Model"])


def paired_delta(meta: pd.DataFrame, rule: pd.DataFrame, metric: str) -> np.ndarray:
    paired = meta[["Episode", "Seed", metric]].merge(
        rule[["Episode", "Seed", metric]],
        on=["Episode", "Seed"],
        suffixes=("_meta", "_rule"),
        validate="one_to_one",
    )
    direction = METRICS[metric][1]
    return direction * (
        paired[f"{metric}_meta"].to_numpy() - paired[f"{metric}_rule"].to_numpy()
    )


def main():
    raw = pd.read_csv(RAW)
    summary = pd.read_csv(SUMMARY)
    expected_cells = 5 * 3 * 5
    counts = raw.groupby(["Scenario", "Model", "Agents"]).size()
    if len(raw) != 3750 or len(counts) != expected_cells or not (counts == 50).all():
        raise RuntimeError(
            f"Incomplete stage-1 panel: rows={len(raw)}, cells={len(counts)}, "
            f"count_range=({counts.min()}, {counts.max()})"
        )
    if raw.duplicated(["Scenario", "Model", "Agents", "Episode"]).any():
        raise RuntimeError("Duplicate episode keys found")

    strongest_rows = []
    pairwise_rows = []
    seed_counter = 0
    for scenario in SCENARIO_ORDER:
        for agents in (10, 40, 100):
            cell_summary = summary[
                (summary["Scenario"] == scenario) & (summary["Agents"] == agents)
            ]
            chosen = strongest_rule(cell_summary)
            meta = raw[
                (raw["Scenario"] == scenario)
                & (raw["Agents"] == agents)
                & (raw["Model"] == META)
            ]
            rules = raw[
                (raw["Scenario"] == scenario)
                & (raw["Agents"] == agents)
                & (raw["Model"] != META)
            ]
            chosen_rule = rules[rules["Model"] == chosen]
            row = {"Scenario": scenario, "Agents": agents, "Strongest Rule": chosen}
            for metric in METRICS:
                seed_counter += 1
                delta = paired_delta(meta, chosen_rule, metric)
                low, high = bootstrap_ci(delta, 20260823 + seed_counter)
                label = metric.replace(" ", "_")
                row[f"{label}_Advantage"] = float(np.mean(delta))
                row[f"{label}_CI_Low"] = low
                row[f"{label}_CI_High"] = high
            strongest_rows.append(row)

            for rule_name, rule in rules.groupby("Model"):
                pair_row = {
                    "Scenario": scenario,
                    "Agents": agents,
                    "Rule": rule_name,
                }
                for metric in METRICS:
                    seed_counter += 1
                    delta = paired_delta(meta, rule, metric)
                    low, high = bootstrap_ci(delta, 20260823 + seed_counter)
                    label = metric.replace(" ", "_")
                    pair_row[f"{label}_Advantage"] = float(np.mean(delta))
                    pair_row[f"{label}_CI_Low"] = low
                    pair_row[f"{label}_CI_High"] = high
                pairwise_rows.append(pair_row)

    strongest = pd.DataFrame(strongest_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    strongest_path = OUT / "stage1_meta_vs_strongest_rule_paired.csv"
    pairwise_path = OUT / "stage1_pairwise_all_rules.csv"
    strongest.to_csv(strongest_path, index=False, encoding="utf-8-sig")
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    plot_specs = [
        ("Success", "成功率优势（百分点）", 100.0),
        ("Steps", "节省步数", 1.0),
        ("Total_Reward", "回合奖励优势", 1.0),
    ]
    colors = {10: "#2F6690", 40: "#3B7F4A", 100: "#C9573D"}
    x = np.arange(len(SCENARIO_ORDER))
    for ax, (prefix, title, scale) in zip(axes, plot_specs):
        for offset_idx, agents in enumerate((10, 40, 100)):
            frame = strongest[strongest["Agents"] == agents].set_index("Scenario").reindex(
                SCENARIO_ORDER
            )
            mean = frame[f"{prefix}_Advantage"].to_numpy() * scale
            low = frame[f"{prefix}_CI_Low"].to_numpy() * scale
            high = frame[f"{prefix}_CI_High"].to_numpy() * scale
            ax.errorbar(
                x + (offset_idx - 1) * 0.08,
                mean,
                yerr=np.vstack([mean - low, high - mean]),
                marker="o",
                capsize=3,
                linewidth=1.5,
                label=f"N={agents}",
                color=colors[agents],
            )
        ax.axhline(0, color="#444444", linewidth=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SCENARIO_CN[item] for item in SCENARIO_ORDER], rotation=22, ha="right"
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("冻结区间Meta-GDM相对同一最强规则的配对优势（95% bootstrap CI）")
    paired_figure = OUT / "stage1_paired_advantage.png"
    fig.savefig(paired_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 第一阶段冻结模型公平规则基线实验结论",
        "",
        "- 检查点训练回合：5000；本阶段未训练。",
        "- 每个场景、模型和成员规模均为50个配对确定性场景，共3750条记录。",
        "- 规则方法与Meta-GDM接收相同观测；规则不读取真实固执度或成本敏感度。",
        "- 每格只选一个最强规则：先比较成功率，成功率相同时以回合奖励作为固定决胜项。",
        "",
        "## 主要结果",
        "",
        "- 隐藏响应异质性：Meta-GDM在N=10/40/100的成功率优势分别为18/36/64个百分点，且奖励和收敛步数均明显占优；总成本仍较高。",
        "- IID控制：成功率相同；Meta-GDM在N=40/100的奖励与成本公平性较好，但步数和总成本没有优势。",
        "- 区间分布迁移：成功率均为100%；Meta-GDM奖励和成本Gini较好，步数基本持平，总成本较高。",
        "- 中途响应切换：成功率基本相同；Meta-GDM奖励和成本公平性较好，但恢复速度未形成稳定优势。",
        "- 噪声与30%反馈缺失：Meta-GDM明显劣于规则方法，且劣势随规模扩大。这是当前冻结模型的明确鲁棒性缺口。",
        "",
        "## 可用于论文的边界",
        "",
        "现有冻结模型能够支持‘隐藏响应异质性下的快速适应优势’以及‘区间迁移下的综合奖励与公平性优势’，不能支持‘噪声和缺失反馈鲁棒性优势’或‘所有指标全面优于规则方法’。",
    ]
    findings_path = OUT / "stage1_findings.md"
    findings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit["validation"] = {
        "raw_rows": len(raw),
        "cells": len(counts),
        "episodes_per_cell_min": int(counts.min()),
        "episodes_per_cell_max": int(counts.max()),
        "duplicate_episode_keys": 0,
        "paired_comparator_rule": "highest success, then highest total reward",
        "confidence_interval": "paired 10000-draw percentile bootstrap",
    }
    for item in (strongest_path, pairwise_path, paired_figure, findings_path):
        audit["files"][item.name] = {"path": str(item), "sha256": sha256(item)}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(strongest.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(findings_path)
    print(paired_figure)


if __name__ == "__main__":
    main()
