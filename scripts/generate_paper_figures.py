"""Regenerate every data-driven figure used by the IEEE manuscript.

The workflow diagram is intentionally excluded because it is a user-supplied
design asset.  All other main-text and appendix plots are rebuilt from the
released CSV files.  Saved PNGs are cropped to their non-white content with a
small safety pad so that labels remain intact while redundant margins are
removed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "paper"
COLORS = ["#426d98", "#d37225", "#3d8550", "#80558c"]
MODEL_ORDER = ["PPO", "DDPG", "SAC", "Meta-GDM"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelweight": "bold",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.55,
        }
    )


def trim_white_border(path: Path, padding: int = 10, threshold: int = 248) -> None:
    """Crop only the uniform white exterior; never resample chart pixels."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    mask = np.any(array < threshold, axis=2)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return
    left = max(int(cols.min()) - padding, 0)
    top = max(int(rows.min()) - padding, 0)
    right = min(int(cols.max()) + padding + 1, image.width)
    bottom = min(int(rows.max()) + padding + 1, image.height)
    image.crop((left, top, right, bottom)).save(path)


def save(fig: plt.Figure, filename: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    trim_white_border(path)
    return path


def grouped_bars(
    frame: pd.DataFrame,
    x_col: str,
    group_col: str,
    value_col: str,
    x_order: list,
    group_order: list,
    ylabel: str,
    filename: str,
    scale: float = 1.0,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.15, 3.45))
    x = np.arange(len(x_order), dtype=float)
    width = 0.78 / len(group_order)
    for idx, group in enumerate(group_order):
        panel = frame[frame[group_col] == group].set_index(x_col)
        values = [float(panel.loc[item, value_col]) * scale for item in x_order]
        offset = (idx - (len(group_order) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, color=COLORS[idx], label=str(group))
    ax.set_xticks(x, [str(item) for item in x_order])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(group_order))
    return save(fig, filename)


def training_figures() -> list[Path]:
    history = pd.read_csv(RESULTS / "main_5000" / "training_history_5000.csv")
    specs = [
        ("Episode Mean Reward", "Episode Mean Reward", "fig02a_episode_reward.png", None),
        ("Final Consensus", "Terminal Interval Consensus", "fig02b_final_consensus.png", (0.0, 1.02)),
        ("Actor Loss", "Actor Loss", "fig02c_actor_loss.png", None),
        ("Critic Loss", "Critic Loss", "fig02d_critic_loss.png", None),
    ]
    outputs: list[Path] = []
    x = pd.to_numeric(history["Episode"], errors="coerce")
    for field, ylabel, filename, ylim in specs:
        y = pd.to_numeric(history[field], errors="coerce")
        rolling = y.rolling(100, min_periods=20).mean()
        fig, ax = plt.subplots(figsize=(6.15, 3.35))
        ax.plot(x, y, color="#a8c4dc", lw=0.42, alpha=0.38, label="Raw value")
        ax.plot(x, rolling, color="#c34a36", lw=1.25, label="100-episode moving average")
        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 5000)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid()
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
        outputs.append(save(fig, filename))
    return outputs


def rl_baseline_figures() -> list[Path]:
    frame = pd.read_csv(RESULTS / "tables" / "table03_rl_baselines.csv")
    frame = frame.rename(
        columns={
            "模型": "Model",
            "决策者数量": "Agents",
            "共识达成率": "Success",
            "平均收敛步数": "Steps",
            "总调整成本": "Cost",
            "平均回合奖励": "Reward",
        }
    )
    frame["Model"] = frame["Model"].replace({"区间Meta-GDM": "Meta-GDM"})
    specs = [
        ("Success", "Success Rate (%)", "fig03a_success_rate.png", 100.0),
        ("Steps", "Mean Convergence Rounds", "fig03b_convergence_rounds.png", 1.0),
        ("Cost", "Per-Capita Adjustment Cost", "fig03c_adjustment_cost.png", 1.0),
        ("Reward", "Mean Episode Reward", "fig03d_episode_reward.png", 1.0),
    ]
    outputs: list[Path] = []
    for metric, ylabel, filename, scale in specs:
        outputs.append(
            grouped_bars(frame, "Model", "Agents", metric, MODEL_ORDER, [10, 40], ylabel, filename, scale)
        )
    return outputs


def heterogeneity_figures() -> list[Path]:
    frame = pd.read_csv(RESULTS / "rule_comparison" / "stage1_summary.csv")
    frame = frame[frame["Scenario"] == "hidden_heterogeneity"].copy()
    order = ["Interval Meta-GDM", "Mean consensus", "Bounded confidence", "Observable cost-aware"]
    specs = [
        ("Success", "Consensus Success Rate", "fig04a_success_rate.png"),
        ("Steps", "Mean Convergence Steps", "fig04b_convergence_steps.png"),
        ("Total Reward", "Mean Episode Reward", "fig04c_episode_reward.png"),
        ("Total Cost", "Mean Total Adjustment Cost", "fig04d_adjustment_cost.png"),
    ]
    return [
        grouped_bars(frame, "Agents", "Model", metric, [10, 40, 100], order, ylabel, filename)
        for metric, ylabel, filename in specs
    ]


def context_ablation_figures() -> list[Path]:
    frame = pd.read_csv(RESULTS / "history_ablation" / "context_ablation_summary.csv")
    specs = [
        ("Success", "Consensus Success Rate", "fig05a_success_rate.png"),
        ("Steps", "Mean Convergence Steps", "fig05b_convergence_steps.png"),
        ("Total Cost", "Mean Total Adjustment Cost", "fig05c_adjustment_cost.png"),
        ("Total Reward", "Mean Episode Reward", "fig05d_episode_reward.png"),
    ]
    return [
        grouped_bars(
            frame,
            "Agents",
            "Context Mode",
            metric,
            [10, 40, 100],
            ["Real history", "Zero history"],
            ylabel,
            filename,
        )
        for metric, ylabel, filename in specs
    ]


def interval_preservation_figures() -> list[Path]:
    summary = pd.read_csv(RESULTS / "interval_audit" / "summary.csv")
    summary = summary[(summary["width_regime"] == "trained_support") & (summary["history_mode"] == "real")]
    summary = summary.sort_values("Agents")
    fig, ax = plt.subplots(figsize=(4.45, 3.25))
    ax.bar(summary["Agents"].astype(str), summary["Final_center_consensus_mean"], color=COLORS[:3])
    ax.set_ylim(0.90, 1.00)
    ax.set_xlabel("Group Size")
    ax.set_ylabel("Center Consensus")
    ax.grid(axis="y")
    outputs = [save(fig, "fig06a_center_consensus.png")]

    episodes = pd.read_csv(RESULTS / "interval_audit" / "episodes.csv")
    episodes = episodes[(episodes["width_regime"] == "trained_support") & (episodes["history_mode"] == "real")]
    fig, ax = plt.subplots(figsize=(4.05, 3.65))
    for color, agents in zip(COLORS[:3], [10, 40, 100]):
        panel = episodes[episodes["Agents"] == agents]
        ax.scatter(
            panel["mean_center_shift"], panel["mean_hausdorff_distortion"],
            s=14, alpha=0.55, color=color, label=f"N={agents}", edgecolors="none",
        )
    limit = float(max(episodes["mean_center_shift"].max(), episodes["mean_hausdorff_distortion"].max()))
    ax.plot([0, limit], [0, limit], "k--", lw=1.1, label="y=x")
    ax.set_xlabel("Mean Center Displacement")
    ax.set_ylabel("Mean Hausdorff Distortion")
    ax.grid()
    ax.legend(loc="upper left")
    outputs.append(save(fig, "fig06b_hausdorff_consistency.png"))
    return outputs


def trajectory_figure(frame: pd.DataFrame, scenario: str, filename: str, max_members: int) -> Path:
    panel = frame[frame["scenario"] == scenario].copy()
    if panel.empty:
        raise ValueError(f"Missing appendix scenario: {scenario}")
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    members = sorted(panel["member"].unique())[:max_members]
    cmap = plt.get_cmap("turbo")
    for index, member in enumerate(members):
        item = panel[panel["member"] == member].sort_values("step")
        color = cmap(index / max(1, len(members) - 1))
        ax.plot(item["step"], item["centre"], color=color, lw=0.7, alpha=0.72)
        if len(members) <= 10:
            ax.fill_between(item["step"], item["lower"], item["upper"], color=color, alpha=0.07)
    group = panel.groupby("step", sort=True)["group_mean_centre"].mean()
    ax.plot(group.index, group.values, "k--", lw=1.35, label="Group center")
    ax.set_xlabel("Negotiation Round")
    ax.set_ylabel("Interval Preference Center")
    ax.set_ylim(0.08, 0.92)
    ax.grid()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16))
    return save(fig, filename)


def appendix_figures() -> list[Path]:
    source = RESULTS / "appendix_process"
    normal = pd.read_csv(source / "normal_trajectories.csv")
    polarized = pd.read_csv(source / "polarized_trajectories.csv")
    outputs = [
        trajectory_figure(normal, "normal_n10", "figA1a_ordinary_n10.png", 10),
        trajectory_figure(polarized, "polarized_n100", "figA1b_polarized_n100.png", 100),
    ]
    steps = pd.read_csv(source / "persona_steps.csv")
    consensus = steps.groupby("step", sort=True)["consensus"].mean()
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.plot(consensus.index, consensus.values, color=COLORS[0], lw=1.6)
    ax.axhline(0.90, color="#c34a36", ls="--", lw=1.0, label="Consensus threshold 0.90")
    ax.set_xlabel("Negotiation Round")
    ax.set_ylabel("Interval Consensus")
    ax.set_ylim(0.0, 1.02)
    ax.grid()
    ax.legend(loc="lower right")
    outputs.append(save(fig, "figA1c_persona_consensus.png"))

    matrix = steps.pivot(index="member", columns="step", values="acceptance").sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.15))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xlabel("Negotiation Round")
    ax.set_ylabel("Member")
    ax.set_yticks(np.arange(len(matrix.index)), [int(i) + 1 for i in matrix.index])
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label="Acceptance Probability")
    outputs.append(save(fig, "figA1d_persona_acceptance.png"))
    return outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(outputs: list[Path]) -> None:
    workflow = OUT / "fig01_workflow.png"
    figures = [workflow, *outputs]
    manifest = {
        "schema": "paper_figure_manifest_v3",
        "workflow_policy": "fig01_workflow.png is user-supplied and is not generated by this script.",
        "plot_policy": "All other manuscript figures are regenerated from released CSVs and cropped to content.",
        "figures": [
            {
                "filename": path.name,
                "size_px": list(Image.open(path).size),
                "sha256": sha256(path),
                "source": "user-supplied" if path == workflow else "scripts/generate_paper_figures.py",
            }
            for path in figures
        ],
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    setup_style()
    outputs: list[Path] = []
    outputs.extend(training_figures())
    outputs.extend(rl_baseline_figures())
    outputs.extend(heterogeneity_figures())
    outputs.extend(context_ablation_figures())
    outputs.extend(interval_preservation_figures())
    outputs.extend(appendix_figures())
    if len(outputs) != 22:
        raise RuntimeError(f"Expected 22 generated manuscript plots, got {len(outputs)}")
    write_manifest(outputs)
    print(json.dumps({"output": str(OUT), "generated": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
