"""Build the figure set that is structurally homologous to the reference paper.

The script consumes only audited 5000-episode results and read-only evaluation
evidence.  It deliberately does not add the project's extra diagnostic pilots to
the paper-facing figure sequence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tmp" / "reference_interval_only_5000_seed20260507_20260822"
MERGED = RUN / "merged_results"
EVIDENCE = RUN / "reference_exact_evidence"
OUT = RUN / "reference_exact_figures"
FONT = Path(r"C:\Windows\Fonts\simsun.ttc")
COLORS = ["#214E7A", "#B54A35", "#4D7C58", "#7A5A91", "#C58C32", "#566573"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup() -> None:
    if FONT.is_file():
        fm.fontManager.addfont(str(FONT))
    plt.rcParams.update(
        {
            "font.family": "SimSun",
            "font.size": 8.5,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.tight_layout(pad=0.65)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def framework() -> Path:
    old = RUN / "paper_figures" / "fig_00_framework.png"
    if not old.is_file():
        raise FileNotFoundError(old)
    target = OUT / "fig01_overall_framework.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(old.read_bytes())
    return target


def training_history() -> pd.DataFrame:
    candidates = list((RUN / "source" / "revision_results" / "checkpoints").glob("*interval_only_5000_training_history.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one principal training history, got {candidates}")
    return pd.read_csv(candidates[0])


def training_curve(df: pd.DataFrame, field: str, ylabel: str, filename: str, ylim=None) -> Path:
    x = pd.to_numeric(df["Episode"], errors="coerce")
    y = pd.to_numeric(df[field], errors="coerce")
    rolling = y.rolling(100, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(6.25, 3.55))
    ax.plot(x, y, color="#A8C4DC", lw=0.38, alpha=0.35, label="原始值")
    ax.plot(x, rolling, color="#B3473C", lw=1.25, label="100回合滑动均值")
    ax.set_xlabel("训练回合")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 5000)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="best", fontsize=7.5)
    return save(fig, filename)


def latent_figures() -> list[Path]:
    df = pd.read_csv(EVIDENCE / "latent_members.csv")
    zcols = sorted([column for column in df if column.startswith("z")], key=lambda value: int(value[1:]))
    if len(zcols) != 8:
        raise RuntimeError(f"Expected z1..z8, got {zcols}")
    scaled = StandardScaler().fit_transform(df[zcols].to_numpy(float))
    pc = PCA(n_components=3, random_state=20260507).fit_transform(scaled)
    df = df.copy()
    for idx in range(3):
        df[f"PC{idx + 1}"] = pc[:, idx]
    traits = [
        "initial_centre",
        "initial_distance",
        "stubbornness",
        "cost_sensitivity",
        "cumulative_abs_movement",
        "cumulative_cost",
        "response_rate",
    ]
    labels = ["初始偏好", "初始偏离", "有限信任", "成本敏感", "累计移动", "累计成本", "响应率"]
    corr = np.asarray([[df[f"PC{p}"].corr(df[t]) for t in traits] for p in (1, 2, 3)])
    fig, ax = plt.subplots(figsize=(7.15, 3.1))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(3), ["PC1", "PC2", "PC3"])
    for r in range(3):
        for c in range(len(labels)):
            ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[r, c]) > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025, label="Pearson相关系数")
    outputs = [save(fig, "fig06_latent_pca_correlation.png")]

    scatter_specs = [
        ("PC1", "initial_centre", "初始区间中心", "fig07_pc1_initial_preference.png"),
        ("PC2", "initial_centre", "初始区间中心", "fig08_pc2_initial_preference.png"),
        ("PC3", "stubbornness", "有限信任参数", "fig09_pc3_stubbornness.png"),
    ]
    for pc_col, trait, xlabel, filename in scatter_specs:
        x = df[trait].to_numpy(float)
        y = df[pc_col].to_numpy(float)
        fig, ax = plt.subplots(figsize=(5.7, 3.5))
        scatter = ax.scatter(x, y, c=df["cumulative_cost"], s=10, alpha=0.55, cmap="viridis", linewidths=0)
        coef = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, coef[0] * grid + coef[1], color="#B3473C", lw=1.2)
        rho = np.corrcoef(x, y)[0, 1]
        ax.text(0.03, 0.95, f"r={rho:.3f}", transform=ax.transAxes, ha="left", va="top")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(pc_col)
        fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.025, label="累计调整成本")
        outputs.append(save(fig, filename))
    return outputs


def plot_trajectory(df: pd.DataFrame, scenario: str, filename: str, max_members=100) -> Path:
    panel = df[df["scenario"] == scenario].copy()
    if panel.empty:
        raise ValueError(f"Missing trajectory: {scenario}")
    fig, ax = plt.subplots(figsize=(6.35, 3.75))
    members = sorted(panel["member"].unique())[:max_members]
    cmap = plt.get_cmap("turbo")
    for idx, member in enumerate(members):
        item = panel[panel["member"] == member].sort_values("step")
        color = cmap(idx / max(1, len(members) - 1))
        ax.plot(item["step"], item["centre"], color=color, lw=0.68, alpha=0.72)
        if len(members) <= 10:
            ax.fill_between(item["step"], item["lower"], item["upper"], color=color, alpha=0.055, linewidth=0)
    group = panel.groupby("step", sort=True).agg(mean=("group_mean_centre", "mean"), consensus=("consensus", "mean"))
    ax.plot(group.index, group["mean"], color="black", lw=1.35, linestyle="--", label="群体中心")
    ax.set_xlabel("协商轮次")
    ax.set_ylabel("区间偏好中心")
    ax.set_ylim(0.08, 0.92)
    ax.legend(frameon=False, loc="best", fontsize=7.5)
    return save(fig, filename)


def persona_figures() -> list[Path]:
    trajectory = pd.read_csv(EVIDENCE / "persona_trajectory.csv")
    steps = pd.read_csv(EVIDENCE / "persona_steps.csv")
    outputs = [plot_trajectory(trajectory, "persona_proxy_n10", "fig15_persona_preference_evolution.png")]

    consensus = steps.groupby("step", sort=True)["consensus"].mean()
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(consensus.index, consensus.values, color=COLORS[0], lw=1.55)
    ax.axhline(0.90, color="#B3473C", ls="--", lw=1.0, label="共识阈值0.90")
    ax.set(xlabel="协商轮次", ylabel="区间共识度", ylim=(0, 1.02))
    ax.legend(frameon=False)
    outputs.append(save(fig, "fig16_persona_consensus.png"))

    matrix = steps.pivot(index="member", columns="step", values="acceptance").sort_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xlabel("协商轮次")
    ax.set_ylabel("人格智能体编号")
    ax.set_yticks(np.arange(len(matrix.index)), [int(i) + 1 for i in matrix.index])
    if matrix.shape[1] > 12:
        ticks = np.linspace(0, matrix.shape[1] - 1, 8, dtype=int)
        ax.set_xticks(ticks, [int(matrix.columns[t]) for t in ticks])
    fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025, label="建议采纳概率")
    outputs.append(save(fig, "fig17_persona_acceptance_heatmap.png"))

    costs = steps.groupby("member", sort=True)["step_cost"].sum()
    fig, ax = plt.subplots(figsize=(5.9, 3.35))
    ax.bar(np.arange(len(costs)) + 1, costs.values, color=COLORS[2], width=0.72)
    ax.set_xlabel("人格智能体编号")
    ax.set_ylabel("累计调整成本")
    ax.set_xticks(np.arange(len(costs)) + 1)
    outputs.append(save(fig, "fig18_persona_cumulative_cost.png"))
    return outputs


def main() -> None:
    setup()
    outputs: list[Path] = [framework()]
    history = training_history()
    outputs.extend(
        [
            training_curve(history, "Episode Mean Reward", "回合平均奖励", "fig02_episode_reward.png"),
            training_curve(history, "Final Consensus", "终态区间共识度", "fig03_final_consensus.png", (-0.03, 1.03)),
            training_curve(history, "Actor Loss", "Actor损失", "fig04_actor_loss.png"),
            training_curve(history, "Critic Loss", "Critic损失", "fig05_critic_loss.png"),
        ]
    )
    outputs.extend(latent_figures())
    normal = pd.read_csv(EVIDENCE / "normal_trajectories.csv")
    outputs.append(plot_trajectory(normal, "normal_n10", "fig10_normal_n10.png"))
    outputs.append(plot_trajectory(normal, "normal_n40", "fig11_normal_n40.png"))
    polarized = pd.read_csv(EVIDENCE / "polarized_trajectories.csv")
    outputs.append(plot_trajectory(polarized, "polarized_n10", "fig12_polarized_n10.png"))
    outputs.append(plot_trajectory(polarized, "polarized_n40", "fig13_polarized_n40.png"))
    outputs.append(plot_trajectory(polarized, "polarized_n100", "fig14_polarized_n100.png"))
    outputs.extend(persona_figures())
    if len(outputs) != 18:
        raise RuntimeError(f"Reference-exact sequence must contain 18 figures, got {len(outputs)}")
    manifest = {
        "schema": "reference_exact_interval_figures_v1",
        "reference_sequence_count": 18,
        "figures": [
            {"number": index, "path": str(path), "sha256": sha256(path)}
            for index, path in enumerate(outputs, start=1)
        ],
        "evidence_audit_sha256": sha256(EVIDENCE / "evidence_audit.json"),
        "training_history_sha256": sha256(
            list((RUN / "source" / "revision_results" / "checkpoints").glob("*interval_only_5000_training_history.csv"))[0]
        ),
        "smoothing": "100-episode rolling mean shown over unmodified raw training values",
        "persona_disclosure": "Figures 15-18 use a deterministic offline persona proxy, not an external LLM call.",
    }
    manifest_path = OUT / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "figures": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
