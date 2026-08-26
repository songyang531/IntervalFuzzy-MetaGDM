"""Frozen real-history versus zero-history test under hidden response heterogeneity."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage1_meta_vs_rules as stage1  # noqa: E402


OUT = ROOT / "outputs" / "hidden_heterogeneity_context_ablation_20260823"
STAGE1_RAW = (
    ROOT
    / "outputs"
    / "stage1_frozen_meta_vs_fair_rules_20260823"
    / "stage1_raw.csv"
)
SCENARIO = stage1.SCENARIOS["hidden_heterogeneity"]
MODES = ["Real history", "Zero history"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_episode(agent, mode: str, n_agents: int, episode: int):
    scenario_offset = stage1.stable_name_seed(SCENARIO.name) % 1_000_000
    env_seed = stage1.BASE_SEED + scenario_offset + episode * 997 + n_agents * 13
    env = stage1.Stage1Env(n_agents, env_seed, SCENARIO)
    state = env.observe_state(env._get_state())
    context = np.zeros((n_agents, 10, 3), dtype=np.float32)
    zero_context = np.zeros_like(context)
    cumulative_costs = np.zeros(n_agents, dtype=float)
    boundary_violations = 0
    total_reward = 0.0
    consensus_trace = []
    success = False
    info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions)}

    for step in range(stage1.MAX_STEPS):
        policy_context = context if mode == "Real history" else zero_context
        action = np.asarray(
            agent.select_action(state, policy_context, evaluate=True)
        ).reshape(n_agents)
        next_true_state, rewards, done, info = env.step(action)
        cumulative_costs += env.calculate_adjustment_costs(info["actual_movements"])
        lower, upper = env._interval_bounds()
        boundary_violations += int(np.sum((lower < env.safe_low) | (upper > env.safe_high)))
        total_reward += float(np.mean(rewards))
        consensus_trace.append(float(info["consensus_level"]))
        obs_move, obs_reward = env.observed_feedback(info["actual_movements"], rewards)
        context = stage1.append_context_step(context, action, obs_move, obs_reward)
        state = env.observe_state(next_true_state)
        if done:
            success = bool(info["success"])
            break

    steps = step + 1
    padded = consensus_trace + [consensus_trace[-1]] * (
        stage1.MAX_STEPS - len(consensus_trace)
    )
    return {
        "Scenario": SCENARIO.name,
        "Context Mode": mode,
        "Agents": n_agents,
        "Episode": episode,
        "Seed": env_seed,
        "Success": int(success),
        "Steps": steps,
        "Total Cost": float(np.sum(cumulative_costs)),
        "Cost Gini": float(env._gini(cumulative_costs)),
        "Boundary Violations": boundary_violations,
        "Total Reward": total_reward,
        "Final Consensus": float(info["consensus_level"]),
        "Consensus AUC": float(np.mean(padded)),
    }


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(raw: pd.DataFrame):
    metrics = [
        "Success",
        "Steps",
        "Total Cost",
        "Cost Gini",
        "Boundary Violations",
        "Total Reward",
        "Final Consensus",
        "Consensus AUC",
    ]
    summary = raw.groupby(["Context Mode", "Agents"], as_index=False)[metrics].mean()
    paired_rows = []
    directions = {
        "Success": 1.0,
        "Steps": -1.0,
        "Total Cost": -1.0,
        "Cost Gini": -1.0,
        "Boundary Violations": -1.0,
        "Total Reward": 1.0,
        "Final Consensus": 1.0,
        "Consensus AUC": 1.0,
    }
    for idx, agents in enumerate((10, 40, 100)):
        real = raw[(raw["Agents"] == agents) & (raw["Context Mode"] == "Real history")]
        zero = raw[(raw["Agents"] == agents) & (raw["Context Mode"] == "Zero history")]
        paired = real.merge(
            zero,
            on=["Scenario", "Agents", "Episode", "Seed"],
            suffixes=("_real", "_zero"),
            validate="one_to_one",
        )
        row = {"Agents": agents}
        for metric_idx, metric in enumerate(metrics):
            direction = directions[metric]
            benefit = direction * (
                paired[f"{metric}_real"].to_numpy()
                - paired[f"{metric}_zero"].to_numpy()
            )
            low, high = bootstrap_ci(
                benefit, 20260823 + idx * len(metrics) + metric_idx
            )
            label = metric.replace(" ", "_")
            row[f"{label}_Benefit"] = float(np.mean(benefit))
            row[f"{label}_CI_Low"] = low
            row[f"{label}_CI_High"] = high
        paired_rows.append(row)
    return summary, pd.DataFrame(paired_rows)


def make_figure(summary: pd.DataFrame, path: Path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    metrics = [
        ("Success", "成功率"),
        ("Steps", "平均步数"),
        ("Total Cost", "总成本"),
        ("Cost Gini", "成本Gini"),
        ("Total Reward", "平均回合奖励"),
        ("Consensus AUC", "共识AUC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    colors = {"Real history": "#2F6690", "Zero history": "#B7C9D6"}
    for ax, (metric, title) in zip(axes.flat, metrics):
        for mode_idx, mode in enumerate(MODES):
            frame = summary[summary["Context Mode"] == mode].set_index("Agents").reindex(
                [10, 40, 100]
            )
            ax.bar(
                x + (mode_idx - 0.5) * width,
                frame[metric],
                width,
                label="真实历史" if mode == "Real history" else "全零历史",
                color=colors[mode],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["N=10", "N=40", "N=100"])
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("隐藏响应异质性：真实历史与全零历史冻结策略消融")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = stage1.load_meta_agent(stage1.CHECKPOINT, device=device)
    for module in (agent.encoder, agent.actor, agent.critic, agent.critic_target):
        module.eval()
    started = time.time()
    records = []
    for agents in (10, 40, 100):
        for mode in MODES:
            for episode in range(50):
                records.append(run_episode(agent, mode, agents, episode))
            print(f"completed N={agents} {mode}", flush=True)
    raw = pd.DataFrame(records)
    summary, paired = summarize(raw)
    raw_path = OUT / "context_ablation_raw.csv"
    summary_path = OUT / "context_ablation_summary.csv"
    paired_path = OUT / "context_ablation_paired_ci.csv"
    figure_path = OUT / "context_ablation.png"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    make_figure(summary, figure_path)

    reference = pd.read_csv(STAGE1_RAW)
    reference = reference[
        (reference["Scenario"] == SCENARIO.name)
        & (reference["Model"] == "Interval Meta-GDM")
    ].sort_values(["Agents", "Episode"])
    reproduced = raw[raw["Context Mode"] == "Real history"].sort_values(
        ["Agents", "Episode"]
    )
    comparison_metrics = [
        "Success",
        "Steps",
        "Total Cost",
        "Cost Gini",
        "Boundary Violations",
        "Total Reward",
        "Final Consensus",
        "Consensus AUC",
    ]
    reproduction_error = float(
        np.max(
            np.abs(
                reference[comparison_metrics].to_numpy(dtype=float)
                - reproduced[comparison_metrics].to_numpy(dtype=float)
            )
        )
    )
    audit = {
        "schema": "hidden_heterogeneity_context_ablation_v1",
        "training_performed": False,
        "deterministic_eval_mode": True,
        "checkpoint": str(stage1.CHECKPOINT),
        "checkpoint_sha256": sha256(stage1.CHECKPOINT),
        "device": device,
        "scenario": SCENARIO.name,
        "agents": [10, 40, 100],
        "episodes_per_cell": 50,
        "context_modes": MODES,
        "raw_rows": len(raw),
        "elapsed_seconds": time.time() - started,
        "paired_protocol": "same environment seed; only encoder context differs",
        "confidence_interval": "paired 10000-draw percentile bootstrap",
        "stage1_real_history_reproduction_max_abs_error": reproduction_error,
        "files": {},
    }
    for item in (raw_path, summary_path, paired_path, figure_path):
        audit["files"][item.name] = {"path": str(item), "sha256": sha256(item)}
    audit_path = OUT / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
