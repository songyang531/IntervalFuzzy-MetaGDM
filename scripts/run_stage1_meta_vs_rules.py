"""Stage-1 frozen-checkpoint evaluation against fair observable-information rules.

This script does not train any model.  It evaluates the existing 5,000-episode
interval Meta-GDM checkpoint and four rule policies on paired deterministic
scenarios.  Rule policies only receive the same observed state and context as
the learned policy; they never read hidden environment personalities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "reference_interval_only_5000_seed20260507_20260822" / "source"
CHECKPOINT = (
    SOURCE
    / "revision_results"
    / "checkpoints"
    / "meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_"
      "ep5000_seed20260507_1df7dc5b_interval_only_5000.pth"
)
OUT_DIR = ROOT / "outputs" / "stage1_frozen_meta_vs_fair_rules_20260823"
BASE_SEED = 20260507
MAX_STEPS = 120
THRESHOLD = 0.90

sys.path.insert(0, str(SOURCE))
from context_utils import append_context_step  # noqa: E402
from env import OpinionDynamicsEnv  # noqa: E402
from revision_experiments import load_meta_agent  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    response_mode: str = "linear"
    interval_low: float = 0.02
    interval_high: float = 0.08
    execution_noise: float = 0.0
    observation_noise: float = 0.0
    feedback_dropout: float = 0.0
    switch_step: int = -1


SCENARIOS = {
    "iid_control": Scenario("iid_control"),
    "hidden_heterogeneity": Scenario(
        "hidden_heterogeneity", response_mode="heterogeneous"
    ),
    "mid_episode_switch": Scenario(
        "mid_episode_switch", response_mode="switch", switch_step=8
    ),
    "noisy_partial_feedback": Scenario(
        "noisy_partial_feedback",
        response_mode="linear",
        execution_noise=0.006,
        observation_noise=0.015,
        feedback_dropout=0.30,
    ),
    "interval_shift": Scenario(
        "interval_shift", response_mode="linear", interval_low=0.08, interval_high=0.12
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_name_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


class Stage1Env(OpinionDynamicsEnv):
    """Interval environment with prespecified hidden response perturbations."""

    def __init__(self, num_agents: int, seed: int, scenario: Scenario):
        super().__init__(
            num_agents=num_agents,
            max_steps=MAX_STEPS,
            consensus_threshold=THRESHOLD,
            seed=seed,
            interval_half_width_range=(scenario.interval_low, scenario.interval_high),
        )
        self.scenario = scenario
        self.stage_rng = np.random.default_rng(seed + 700_001)
        self.response_types = self.stage_rng.integers(0, 4, size=num_agents)
        self.response_thresholds = self.stage_rng.uniform(0.18, 0.55, size=num_agents)
        self.response_profiles = self.stage_rng.uniform(0.20, 0.95, size=num_agents)
        self.previous_suggestions = np.zeros(num_agents, dtype=float)

    def _response_movements(self, suggestions: np.ndarray) -> np.ndarray:
        suggestions = np.asarray(suggestions, dtype=float).reshape(self.num_agents)
        base_acceptance = 1.0 - self.personalities["stubbornness"]
        mode = self.scenario.response_mode

        if mode == "heterogeneous":
            magnitude = np.abs(suggestions)
            sign = np.sign(suggestions)
            movements = suggestions * base_acceptance
            threshold_mask = self.response_types == 1
            movements[threshold_mask] = (
                sign[threshold_mask]
                * magnitude[threshold_mask]
                * base_acceptance[threshold_mask]
                * (magnitude[threshold_mask] >= self.response_thresholds[threshold_mask])
            )
            sigmoid_mask = self.response_types == 2
            sigmoid_gain = 1.0 / (
                1.0
                + np.exp(
                    -10.0
                    * (magnitude[sigmoid_mask] - self.response_thresholds[sigmoid_mask])
                )
            )
            movements[sigmoid_mask] = (
                suggestions[sigmoid_mask]
                * base_acceptance[sigmoid_mask]
                * sigmoid_gain
            )
            delayed_mask = self.response_types == 3
            movements[delayed_mask] = (
                (0.35 * suggestions[delayed_mask] + 0.65 * self.previous_suggestions[delayed_mask])
                * base_acceptance[delayed_mask]
            )
        elif mode == "switch":
            before = self.current_step < self.scenario.switch_step
            if before:
                acceptance = 0.15 + 0.75 * self.response_profiles
            else:
                acceptance = 0.15 + 0.75 * (1.0 - self.response_profiles)
            movements = suggestions * base_acceptance * acceptance
        else:
            movements = suggestions * base_acceptance

        self.previous_suggestions = suggestions.copy()
        movements = movements * self.movement_scale
        if self.scenario.execution_noise > 0:
            movements += self.stage_rng.normal(
                0.0, self.scenario.execution_noise, size=self.num_agents
            )
        return np.clip(movements, -self.movement_scale, self.movement_scale)

    def observe_state(self, true_state: np.ndarray) -> np.ndarray:
        observed = np.asarray(true_state, dtype=float).copy()
        if self.scenario.observation_noise <= 0:
            return observed
        centres = np.clip(
            observed[:, 0]
            + self.stage_rng.normal(0.0, self.scenario.observation_noise, self.num_agents),
            observed[:, 1],
            1.0 - observed[:, 1],
        )
        observed[:, 0] = centres
        observed[:, 2] = centres - np.mean(centres)
        return observed

    def observed_feedback(self, movements: np.ndarray, rewards: np.ndarray):
        obs_move = np.asarray(movements, dtype=float).copy()
        obs_reward = np.asarray(rewards, dtype=float).copy()
        if self.scenario.feedback_dropout > 0:
            missing = self.stage_rng.random(self.num_agents) < self.scenario.feedback_dropout
            obs_move[missing] = 0.0
            obs_reward[missing] = 0.0
        return obs_move, obs_reward

    def step(self, suggestions):
        suggestions = np.asarray(suggestions, dtype=float).reshape(self.num_agents)
        old_opinions = self.opinions.copy()
        old_std = np.std(old_opinions)
        old_mean = np.mean(old_opinions)
        actual_movements = self._response_movements(suggestions)
        self.opinions = np.clip(
            self.opinions + actual_movements,
            self.interval_half_widths,
            1.0 - self.interval_half_widths,
        )
        new_std = np.std(self.opinions)
        new_mean = np.mean(self.opinions)
        consensus_level = self._calculate_consensus_level_from_opinions(self.opinions)
        rewards, reward_components = self._calculate_rewards_v8(
            old_opinions,
            self.opinions,
            old_std,
            new_std,
            old_mean,
            new_mean,
            actual_movements,
            consensus_level,
        )
        self.current_step += 1
        success = consensus_level >= self.consensus_threshold
        if self.scenario.response_mode == "switch":
            success = success and self.current_step > self.scenario.switch_step
        done = success or self.current_step >= self.max_steps
        info = {
            "suggestions": suggestions,
            "actual_movements": actual_movements,
            "interval_half_widths": self.interval_half_widths.copy(),
            "interval_lower": self._interval_bounds()[0],
            "interval_upper": self._interval_bounds()[1],
            "interval_dispersion": self._interval_dispersion(),
            "step": self.current_step,
            "consensus_level": consensus_level,
            "std": new_std,
            "success": success,
            "reward_components": reward_components,
        }
        return self._get_state(), rewards, done, info


def action_towards(centres: np.ndarray, target, gain=1.0, scale=None):
    target = np.asarray(target, dtype=float)
    if target.ndim == 0:
        target = np.full(centres.shape, float(target))
    action = gain * (target - centres) / 0.1
    if scale is not None:
        action *= np.asarray(scale, dtype=float)
    return np.clip(action, -1.0, 1.0)


def rule_action(name: str, state: np.ndarray, context: np.ndarray, rng) -> np.ndarray:
    centres = np.asarray(state[:, 0], dtype=float)
    if name == "Random":
        return rng.uniform(-1.0, 1.0, centres.size)
    if name == "Mean consensus":
        return action_towards(centres, np.mean(centres))
    if name == "Bounded confidence":
        targets = np.zeros_like(centres)
        global_mean = np.mean(centres)
        for idx, centre in enumerate(centres):
            neighbours = np.abs(centres - centre) <= 0.35
            targets[idx] = 0.7 * np.mean(centres[neighbours]) + 0.3 * global_mean
        return action_towards(centres, targets)
    if name == "Observable cost-aware":
        recent_rewards = context[:, :, 2] * 10.0
        recent_moves = np.abs(context[:, :, 1])
        burden = np.mean(np.maximum(-recent_rewards, 0.0), axis=1)
        response = np.mean(recent_moves, axis=1) / 0.1
        estimated_cost = 1.0 + burden / 5.0
        weights = 1.0 / estimated_cost
        target = np.average(centres, weights=weights)
        scale = np.clip((0.45 + response) / estimated_cost, 0.20, 1.0)
        return action_towards(centres, target, scale=scale)
    raise KeyError(name)


MODELS = [
    "Interval Meta-GDM",
    "Bounded confidence",
    "Mean consensus",
    "Observable cost-aware",
    "Random",
]


def run_episode(agent, model: str, scenario: Scenario, n_agents: int, episode: int):
    scenario_offset = stable_name_seed(scenario.name) % 1_000_000
    env_seed = BASE_SEED + scenario_offset + episode * 997 + n_agents * 13
    env = Stage1Env(n_agents, env_seed, scenario)
    state = env.observe_state(env._get_state())
    context = np.zeros((n_agents, 10, 3), dtype=np.float32)
    rule_rng = np.random.default_rng(env_seed + stable_name_seed(model))
    cumulative_costs = np.zeros(n_agents, dtype=float)
    boundary_violations = 0
    total_reward = 0.0
    consensus_trace = []
    success = False
    info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions)}

    for step in range(MAX_STEPS):
        if model == "Interval Meta-GDM":
            action = np.asarray(agent.select_action(state, context, evaluate=True)).reshape(n_agents)
        else:
            action = rule_action(model, state, context, rule_rng)
        next_true_state, rewards, done, info = env.step(action)
        cumulative_costs += env.calculate_adjustment_costs(info["actual_movements"])
        lower, upper = env._interval_bounds()
        boundary_violations += int(np.sum((lower < env.safe_low) | (upper > env.safe_high)))
        total_reward += float(np.mean(rewards))
        consensus_trace.append(float(info["consensus_level"]))
        obs_move, obs_reward = env.observed_feedback(info["actual_movements"], rewards)
        context = append_context_step(context, action, obs_move, obs_reward)
        state = env.observe_state(next_true_state)
        if done:
            success = bool(info["success"])
            break

    steps = step + 1
    padded_trace = consensus_trace + [consensus_trace[-1]] * (MAX_STEPS - len(consensus_trace))
    recovery = np.nan
    if scenario.response_mode == "switch" and success:
        recovery = max(0, steps - scenario.switch_step)
    return {
        "Scenario": scenario.name,
        "Model": model,
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
        "Consensus AUC": float(np.mean(padded_trace)),
        "Switch Recovery Steps": recovery,
        "Mean Half Width": float(np.mean(env.interval_half_widths)),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "Success": "mean",
        "Steps": "mean",
        "Total Cost": "mean",
        "Cost Gini": "mean",
        "Boundary Violations": "mean",
        "Total Reward": "mean",
        "Final Consensus": "mean",
        "Consensus AUC": "mean",
        "Switch Recovery Steps": "mean",
        "Mean Half Width": "mean",
    }
    return raw.groupby(["Scenario", "Model", "Agents"], as_index=False).agg(metrics)


def meta_vs_best_rule(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, agents), frame in summary.groupby(["Scenario", "Agents"]):
        meta = frame[frame["Model"] == "Interval Meta-GDM"].iloc[0]
        rules = frame[frame["Model"] != "Interval Meta-GDM"]
        # Compare secondary metrics only with rules having comparable completion.
        # The complete unfiltered summary remains available in stage1_summary.csv.
        eligible = rules[rules["Success"] >= max(0.0, meta["Success"] - 0.05)]
        if eligible.empty:
            eligible = rules
        rows.append(
            {
                "Scenario": scenario,
                "Agents": agents,
                "Meta Success": meta["Success"],
                "Best Rule Success": rules["Success"].max(),
                "Success Gain": meta["Success"] - rules["Success"].max(),
                "Meta Steps": meta["Steps"],
                "Best Success-Matched Rule Steps": eligible["Steps"].min(),
                "Steps Saved": eligible["Steps"].min() - meta["Steps"],
                "Meta Cost": meta["Total Cost"],
                "Best Success-Matched Rule Cost": eligible["Total Cost"].min(),
                "Cost Saved": eligible["Total Cost"].min() - meta["Total Cost"],
                "Meta Gini": meta["Cost Gini"],
                "Best Success-Matched Rule Gini": eligible["Cost Gini"].min(),
                "Gini Gain": eligible["Cost Gini"].min() - meta["Cost Gini"],
                "Meta Reward": meta["Total Reward"],
                "Best Success-Matched Rule Reward": eligible["Total Reward"].max(),
                "Reward Gain": meta["Total Reward"] - eligible["Total Reward"].max(),
                "Meta Consensus AUC": meta["Consensus AUC"],
                "Best Success-Matched Rule Consensus AUC": eligible["Consensus AUC"].max(),
                "AUC Gain": meta["Consensus AUC"] - eligible["Consensus AUC"].max(),
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    scenario_order = list(SCENARIOS)
    metrics = ["Success", "Steps", "Total Cost", "Total Reward"]
    titles = ["成功率", "平均步数", "平均总成本", "平均回合奖励"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), constrained_layout=True)
    colors = {
        "Interval Meta-GDM": "#2F6690",
        "Bounded confidence": "#6A994E",
        "Mean consensus": "#F4A261",
        "Observable cost-aware": "#9B5DE5",
        "Random": "#999999",
    }
    for row_idx, agents in enumerate((10, 40, 100)):
        for col_idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row_idx, col_idx]
            for model in MODELS:
                frame = summary[
                    (summary["Agents"] == agents) & (summary["Model"] == model)
                ].set_index("Scenario").reindex(scenario_order)
                ax.plot(
                    range(len(scenario_order)),
                    frame[metric],
                    marker="o",
                    linewidth=1.8,
                    label=model,
                    color=colors[model],
                )
            ax.set_title(f"N={agents} · {title}")
            ax.set_xticks(range(len(scenario_order)))
            ax.set_xticklabels(
                ["IID", "隐藏异质", "中途切换", "噪声缺失", "区间迁移"],
                rotation=18,
                ha="right",
                fontsize=8,
            )
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("冻结5000回合区间Meta-GDM与公平规则基线：第一阶段评估", fontsize=15)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agents", default="10,40,100")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    selected_agents = [int(value) for value in args.agents.split(",") if value]
    selected_scenarios = [SCENARIOS[name] for name in args.scenarios.split(",") if name]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    agent = load_meta_agent(CHECKPOINT, device=device)
    # Deterministic action selection does not disable Transformer dropout.
    for module in (agent.encoder, agent.actor, agent.critic, agent.critic_target):
        module.eval()
    started = time.time()
    records = []
    total = len(selected_scenarios) * len(selected_agents) * len(MODELS) * args.episodes
    completed = 0
    for scenario in selected_scenarios:
        for agents in selected_agents:
            for model in MODELS:
                for episode in range(args.episodes):
                    records.append(run_episode(agent, model, scenario, agents, episode))
                    completed += 1
                print(
                    f"[{completed:>5}/{total}] {scenario.name:>24} N={agents:<3} {model}",
                    flush=True,
                )

    raw = pd.DataFrame(records)
    summary = summarize(raw)
    comparison = meta_vs_best_rule(summary)
    raw_path = output_dir / "stage1_raw.csv"
    summary_path = output_dir / "stage1_summary.csv"
    comparison_path = output_dir / "stage1_meta_vs_best_rule.csv"
    figure_path = output_dir / "stage1_overview.png"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    make_figure(summary, figure_path)
    audit = {
        "schema": "stage1_frozen_meta_vs_fair_rules_v1",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_training_episodes": 5000,
        "training_performed": False,
        "deterministic_eval_mode": True,
        "device": device,
        "base_seed": BASE_SEED,
        "episodes_per_cell": args.episodes,
        "agents": selected_agents,
        "models": MODELS,
        "scenarios": [asdict(value) for value in selected_scenarios],
        "fairness_protocol": (
            "All policies receive the same observed state and context. Rule policies never "
            "read hidden stubbornness or cost_sensitivity. Each policy/scenario/cell uses "
            "the same deterministic environment seed."
        ),
        "files": {},
    }
    for item in (raw_path, summary_path, comparison_path, figure_path):
        audit["files"][item.name] = {"path": str(item), "sha256": sha256(item)}
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
