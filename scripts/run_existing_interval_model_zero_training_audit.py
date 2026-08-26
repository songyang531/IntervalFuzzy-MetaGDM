"""Run a read-only interval audit with the existing 5,000-episode checkpoint.

No optimizer/update method is called.  The audit contains paired interval-width
stress tests and real/zero/shuffled-history causal evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch

from zero_training_interval_metrics import (
    interval_information_distortion,
    interval_snapshot_metrics,
)


WIDTH_REGIMES: Dict[str, Tuple[float, float]] = {
    "crisp_limit": (0.0, 0.0),
    "narrow": (0.01, 0.04),
    "trained_support": (0.02, 0.08),
    "wide_ood": (0.08, 0.16),
}
MEMBER_COUNTS = (10, 40, 100)
HISTORY_MODES = ("real", "zero", "shuffled")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(agent) -> str:
    digest = hashlib.sha256()
    for module_name in ("encoder", "actor", "critic", "critic_target"):
        module = getattr(agent, module_name)
        for key, value in sorted(module.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(module_name.encode())
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def set_eval_mode(agent) -> None:
    for name in ("encoder", "actor", "critic", "critic_target"):
        getattr(agent, name).eval()


def load_agent_without_training_runner(agent_module, checkpoint: Path, device: str):
    """Strictly load the frozen agent without importing tqdm/training entrypoints."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = agent_module.MetaAgent(
        state_dim=int(payload.get("state_dim", 5)),
        action_dim=int(payload.get("action_dim", 1)),
        z_dim=int(payload.get("z_dim", 8)),
        hidden_dim=256,
        context_window=int(payload.get("context_window", 10)),
        aggregation=payload.get("aggregation", "social_attention"),
        num_heads=int(payload.get("num_heads", 4)),
        context_mode=payload.get("context_mode", "learned"),
        device=device,
    )
    encoder_state = payload.get("encoder_state_dict", payload.get("encoder"))
    actor_state = payload.get("actor_state_dict", payload.get("actor"))
    critic_state = payload.get("critic_state_dict", payload.get("critic"))
    if encoder_state is None or actor_state is None or critic_state is None:
        raise KeyError("checkpoint lacks encoder, actor, or critic state")
    agent.encoder.load_state_dict(encoder_state, strict=True)
    agent.actor.load_state_dict(actor_state, strict=True)
    agent.critic.load_state_dict(critic_state, strict=True)
    target_state = payload.get("critic_target_state_dict", payload.get("critic_target"))
    if target_state is not None:
        agent.critic_target.load_state_dict(target_state, strict=True)
    else:
        agent.critic_target.load_state_dict(agent.critic.state_dict(), strict=True)
    return agent


def episode_condition_pairs() -> Iterable[Tuple[str, str]]:
    for width_name in WIDTH_REGIMES:
        yield width_name, "real"
    yield "trained_support", "zero"
    yield "trained_support", "shuffled"


def run_episode(
    *,
    agent,
    env_class,
    append_context_step,
    num_agents: int,
    episode: int,
    seed_base: int,
    width_name: str,
    width_range: Tuple[float, float],
    history_mode: str,
    max_steps: int,
    threshold: float,
):
    env_seed = int(seed_base + episode * 997 + num_agents * 13)
    env = env_class(
        num_agents=num_agents,
        max_steps=max_steps,
        consensus_threshold=threshold,
        seed=env_seed,
        interval_half_width_range=width_range,
    )
    state = env._get_state()
    initial_centres = env.opinions.copy()
    initial_widths = env.interval_half_widths.copy()
    initial_metrics = interval_snapshot_metrics(
        initial_centres, initial_widths, decay=env.consensus_decay, threshold=threshold
    )
    context = np.zeros((num_agents, agent.context_window, 3), dtype=np.float32)
    permutation_rng = np.random.default_rng(env_seed + 7_919_113)
    permutation = permutation_rng.permutation(num_agents)
    cumulative_costs = np.zeros(num_agents, dtype=np.float64)
    total_reward = 0.0
    trajectory_rows = []
    info = {
        "success": False,
        "consensus_level": env._calculate_consensus_level_from_opinions(env.opinions),
    }

    for step in range(max_steps):
        if history_mode == "real":
            policy_context = context
        elif history_mode == "zero":
            policy_context = np.zeros_like(context)
        elif history_mode == "shuffled":
            policy_context = context[permutation]
        else:
            raise ValueError(f"unknown history mode: {history_mode}")

        action = agent.select_action(state, policy_context, evaluate=True)
        next_state, reward, done, info = env.step(action)
        step_costs = env.calculate_adjustment_costs(info["actual_movements"])
        cumulative_costs += step_costs
        total_reward += float(np.mean(reward))
        snapshot = interval_snapshot_metrics(
            env.opinions, env.interval_half_widths,
            decay=env.consensus_decay, threshold=threshold,
        )
        trajectory_rows.append({
            "width_regime": width_name,
            "history_mode": history_mode,
            "Agents": num_agents,
            "Episode": episode,
            "env_seed": env_seed,
            "step": step + 1,
            "success": int(info["success"]),
            "environment_consensus": float(info["consensus_level"]),
            "endpoint_consensus": snapshot["endpoint_consensus"],
            "center_consensus": snapshot["center_consensus"],
            "width_consensus": snapshot["width_consensus"],
            "lower_endpoint_consensus": snapshot["lower_endpoint_consensus"],
            "upper_endpoint_consensus": snapshot["upper_endpoint_consensus"],
            "common_overlap_ratio": snapshot["common_overlap_ratio"],
            "mean_abs_action": float(np.mean(np.abs(action))),
            "mean_abs_actual_movement": float(np.mean(np.abs(info["actual_movements"]))),
            "step_cost": float(np.sum(step_costs)),
            "mean_reward": float(np.mean(reward)),
        })
        context = append_context_step(
            context, info["suggestions"], info["actual_movements"], reward
        )
        state = next_state
        if done:
            break

    steps = int(info.get("step", max_steps))
    final_metrics = interval_snapshot_metrics(
        env.opinions, env.interval_half_widths,
        decay=env.consensus_decay, threshold=threshold,
    )
    distortion = interval_information_distortion(
        initial_centres, initial_widths, env.opinions, env.interval_half_widths
    )
    success = int(bool(info["success"]))
    episode_row = {
        "width_regime": width_name,
        "width_low": width_range[0],
        "width_high": width_range[1],
        "history_mode": history_mode,
        "Agents": num_agents,
        "Episode": episode,
        "env_seed": env_seed,
        "Success": success,
        "Steps": steps,
        "Success_AUC": success * (max_steps - steps + 1) / (max_steps + 1),
        "Total_Cost": float(np.sum(cumulative_costs)),
        "Total_Reward": total_reward,
        "Final_Environment_Consensus": float(info["consensus_level"]),
    }
    episode_row.update({f"Initial_{key}": value for key, value in initial_metrics.items()})
    episode_row.update({f"Final_{key}": value for key, value in final_metrics.items()})
    episode_row.update(distortion)
    return episode_row, trajectory_rows


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "Success": ["mean", "sum"],
        "Success_AUC": ["mean"],
        "Steps": ["mean"],
        "Total_Cost": ["mean"],
        "Total_Reward": ["mean"],
        "Final_Environment_Consensus": ["mean"],
        "Final_center_consensus": ["mean"],
        "Final_width_consensus": ["mean"],
        "Final_lower_endpoint_consensus": ["mean"],
        "Final_upper_endpoint_consensus": ["mean"],
        "Final_endpoint_consensus": ["mean"],
        "Final_common_overlap_ratio": ["mean"],
        "Final_mean_pairwise_hausdorff": ["mean"],
        "Final_max_pairwise_hausdorff": ["mean"],
        "Initial_width_ceiling_reaches_threshold": ["mean"],
        "mean_hausdorff_distortion": ["mean"],
        "mean_center_shift": ["mean"],
        "mean_half_width_shift": ["mean"],
        "translation_identity_max_error": ["max"],
    }
    result = episodes.groupby(
        ["width_regime", "history_mode", "Agents"], sort=False
    ).agg(metrics)
    result.columns = ["_".join(name) for name in result.columns]
    return result.reset_index()


def history_paired_differences(episodes: pd.DataFrame) -> pd.DataFrame:
    source = episodes[episodes["width_regime"] == "trained_support"].copy()
    keys = ["Agents", "Episode", "env_seed"]
    real = source[source["history_mode"] == "real"].set_index(keys)
    rows = []
    for mode in ("zero", "shuffled"):
        other = source[source["history_mode"] == mode].set_index(keys)
        if set(real.index) != set(other.index):
            raise RuntimeError(f"history mode {mode} does not share the real-history panel")
        for key in sorted(real.index):
            a = real.loc[key]
            b = other.loc[key]
            rows.append({
                "Agents": key[0], "Episode": key[1], "env_seed": key[2],
                "comparison": f"real_minus_{mode}",
                "success_difference": float(a["Success"] - b["Success"]),
                "auc_difference": float(a["Success_AUC"] - b["Success_AUC"]),
                "steps_improvement": float(b["Steps"] - a["Steps"]),
                "cost_difference": float(a["Total_Cost"] - b["Total_Cost"]),
                "endpoint_consensus_difference": float(
                    a["Final_endpoint_consensus"] - b["Final_endpoint_consensus"]
                ),
            })
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        rendered = frame.copy()
        for column in rendered.columns:
            if pd.api.types.is_float_dtype(rendered[column]):
                rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
        headers = [str(column) for column in rendered.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rendered.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    real = summary[summary["history_mode"] == "real"]
    history_summary = paired.groupby(["comparison", "Agents"], sort=False).mean(numeric_only=True).reset_index()
    lines = [
        "# 现有5000回合区间模型：零训练冻结策略审计",
        "",
        "本审计没有调用任何训练或更新函数。区间宽度压力测试和历史消融均使用冻结检查点。",
        "",
        "## 结构边界",
        "",
        "当前模型每位成员只有一个标量区间，不包含备选项两两比较矩阵，因此文献中的区间互反性与加性/乘性传递一致性不可定义。本报告不生成伪一致性指标。",
        "",
        "## 区间宽度压力测试（真实历史）",
        "",
        markdown_table(real),
        "",
        "## 历史因果消融（真实历史减去对照）",
        "",
        markdown_table(history_summary),
        "",
        "## 指标说明",
        "",
        "- endpoint_consensus 与现有环境的区间端点RMS口径一致。",
        "- width_limited_consensus_ceiling 是所有中心完全相等时，固定宽度异质性允许达到的最高共识。",
        "- Hausdorff失真比较每位成员初始与终止区间上下端点。当前动作只平移区间，因此它应严格等于中心移动量。",
        "- shuffled只打乱成员历史归属；zero将上下文恒置零；两者都不更新模型。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.finalize_only:
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(source_root))
    agent_module = importlib.import_module("agent")
    env_module = importlib.import_module("env")
    context_module = importlib.import_module("context_utils")

    agent = load_agent_without_training_runner(agent_module, checkpoint, args.device)
    set_eval_mode(agent)
    checkpoint_hash_before = sha256_file(checkpoint)
    model_hash_before = model_state_sha256(agent)
    if args.finalize_only:
        episodes_path = output / "episodes.csv"
        trajectories_path = output / "trajectories.csv"
        if not episodes_path.is_file() or not trajectories_path.is_file():
            raise FileNotFoundError("finalize-only requires existing episodes.csv and trajectories.csv")
        episodes = pd.read_csv(episodes_path)
        trajectories = pd.read_csv(trajectories_path)
    else:
        episode_rows = []
        trajectory_rows = []
        with torch.inference_mode():
            total = len(tuple(episode_condition_pairs())) * len(MEMBER_COUNTS) * args.episodes
            completed = 0
            for width_name, history_mode in episode_condition_pairs():
                width_range = WIDTH_REGIMES[width_name]
                for num_agents in MEMBER_COUNTS:
                    for episode in range(args.episodes):
                        episode_row, step_rows = run_episode(
                            agent=agent,
                            env_class=env_module.OpinionDynamicsEnv,
                            append_context_step=context_module.append_context_step,
                            num_agents=num_agents,
                            episode=episode,
                            seed_base=args.seed,
                            width_name=width_name,
                            width_range=width_range,
                            history_mode=history_mode,
                            max_steps=args.max_steps,
                            threshold=args.threshold,
                        )
                        episode_rows.append(episode_row)
                        trajectory_rows.extend(step_rows)
                        completed += 1
                        if completed % 50 == 0 or completed == total:
                            print(f"completed {completed}/{total}", flush=True)
        episodes = pd.DataFrame(episode_rows)
        trajectories = pd.DataFrame(trajectory_rows)

    checkpoint_hash_after = sha256_file(checkpoint)
    model_hash_after = model_state_sha256(agent)
    if checkpoint_hash_before != checkpoint_hash_after or model_hash_before != model_hash_after:
        raise RuntimeError("checkpoint or model parameters changed during frozen evaluation")

    summary = summarize(episodes)
    paired = history_paired_differences(episodes)
    episodes.to_csv(output / "episodes.csv", index=False)
    trajectories.to_csv(output / "trajectories.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "history_paired_differences.csv", index=False)
    write_report(summary, paired, output)
    audit = {
        "protocol": "existing_interval_checkpoint_zero_training_audit_v1",
        "source_root": str(source_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "training_or_update_calls": 0,
        "all_modules_eval_mode": True,
        "torch_inference_mode": True,
        "episodes_per_condition_and_size": args.episodes,
        "total_episodes": len(episodes),
        "total_transitions": len(trajectories),
        "member_counts": list(MEMBER_COUNTS),
        "width_regimes": {key: list(value) for key, value in WIDTH_REGIMES.items()},
        "history_modes": list(HISTORY_MODES),
        "pairwise_interval_preference_matrix_available": False,
        "reciprocity_and_additive_consistency_status": "not_defined_for_scalar_member_intervals",
        "random_seed": args.seed,
        "max_steps": args.max_steps,
        "consensus_threshold": args.threshold,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "finalized_from_existing_complete_csvs": bool(args.finalize_only),
        "output_files": {},
    }
    for name in (
        "episodes.csv", "trajectories.csv", "summary.csv",
        "history_paired_differences.csv", "REPORT.md",
    ):
        audit["output_files"][name] = sha256_file(output / name)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "episodes": len(episodes), "transitions": len(trajectories)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
