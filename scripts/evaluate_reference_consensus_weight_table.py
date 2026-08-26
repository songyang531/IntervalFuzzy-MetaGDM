"""Create a deterministic, consolidated summary for the five consensus weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tmp" / "reference_interval_only_5000_seed20260507_20260822"
SOURCE = RUN / "source"
MAIN = (
    SOURCE / "revision_results" / "checkpoints"
    / "meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth"
)
PRIMARY = RUN / "reference_exact_missing" / "consensus_weight_reference" / "revision_results"
SHARDS = RUN / "reference_exact_weight_shards"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_weight(path: Path) -> float:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return float(payload["reward_parameters"]["consensus_improvement_weight"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--main-checkpoint", type=Path, default=MAIN)
    parser.add_argument("--primary-root", type=Path, default=PRIMARY)
    parser.add_argument("--shard-root", type=Path, default=SHARDS)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    candidates = [args.main_checkpoint.resolve()]
    candidates += sorted(args.primary_root.glob("checkpoints/*.pth"))
    candidates += sorted(args.shard_root.glob("**/checkpoints/*.pth"))
    checkpoints: dict[float, Path] = {}
    for path in candidates:
        if not path.is_file():
            continue
        weight = checkpoint_weight(path)
        if weight in checkpoints and sha256(checkpoints[weight]) != sha256(path):
            raise RuntimeError(f"Conflicting checkpoints for consensus weight {weight}")
        checkpoints[weight] = path
    expected = {30.0, 40.0, 50.0, 60.0, 70.0}
    if set(checkpoints) != expected:
        raise RuntimeError(f"Incomplete weight checkpoints: got {sorted(checkpoints)}, expected {sorted(expected)}")

    sys.path.insert(0, str(args.source.resolve()))
    from context_utils import append_context_step
    from revision_experiments import gini, load_meta_agent, prepare_env

    rows = []
    for weight in sorted(checkpoints):
        checkpoint = checkpoints[weight]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        agent = load_meta_agent(checkpoint, device=args.device)
        agent.encoder.eval()
        agent.actor.eval()
        agent.critic.eval()
        agent.critic_target.eval()
        for agents in (10, 40, 100):
            episode_rows = []
            for episode in range(args.episodes):
                env_seed = args.seed + episode * 997 + agents * 13
                env, state = prepare_env(
                    agents,
                    120,
                    0.90,
                    env_seed,
                    env_kwargs={
                        "consensus_decay": 5.0,
                        "cost_coefficient": 75.0,
                        "consensus_improvement_weight": weight,
                        "boundary_penalty_weight": 20.0,
                        "interval_half_width_range": (0.02, 0.08),
                    },
                )
                context = np.zeros((agents, agent.context_window, 3), dtype=np.float32)
                cumulative_cost = np.zeros(agents, dtype=float)
                boundary = 0
                total_reward = 0.0
                info = {"success": False, "consensus_level": env._calculate_consensus_level_from_opinions(env.opinions)}
                for _ in range(120):
                    action = agent.select_action(state, context, evaluate=True)
                    state, reward, done, info = env.step(action)
                    actual = np.asarray(info["actual_movements"])
                    cumulative_cost += env.calculate_adjustment_costs(actual)
                    lower, upper = env._interval_bounds()
                    boundary += int(np.sum((lower < env.safe_low) | (upper > env.safe_high)))
                    total_reward += float(np.mean(reward))
                    context = append_context_step(context, info["suggestions"], actual, reward)
                    if done:
                        break
                episode_rows.append(
                    {
                        "Success": int(bool(info.get("success", False))),
                        "Steps": env.current_step,
                        "Total Cost": float(np.sum(cumulative_cost)),
                        "Cost Gini": gini(cumulative_cost),
                        "Boundary Violations": boundary,
                        "Total Reward": total_reward,
                        "Final Consensus": float(info["consensus_level"]),
                        "Final Interval Dispersion": float(env._interval_dispersion()),
                        "Mean Half Width": float(np.mean(env.interval_half_widths)),
                    }
                )
            row = {"Model": f"consensus={weight:g}", "Agents": agents}
            row.update({column: float(np.mean([item[column] for item in episode_rows])) for column in episode_rows[0]})
            rows.append(row)

    output = args.primary_root / "consensus_weight_reference_interval5000_summary.csv"
    raw_output = args.primary_root / "consensus_weight_reference_interval5000_deterministic_audit_rows.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"weight": weight, "checkpoint": str(path), "checkpoint_sha256": sha256(path)}
            for weight, path in sorted(checkpoints.items())
        ]
    ).to_csv(raw_output, index=False, encoding="utf-8-sig")
    audit = {
        "schema": "reference_consensus_weight_table_v1",
        "weights": sorted(checkpoints),
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "deterministic_actions": True,
        "modules_eval": True,
        "training_updates": 0,
        "locked_test_opened": False,
        "output": str(output),
        "output_sha256": sha256(output),
        "checkpoint_audit_csv": str(raw_output),
        "checkpoint_audit_sha256": sha256(raw_output),
        "training_core_source_sha256": {
            name: sha256(args.source.resolve() / name)
            for name in ("revision_experiments.py", "agent.py", "model.py", "env.py", "replay_buffer.py")
        },
    }
    (args.primary_root / "consensus_weight_reference_interval5000_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
