"""Re-evaluate consensus-decay checkpoints for the reference Table 10 analogue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tmp" / "reference_interval_only_5000_seed20260507_20260822"
SOURCE = RUN / "source"
MISSING = RUN / "reference_exact_missing" / "consensus_decay_reference" / "revision_results"
MAIN_CHECKPOINT = (
    SOURCE / "revision_results" / "checkpoints"
    / "meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth"
)
OUT = RUN / "reference_exact_tables" / "decay_dynamics_source.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--missing-root", type=Path, default=MISSING)
    parser.add_argument("--main-checkpoint", type=Path, default=MAIN_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    audit_candidates = sorted(args.missing_root.glob("consensus_decay_reference_interval5000*_audit.json"))
    if len(audit_candidates) != 1:
        raise RuntimeError(f"Expected one completed decay audit, got {audit_candidates}")
    source_audit = json.loads(audit_candidates[0].read_text(encoding="utf-8"))
    checkpoints = {
        3.0: Path(source_audit["checkpoint_audit"]["consensus_decay=3"]["path"]),
        5.0: args.main_checkpoint.resolve(),
        7.0: Path(source_audit["checkpoint_audit"]["consensus_decay=7"]["path"]),
    }
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from context_utils import append_context_step
    from revision_experiments import gini, load_meta_agent, prepare_env

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    rows: list[dict] = []
    model_hash_before: dict[str, str] = {}
    model_hash_after: dict[str, str] = {}
    for decay, checkpoint in checkpoints.items():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        agent = load_meta_agent(checkpoint, device=device)
        agent.encoder.eval()
        agent.actor.eval()
        agent.critic.eval()
        agent.critic_target.eval()
        model_hash_before[str(decay)] = sha256(checkpoint)
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
                        "consensus_decay": decay,
                        "cost_coefficient": 75.0,
                        "consensus_improvement_weight": 50.0,
                        "boundary_penalty_weight": 20.0,
                        "interval_half_width_range": (0.02, 0.08),
                    },
                )
                initial_consensus = float(env._calculate_consensus_level_from_opinions(env.opinions))
                initial_std = float(np.std(env.opinions))
                context = np.zeros((agents, agent.context_window, 3), dtype=np.float32)
                cumulative_cost = np.zeros(agents, dtype=float)
                boundary_violations = 0
                total_reward = 0.0
                info = {"success": False, "consensus_level": initial_consensus}
                for step in range(120):
                    action = agent.select_action(state, context, evaluate=True)
                    state, reward, done, info = env.step(action)
                    actual = np.asarray(info["actual_movements"])
                    cumulative_cost += env.calculate_adjustment_costs(actual)
                    lower, upper = env._interval_bounds()
                    boundary_violations += int(np.sum((lower < env.safe_low) | (upper > env.safe_high)))
                    total_reward += float(np.mean(reward))
                    context = append_context_step(context, info["suggestions"], actual, reward)
                    if done:
                        break
                episode_rows.append(
                    {
                        "success": int(bool(info.get("success", False))),
                        "steps": int(env.current_step),
                        "total_cost": float(np.sum(cumulative_cost)),
                        "cost_gini": float(gini(cumulative_cost)),
                        "boundary_violations": boundary_violations,
                        "total_reward": total_reward,
                        "initial_consensus": initial_consensus,
                        "final_consensus": float(info["consensus_level"]),
                        "initial_preference_std": initial_std,
                        "final_preference_std": float(np.std(env.opinions)),
                    }
                )
            keys = list(episode_rows[0])
            row = {"consensus_decay": decay, "agents": agents}
            row.update({key: float(np.mean([item[key] for item in episode_rows])) for key in keys})
            rows.append(row)
        model_hash_after[str(decay)] = sha256(checkpoint)
        if model_hash_before[str(decay)] != model_hash_after[str(decay)]:
            raise RuntimeError(f"Checkpoint mutated during evaluation: decay={decay}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema": "reference_consensus_decay_table_v1",
        "source_audit": str(audit_candidates[0]),
        "source_audit_sha256": sha256(audit_candidates[0]),
        "checkpoints": {
            str(decay): {"path": str(path), "sha256": model_hash_before[str(decay)]}
            for decay, path in checkpoints.items()
        },
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "deterministic_actions": True,
        "modules_eval": True,
        "training_updates": 0,
        "locked_test_opened": False,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
