"""Independent checks of the released paired evaluation records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, label: str, atol=1e-9):
    if not np.isclose(float(left), float(right), atol=atol, rtol=0):
        raise AssertionError(f"{label}: {left} != {right}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if json.loads((run / "status.json").read_text(encoding="utf-8"))["state"] != "completed":
        raise AssertionError("Evaluation is not complete")
    for relative, expected in manifest["source_sha256"].items():
        if sha256(repo / relative) != expected:
            raise AssertionError(f"Source hash changed: {relative}")
    checkpoint = repo / "checkpoints/meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth"
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise AssertionError("Checkpoint hash changed")
    if manifest["model_state_hash_before"] != manifest["model_state_hash_after"]:
        raise AssertionError("Frozen model state changed")
    episodes = pd.read_csv(run / "episodes.csv")
    traces = pd.read_csv(run / "trajectories.csv")
    summary = pd.read_csv(run / "summary.csv")
    key = ["scenario", "N", "episode", "method"]
    expected_rows = (len(manifest["scenarios"]) * len(manifest["group_sizes"])
                     * manifest["evaluation_cases_per_cell"] * len(manifest["methods"]))
    if len(episodes) != expected_rows or episodes.duplicated(key).any():
        raise AssertionError("Missing or duplicated evaluation cases")
    expected_methods = set(manifest["methods"])
    for (scenario, n, episode), group in episodes.groupby(key[:3]):
        if set(group.method) != expected_methods or group.initial_sha256.nunique() != 1:
            raise AssertionError(f"Unpaired initial state: {(scenario, n, episode)}")
    trace_groups = {k: group for k, group in traces.groupby(key)}
    failure_count = 0
    unsafe_count = 0
    clipping_count = 0
    for record in episodes.itertuples(index=False):
        ident = (record.scenario, record.N, record.episode, record.method)
        trajectory = trace_groups.get(ident)
        if trajectory is None or len(trajectory) != record.steps:
            raise AssertionError(f"Incorrect number of rounds: {ident}")
        close(trajectory.cost.sum(), record.total_cost, f"cost {ident}")
        close(trajectory.mean_reward.sum(), record.total_reward, f"reward {ident}")
        close(trajectory.consensus.iloc[-1], record.final_consensus, f"final consensus {ident}")
        padded = list(trajectory.consensus) + [trajectory.consensus.iloc[-1]] * (120 - record.steps)
        close(np.mean(padded), record.consensus_auc, f"consensus AUC {ident}")
        member_costs = np.asarray(json.loads(record.member_cumulative_costs_json), dtype=float)
        if len(member_costs) != record.N or np.min(member_costs) < -1e-9:
            raise AssertionError(f"Invalid member costs: {ident}")
        close(member_costs.sum(), record.total_cost, f"member cost sum {ident}")
        gini = (0.0 if member_costs.sum() <= 1e-15 else
                np.abs(member_costs[:, None] - member_costs).sum()
                / (2 * len(member_costs) * member_costs.sum()))
        close(gini, record.cost_gini, f"standard cumulative Gini {ident}")
        if record.success not in (0, 1) or (record.success == 0 and record.steps != 120):
            raise AssertionError(f"Invalid completion label: {ident}")
        failure_count += int(record.success == 0)
        unsafe_count += int(record.unsafe_member_steps)
        clipping_count += int(record.boundary_clipping_count)
    groups = episodes.groupby(["scenario", "N", "method"])
    if len(summary) != len(groups):
        raise AssertionError("Incomplete summary")
    for record in summary.itertuples(index=False):
        group = groups.get_group((record.scenario, record.N, record.method))
        if len(group) != record.cases or int(group.success.sum()) != record.successes:
            raise AssertionError("Incorrect summary counts")
        for raw, published in ((group.success.mean(), record.success_rate),
                               (group.steps.mean(), record.mean_rounds),
                               (group.total_cost.mean(), record.mean_physical_cost),
                               (group.cost_gini.mean(), record.mean_cumulative_cost_gini),
                               (group.total_reward.mean(), record.mean_reward)):
            close(raw, published, "summary mean")
    audit = {
        "status": "passed", "cases": len(episodes), "groups": len(summary),
        "paired_initial_cases": len(episodes) // len(expected_methods),
        "failures_included": failure_count,
        "unsafe_member_steps": unsafe_count,
        "physical_boundary_clipping_count": clipping_count,
        "trajectory_rows": len(traces),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checked": ["source and checkpoint hashes", "frozen model state",
                    "scenario pairing", "all failures", "trajectory sums and AUC",
                    "member-level physical cost and standard cumulative Gini",
                    "all summary means and safety counts"],
    }
    (run / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
