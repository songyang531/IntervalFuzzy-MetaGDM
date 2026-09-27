"""Re-run the paired literature-informed controls without retraining Meta-GDM.

The published papers are not claimed to be reproduced in their native task.
FZZ-MCC is a selected framework instantiation; Guo and RTS are task adaptations.
The optional gain compensation is our common execution adapter, not theirs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fzz_mcc
import robust_cost
import run_pilot as pilot
from run_robustness import FeedbackProxy


METHODS = (
    "Meta-GDM",
    "Guo2024 direct adaptation",
    "Guo2024 + observed-gain adapter",
    "FZZ2023 endpoint-L1 instantiation",
    "FZZ2023 + observed-gain adapter",
    "RTS2024 restricted adaptation",
    "RTS2024 + observed-gain adapter",
)
SCENARIOS = ("iid_control", "hidden_heterogeneity")
SIZES = (10, 40, 100)
BASE_SEED = 20260926
EXPECTED_CHECKPOINT_SHA256 = (
    "7127330b0a3d1ec408f27b7617317c3ea13b87fba3d6542534f05fa490623d1e"
)


class ControllerProxy:
    def __init__(self, constructor, feedback: bool):
        self.controller = constructor(feedback=feedback)

    def select_action(self, state, context, evaluate=True):
        action = np.asarray(self.controller.action(state.copy(), context.copy()), dtype=float)
        if action.shape != (len(state),) or not np.isfinite(action).all():
            raise AssertionError("Invalid adapted-controller action")
        if np.any(np.abs(action) > 1 + 1e-9):
            raise AssertionError("Adapted-controller action exceeds [-1, 1]")
        return action


def evaluate_one(model, name: str, scenario: str, n: int, episode: int):
    if name == "Meta-GDM":
        row, trace = pilot.run_episode(model, "Meta-GDM", scenario, n, episode, BASE_SEED)
    elif name == "Guo2024 direct adaptation":
        row, trace = pilot.run_episode(None, "Guo2024-adapted", scenario, n, episode, BASE_SEED)
    else:
        if name == "Guo2024 + observed-gain adapter":
            proxy = FeedbackProxy("guo", feedback=True)
        elif name.startswith("FZZ2023"):
            proxy = ControllerProxy(fzz_mcc.Controller, "adapter" in name)
        elif name.startswith("RTS2024"):
            proxy = ControllerProxy(robust_cost.Controller, "adapter" in name)
        else:
            raise ValueError(name)
        row, trace = pilot.run_episode(proxy, "Meta-GDM", scenario, n, episode, BASE_SEED)
        stats = proxy.controller.stats if hasattr(proxy, "controller") else proxy.stats
        row.update({"guard_" + key: value for key, value in stats.items()})
    row["method"] = name
    for step in trace:
        step["method"] = name
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes", default=20, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 20:
        raise ValueError("The documented paired test pool has 20 cases per cell")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    actual_hash = pilot.sha256_file(pilot.CHECKPOINT)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError("The released checkpoint is not the audited paper checkpoint")
    model = pilot.load_agent_without_training_runner(
        pilot.agent_module, pilot.CHECKPOINT, args.device
    )
    pilot.set_eval_mode(model)
    model_hash_before = pilot.model_state_sha256(model)
    sources = [Path(__file__), Path(pilot.__file__),
               Path(FeedbackProxy.__module__.replace(".", "/") + ".py"),
               Path(fzz_mcc.__file__), Path(robust_cost.__file__),
               Path(pilot.__file__).with_name("interval_qp.py"),
               pilot.RELEASE / "code/env.py", pilot.RELEASE / "code/agent.py",
               pilot.RELEASE / "scripts/run_stage1_meta_vs_rules.py"]
    sources = [path if path.is_absolute() else Path(__file__).with_name(path.name)
               for path in sources]
    source_hashes = {str(path.relative_to(pilot.RELEASE)): pilot.sha256_file(path)
                     for path in sources}
    manifest = {
        "design": "exploratory paired evaluation; no Meta-GDM retraining",
        "base_seed": BASE_SEED, "training_seed": 20260507,
        "evaluation_cases_per_cell": args.episodes,
        "scenarios": SCENARIOS, "group_sizes": SIZES, "methods": METHODS,
        "checkpoint_sha256": actual_hash,
        "model_state_hash_before": model_hash_before,
        "source_sha256": source_hashes,
        "success_threshold": 0.9, "horizon": 120,
        "cost": "nonlinear physical cost of realized response movements",
        "fairness": "standard Gini of per-member cumulative physical cost",
        "method_boundary": (
            "Guo and RTS are restricted common-task adaptations; FZZ is a framework "
            "instantiation. Observed-gain compensation is an external shared adapter, "
            "not an element of any cited paper. Shen payment RL is not tested."
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows, trajectories = [], []
    started = time.time()
    for scenario in SCENARIOS:
        for n in SIZES:
            for episode in range(args.episodes):
                fingerprints = set()
                for name in METHODS:
                    row, trace = evaluate_one(model, name, scenario, n, episode)
                    fingerprints.add(row["initial_sha256"])
                    rows.append(row)
                    trajectories.extend(trace)
                if len(fingerprints) != 1:
                    raise AssertionError("Initial scenarios differ across methods")
                if (episode + 1) % 5 == 0 or episode + 1 == args.episodes:
                    pd.DataFrame(rows).to_csv(args.out / "episodes_partial.csv", index=False)
                    status = {"state": "running", "scenario": scenario, "N": n,
                              "episode": episode + 1, "records": len(rows),
                              "elapsed_seconds": time.time() - started}
                    (args.out / "status.json").write_text(
                        json.dumps(status, indent=2), encoding="utf-8"
                    )
                    print(json.dumps(status), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "episodes.csv", index=False)
    pd.DataFrame(trajectories).to_csv(args.out / "trajectories.csv", index=False)
    summary = frame.groupby(["scenario", "N", "method"], sort=False).agg(
        cases=("success", "count"), successes=("success", "sum"),
        success_rate=("success", "mean"), mean_rounds=("steps", "mean"),
        mean_physical_cost=("total_cost", "mean"),
        mean_cumulative_cost_gini=("cost_gini", "mean"),
        mean_reward=("total_reward", "mean"),
        unsafe_member_steps=("unsafe_member_steps", "sum"),
        boundary_clipping_count=("boundary_clipping_count", "sum"),
    ).reset_index()
    summary.to_csv(args.out / "summary.csv", index=False)
    model_hash_after = pilot.model_state_sha256(model)
    if model_hash_after != model_hash_before:
        raise AssertionError("Frozen Meta-GDM parameters changed during evaluation")
    if any(pilot.sha256_file(path) != source_hashes[str(path.relative_to(pilot.RELEASE))]
           for path in sources):
        raise AssertionError("Evaluation source changed during run")
    manifest["model_state_hash_after"] = model_hash_after
    manifest["elapsed_seconds"] = time.time() - started
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "status.json").write_text(
        json.dumps({"state": "completed", "records": len(rows),
                    "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
