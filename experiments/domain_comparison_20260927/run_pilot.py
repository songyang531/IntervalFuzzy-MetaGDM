"""Paired frozen-policy pilot; published adaptation != native reproduction.

No training, no edits to historical code/checkpoints. Run from any directory.
Policies receive state/context only; evaluator alone sees hidden traits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT
sys.path.insert(0, str(RELEASE / "code"))
sys.path.insert(0, str(RELEASE / "scripts"))
import agent as agent_module
from context_utils import append_context_step
from run_existing_interval_model_zero_training_audit import (
    load_agent_without_training_runner, model_state_sha256, set_eval_mode, sha256_file,
)
from run_stage1_meta_vs_rules import Stage1Env, SCENARIOS, stable_name_seed
from interval_qp import interval_qp_target

CHECKPOINT = RELEASE / "checkpoints/meta_gdm_social_attention_ctxlearned_z8_cw10_h4_lr0p0003_ep5000_seed20260507_1df7dc5b_interval_only_5000.pth"
METHODS = ["Meta-GDM", "Mean rule", "Static MCC-L1", "Guo2024-adapted", "Interval-QP reference"]


class MinimumCostLP:
    """Guo2024 Eqs5--13/Model15 adaptation, or a static L1 MCC reference.

    Internal nominal prices are NOT estimates of hidden environment q.
    Safe endpoint bounds and closed-loop execution are explicit adaptations.
    """
    relative_price_floor = 1e-8
    solver_tolerance = 1e-9

    def __init__(self, dynamic):
        self.dynamic = dynamic
        self.k = 0
        self.stats = {"cl_clipped": 0, "zero_acl": 0, "unanimous": 0,
                      "price_floor": 0, "weight_floor": 0, "lp_calls": 0,
                      "min_relative_price": 1., "max_objective_residual": 0.}

    def action(self, state, context):
        c, h = np.asarray(state[:, 0], float), np.asarray(state[:, 1], float)
        n = c.size
        if self.k == 0:
            self.initial = c.copy()
            self.weights = np.full(n, 1/n)
            self.prices = np.ones(n)
            beta = np.ones(n)
        else:
            # Previous weights applied to actually observed post-response centres.
            collective = float(self.weights @ c)
            denominator = collective - self.initial
            levels = np.ones(n)
            np.divide(c-self.initial, denominator, out=levels,
                      where=np.abs(denominator) > 1e-12)
            self.stats["cl_clipped"] += int(np.sum((levels < 0) | (levels > 1)))
            levels = np.clip(levels, 0, 1)
            if levels.mean() <= 1e-12:
                beta = np.ones(n)
                self.stats["zero_acl"] += 1
            else:
                beta = levels/levels.mean()
        if self.dynamic:
            distances = np.abs(c[:, None]-c).sum(axis=1)
            if distances.max() <= 1e-12:
                self.stats["unanimous"] += 1
            else:
                self.weights /= np.maximum(distances, 1e-12)
                self.weights /= self.weights.sum()
                self.stats["weight_floor"] += int(np.sum(self.weights < 1e-12))
                self.weights = np.maximum(self.weights, 1e-12)
                self.weights /= self.weights.sum()
            alpha = np.where(self.weights < .8/n, .5,
                             np.where(self.weights < 1.2/n, 1., 1.5))
            self.prices *= (alpha+beta)/2
            self.prices /= self.prices.max()  # Common scaling leaves LP unchanged.
            self.stats["price_floor"] += int(np.sum(self.prices < self.relative_price_floor))
            self.prices = np.maximum(self.prices, self.relative_price_floor)
            self.stats["min_relative_price"] = min(self.stats["min_relative_price"], float(self.prices.min()))
        self.k += 1
        tolerance = max(.2 * .5**(self.k-1), 1e-6)
        eye = np.eye(n)
        deviation = eye - np.tile(self.weights, (n, 1))
        matrix = np.vstack([np.c_[eye, -eye], np.c_[-eye, -eye],
                            np.c_[deviation, np.zeros((n, n))],
                            np.c_[-deviation, np.zeros((n, n))]])
        rhs = np.r_[c, -c, np.full(2*n, tolerance)]
        bounds = list(zip(.1+h, .9-h)) + [(0, None)]*n
        result = linprog(np.r_[np.zeros(n), self.prices], A_ub=matrix,
                         b_ub=rhs, bounds=bounds, method="highs",
                         options={"dual_feasibility_tolerance": self.solver_tolerance,
                                  "primal_feasibility_tolerance": self.solver_tolerance})
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        target = result.x[:n]
        assert np.max(np.abs(target-self.weights@target)) <= tolerance+1e-7
        assert np.all(target-h >= .1-1e-7) and np.all(target+h <= .9+1e-7)
        objective_residual = abs(result.fun - self.prices @ np.abs(target-c))
        assert objective_residual < 1e-7
        self.stats["max_objective_residual"] = max(self.stats["max_objective_residual"], float(objective_residual))
        self.stats["lp_calls"] += 1
        return np.clip((target-c)/.1, -1, 1)


def gini_standard(values):
    values = np.asarray(values)
    if values.sum() <= 1e-15:
        return 0.
    return float(np.abs(values[:, None]-values).sum() / (2*len(values)*values.sum()))


def initial_fingerprint(env):
    digest = hashlib.sha256()
    for x in [env.opinions, env.interval_half_widths,
              env.personalities["stubbornness"], env.personalities["cost_sensitivity"],
              env.response_types, env.response_thresholds]:
        digest.update(np.asarray(x).tobytes())
    return digest.hexdigest()


def run_episode(model, method, scenario_name, n, episode, base_seed):
    seed = int(base_seed+stable_name_seed(scenario_name)%1_000_000+episode*997+n*13)
    env = Stage1Env(n, seed, SCENARIOS[scenario_name])
    fingerprint = initial_fingerprint(env)
    widths = env.interval_half_widths.copy()
    std_h = float(np.std(widths))
    ceiling = 1. if std_h < .01 else float(np.exp(-5*std_h))
    state = env._get_state()
    context = np.zeros((n, 10, 3), dtype=np.float32)
    controller = MinimumCostLP(method == "Guo2024-adapted") if "MCC" in method or method == "Guo2024-adapted" else None
    costs, displacement_costs = np.zeros(n), np.zeros(n)
    reward_sum, clipping_count, unsafe_member_steps = 0., 0, 0
    action_seconds = 0.
    consensuses = []
    qp_infeasible = 0
    trajectory = []
    for step in range(120):
        tick = time.perf_counter()
        if method == "Meta-GDM":
            with torch.inference_mode():
                action = model.select_action(state, context, evaluate=True)
        elif method == "Mean rule":
            action = np.clip((state[:, 0].mean()-state[:, 0])/.1, -1, 1)
        elif method == "Interval-QP reference":
            target, diagnostic = interval_qp_target(state[:, 0], state[:, 1], threshold=.905)
            qp_infeasible += int(not diagnostic["feasible"])
            action = np.clip((target-state[:, 0])/.1, -1, 1)
        else:
            action = controller.action(state, context)
        action_seconds += time.perf_counter()-tick
        action = np.asarray(action, float).reshape(n)
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1+1e-7
        old_c = env.opinions.copy()
        next_state, reward, done, info = env.step(action)
        m = info["actual_movements"]
        displacement = env.opinions-old_c
        clipping_count += int(np.sum(np.abs(m-displacement) > 1e-9))
        qcost = env.calculate_adjustment_costs(m)
        costs += qcost
        displacement_costs += env.calculate_adjustment_costs(displacement)
        unsafe_member_steps += int(np.sum((env.opinions-widths < .1-1e-9) |
                                           (env.opinions+widths > .9+1e-9)))
        reward_sum += float(np.mean(reward))
        consensuses.append(float(info["consensus_level"]))
        trajectory.append(dict(scenario=scenario_name, N=n, episode=episode,
                               method=method, step=step+1, consensus=consensuses[-1],
                               cost=float(qcost.sum()), mean_reward=float(np.mean(reward))))
        context = append_context_step(context, action, m, reward)
        state = next_state
        if done:
            break
    assert np.array_equal(widths, env.interval_half_widths)
    steps = len(consensuses)
    stats = controller.stats if controller else {}
    return dict(scenario=scenario_name, N=n, episode=episode, seed=seed, method=method,
                initial_sha256=fingerprint, success=int(info["success"]), steps=steps,
                total_cost=float(costs.sum()), total_displacement_cost=float(displacement_costs.sum()),
                cost_gini=gini_standard(costs),
                member_cumulative_costs_json=json.dumps(costs.tolist(), separators=(",", ":")),
                legacy_shifted_gini=float(env._gini(costs)),
                total_reward=reward_sum, final_consensus=consensuses[-1],
                consensus_auc=float(np.mean(consensuses+[consensuses[-1]]*(120-steps))),
                width_ceiling=ceiling, width_feasible=int(ceiling >= .9),
                boundary_clipping_count=clipping_count, unsafe_member_steps=unsafe_member_steps,
                action_seconds=action_seconds, qp_infeasible_calls=qp_infeasible,
                **{f"guard_{key}": value for key, value in stats.items()}), trajectory


def save_results(rows, traces, out, bootstrap=3000):
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"episodes.csv", index=False)
    pd.DataFrame(traces).to_csv(out/"trajectories.csv", index=False)
    keys = ["scenario", "N", "method"]
    summary = frame.groupby(keys).agg(
        episodes=("success", "count"), successes=("success", "sum"),
        success_rate=("success", "mean"), mean_steps=("steps", "mean"),
        mean_total_cost=("total_cost", "mean"), mean_cost_gini=("cost_gini", "mean"),
        mean_reward=("total_reward", "mean"), mean_consensus_auc=("consensus_auc", "mean"),
        mean_final_consensus=("final_consensus", "mean"),
        mean_policy_seconds=("action_seconds", "mean"),
        width_feasible_cases=("width_feasible", "sum"),
        boundary_clipping_count=("boundary_clipping_count", "sum"),
        unsafe_member_steps=("unsafe_member_steps", "sum"),
    ).reset_index()
    summary.to_csv(out/"summary.csv", index=False)
    pairs = []
    rng = np.random.default_rng(20260926)
    for (scenario, n), subset in frame.groupby(["scenario", "N"]):
        meta = subset[subset.method == "Meta-GDM"].set_index("episode")
        for method in METHODS[1:]:
            baseline = subset[subset.method == method].set_index("episode")
            common = meta.index.intersection(baseline.index)
            if not len(common):
                continue
            left, right = meta.loc[common], baseline.loc[common]
            assert np.array_equal(left.initial_sha256, right.initial_sha256)
            indices = rng.integers(0, len(common), (bootstrap, len(common)))
            for metric in ["success", "steps", "total_cost", "cost_gini", "consensus_auc"]:
                delta = left[metric].to_numpy()-right[metric].to_numpy()
                lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                pairs.append(dict(scenario=scenario, N=n, baseline=method, metric=metric,
                                  paired_cases=len(common), delta_meta_minus_baseline=float(delta.mean()),
                                  ci95_low=float(lo), ci95_high=float(hi), subset="all"))
            both = (left.success == 1) & (right.success == 1)
            if both.any():
                cost_delta = (left.total_cost[both]-right.total_cost[both]).to_numpy()
                indices = rng.integers(0, len(cost_delta), (bootstrap, len(cost_delta)))
                lo, hi = np.quantile(cost_delta[indices].mean(axis=1), [.025, .975])
                pairs.append(dict(scenario=scenario, N=n, baseline=method, metric="total_cost",
                                  paired_cases=int(both.sum()), delta_meta_minus_baseline=float(cost_delta.mean()),
                                  ci95_low=float(lo), ci95_high=float(hi), subset="both_succeeded_selection_caveat"))
    pd.DataFrame(pairs).to_csv(out/"paired_comparisons.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_agent_without_training_runner(agent_module, CHECKPOINT, args.device)
    set_eval_mode(model)
    original_hash = model_state_sha256(model)
    source_paths = [Path(__file__), Path(__file__).with_name("interval_qp.py"),
                    RELEASE/"code/env.py", RELEASE/"code/agent.py", RELEASE/"code/model.py",
                    RELEASE/"scripts/run_stage1_meta_vs_rules.py"]
    manifest = dict(protocol="paired domain baseline pilot v1", base_seed=args.seed,
                    scenarios=["iid_control", "hidden_heterogeneity"], N=[10, 40, 100],
                    episodes_per_cell=args.episodes, methods=METHODS,
                    checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256_file(CHECKPOINT),
                    frozen_parameters_before=original_hash, training_performed=False,
                    source_sha256={str(p): sha256_file(p) for p in source_paths},
                    scipy_version=__import__("scipy").__version__, torch_version=torch.__version__,
                    numpy_version=np.__version__, device=args.device,
                    guo_source="https://doi.org/10.1016/j.inffus.2023.102185",
                    guo_native_case="Native-case verification is documented separately; not needed for the paired run",
                    guo_tolerance_initial=.2, guo_tolerance_decay=.5, guo_tolerance_floor=1e-6,
                    lp_feasibility_tolerance=1e-9, guo_relative_price_floor=1e-8,
                    guo_lambda=.5, safe_endpoint_bounds=[.1, .9], qp_target_threshold=.905,
                    success_threshold=.9, max_steps=120, cost="original environment cost on response m",
                    warning="One published adaptation; static MCC and interval QP are references, not additional SOTA reproductions.")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows, traces = [], []
    started = time.time()
    for scenario in manifest["scenarios"]:
        for n in manifest["N"]:
            for episode in range(args.episodes):
                fingerprints = []
                for method in METHODS:
                    row, trajectory = run_episode(model, method, scenario, n, episode, args.seed)
                    rows.append(row)
                    traces.extend(trajectory)
                    fingerprints.append(row["initial_sha256"])
                assert len(set(fingerprints)) == 1
                if (episode+1) % 5 == 0 or episode == args.episodes-1:
                    pd.DataFrame(rows).to_csv(args.out/"episodes_partial.csv", index=False)
                    status = dict(state="running", scenario=scenario, N=n, episode=episode+1,
                                  completed_episodes=len(rows), elapsed_seconds=time.time()-started)
                    (args.out/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                    print(json.dumps(status), flush=True)
    summary = save_results(rows, traces, args.out)
    final_hash = model_state_sha256(model)
    assert final_hash == original_hash
    manifest.update(frozen_parameters_after=final_hash, elapsed_seconds=time.time()-started,
                    source_sha256_after={str(p): sha256_file(p) for p in source_paths})
    assert manifest["source_sha256"] == manifest["source_sha256_after"]
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out/"status.json").write_text(json.dumps(dict(state="completed", episodes=len(rows),
           elapsed_seconds=time.time()-started), indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
