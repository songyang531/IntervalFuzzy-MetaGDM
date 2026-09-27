"""Exploratory controls after the primary pilot; NEVER pooled as preregistered.

Observational feedback compensation checks whether the direct target-to-action
adapter artificially handicaps domain optimizers. It is our added controller,
not part of Guo2024. Floor sensitivity checks numerical robustness separately.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import run_pilot as pilot


class FeedbackProxy:
    def __init__(self, kind, feedback=True, price_floor=1e-8):
        self.kind = kind
        self.feedback = feedback
        self.controller = pilot.MinimumCostLP(True) if kind == "guo" else None
        if self.controller:
            self.controller.relative_price_floor = price_floor
        self.stats = {"gain_clipped_low": 0, "gain_clipped_high": 0, "qp_infeasible": 0}

    def select_action(self, state, context, evaluate=True):
        if self.controller:
            direct = self.controller.action(state, context)
        else:
            target, diagnostics = pilot.interval_qp_target(state[:, 0], state[:, 1], threshold=.905)
            self.stats["qp_infeasible"] += int(not diagnostics["feasible"])
            direct = np.clip((target-state[:, 0])/.1, -1, 1)
        if not self.feedback:
            return direct
        a, m = context[:, :, 0], context[:, :, 1]
        denominator = .1*np.sum(a*a, axis=1)
        gain = np.ones(len(state))
        np.divide(np.sum(a*m, axis=1), denominator, out=gain, where=denominator>1e-10)
        self.stats["gain_clipped_low"] += int(np.sum(gain<.05))
        self.stats["gain_clipped_high"] += int(np.sum(gain>1))
        gain = np.clip(gain, .05, 1.)
        return np.clip(direct/gain, -1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    primary = pd.read_csv(args.primary/"episodes.csv")
    manifest = json.loads((args.primary/"manifest.json").read_text(encoding="utf-8"))
    controls = {
        "Guo2024 + feedback (exploratory)": ("guo", True, 1e-8),
        "Interval-QP + feedback (exploratory)": ("qp", True, 1e-8),
        "Guo2024 floor1e-6 (sensitivity)": ("guo", False, 1e-6),
    }
    metadata = dict(primary=str(args.primary), exploratory_after_pilot=True,
                    not_native_published_methods=True, training_performed=False,
                    controls=controls, gain_estimator="sum(a*m)/(0.1*sum(a*a)) over observed 10-step history",
                    gain_bounds=[.05, 1.], no_data_prior_gain=1., test_tuning_performed=False,
                    checkpoint_evaluations_reused_without_rerun=True,
                    source_sha256={str(path): pilot.sha256_file(path) for path in
                                   [Path(__file__), Path(pilot.__file__), Path(pilot.__file__).with_name("interval_qp.py")]})
    (args.out/"manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows, traces = [], []
    started = time.time()
    for scenario in manifest["scenarios"]:
        for n in manifest["N"]:
            for episode in range(manifest["episodes_per_cell"]):
                expected = primary[(primary.scenario==scenario)&(primary.N==n)&(primary.episode==episode)].iloc[0]
                for name, parameters in controls.items():
                    proxy = FeedbackProxy(*parameters)
                    row, trace = pilot.run_episode(proxy, "Meta-GDM", scenario, n, episode, manifest["base_seed"])
                    assert row["initial_sha256"] == expected.initial_sha256
                    row["method"] = name
                    row.update({"guard_"+key: value for key, value in proxy.stats.items()})
                    if proxy.controller:
                        row.update({"guard_"+key: value for key, value in proxy.controller.stats.items()})
                    for t in trace:
                        t["method"] = name
                    rows.append(row)
                    traces.extend(trace)
                if (episode+1)%5==0:
                    pd.DataFrame(rows).to_csv(args.out/"episodes_partial.csv", index=False)
                    print(json.dumps(dict(scenario=scenario, N=n, episode=episode+1,
                                         completed_episodes=len(rows), seconds=time.time()-started)), flush=True)
    pd.DataFrame(rows).to_csv(args.out/"new_control_episodes.csv", index=False)
    primary_traces = pd.read_csv(args.primary/"trajectories.csv").to_dict("records")
    pilot.METHODS = pilot.METHODS + list(controls)
    summary = pilot.save_results(primary.to_dict("records")+rows, primary_traces+traces, args.out)
    metadata.update(elapsed_seconds=time.time()-started, new_episodes=len(rows), reused_episodes=len(primary))
    (args.out/"manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out/"status.json").write_text(json.dumps(dict(state="completed", **metadata), indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
