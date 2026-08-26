import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from plot_meta_gdm_diagnostics import (
    load_meta_agent,
    plot_paper_aggregate_separate,
    plot_training_history_separate,
    run_aggregate_trajectories,
)
from revision_experiments import train_meta_variant


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain Meta-GDM with the old main parameters and export separate paper-style curves."
    )
    parser.add_argument("--tag", default=None,
                        help="Run tag used for checkpoint and output folder names.")
    parser.add_argument("--output-root", default="revision_results/figures",
                        help="Root folder for the new output directory.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--agents", nargs="+", type=int, default=[10, 40],
                        help="Agent sizes used for evaluation curves after training.")
    parser.add_argument("--train-agents", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Evaluation success threshold; not drawn as a threshold line in the exported curves.")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--smooth-window", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.tag or f"paper_curves_retrain5000_oldparams_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_root) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(train_meta_variant(
        z_dim=8,
        aggregation="social_attention",
        context_window=10,
        num_heads=4,
        context_mode="learned",
        lr=3e-4,
        gamma=0.95,
        tau=0.01,
        alpha=0.1,
        train_episodes=args.train_episodes,
        num_agents=args.train_agents,
        max_steps=args.max_steps,
        threshold=args.threshold,
        seed=args.seed,
        device_arg=args.device,
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_sensitivity_range=(0.1, 0.9),
        warmup_fraction=0.03,
        curriculum_thresholds=(0.70, 0.80, 0.85, 0.90),
        checkpoint_tag=tag,
    ))
    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_training_history.csv")

    copied_checkpoint = output_dir / checkpoint_path.name
    copied_history = output_dir / history_path.name
    shutil.copy2(checkpoint_path, copied_checkpoint)
    shutil.copy2(history_path, copied_history)

    plot_training_history_separate(
        copied_history,
        output_dir / "training_curves",
        smooth_window=args.smooth_window,
    )

    agent = load_meta_agent(str(checkpoint_path), device=args.device)
    for n in args.agents:
        combined, summary = run_aggregate_trajectories(
            agent=agent,
            num_agents=n,
            num_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            threshold=args.threshold,
            seed=args.seed,
            opinion_range=(0.1, 0.9),
            stubbornness_range=(0.1, 0.9),
            cost_range=(0.1, 0.9),
        )
        combined.to_csv(
            output_dir / f"aggregate_trajectories_N{n}_{args.eval_episodes}eps_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            output_dir / f"aggregate_trajectories_N{n}_{args.eval_episodes}eps_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_paper_aggregate_separate(
            summary,
            f"Meta-GDM, N={n}, {args.eval_episodes} episodes",
            output_dir / f"N{n}_separate_curves",
        )

    run_config = {
        "tag": tag,
        "checkpoint": str(checkpoint_path),
        "copied_checkpoint": str(copied_checkpoint),
        "training_history": str(copied_history),
        "agents": args.agents,
        "train_agents": args.train_agents,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "threshold": args.threshold,
        "seed": args.seed,
        "aggregation": "social_attention",
        "context_mode": "learned",
        "z_dim": 8,
        "context_window": 10,
        "num_heads": 4,
        "lr": 3e-4,
        "gamma": 0.95,
        "tau": 0.01,
        "alpha": 0.1,
        "warmup_fraction": 0.03,
        "curriculum_thresholds": [0.70, 0.80, 0.85, 0.90],
        "curriculum_step_fracs": ["0.75", "5/6", "11/12", "1.0"],
        "opinion_range": [0.1, 0.9],
        "stubbornness_range": [0.1, 0.9],
        "cost_range": [0.1, 0.9],
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    summary_rows = []
    for n in args.agents:
        df = pd.read_csv(output_dir / f"aggregate_trajectories_N{n}_{args.eval_episodes}eps_raw.csv")
        episode_rows = []
        for _, group in df.groupby("Episode"):
            group = group.sort_values("Step")
            success_rows = group[group["Success"].astype(float) > 0]
            final_row = group.iloc[-1]
            episode_rows.append({
                "Success": 0 if success_rows.empty else 1,
                "Steps": args.max_steps if success_rows.empty else int(success_rows.iloc[0]["Step"]),
                "Cumulative Cost": float(final_row["Cumulative Cost"]),
                "Cumulative Reward": float(final_row["Cumulative Reward"]),
                "Final Consensus": float(final_row["Consensus"]),
            })
        final_rows = pd.DataFrame(episode_rows)
        summary_rows.append({
            "Agents": n,
            "Success": float(final_rows["Success"].mean()),
            "Steps": float(final_rows["Steps"].mean()),
            "Cumulative Cost": float(final_rows["Cumulative Cost"].mean()),
            "Cumulative Reward": float(final_rows["Cumulative Reward"].mean()),
            "Final Consensus": float(final_rows["Consensus"].mean()),
        })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "evaluation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone.")
    print(f"Output folder: {output_dir}")
    print(f"Checkpoint: {copied_checkpoint}")
    print(f"Training history: {copied_history}")


if __name__ == "__main__":
    main()
