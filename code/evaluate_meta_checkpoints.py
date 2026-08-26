import argparse

import pandas as pd

from revision_experiments import evaluate_policy, resolve_device, save_summary


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(
            "Each --model value must use the format 'Label=checkpoint_path'. "
            f"Got: {spec}"
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --model value: {spec}")
    return label, path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Meta-GDM checkpoints without retraining.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model specification in the form 'Label=checkpoint_path'. Repeat for multiple models."
    )
    parser.add_argument("--agents", nargs="+", type=int, default=[10, 40, 100])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opinion-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--stubbornness-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--cost-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--name", default="meta_checkpoint_eval")
    args = parser.parse_args()

    device = resolve_device(args.device)
    frames = []
    for spec in args.model:
        label, checkpoint = parse_model_spec(spec)
        for n in args.agents:
            frames.append(evaluate_policy(
                label,
                n,
                args.episodes,
                args.max_steps,
                args.threshold,
                args.seed,
                checkpoint_path=checkpoint,
                opinion_range=tuple(args.opinion_range),
                stubbornness_range=tuple(args.stubbornness_range),
                cost_sensitivity_range=tuple(args.cost_range),
                device=device,
            ))

    save_summary(pd.concat(frames, ignore_index=True), args.name)


if __name__ == "__main__":
    main()
