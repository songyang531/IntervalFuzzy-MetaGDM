import argparse
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from env import OpinionDynamicsEnv
from replay_buffer import MetaReplayBuffer


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is not available. Run this script in the training environment.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def update_smoke(args):
    require_torch()
    from agent import MetaAgent

    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = OpinionDynamicsEnv(
        num_agents=args.agents,
        max_steps=args.max_steps,
        seed=args.seed,
        consensus_threshold=args.threshold,
        opinion_range=tuple(args.opinion_range),
        stubbornness_range=tuple(args.stubbornness_range),
        cost_sensitivity_range=tuple(args.cost_range),
    )
    agent = MetaAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        z_dim=args.z_dim,
        context_window=args.context_window,
        num_heads=args.num_heads,
        context_mode=args.context_mode,
        device=device,
    )
    buffer = MetaReplayBuffer(capacity=200, context_window=args.context_window)

    for _ in range(args.episodes):
        state = env.reset()
        episode = []
        for _ in range(env.max_steps):
            action = np.random.uniform(-1.0, 1.0, env.num_agents)
            next_state, reward, done, info = env.step(action)
            not_done = float(not done)
            episode.append((state, action, reward, next_state, not_done, info))
            state = next_state
            if done:
                break
        buffer.push_episode(episode)

    batch = buffer.sample(args.batch_size, device=device)
    losses = agent.update(batch)
    print("device", device)
    print("batch_shapes", [tuple(x.shape) for x in batch])
    print("losses", losses)
    print("last_reward_components", info.get("reward_components", {}))


def action_probe(args):
    require_torch()
    from agent import MetaAgent

    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = OpinionDynamicsEnv(
        num_agents=args.agents,
        max_steps=args.max_steps,
        seed=args.seed,
        consensus_threshold=args.threshold,
        opinion_range=tuple(args.opinion_range),
        stubbornness_range=tuple(args.stubbornness_range),
        cost_sensitivity_range=tuple(args.cost_range),
    )
    agent = MetaAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        z_dim=args.z_dim,
        context_window=args.context_window,
        num_heads=args.num_heads,
        context_mode=args.context_mode,
        device=device,
    )
    start_ep = agent.load_checkpoint(args.checkpoint) if args.checkpoint else 0

    sign_hits = []
    corrs = []
    model_improvements = []
    mean_improvements = []
    model_costs = []
    mean_costs = []
    action_abs = []

    for ep in range(args.episodes):
        env = OpinionDynamicsEnv(
            num_agents=args.agents,
            max_steps=args.max_steps,
            seed=args.seed + ep * 997,
            consensus_threshold=args.threshold,
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
        )
        state = env.reset()
        context = np.zeros((env.num_agents, args.context_window, 3), dtype=np.float32)
        old_consensus = env._calculate_consensus_level_from_opinions(env.opinions)
        mean_direction = np.mean(env.opinions) - env.opinions

        action = agent.select_action(state, context, evaluate=True)
        if np.std(action) > 1e-8 and np.std(mean_direction) > 1e-8:
            corrs.append(float(np.corrcoef(action, mean_direction)[0, 1]))
        sign_hits.append(float(np.mean(np.sign(action) == np.sign(mean_direction))))
        action_abs.append(float(np.mean(np.abs(action))))
        _, _, _, info = env.step(action)
        model_improvements.append(info["consensus_level"] - old_consensus)
        model_costs.append(float(np.sum(
            env.calculate_adjustment_costs(info["actual_movements"])
        )))

        env = OpinionDynamicsEnv(
            num_agents=args.agents,
            max_steps=args.max_steps,
            seed=args.seed + ep * 997,
            consensus_threshold=args.threshold,
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
        )
        old_consensus = env._calculate_consensus_level_from_opinions(env.opinions)
        mean_action = np.clip(
            (np.mean(env.opinions) - env.opinions) / max(env.movement_scale, 1e-8),
            -1.0,
            1.0,
        )
        _, _, _, info = env.step(mean_action)
        mean_improvements.append(info["consensus_level"] - old_consensus)
        mean_costs.append(float(np.sum(
            env.calculate_adjustment_costs(info["actual_movements"])
        )))

    print("checkpoint", args.checkpoint or "<untrained>")
    print("loaded_next_episode", start_ep)
    print("device", device)
    print("agents", args.agents)
    print("action_abs_mean", float(np.mean(action_abs)))
    print("direction_sign_accuracy", float(np.mean(sign_hits)))
    print("direction_corr", float(np.mean(corrs)) if corrs else None)
    print("first_step_model_improve", float(np.mean(model_improvements)))
    print("first_step_model_cost", float(np.mean(model_costs)))
    print("first_step_mean_improve", float(np.mean(mean_improvements)))
    print("first_step_mean_cost", float(np.mean(mean_costs)))


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight Meta-GDM diagnostics.")
    parser.add_argument("--mode", choices=["update_smoke", "action_probe"], default="update_smoke")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--context-mode", choices=["learned", "deterministic", "zero"], default="learned")
    parser.add_argument("--opinion-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--stubbornness-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--cost-range", nargs=2, type=float, default=[0.1, 0.9])
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "update_smoke":
        update_smoke(cli_args)
    elif cli_args.mode == "action_probe":
        action_probe(cli_args)
