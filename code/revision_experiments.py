import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:
    torch = None

from env import OpinionDynamicsEnv
from context_utils import append_context_step


RESULT_DIR = Path("revision_results")
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available in the active Python environment. "
            "Switch to the environment used for the original training, or install requirements.txt."
        )


def resolve_device(device_arg="auto"):
    require_torch()
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = values - values.min()
    if np.allclose(values.sum(), 0.0):
        return 0.0
    values = np.sort(values)
    index = np.arange(1, values.size + 1)
    return float((2 * np.sum(index * values) / (values.size * np.sum(values))) - (values.size + 1) / values.size)


def prepare_env(
        num_agents,
        max_steps,
        threshold,
        seed,
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_sensitivity_range=(0.1, 0.9),
        env_kwargs=None
):
    env_kwargs = env_kwargs or {}
    env = OpinionDynamicsEnv(
        num_agents=num_agents,
        max_steps=max_steps,
        consensus_threshold=threshold,
        seed=seed,
        **env_kwargs
    )
    rng = np.random.default_rng(seed)
    centre_low = opinion_range[0] + env.interval_half_widths
    centre_high = opinion_range[1] - env.interval_half_widths
    env.opinions = rng.uniform(centre_low, centre_high, num_agents)
    while np.std(env.opinions) < 0.1:
        env.opinions = rng.uniform(centre_low, centre_high, num_agents)
    env.personalities["stubbornness"] = rng.uniform(*stubbornness_range, num_agents)
    env.personalities["cost_sensitivity"] = rng.uniform(*cost_sensitivity_range, num_agents)
    return env, env._get_state()


def prepare_case_env(case_df, max_steps, threshold, seed, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    env = OpinionDynamicsEnv(
        num_agents=len(case_df),
        max_steps=max_steps,
        consensus_threshold=threshold,
        seed=seed,
        **env_kwargs
    )
    env.opinions = case_df["opinion"].to_numpy(dtype=float)
    env.personalities["stubbornness"] = case_df["stubbornness"].to_numpy(dtype=float)
    env.personalities["cost_sensitivity"] = case_df["cost_sensitivity"].to_numpy(dtype=float)
    return env, env._get_state()


def load_case_records(case_csv):
    df = pd.read_csv(case_csv)
    required = {"opinion", "stubbornness", "cost_sensitivity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Case CSV is missing required columns: {sorted(missing)}")
    if "case_id" not in df.columns:
        df["case_id"] = "case_1"
    return [(case_id, group.reset_index(drop=True)) for case_id, group in df.groupby("case_id", sort=False)]


def action_towards(env, target, gain=1.0, per_agent_scale=None):
    target = np.asarray(target)
    if target.ndim == 0:
        target = np.full(env.num_agents, float(target))
    raw = gain * (target - env.opinions) / max(env.movement_scale, 1e-8)
    if per_agent_scale is not None:
        raw = raw * per_agent_scale
    return np.clip(raw, -1.0, 1.0)


def random_policy(env, rng):
    return rng.uniform(-1.0, 1.0, env.num_agents)


def mean_consensus_policy(env, rng):
    return action_towards(env, np.mean(env.opinions), gain=1.0)

def mccm_policy(env, rng):
    """
    传统最小成本共识模型（Minimum Cost Consensus Model）策略
    假设二次成本，最优共识 = 加权平均 (权重 = 1 / cost_sensitivity)
    每步根据当前意见重新计算目标，并向该目标移动
    """
    # 获取成本敏感度，避免除零
    cost_sens = np.maximum(env.personalities["cost_sensitivity"], 1e-6)
    # 权重 = 1 / 成本敏感度
    weights = 1.0 / cost_sens
    # 加权平均目标
    target = np.average(env.opinions, weights=weights)
    # 向目标移动，但不对个体差异缩放（传统MCCM中所有个体都调到同一值）
    # 注意：实际移动幅度还会受顽固性影响（在env.step中处理）
    return action_towards(env, target, gain=1.0, per_agent_scale=None)

def bounded_confidence_policy(env, rng, epsilon=0.35):
    targets = np.zeros(env.num_agents)
    global_mean = np.mean(env.opinions)
    for i, opinion in enumerate(env.opinions):
        neighbors = np.abs(env.opinions - opinion) <= epsilon
        local_mean = np.mean(env.opinions[neighbors])
        targets[i] = 0.7 * local_mean + 0.3 * global_mean
    return action_towards(env, targets, gain=1.0)


def cluster_consensus_policy(env, rng, num_clusters=None):
    num_clusters = num_clusters or max(2, int(np.sqrt(env.num_agents)))
    order = np.argsort(env.opinions)
    targets = np.zeros(env.num_agents)
    global_mean = np.mean(env.opinions)
    for cluster in np.array_split(order, num_clusters):
        cluster_mean = np.mean(env.opinions[cluster])
        targets[cluster] = 0.6 * cluster_mean + 0.4 * global_mean
    return action_towards(env, targets, gain=1.0)


def cost_aware_policy(env, rng):
    cost = env.personalities["cost_sensitivity"]
    weights = 1.0 / (cost + 1e-3)
    target = np.average(env.opinions, weights=weights)
    per_agent_scale = 1.0 / (1.0 + cost)
    per_agent_scale = per_agent_scale / per_agent_scale.max()
    return action_towards(env, target, gain=1.0, per_agent_scale=per_agent_scale)


CLASSIC_POLICIES = {
    "Random": random_policy,
    "Mean consensus": mean_consensus_policy,
    "Bounded confidence": bounded_confidence_policy,
    "Cluster consensus": cluster_consensus_policy,
    "Cost-aware heuristic": cost_aware_policy,
    "Minimum Cost Consensus (MCCM)": mccm_policy,
}


def load_meta_agent(checkpoint_path, device=None, z_dim=8, hidden_dim=256,
                    aggregation="social_attention", num_heads=4, context_mode="learned"):
    require_torch()
    from agent import MetaAgent

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    z_dim = int(checkpoint.get("z_dim", z_dim))
    aggregation = checkpoint.get("aggregation", aggregation)
    num_heads = int(checkpoint.get("num_heads", num_heads))
    context_mode = checkpoint.get("context_mode", context_mode)

    agent = MetaAgent(
        state_dim=int(checkpoint.get("state_dim", 5)),
        action_dim=int(checkpoint.get("action_dim", 1)),
        z_dim=z_dim,
        context_window=int(checkpoint.get("context_window", 10)),
        hidden_dim=hidden_dim,
        aggregation=aggregation,
        num_heads=num_heads,
        context_mode=context_mode,
        device=device
    )
    if "encoder_state_dict" in checkpoint:
        agent._load_compatible_state_dict(agent.encoder, checkpoint["encoder_state_dict"], "encoder")
        agent._load_compatible_state_dict(agent.actor, checkpoint["actor_state_dict"], "actor")
        if "critic_state_dict" in checkpoint:
            agent._load_compatible_state_dict(agent.critic, checkpoint["critic_state_dict"], "critic")
    else:
        agent._load_compatible_state_dict(agent.encoder, checkpoint["encoder"], "encoder")
        agent._load_compatible_state_dict(agent.actor, checkpoint["actor"], "actor")
        if "critic" in checkpoint:
            agent._load_compatible_state_dict(agent.critic, checkpoint["critic"], "critic")
    return agent


def evaluate_policy(
        model_name,
        num_agents,
        num_episodes,
        max_steps,
        threshold,
        seed_base,
        checkpoint_path=None,
        policy_fn=None,
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_sensitivity_range=(0.1, 0.9),
        env_kwargs=None,
        case_records=None,
        device=None
):
    rng = np.random.default_rng(seed_base)
    records = []
    agent = None
    if checkpoint_path is not None:
        agent = load_meta_agent(checkpoint_path, device=device)

    requested_agents = num_agents
    episode_iter = tqdm(
        range(num_episodes),
        desc=f"eval {model_name} N={num_agents}",
        dynamic_ncols=True,
        leave=False
    )
    for ep in episode_iter:
        env_seed = seed_base + ep * 997 + requested_agents * 13
        case_id = ""
        if case_records:
            case_id, case_df = case_records[ep % len(case_records)]
            env, state = prepare_case_env(case_df, max_steps, threshold, env_seed, env_kwargs)
        else:
            env, state = prepare_env(
                num_agents,
                max_steps,
                threshold,
                env_seed,
                opinion_range=opinion_range,
                stubbornness_range=stubbornness_range,
                cost_sensitivity_range=cost_sensitivity_range,
                env_kwargs=env_kwargs
            )
        context_window = getattr(agent, "context_window", 10)
        current_agents = env.num_agents
        context = np.zeros((current_agents, context_window, 3))
        cumulative_costs = np.zeros(current_agents)
        boundary_violations = 0
        total_reward = 0.0
        success = False
        steps = max_steps
        info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions)}

        for step in range(max_steps):
            if agent is not None:
                action = agent.select_action(state, context, evaluate=True)
            else:
                action = policy_fn(env, rng)

            next_state, reward, done, info = env.step(action)
            step_costs = env.calculate_adjustment_costs(info["actual_movements"])
            cumulative_costs += step_costs
            lower, upper = env._interval_bounds()
            boundary_violations += int(np.sum((lower < env.safe_low) | (upper > env.safe_high)))
            total_reward += float(np.mean(reward))

            context = append_context_step(
                context,
                info["suggestions"],
                info["actual_movements"],
                reward
            )
            state = next_state

            if done:
                success = info["success"]
                steps = step + 1
                break

        episode_iter.set_postfix({
            "success": int(success),
            "steps": steps,
            "cons": f"{info.get('consensus_level', 0.0):.3f}",
            "cost": f"{np.sum(cumulative_costs):.2f}"
        })

        records.append({
            "Model": model_name,
            "Agents": current_agents,
            "Case": case_id,
            "Episode": ep,
            "Success": int(success),
            "Steps": steps,
            "Total Cost": float(np.sum(cumulative_costs)),
            "Cost Gini": gini(cumulative_costs),
            "Boundary Violations": boundary_violations,
            "Total Reward": total_reward,
            "Final Consensus": float(info.get("consensus_level", 0.0)),
            "Final Interval Dispersion": float(info.get("interval_dispersion", env._interval_dispersion())),
            "Mean Half Width": float(np.mean(env.interval_half_widths))
        })
    return pd.DataFrame(records)


def train_meta_variant(
        z_dim=8,
        aggregation="social_attention",
        context_window=10,
        num_heads=4,
        context_mode="learned",
        lr=3e-4,
        gamma=0.95,
        tau=0.01,
        alpha=0.1,
        train_episodes=5000,
        num_agents=10,
        max_steps=120,
        threshold=0.9,
        seed=20260507,
        env_kwargs=None,
        save_dir=CHECKPOINT_DIR,
        device_arg="auto",
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_sensitivity_range=(0.1, 0.9),
        interval_half_width_range=(0.02, 0.08),
        warmup_fraction=0.03,
        curriculum_thresholds=(0.70, 0.80, 0.85, 0.90),
        curriculum_step_fracs=(0.75, 5.0 / 6.0, 11.0 / 12.0, 1.00),
        checkpoint_tag=""
):
    require_torch()
    from agent import MetaAgent
    from replay_buffer import MetaReplayBuffer

    set_global_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = env_kwargs or {}
    device = resolve_device(device_arg)
    print(f"Using device: {device}")
    env = OpinionDynamicsEnv(
        num_agents=num_agents,
        max_steps=max_steps,
        consensus_threshold=threshold,
        seed=seed,
        opinion_range=opinion_range,
        stubbornness_range=stubbornness_range,
        cost_sensitivity_range=cost_sensitivity_range,
        interval_half_width_range=interval_half_width_range,
        **env_kwargs
    )
    agent = MetaAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        z_dim=z_dim,
        context_window=context_window,
        hidden_dim=256,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        kl_weight=0.05,
        aggregation=aggregation,
        num_heads=num_heads,
        context_mode=context_mode,
        device=device
    )
    replay_buffer = MetaReplayBuffer(capacity=10000, context_window=agent.context_window)
    batch_size = 32
    warmup_episodes = min(200, max(10, int(train_episodes * warmup_fraction)))
    curriculum = []
    for stage_threshold, step_frac in zip(curriculum_thresholds, curriculum_step_fracs):
        curriculum.append((min(stage_threshold, threshold), max(5, int(max_steps * step_frac))))
    curriculum[-1] = (threshold, max_steps)
    stage_len = max(1, train_episodes // len(curriculum))

    progress = tqdm(
        range(train_episodes),
        desc=f"train {aggregation} z={z_dim} cw={context_window} h={num_heads}",
        dynamic_ncols=True
    )
    training_records = []
    for episode in progress:
        stage_idx = min(episode // stage_len, len(curriculum) - 1)
        env.consensus_threshold, env.max_steps = curriculum[stage_idx]
        state = env.reset()
        context = np.zeros((num_agents, agent.context_window, 3))
        episode_data = []
        episode_reward = 0.0
        update_stats = {"critic_loss": np.nan, "actor_loss": np.nan, "kl_loss": np.nan}
        info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions), "success": False}
        for step in range(env.max_steps):
            if episode < warmup_episodes:
                action = np.random.uniform(-1.0, 1.0, num_agents)
            else:
                exploration_noise = max(0.2 * (1.0 - episode / max(train_episodes, 1)), 0.03)
                action = agent.select_action(state, context, evaluate=False, exploration_noise=exploration_noise)

            next_state, reward, done, info = env.step(action)
            not_done = float(not done)
            episode_data.append((state, action, reward, next_state, not_done, info))
            episode_reward += float(np.mean(reward))

            context = append_context_step(
                context,
                info["suggestions"],
                info["actual_movements"],
                reward
            )
            state = next_state
            if done:
                break

        replay_buffer.push_episode(episode_data)
        if episode >= warmup_episodes and len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size, device=device)
            update_stats = agent.update(batch)

        training_records.append({
            "Episode": episode + 1,
            "Stage": stage_idx + 1,
            "Curriculum Threshold": env.consensus_threshold,
            "Curriculum Max Steps": env.max_steps,
            "Steps": step + 1,
            "Episode Mean Reward": episode_reward,
            "Final Consensus": float(info.get("consensus_level", 0.0)),
            "Success": int(bool(info.get("success", False))),
            "Critic Loss": float(update_stats.get("critic_loss", np.nan)),
            "Actor Loss": float(update_stats.get("actor_loss", np.nan)),
            "KL Loss": float(update_stats.get("kl_loss", np.nan)),
        })

        if episode % 10 == 0 or episode == train_episodes - 1:
            progress.set_postfix({
                "cons": f"{info.get('consensus_level', 0.0):.3f}",
                "success": int(info.get("success", False)),
                "thr": f"{env.consensus_threshold:.2f}",
                "steps": step + 1,
                "reward": f"{episode_reward:.1f}",
                "buffer": len(replay_buffer)
            })

    env_tag = hashlib.md5(json.dumps(env.get_reward_parameters(), sort_keys=True).encode("utf-8")).hexdigest()[:8]
    lr_tag = f"{lr:g}".replace(".", "p").replace("-", "m")
    context_tag = f"ctx{context_mode}"
    suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    checkpoint_path = save_dir / (
        f"meta_gdm_{aggregation}_{context_tag}_z{z_dim}_cw{context_window}_h{num_heads}_"
        f"lr{lr_tag}_ep{train_episodes}_seed{seed}_{env_tag}{suffix}.pth"
    )
    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_training_history.csv")
    pd.DataFrame(training_records).to_csv(history_path, index=False, encoding="utf-8-sig")
    torch.save({
        "encoder_state_dict": agent.encoder.state_dict(),
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "critic_target_state_dict": agent.critic_target.state_dict(),
        "z_dim": z_dim,
        "context_window": agent.context_window,
        "aggregation": aggregation,
        "num_heads": num_heads,
        "context_mode": context_mode,
        "learning_rate": lr,
        "gamma": gamma,
        "tau": tau,
        "alpha": alpha,
        "checkpoint_tag": checkpoint_tag,
        "warmup_fraction": warmup_fraction,
        "curriculum": curriculum,
        "episode": train_episodes,
        "seed": seed,
        "reward_parameters": env.get_reward_parameters(),
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "interval_extension": {
            "representation": "center_half_width",
            "half_width_range": list(interval_half_width_range),
            "action_semantics": "scalar_interval_translation",
            "context_schema_unchanged": True
        },
        "training_history_csv": str(history_path)
    }, checkpoint_path)
    print(f"Saved training history to: {history_path}")
    return str(checkpoint_path)


def save_summary(df, name):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULT_DIR / f"{name}_raw.csv"
    summary_path = RESULT_DIR / f"{name}_summary.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    aggregations = {
        "Success": "mean",
        "Steps": "mean",
        "Total Cost": "mean",
        "Cost Gini": "mean",
        "Boundary Violations": "mean",
        "Total Reward": "mean",
        "Final Consensus": "mean",
    }
    for optional in ("Final Interval Dispersion", "Mean Half Width"):
        if optional in df.columns:
            aggregations[optional] = "mean"
    summary = df.groupby(["Model", "Agents"]).agg(aggregations).reset_index()
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved:\n  {raw_path}\n  {summary_path}")
    return summary


def run_baselines(args):
    frames = []
    case_records = load_case_records(args.case_csv) if args.case_csv else None
    agent_counts = [0] if case_records else args.agents
    device = resolve_device(args.device) if args.checkpoint and not args.skip_meta else None
    if device is not None:
        print(f"Using device: {device}")
    for n in agent_counts:
        if args.checkpoint and not args.skip_meta:
            frames.append(evaluate_policy(
                "Meta-GDM",
                n,
                args.episodes,
                args.max_steps,
                args.threshold,
                args.seed,
                checkpoint_path=args.checkpoint,
                case_records=case_records,
                opinion_range=tuple(args.opinion_range),
                stubbornness_range=tuple(args.stubbornness_range),
                cost_sensitivity_range=tuple(args.cost_range),
                env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)},
                device=device
            ))
        for name, fn in CLASSIC_POLICIES.items():
            frames.append(evaluate_policy(
                name,
                n,
                args.episodes,
                args.max_steps,
                args.threshold,
                args.seed,
                policy_fn=fn,
                case_records=case_records,
                opinion_range=tuple(args.opinion_range),
                stubbornness_range=tuple(args.stubbornness_range),
                cost_sensitivity_range=tuple(args.cost_range),
                env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)}
            ))
    return save_summary(pd.concat(frames, ignore_index=True), "baseline_comparison")


def run_train_meta(args):
    checkpoint = train_meta_variant(
        z_dim=8,
        aggregation=args.aggregation,
        train_episodes=args.train_episodes,
        num_agents=args.train_agents,
        seed=args.seed,
        max_steps=args.max_steps,
        threshold=args.threshold,
        device_arg=args.device,
        context_window=args.context_window,
        num_heads=args.num_heads,
        context_mode=args.context_mode,
        lr=args.lr,
        warmup_fraction=args.warmup_fraction,
        curriculum_thresholds=tuple(args.curriculum_thresholds),
        curriculum_step_fracs=tuple(args.curriculum_step_fracs),
        opinion_range=tuple(args.opinion_range),
        stubbornness_range=tuple(args.stubbornness_range),
        cost_sensitivity_range=tuple(args.cost_range),
        interval_half_width_range=tuple(args.interval_half_width_range),
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        checkpoint_tag=args.checkpoint_tag,
    )
    frames = []
    device = resolve_device(args.device)
    for n in args.agents:
        frames.append(evaluate_policy(
            f"Meta-GDM/{args.aggregation}",
            n,
            args.episodes,
            args.max_steps,
            args.threshold,
            args.seed,
            checkpoint_path=checkpoint,
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)},
            device=device
        ))
    print(f"\nCheckpoint:\n  {checkpoint}")
    output_name = "meta_gdm_train_eval" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def run_attention_ablation(args):
    frames = []
    for aggregation in args.attention_aggregations:
        print(f"\n=== Training attention ablation variant: {aggregation} ===")
        checkpoint = train_meta_variant(
            z_dim=8,
            aggregation=aggregation,
            train_episodes=args.train_episodes,
            num_agents=args.train_agents,
            seed=args.seed,
            max_steps=args.max_steps,
            threshold=args.threshold,
            device_arg=args.device,
            context_window=args.context_window,
            num_heads=args.num_heads,
            context_mode=args.context_mode,
            lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            curriculum_thresholds=tuple(args.curriculum_thresholds),
            curriculum_step_fracs=tuple(args.curriculum_step_fracs),
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            interval_half_width_range=tuple(args.interval_half_width_range),
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            checkpoint_tag=args.checkpoint_tag,
        )
        for n in args.agents:
            frames.append(evaluate_policy(
                f"Meta-GDM/{aggregation}",
                n,
                args.episodes,
                args.max_steps,
                args.threshold,
                args.seed,
                checkpoint_path=checkpoint,
                opinion_range=tuple(args.opinion_range),
                stubbornness_range=tuple(args.stubbornness_range),
                cost_sensitivity_range=tuple(args.cost_range),
                device=resolve_device(args.device)
            ))
    output_name = "attention_ablation" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def run_core_ablation(args):
    frames = []
    variants = [
        ("Full Meta-GDM", "social_attention", "learned"),
        ("w/o context encoder", "social_attention", "zero"),
        ("w/o social attention", "no_attention", "learned"),
        ("w/o both", "no_attention", "zero"),
    ]
    device = resolve_device(args.device)
    for label, aggregation, context_mode in variants:
        print(f"\n=== Training core ablation variant: {label} ===")
        checkpoint = train_meta_variant(
            z_dim=8,
            aggregation=aggregation,
            train_episodes=args.train_episodes,
            num_agents=args.train_agents,
            seed=args.seed,
            max_steps=args.max_steps,
            threshold=args.threshold,
            device_arg=args.device,
            context_window=args.context_window,
            num_heads=args.num_heads,
            context_mode=context_mode,
            lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            curriculum_thresholds=tuple(args.curriculum_thresholds),
            curriculum_step_fracs=tuple(args.curriculum_step_fracs),
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            interval_half_width_range=tuple(args.interval_half_width_range),
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            checkpoint_tag=args.checkpoint_tag,
        )
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
                env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)},
                device=device
            ))
    output_name = "core_ablation" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def run_latent_dim_sensitivity(args):
    frames = []
    for z_dim in args.z_dims:
        print(f"\n=== Training latent-dimension variant: z={z_dim} ===")
        checkpoint = train_meta_variant(
            z_dim=z_dim,
            aggregation="social_attention",
            train_episodes=args.train_episodes,
            num_agents=args.train_agents,
            seed=args.seed,
            max_steps=args.max_steps,
            threshold=args.threshold,
            device_arg=args.device,
            context_window=args.context_window,
            num_heads=args.num_heads,
            context_mode=args.context_mode,
            lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            curriculum_thresholds=tuple(args.curriculum_thresholds),
            curriculum_step_fracs=tuple(args.curriculum_step_fracs),
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            checkpoint_tag=args.checkpoint_tag,
        )
        for n in args.agents:
            frames.append(evaluate_policy(
                f"z={z_dim}",
                n,
                args.episodes,
                args.max_steps,
                args.threshold,
                args.seed,
                checkpoint_path=checkpoint,
                opinion_range=tuple(args.opinion_range),
                stubbornness_range=tuple(args.stubbornness_range),
                cost_sensitivity_range=tuple(args.cost_range),
                device=resolve_device(args.device)
            ))
    output_name = "latent_dim_sensitivity" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def run_reward_sensitivity(args):
    frames = []
    sweeps = []
    if "cost" in args.reward_sweep:
        for value in args.cost_coefficients:
            sweeps.append((f"cost={value:g}", {"cost_coefficient": value}))
    if "boundary" in args.reward_sweep:
        for value in args.boundary_penalties:
            sweeps.append((f"boundary={value:g}", {"boundary_penalty_weight": value}))
    if "consensus" in args.reward_sweep:
        for value in args.consensus_weights:
            sweeps.append((f"consensus={value:g}", {"consensus_improvement_weight": value}))
    if "fairness" in args.reward_sweep:
        for value in args.fairness_weights:
            sweeps.append((f"fairness={value:g}", {"fairness_weight": value}))
    if "outlier" in args.reward_sweep:
        for value in args.outlier_weights:
            sweeps.append((f"outlier={value:g}", {"distance_penalty_weight": value}))
    if "cost_power" in args.reward_sweep:
        for value in args.cost_powers:
            sweeps.append((f"cost_power={value:g}", {"cost_power": value}))

    for label, env_kwargs in sweeps:
        print(f"\n=== Training reward-sensitivity variant: {label} ===")
        checkpoint = train_meta_variant(
            z_dim=8,
            aggregation="social_attention",
            train_episodes=args.train_episodes,
            num_agents=args.train_agents,
            seed=args.seed,
            max_steps=args.max_steps,
            threshold=args.threshold,
            env_kwargs=env_kwargs,
            device_arg=args.device,
            context_window=args.context_window,
            num_heads=args.num_heads,
            context_mode=args.context_mode,
            lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            curriculum_thresholds=tuple(args.curriculum_thresholds),
            curriculum_step_fracs=tuple(args.curriculum_step_fracs),
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            checkpoint_tag=args.checkpoint_tag,
        )
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
                env_kwargs=env_kwargs,
                device=resolve_device(args.device)
            ))
    output_name = "reward_sensitivity" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def run_training_sensitivity(args):
    frames = []
    sweeps = []
    if "context_window" in args.training_sweep:
        for value in args.context_windows:
            sweeps.append((f"context_window={value:g}", {"context_window": value}))
    if "attention_heads" in args.training_sweep:
        for value in args.attention_heads:
            sweeps.append((f"attention_heads={value:g}", {"num_heads": value}))
    if "learning_rate" in args.training_sweep:
        for value in args.learning_rates:
            sweeps.append((f"lr={value:g}", {"lr": value}))

    for label, train_kwargs in sweeps:
        print(f"\n=== Training hyperparameter-sensitivity variant: {label} ===")
        checkpoint = train_meta_variant(
            z_dim=8,
            aggregation="social_attention",
            train_episodes=args.train_episodes,
            num_agents=args.train_agents,
            seed=args.seed,
            max_steps=args.max_steps,
            threshold=args.threshold,
            device_arg=args.device,
            context_window=train_kwargs.get("context_window", args.context_window),
            num_heads=train_kwargs.get("num_heads", args.num_heads),
            context_mode=args.context_mode,
            lr=train_kwargs.get("lr", args.lr),
            warmup_fraction=args.warmup_fraction,
            curriculum_thresholds=tuple(args.curriculum_thresholds),
            curriculum_step_fracs=tuple(args.curriculum_step_fracs),
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            checkpoint_tag=args.checkpoint_tag,
        )
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
                device=resolve_device(args.device)
            ))
    output_name = "training_sensitivity" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def run_context_ablation(args):
    frames = []
    variants = [
        ("Full context VAE", "learned"),
        ("Deterministic context", "deterministic"),
        ("No context", "zero"),
    ]
    for label, context_mode in variants:
        print(f"\n=== Training context ablation variant: {label} ===")
        checkpoint = train_meta_variant(
            z_dim=8,
            aggregation=args.aggregation,
            train_episodes=args.train_episodes,
            num_agents=args.train_agents,
            seed=args.seed,
            max_steps=args.max_steps,
            threshold=args.threshold,
            device_arg=args.device,
            context_window=args.context_window,
            num_heads=args.num_heads,
            context_mode=context_mode,
            lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            curriculum_thresholds=tuple(args.curriculum_thresholds),
            curriculum_step_fracs=tuple(args.curriculum_step_fracs),
            opinion_range=tuple(args.opinion_range),
            stubbornness_range=tuple(args.stubbornness_range),
            cost_sensitivity_range=tuple(args.cost_range),
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            checkpoint_tag=args.checkpoint_tag,
        )
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
                device=resolve_device(args.device)
            ))
    output_name = "context_ablation" + (f"_{args.checkpoint_tag}" if args.checkpoint_tag else "")
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def evaluate_rl_checkpoint(
        model_name,
        checkpoint_path,
        num_agents,
        num_episodes,
        max_steps,
        threshold,
        seed_base,
        opinion_range,
        stubbornness_range,
        cost_sensitivity_range,
        env_kwargs=None,
        device_arg="auto"
):
    from rl_baselines import load_policy, select_action

    algorithm, actor, device = load_policy(checkpoint_path, device_arg)
    records = []
    for ep in tqdm(range(num_episodes), desc=f"eval {model_name} N={num_agents}", leave=False):
        env_seed = seed_base + ep * 997 + num_agents * 13
        env, state = prepare_env(
            num_agents,
            max_steps,
            threshold,
            env_seed,
            opinion_range=opinion_range,
            stubbornness_range=stubbornness_range,
            cost_sensitivity_range=cost_sensitivity_range,
            env_kwargs=env_kwargs
        )
        cumulative_costs = np.zeros(env.num_agents)
        boundary_violations = 0
        total_reward = 0.0
        success = False
        steps = max_steps
        info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions)}

        for step in range(max_steps):
            action = select_action(actor, algorithm, state, device)
            next_state, reward, done, info = env.step(action)
            cumulative_costs += env.calculate_adjustment_costs(info["actual_movements"])
            lower, upper = env._interval_bounds()
            boundary_violations += int(np.sum((lower < env.safe_low) | (upper > env.safe_high)))
            total_reward += float(np.mean(reward))
            state = next_state
            if done:
                success = info["success"]
                steps = step + 1
                break

        records.append({
            "Model": model_name,
            "Agents": env.num_agents,
            "Case": "",
            "Episode": ep,
            "Success": int(success),
            "Steps": steps,
            "Total Cost": float(np.sum(cumulative_costs)),
            "Cost Gini": gini(cumulative_costs),
            "Boundary Violations": boundary_violations,
            "Total Reward": total_reward,
            "Final Consensus": float(info.get("consensus_level", 0.0)),
            "Final Interval Dispersion": float(info.get("interval_dispersion", env._interval_dispersion())),
            "Mean Half Width": float(np.mean(env.interval_half_widths))
        })
    return pd.DataFrame(records)


def run_rl_baselines(args):
    from rl_baselines import train_ddpg, train_ppo, train_sac

    frames = []
    for n in args.agents:
        for algorithm in args.rl_algorithms:
            print(f"\n=== Training RL baseline: {algorithm} N={n} ===")
            if algorithm == "SAC":
                checkpoint = train_sac(
                    num_agents=n,
                    train_episodes=args.train_episodes,
                    max_steps=args.max_steps,
                    threshold=args.threshold,
                    seed=args.seed,
                    device_arg=args.device,
                    lr=args.lr,
                    opinion_range=tuple(args.opinion_range),
                    stubbornness_range=tuple(args.stubbornness_range),
                    cost_range=tuple(args.cost_range),
                    env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)}
                )
            elif algorithm == "PPO":
                checkpoint = train_ppo(
                    num_agents=n,
                    train_episodes=args.train_episodes,
                    max_steps=args.max_steps,
                    threshold=args.threshold,
                    seed=args.seed,
                    device_arg=args.device,
                    lr=args.lr,
                    opinion_range=tuple(args.opinion_range),
                    stubbornness_range=tuple(args.stubbornness_range),
                    cost_range=tuple(args.cost_range),
                    env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)}
                )
            elif algorithm == "DDPG":
                checkpoint = train_ddpg(
                    num_agents=n,
                    train_episodes=args.train_episodes,
                    max_steps=args.max_steps,
                    threshold=args.threshold,
                    seed=args.seed,
                    device_arg=args.device,
                    lr=args.lr,
                    opinion_range=tuple(args.opinion_range),
                    stubbornness_range=tuple(args.stubbornness_range),
                    cost_range=tuple(args.cost_range),
                    env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)}
                )
            else:
                raise ValueError(f"Unsupported RL algorithm: {algorithm}")

            frames.append(evaluate_rl_checkpoint(
                algorithm,
                checkpoint,
                n,
                args.episodes,
                args.max_steps,
                args.threshold,
                args.seed,
                opinion_range=tuple(args.opinion_range),
                stubbornness_range=tuple(args.stubbornness_range),
                cost_sensitivity_range=tuple(args.cost_range),
                env_kwargs={"interval_half_width_range": tuple(args.interval_half_width_range)},
                device_arg=args.device
            ))
    selected = [alg.lower() for alg in args.rl_algorithms]
    output_name = "rl_baselines" if set(selected) == {"sac", "ppo", "ddpg"} else "rl_baselines_" + "_".join(selected)
    return save_summary(pd.concat(frames, ignore_index=True), output_name)


def write_run_metadata(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    meta = vars(args).copy()
    meta["torch_available"] = torch is not None
    if torch is not None:
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
    with open(RESULT_DIR / f"{args.mode}_run_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Revision experiments for Meta-GDM reviewer response.")
    parser.add_argument("--mode", choices=[
        "train_meta",
        "baselines",
        "attention_ablation",
        "core_ablation",
        "context_ablation",
        "latent_dim",
        "reward_sensitivity",
        "training_sensitivity",
        "rl_baselines"
    ],
                        default="baselines")
    parser.add_argument("--checkpoint", default="./logs/checkpoint_ep6000.pth")
    parser.add_argument("--skip-meta", action="store_true",
                        help="Run only non-neural classic baselines in baselines mode.")
    parser.add_argument("--case-csv", default=None,
                        help="Optional real/published case CSV with opinion, stubbornness and cost_sensitivity columns.")
    parser.add_argument("--agents", nargs="+", type=int, default=[10, 40])
    parser.add_argument("--train-agents", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--opinion-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--stubbornness-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--cost-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--interval-half-width-range", nargs=2, type=float, default=[0.02, 0.08])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device for neural models. Use cuda to require GPU, auto to use GPU when available.")
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--context-mode", choices=["learned", "deterministic", "zero"], default="learned")
    parser.add_argument("--aggregation", choices=["social_attention", "standard_mha", "mean_pooling", "no_attention"],
                        default="social_attention")
    parser.add_argument("--attention-aggregations", nargs="+",
                        choices=["social_attention", "standard_mha", "mean_pooling", "no_attention"],
                        default=["social_attention", "standard_mha", "mean_pooling", "no_attention"],
                        help="Aggregation variants to train in attention_ablation mode.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--checkpoint-tag", default="",
                        help="Optional suffix such as v2 for checkpoint and result filenames.")
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--curriculum-thresholds", nargs=4, type=float, default=[0.70, 0.80, 0.85, 0.90])
    parser.add_argument("--curriculum-step-fracs", nargs=4, type=float, default=[0.75, 5.0 / 6.0, 11.0 / 12.0, 1.00])
    parser.add_argument("--z-dims", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--cost-coefficients", nargs="+", type=float, default=[25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--boundary-penalties", nargs="+", type=float, default=[10.0, 20.0, 40.0])
    parser.add_argument("--consensus-weights", nargs="+", type=float, default=[25.0, 50.0, 100.0])
    parser.add_argument("--fairness-weights", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--outlier-weights", nargs="+", type=float, default=[0.5, 2.0, 5.0])
    parser.add_argument("--cost-powers", nargs="+", type=float, default=[1.5, 2.0, 3.0])
    parser.add_argument("--reward-sweep", nargs="+",
                        choices=["cost", "boundary", "consensus", "fairness", "outlier", "cost_power"],
                        default=["cost", "boundary"])
    parser.add_argument("--context-windows", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--attention-heads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--training-sweep", nargs="+",
                        choices=["context_window", "attention_heads", "learning_rate"],
                        default=["context_window", "attention_heads", "learning_rate"])
    parser.add_argument("--rl-algorithms", nargs="+", choices=["SAC", "PPO", "DDPG"], default=["SAC", "PPO", "DDPG"])
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    set_global_seed(cli_args.seed)
    write_run_metadata(cli_args)
    if cli_args.mode == "train_meta":
        run_train_meta(cli_args)
    elif cli_args.mode == "baselines":
        run_baselines(cli_args)
    elif cli_args.mode == "attention_ablation":
        run_attention_ablation(cli_args)
    elif cli_args.mode == "core_ablation":
        run_core_ablation(cli_args)
    elif cli_args.mode == "context_ablation":
        run_context_ablation(cli_args)
    elif cli_args.mode == "latent_dim":
        run_latent_dim_sensitivity(cli_args)
    elif cli_args.mode == "reward_sensitivity":
        run_reward_sensitivity(cli_args)
    elif cli_args.mode == "training_sensitivity":
        run_training_sensitivity(cli_args)
    elif cli_args.mode == "rl_baselines":
        run_rl_baselines(cli_args)
