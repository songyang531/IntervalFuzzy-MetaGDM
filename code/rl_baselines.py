from collections import deque
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    optim = None

from env import OpinionDynamicsEnv


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is not available in the active Python environment.")


def resolve_device(device_arg="auto"):
    require_torch()
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, final_tanh=False):
        super().__init__()
        self.final_tanh = final_tanh
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        y = self.net(x)
        return torch.tanh(y) if self.final_tanh else y


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, state_dependent_std=False):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        if state_dependent_std:
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs):
        h = self.body(obs)
        if self.state_dependent_std:
            log_std = torch.clamp(self.log_std_head(h), -20.0, 2.0)
        else:
            log_std = torch.clamp(self.log_std, -5.0, 1.0)
        return self.mean(h), log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def log_prob(self, obs, action):
        action = torch.clamp(action, -0.999, 0.999)
        raw = 0.5 * torch.log((1 + action) / (1 - action))
        mean, log_std = self.forward(obs)
        std = log_std.exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        entropy = normal.entropy().sum(dim=1, keepdim=True)
        return log_prob.sum(dim=1, keepdim=True), entropy

    def deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SimpleReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, not_done):
        self.buffer.append((obs, action, reward, next_obs, not_done))

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        obs, action, reward, next_obs, not_done = zip(*(self.buffer[i] for i in indices))
        return (
            torch.FloatTensor(np.array(obs)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_obs)).to(device),
            torch.FloatTensor(np.array(not_done)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


def make_env(num_agents, max_steps, threshold, seed, opinion_range, stubbornness_range, cost_range, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    return OpinionDynamicsEnv(
        num_agents=num_agents,
        max_steps=max_steps,
        consensus_threshold=threshold,
        seed=seed,
        opinion_range=opinion_range,
        stubbornness_range=stubbornness_range,
        cost_sensitivity_range=cost_range,
        **env_kwargs,
    )


def flatten_state(state):
    return np.asarray(state, dtype=np.float32).reshape(-1)


def curriculum_stage(episode, train_episodes, threshold, max_steps):
    thresholds = (0.70, 0.80, 0.85, threshold)
    step_values = (90, 100, 110, max_steps)
    stage_len = max(1, train_episodes // len(thresholds))
    stage_idx = min(episode // stage_len, len(thresholds) - 1)
    return min(thresholds[stage_idx], threshold), min(step_values[stage_idx], max_steps)


def train_ddpg(
        num_agents=10,
        train_episodes=5000,
        max_steps=120,
        threshold=0.9,
        seed=20260507,
        device_arg="auto",
        lr=3e-4,
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_range=(0.1, 0.9),
        env_kwargs=None,
        save_dir=Path("revision_results") / "checkpoints"
):
    set_seed(seed)
    device = resolve_device(device_arg)
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    probe_env = make_env(
        num_agents, max_steps, threshold, seed,
        opinion_range, stubbornness_range, cost_range, env_kwargs
    )
    obs_dim = num_agents * probe_env.state_dim
    action_dim = num_agents
    actor = MLP(obs_dim, action_dim, final_tanh=True).to(device)
    actor_target = MLP(obs_dim, action_dim, final_tanh=True).to(device)
    critic = MLP(obs_dim + action_dim, 1).to(device)
    critic_target = MLP(obs_dim + action_dim, 1).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(critic.parameters(), lr=lr)
    replay = SimpleReplayBuffer()
    batch_size = 64
    warmup = min(200, max(20, int(train_episodes * 0.03)))
    gamma = 0.95
    tau = 0.01

    progress = tqdm(
        range(train_episodes),
        desc=f"train DDPG N={num_agents}",
        dynamic_ncols=True
    )
    for episode in progress:
        env_threshold, env_steps = curriculum_stage(episode, train_episodes, threshold, max_steps)
        env = make_env(
            num_agents, env_steps, env_threshold, seed + episode * 997,
            opinion_range, stubbornness_range, cost_range, env_kwargs
        )
        state = env.reset()
        obs = flatten_state(state)
        episode_reward = 0.0
        info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions), "success": False}
        step_count = 0
        updates = 0
        for step_count in range(1, env.max_steps + 1):
            if episode < warmup:
                action = rng.uniform(-1.0, 1.0, action_dim)
            else:
                with torch.no_grad():
                    action = actor(torch.FloatTensor(obs).unsqueeze(0).to(device)).cpu().numpy()[0]
                noise_std = max(0.25 * (1.0 - episode / max(train_episodes, 1)), 0.05)
                action = np.clip(action + rng.normal(0.0, noise_std, action_dim), -1.0, 1.0)
            next_state, reward_vec, done, info = env.step(action)
            next_obs = flatten_state(next_state)
            step_reward = float(np.mean(reward_vec))
            replay.push(obs, action, step_reward, next_obs, float(not done))
            episode_reward += step_reward
            obs = next_obs

            if len(replay) >= batch_size:
                b_obs, b_action, b_reward, b_next_obs, b_not_done = replay.sample(batch_size, device)
                with torch.no_grad():
                    next_action = actor_target(b_next_obs)
                    target_q = critic_target(torch.cat([b_next_obs, next_action], dim=1))
                    y = b_reward + b_not_done * gamma * target_q
                q = critic(torch.cat([b_obs, b_action], dim=1))
                critic_loss = F.mse_loss(q, y)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                actor_loss = -critic(torch.cat([b_obs, actor(b_obs)], dim=1)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                for target, param in zip(actor_target.parameters(), actor.parameters()):
                    target.data.copy_(target.data * (1.0 - tau) + param.data * tau)
                for target, param in zip(critic_target.parameters(), critic.parameters()):
                    target.data.copy_(target.data * (1.0 - tau) + param.data * tau)
                updates += 1
            if done:
                break

        if episode % 10 == 0 or episode == train_episodes - 1:
            progress.set_postfix({
                "cons": f"{info.get('consensus_level', 0.0):.3f}",
                "success": int(info.get("success", False)),
                "thr": f"{env_threshold:.2f}",
                "steps": step_count,
                "reward": f"{episode_reward:.1f}",
                "buffer": len(replay),
                "upd": updates
            })

    checkpoint = save_dir / f"ddpg_N{num_agents}_ep{train_episodes}_seed{seed}.pth"
    torch.save({
        "algorithm": "DDPG",
        "actor": actor.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_agents": num_agents,
        "state_dim": probe_env.state_dim,
        "seed": seed,
    }, checkpoint)
    return str(checkpoint)


def train_sac(
        num_agents=10,
        train_episodes=5000,
        max_steps=120,
        threshold=0.9,
        seed=20260507,
        device_arg="auto",
        lr=3e-4,
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_range=(0.1, 0.9),
        env_kwargs=None,
        save_dir=Path("revision_results") / "checkpoints"
):
    set_seed(seed)
    device = resolve_device(device_arg)
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    probe_env = make_env(
        num_agents, max_steps, threshold, seed,
        opinion_range, stubbornness_range, cost_range, env_kwargs
    )
    obs_dim = num_agents * probe_env.state_dim
    action_dim = num_agents
    actor = GaussianActor(obs_dim, action_dim, state_dependent_std=True).to(device)
    q1 = MLP(obs_dim + action_dim, 1).to(device)
    q2 = MLP(obs_dim + action_dim, 1).to(device)
    q1_target = MLP(obs_dim + action_dim, 1).to(device)
    q2_target = MLP(obs_dim + action_dim, 1).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    replay = SimpleReplayBuffer()
    batch_size = 64
    warmup = min(200, max(20, int(train_episodes * 0.03)))
    gamma = 0.95
    tau = 0.01
    alpha = 0.1

    progress = tqdm(
        range(train_episodes),
        desc=f"train SAC N={num_agents}",
        dynamic_ncols=True
    )
    for episode in progress:
        env_threshold, env_steps = curriculum_stage(episode, train_episodes, threshold, max_steps)
        env = make_env(
            num_agents, env_steps, env_threshold, seed + episode * 997,
            opinion_range, stubbornness_range, cost_range, env_kwargs
        )
        obs = flatten_state(env.reset())
        episode_reward = 0.0
        info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions), "success": False}
        step_count = 0
        updates = 0
        for step_count in range(1, env.max_steps + 1):
            if episode < warmup:
                action = rng.uniform(-1.0, 1.0, action_dim)
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action_t, _ = actor.sample(obs_t)
                action = action_t.cpu().numpy()[0]

            next_state, reward_vec, done, info = env.step(action)
            next_obs = flatten_state(next_state)
            step_reward = float(np.mean(reward_vec))
            replay.push(obs, action, step_reward, next_obs, float(not done))
            episode_reward += step_reward
            obs = next_obs

            if len(replay) >= batch_size:
                b_obs, b_action, b_reward, b_next_obs, b_not_done = replay.sample(batch_size, device)
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(b_next_obs)
                    next_q_input = torch.cat([b_next_obs, next_action], dim=1)
                    next_q = torch.min(q1_target(next_q_input), q2_target(next_q_input))
                    y = b_reward + b_not_done * gamma * (next_q - alpha * next_log_prob)

                q_input = torch.cat([b_obs, b_action], dim=1)
                q_loss = F.mse_loss(q1(q_input), y) + F.mse_loss(q2(q_input), y)
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                pi, log_pi = actor.sample(b_obs)
                q_pi_input = torch.cat([b_obs, pi], dim=1)
                actor_loss = (alpha * log_pi - torch.min(q1(q_pi_input), q2(q_pi_input))).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                for target, param in zip(q1_target.parameters(), q1.parameters()):
                    target.data.copy_(target.data * (1.0 - tau) + param.data * tau)
                for target, param in zip(q2_target.parameters(), q2.parameters()):
                    target.data.copy_(target.data * (1.0 - tau) + param.data * tau)
                updates += 1
            if done:
                break

        if episode % 10 == 0 or episode == train_episodes - 1:
            progress.set_postfix({
                "cons": f"{info.get('consensus_level', 0.0):.3f}",
                "success": int(info.get("success", False)),
                "thr": f"{env_threshold:.2f}",
                "steps": step_count,
                "reward": f"{episode_reward:.1f}",
                "buffer": len(replay),
                "upd": updates
            })

    checkpoint = save_dir / f"sac_N{num_agents}_ep{train_episodes}_seed{seed}.pth"
    torch.save({
        "algorithm": "SAC",
        "actor": actor.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_agents": num_agents,
        "state_dim": probe_env.state_dim,
        "seed": seed,
        "state_dependent_std": True,
        "gamma": gamma,
        "tau": tau,
        "alpha": alpha,
    }, checkpoint)
    return str(checkpoint)


def train_ppo(
        num_agents=10,
        train_episodes=5000,
        max_steps=120,
        threshold=0.9,
        seed=20260507,
        device_arg="auto",
        lr=3e-4,
        opinion_range=(0.1, 0.9),
        stubbornness_range=(0.1, 0.9),
        cost_range=(0.1, 0.9),
        env_kwargs=None,
        save_dir=Path("revision_results") / "checkpoints"
):
    set_seed(seed)
    device = resolve_device(device_arg)
    save_dir.mkdir(parents=True, exist_ok=True)
    probe_env = make_env(
        num_agents, max_steps, threshold, seed,
        opinion_range, stubbornness_range, cost_range, env_kwargs
    )
    obs_dim = num_agents * probe_env.state_dim
    action_dim = num_agents
    actor = GaussianActor(obs_dim, action_dim).to(device)
    value_fn = MLP(obs_dim, 1).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    value_opt = optim.Adam(value_fn.parameters(), lr=lr)
    gamma = 0.95
    clip_eps = 0.2
    update_every = 10
    epochs = 4
    storage = []

    progress = tqdm(
        range(train_episodes),
        desc=f"train PPO N={num_agents}",
        dynamic_ncols=True
    )
    for episode in progress:
        env_threshold, env_steps = curriculum_stage(episode, train_episodes, threshold, max_steps)
        env = make_env(
            num_agents, env_steps, env_threshold, seed + episode * 997,
            opinion_range, stubbornness_range, cost_range, env_kwargs
        )
        obs = flatten_state(env.reset())
        episode_transitions = []
        episode_reward = 0.0
        info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions), "success": False}
        step_count = 0
        for step_count in range(1, env.max_steps + 1):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_t, log_prob_t = actor.sample(obs_t)
                value_t = value_fn(obs_t)
            action = action_t.cpu().numpy()[0]
            next_state, reward_vec, done, info = env.step(action)
            reward = float(np.mean(reward_vec))
            episode_reward += reward
            episode_transitions.append((
                obs,
                action,
                reward,
                float(done),
                float(log_prob_t.cpu().item()),
                float(value_t.cpu().item()),
            ))
            obs = flatten_state(next_state)
            if done:
                break

        ret = 0.0
        processed = []
        for obs, action, reward, done, log_prob, value in reversed(episode_transitions):
            ret = reward + gamma * ret * (1.0 - done)
            processed.append((obs, action, ret, log_prob, value))
        storage.extend(reversed(processed))

        if (episode + 1) % update_every == 0 and storage:
            obs_b, action_b, return_b, old_log_b, value_b = zip(*storage)
            obs_b = torch.FloatTensor(np.array(obs_b)).to(device)
            action_b = torch.FloatTensor(np.array(action_b)).to(device)
            return_b = torch.FloatTensor(np.array(return_b)).unsqueeze(1).to(device)
            old_log_b = torch.FloatTensor(np.array(old_log_b)).unsqueeze(1).to(device)
            advantage_b = return_b - torch.FloatTensor(np.array(value_b)).unsqueeze(1).to(device)
            advantage_b = (advantage_b - advantage_b.mean()) / (advantage_b.std() + 1e-6)

            for _ in range(epochs):
                new_log, entropy = actor.log_prob(obs_b, action_b)
                ratio = torch.exp(new_log - old_log_b)
                unclipped = ratio * advantage_b
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage_b
                actor_loss = -torch.min(unclipped, clipped).mean() - 0.001 * entropy.mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                value = value_fn(obs_b)
                value_loss = F.mse_loss(value, return_b)
                value_opt.zero_grad()
                value_loss.backward()
                value_opt.step()
            storage = []

        if episode % 10 == 0 or episode == train_episodes - 1:
            progress.set_postfix({
                "cons": f"{info.get('consensus_level', 0.0):.3f}",
                "success": int(info.get("success", False)),
                "thr": f"{env_threshold:.2f}",
                "steps": step_count,
                "reward": f"{episode_reward:.1f}",
                "storage": len(storage)
            })

    checkpoint = save_dir / f"ppo_N{num_agents}_ep{train_episodes}_seed{seed}.pth"
    torch.save({
        "algorithm": "PPO",
        "actor": actor.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_agents": num_agents,
        "state_dim": probe_env.state_dim,
        "seed": seed,
    }, checkpoint)
    return str(checkpoint)


def load_policy(checkpoint_path, device_arg="auto"):
    device = resolve_device(device_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algorithm = checkpoint["algorithm"]
    obs_dim = int(checkpoint["obs_dim"])
    action_dim = int(checkpoint["action_dim"])
    if algorithm == "SAC":
        actor_state = checkpoint["actor"]
        state_dependent_std = bool(
            checkpoint.get("state_dependent_std", False)
            or any(key.startswith("log_std_head.") for key in actor_state)
        )
        actor = GaussianActor(obs_dim, action_dim, state_dependent_std=state_dependent_std).to(device)
    elif algorithm == "PPO":
        actor = GaussianActor(obs_dim, action_dim).to(device)
    else:
        actor = MLP(obs_dim, action_dim, final_tanh=True).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return algorithm, actor, device


def select_action(actor, algorithm, state, device):
    obs = torch.FloatTensor(flatten_state(state)).unsqueeze(0).to(device)
    with torch.no_grad():
        if algorithm in {"PPO", "SAC"}:
            action = actor.deterministic(obs)
        else:
            action = actor(obs)
    return action.cpu().numpy()[0]
