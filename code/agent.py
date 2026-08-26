import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from model import ContextEncoder, build_social_module


class Actor(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, hidden_dim=256, aggregation="social_attention", num_heads=4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.agent_feat_dim = state_dim + z_dim
        self.use_group_context = aggregation != "no_attention"
        if self.use_group_context:
            self.social_attn = build_social_module(self.agent_feat_dim, hidden_dim, num_heads=num_heads, aggregation=aggregation)
            decision_input_dim = hidden_dim + self.agent_feat_dim
        else:
            self.social_attn = None
            decision_input_dim = self.agent_feat_dim
        self.l1 = nn.Linear(decision_input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, z):
        batch_size = state.shape[0]
        num_agents = z.shape[1]
        state_reshaped = state.view(batch_size, num_agents, self.state_dim)
        agent_inputs = torch.cat([state_reshaped, z], dim=2)
        if self.use_group_context:
            s_global = self.social_attn(agent_inputs)
            s_global_expanded = s_global.unsqueeze(1).expand(-1, num_agents, -1)
            actor_input = torch.cat([s_global_expanded, agent_inputs], dim=-1)
        else:
            actor_input = agent_inputs
        x = F.relu(self.l1(actor_input))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state, z):
        mean, log_std = self.forward(state, z)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1).sum(dim=1, keepdim=True)
        return y_t.squeeze(-1), log_prob, torch.tanh(mean).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, hidden_dim=256, aggregation="social_attention", num_heads=4):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.agent_feat_dim = state_dim + z_dim
        self.critic_feat_dim = self.agent_feat_dim + action_dim
        self.use_group_context = aggregation != "no_attention"
        if self.use_group_context:
            self.attn1 = build_social_module(self.critic_feat_dim, hidden_dim, num_heads=num_heads, aggregation=aggregation)
            self.attn2 = build_social_module(self.critic_feat_dim, hidden_dim, num_heads=num_heads, aggregation=aggregation)
        else:
            self.attn1 = None
            self.attn2 = None
            self.agent_q1_encoder = nn.Linear(self.critic_feat_dim, hidden_dim)
            self.agent_q2_encoder = nn.Linear(self.critic_feat_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, z, action):
        batch_size = state.shape[0]
        num_agents = z.shape[1]
        state_reshaped = state.view(batch_size, num_agents, self.state_dim)
        agent_inputs = torch.cat([state_reshaped, z], dim=2)
        if action.dim() == 2:
            action = action.unsqueeze(2)

        critic_inputs = torch.cat([agent_inputs, action], dim=-1)
        if self.use_group_context:
            s_global_1 = self.attn1(critic_inputs)
            s_global_2 = self.attn2(critic_inputs)
        else:
            s_global_1 = F.relu(self.agent_q1_encoder(critic_inputs)).mean(dim=1)
            s_global_2 = F.relu(self.agent_q2_encoder(critic_inputs)).mean(dim=1)

        q1 = self.l2(F.relu(self.l1(s_global_1)))

        q2 = self.l4(F.relu(self.l3(s_global_2)))

        return q1, q2


class MetaAgent:
    def __init__(self, state_dim, action_dim, z_dim=8, hidden_dim=256, lr=3e-4,
                 gamma=0.95, tau=0.01, alpha=0.1,
                 context_window=10, kl_weight=0.05, personality_weight=0.0,
                 aggregation="social_attention", num_heads=4, context_mode="learned", device="cuda"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.z_dim = z_dim
        self.kl_weight = kl_weight
        self.context_window = context_window
        self.aggregation = aggregation
        self.num_heads = num_heads
        self.context_mode = context_mode
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.encoder = ContextEncoder(input_dim=3, z_dim=z_dim, num_heads=num_heads).to(device)
        self.actor = Actor(state_dim, z_dim, action_dim, hidden_dim, aggregation=aggregation, num_heads=num_heads).to(device)
        self.critic = Critic(state_dim, z_dim, action_dim, hidden_dim, aggregation=aggregation, num_heads=num_heads).to(device)
        self.critic_target = Critic(state_dim, z_dim, action_dim, hidden_dim, aggregation=aggregation, num_heads=num_heads).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, context, evaluate=False, exploration_noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        context = torch.FloatTensor(context).to(self.device)

        with torch.no_grad():
            context_batch = context.unsqueeze(0)
            batch_size = 1
            num_agents = context.shape[0]
            seq_len = context.shape[1]
            feat_dim = context.shape[2]

            z, _, _, _ = self._encode_context(
                context_batch.view(batch_size, num_agents, seq_len, feat_dim),
                sample=False
            )

            if evaluate:
                _, _, action = self.actor.sample(state, z)
            else:
                action, _, _ = self.actor.sample(state, z)

        action = action.cpu().numpy()[0]

        if exploration_noise > 0:
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action

    def update(self, batch):
        if len(batch) == 6:
            context, state, action, reward, next_state, not_done = batch
            next_context = context
        else:
            context, state, action, reward, next_state, next_context, not_done = batch

        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        elif reward.dim() == 2:
            reward = reward.mean(dim=1, keepdim=True)
        else:
            reward = reward.view(reward.size(0), -1).mean(dim=1, keepdim=True)
        not_done = not_done.view(-1, 1)

        batch_size, num_agents, seq_len, feat_dim = context.shape
        z, mu, std, kl_loss = self._encode_context(context, sample=True)

        with torch.no_grad():
            next_z, _, _, _ = self._encode_context(next_context, sample=False)
            next_action, next_log_pi, _ = self.actor.sample(next_state, next_z)
            q1_next, q2_next = self.critic_target(next_state, next_z, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            next_q_value = reward + not_done * self.gamma * min_q_next

        q1, q2 = self.critic(state, z, action)
        qf_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        encoder_total_loss = qf_loss + self.kl_weight * kl_loss
        self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        encoder_total_loss.backward()
        if self.context_mode != "zero":
            self.encoder_optimizer.step()
        self.critic_optimizer.step()
        pi, log_pi, _ = self.actor.sample(state, z.detach())
        q1_pi, q2_pi = self.critic(state, z.detach(), pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)
        return {
            'critic_loss': qf_loss.item(),
            'actor_loss': policy_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def _encode_context(self, context, sample=True):
        batch_size, num_agents, seq_len, feat_dim = context.shape
        if self.context_mode == "zero":
            z = torch.zeros(batch_size, num_agents, self.z_dim, device=context.device)
            zero = torch.zeros((), device=context.device)
            return z, None, None, zero

        flat_context = context.view(-1, seq_len, feat_dim)
        mu, std = self.encoder(flat_context)
        if self.context_mode == "deterministic" or not sample:
            z_flat = mu
            kl_loss = torch.zeros((), device=context.device)
        elif self.context_mode == "learned":
            z_flat = self.encoder.sample_z(mu, std)
            kl_loss = -0.5 * torch.sum(
                1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2),
                dim=1
            ).mean()
        else:
            raise ValueError(f"Unknown context_mode: {self.context_mode}")

        z = z_flat.view(batch_size, num_agents, -1)
        return z, mu, std, kl_loss

    def _load_compatible_state_dict(self, module, state_dict, name, allow_partial=False):
        try:
            module.load_state_dict(state_dict)
            return
        except RuntimeError as exc:
            if not allow_partial:
                print(f"Warning: skipped loading {name}; checkpoint architecture is incompatible.")
                return
            current = module.state_dict()
            compatible = {
                key: value
                for key, value in state_dict.items()
                if key in current and current[key].shape == value.shape
            }
            if not compatible:
                print(f"Warning: skipped loading {name}; checkpoint architecture is incompatible.")
                return
            current.update(compatible)
            module.load_state_dict(current)
            print(
                f"Warning: partially loaded {name} "
                f"({len(compatible)}/{len(current)} tensors); checkpoint architecture differs."
            )

    def save_checkpoint(self, filepath, episode):
        checkpoint = {
            'episode': episode,
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'z_dim': self.z_dim,
            'context_window': self.context_window,
            'aggregation': self.aggregation,
            'num_heads': self.num_heads,
            'context_mode': self.context_mode,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'encoder_opt': self.encoder_optimizer.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"✅ 模型已保存至 {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            return 0
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        encoder_key = 'encoder_state_dict' if 'encoder_state_dict' in checkpoint else 'encoder'
        actor_key = 'actor_state_dict' if 'actor_state_dict' in checkpoint else 'actor'
        critic_key = 'critic_state_dict' if 'critic_state_dict' in checkpoint else 'critic'

        self._load_compatible_state_dict(self.encoder, checkpoint[encoder_key], "encoder", allow_partial=True)
        self._load_compatible_state_dict(self.actor, checkpoint[actor_key], "actor", allow_partial=True)
        self._load_compatible_state_dict(self.critic, checkpoint[critic_key], "critic")

        if 'critic_target_state_dict' in checkpoint:
            self._load_compatible_state_dict(
                self.critic_target,
                checkpoint['critic_target_state_dict'],
                "critic_target"
            )
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())

        optimizer_keys = [
            (self.encoder_optimizer, 'encoder_optimizer', 'encoder_opt'),
            (self.actor_optimizer, 'actor_optimizer', 'actor_opt'),
            (self.critic_optimizer, 'critic_optimizer', 'critic_opt'),
        ]
        for optimizer, new_key, old_key in optimizer_keys:
            opt_key = new_key if new_key in checkpoint else old_key
            if opt_key in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint[opt_key])
                except ValueError:
                    print(f"Warning: skipped loading {opt_key}; optimizer architecture differs.")

        return checkpoint.get('episode', 0) + 1
