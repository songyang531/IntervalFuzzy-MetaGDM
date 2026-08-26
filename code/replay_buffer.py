import numpy as np
import torch
from collections import deque
from typing import List, Tuple
from context_utils import build_context_step


class MetaReplayBuffer:
    def __init__(self,
                 capacity: int = 10000,
                 context_window: int = 10):

        self.capacity = capacity
        self.context_window = context_window
        self.buffer = deque(maxlen=capacity)

    def push_episode(self, episode_data: List[Tuple]):
        self.buffer.append(episode_data)

    def sample(self, batch_size: int, device: str = 'cpu'):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        batch_contexts = []
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_next_contexts = []
        batch_not_dones = []

        for idx in indices:
            episode = self.buffer[idx]
            t = np.random.randint(0, len(episode))
            state, action, reward, next_state, not_done, info = episode[t]
            context = self._build_context(episode, t)
            next_context = self._build_context(episode, t + 1)
            batch_contexts.append(context)
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_next_contexts.append(next_context)
            batch_not_dones.append(not_done)

        return (
            torch.FloatTensor(np.array(batch_contexts)).to(device),
            torch.FloatTensor(np.array(batch_states)).to(device),
            torch.FloatTensor(np.array(batch_actions)).to(device),
            torch.FloatTensor(np.array(batch_rewards)).to(device),
            torch.FloatTensor(np.array(batch_next_states)).to(device),
            torch.FloatTensor(np.array(batch_next_contexts)).to(device),
            torch.FloatTensor(np.array(batch_not_dones)).to(device)
        )

    def _build_context(self, episode: List, current_t: int) -> np.ndarray:
        num_agents = len(episode[0][0])
        context = np.zeros((num_agents, self.context_window, 3))
        start_t = max(0, current_t - self.context_window)
        history_indices = list(range(start_t, current_t))
        offset = self.context_window - len(history_indices)

        for i, t in enumerate(history_indices):
            _, action, reward, _, _, info = episode[t]
            context[:, offset + i, :] = build_context_step(
                info['suggestions'],
                info['actual_movements'],
                reward
            )

        return context

    def __len__(self):
        return len(self.buffer)
