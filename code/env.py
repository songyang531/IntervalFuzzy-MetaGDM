import numpy as np
from typing import Dict, Tuple


class OpinionDynamicsEnv:
    def __init__(
            self,
            num_agents: int = 10,
            max_steps: int = 30,
            consensus_threshold: float = 0.90,
            seed: int = None,
            consensus_decay: float = 5.0,
            safe_low: float = 0.1,
            safe_high: float = 0.9,
            boundary_penalty_weight: float = 20.0,
            movement_scale: float = 0.1,
            cost_coefficient: float = 75.0,
            cost_power: float = 2.0,
            distance_penalty_weight: float = 2.0,
            consensus_improvement_weight: float = 50.0,
            fairness_weight: float = 1.0,
            mission_bonus_value: float = 10.0,
            time_penalty: float = 0.1,
            reward_clip: float = 20.0,
            opinion_range=(0.1, 0.9),
            interval_half_width_range=(0.02, 0.08),
            stubbornness_range=(0.1, 0.9),
            cost_sensitivity_range=(0.1, 0.9)
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.consensus_threshold = consensus_threshold
        self.current_step = 0
        # Minimal interval extension of the reference environment.  The action
        # remains one-dimensional and translates the whole interval; only the
        # preference state and interval-aware reward/metrics are changed.
        self.state_dim = 5
        self.action_dim = 1
        self.consensus_decay = consensus_decay
        self.safe_low = safe_low
        self.safe_high = safe_high
        self.boundary_penalty_weight = boundary_penalty_weight
        self.movement_scale = movement_scale
        self.cost_coefficient = cost_coefficient
        self.cost_power = cost_power
        self.distance_penalty_weight = distance_penalty_weight
        self.consensus_improvement_weight = consensus_improvement_weight
        self.fairness_weight = fairness_weight
        self.mission_bonus_value = mission_bonus_value
        self.time_penalty = time_penalty
        self.reward_clip = reward_clip
        self.opinion_range = opinion_range
        self.interval_half_width_range = interval_half_width_range
        self.stubbornness_range = stubbornness_range
        self.cost_sensitivity_range = cost_sensitivity_range
        self.rng = np.random.default_rng(seed)
        # Keep the reference RNG stream for personalities and centres exactly
        # unchanged.  Interval widths use an independent deterministic stream.
        width_seed = np.random.SeedSequence(seed).spawn(1)[0] if seed is not None else None
        self.width_rng = np.random.default_rng(width_seed)
        self.opinions = None
        self.interval_half_widths = None
        self.personalities = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.personalities = {
            'stubbornness': self.rng.uniform(*self.stubbornness_range, self.num_agents),
            'cost_sensitivity': self.rng.uniform(*self.cost_sensitivity_range, self.num_agents)
        }
        self.interval_half_widths = self.width_rng.uniform(
            *self.interval_half_width_range, self.num_agents
        )
        centre_low = self.opinion_range[0] + self.interval_half_widths
        centre_high = self.opinion_range[1] - self.interval_half_widths
        if np.any(centre_low > centre_high):
            raise ValueError("Interval half widths are incompatible with opinion_range")
        self.opinions = self.rng.uniform(centre_low, centre_high, self.num_agents)
        while np.std(self.opinions) < 0.1:
            self.opinions = self.rng.uniform(centre_low, centre_high, self.num_agents)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        consensus_mean = np.mean(self.opinions)
        distance_to_mean = self.opinions - consensus_mean
        mean_half_width = np.mean(self.interval_half_widths)
        half_width_distance = self.interval_half_widths - mean_half_width
        remaining_time = (self.max_steps - self.current_step) / self.max_steps
        state = np.stack([
            self.opinions,
            self.interval_half_widths,
            distance_to_mean,
            half_width_distance,
            np.full(self.num_agents, remaining_time)
        ], axis=1)
        return state

    def _interval_bounds(self, opinions=None):
        centres = self.opinions if opinions is None else np.asarray(opinions, dtype=float)
        widths = np.asarray(self.interval_half_widths, dtype=float)
        return centres - widths, centres + widths

    def _interval_dispersion(self, opinions=None):
        lower, upper = self._interval_bounds(opinions)
        return float(np.sqrt(0.5 * (np.std(lower) ** 2 + np.std(upper) ** 2)))

    def step(self, suggestions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        old_opinions = self.opinions.copy()
        old_std = np.std(old_opinions)
        old_mean = np.mean(old_opinions)
        stubbornness = self.personalities['stubbornness']
        actual_movements = np.clip(
            suggestions * (1.0 - stubbornness) * self.movement_scale,
            -self.movement_scale,
            self.movement_scale
        )
        # The full interval must remain inside the physical opinion domain.
        self.opinions = np.clip(
            self.opinions + actual_movements,
            self.interval_half_widths,
            1.0 - self.interval_half_widths
        )

        new_std = np.std(self.opinions)
        new_mean = np.mean(self.opinions)
        consensus_level = self._calculate_consensus_level_from_opinions(self.opinions)

        rewards, reward_components = self._calculate_rewards_v8(
            old_opinions, self.opinions,
            old_std, new_std,
            old_mean, new_mean,
            actual_movements, consensus_level
        )

        self.current_step += 1
        success = consensus_level >= self.consensus_threshold
        done = success or (self.current_step >= self.max_steps)

        info = {
            'suggestions': suggestions,
            'actual_movements': actual_movements,
            'interval_half_widths': self.interval_half_widths.copy(),
            'interval_lower': self._interval_bounds()[0],
            'interval_upper': self._interval_bounds()[1],
            'interval_dispersion': self._interval_dispersion(),
            'step': self.current_step,
            'consensus_level': consensus_level,
            'std': new_std,
            'success': success,
            'personalities': self.personalities,
            'reward_components': reward_components
        }
        return self._get_state(), rewards, done, info

    def _calculate_rewards_v8(self, old_op, new_op, old_std, new_std, old_mean, new_mean, movements, consensus_level):
        rewards = np.zeros(self.num_agents)
        old_consensus = self._calculate_consensus_level_from_opinions(old_op)
        consensus_delta = consensus_level - old_consensus

        new_lower, new_upper = self._interval_bounds(new_op)
        mean_lower = np.mean(new_lower)
        mean_upper = np.mean(new_upper)
        # RMS endpoint distance is the Euclidean interval distance divided by
        # sqrt(2).  For zero-width intervals it is exactly |o_i - mean(o)|.
        dist_to_mean = np.sqrt(
            0.5 * ((new_lower - mean_lower) ** 2 + (new_upper - mean_upper) ** 2)
        )
        outlier_penalty = -dist_to_mean * self.distance_penalty_weight

        boundary_violation = (
                np.maximum(self.safe_low - new_lower, 0.0) +
                np.maximum(new_upper - self.safe_high, 0.0)
        )
        boundary_penalty = -boundary_violation * self.boundary_penalty_weight

        consensus_reward = consensus_delta * self.consensus_improvement_weight
        mission_bonus = 0.0
        if consensus_level >= self.consensus_threshold:
            mission_bonus = self.mission_bonus_value

        adjustment_cost = self.calculate_adjustment_costs(movements)
        movement_cost = -adjustment_cost
        cost_gini = self._gini(adjustment_cost)
        fairness_penalty = -self.fairness_weight * cost_gini
        shared_reward = consensus_reward + mission_bonus + fairness_penalty - self.time_penalty

        for i in range(self.num_agents):
            r = (shared_reward +
                 outlier_penalty[i] +
                 boundary_penalty[i] +
                 movement_cost[i])
            rewards[i] = np.clip(r, -self.reward_clip, self.reward_clip)

        reward_components = {
            'old_consensus': old_consensus,
            'consensus_delta': consensus_delta,
            'consensus_reward': consensus_reward,
            'mission_bonus': mission_bonus,
            'mean_movement': float(np.mean(np.abs(movements))),
            'interval_dispersion': self._interval_dispersion(new_op),
            'mean_half_width': float(np.mean(self.interval_half_widths)),
            'mean_movement_intensity': float(np.mean(self._movement_intensity(movements))),
            'mean_outlier_penalty': float(np.mean(outlier_penalty)),
            'mean_boundary_penalty': float(np.mean(boundary_penalty)),
            'mean_adjustment_cost': float(np.mean(adjustment_cost)),
            'cost_gini': float(cost_gini),
            'fairness_penalty': float(fairness_penalty),
            'shared_reward': float(shared_reward),
            'mean_reward': float(np.mean(rewards))
        }
        return rewards, reward_components

    def calculate_adjustment_costs(self, movements):
        cost_sens = self.personalities['cost_sensitivity']
        intensity = self._movement_intensity(movements)
        return cost_sens * self.cost_coefficient * self.movement_scale * (intensity ** self.cost_power)

    def _movement_intensity(self, movements):
        if self.movement_scale <= 0:
            return np.zeros_like(movements, dtype=float)
        return np.clip(np.abs(movements) / self.movement_scale, 0.0, 1.0)

    @staticmethod
    def _gini(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return 0.0
        values = values - values.min()
        if np.allclose(values.sum(), 0.0):
            return 0.0
        values = np.sort(values)
        index = np.arange(1, values.size + 1)
        return float((2 * np.sum(index * values) / (values.size * np.sum(values))) - (values.size + 1) / values.size)

    def _calculate_consensus_level_from_opinions(self, opinions):
        std = self._interval_dispersion(opinions)
        if std < 0.01:
            dispersion_score = 1.0
        else:
            dispersion_score = np.exp(-self.consensus_decay * std)
        lower, upper = self._interval_bounds(opinions)
        min_op = np.min(lower)
        max_op = np.max(upper)
        low_penalty = max(0.0, self.safe_low - min_op) / self.safe_low
        high_penalty = max(0.0, max_op - self.safe_high) / (1.0 - self.safe_high)
        safety_score = 1.0 - max(low_penalty, high_penalty)
        safety_score = np.clip(safety_score, 0.0, 1.0)
        return dispersion_score * safety_score

    def get_reward_parameters(self) -> Dict[str, float]:
        return {
            'consensus_decay': self.consensus_decay,
            'safe_low': self.safe_low,
            'safe_high': self.safe_high,
            'boundary_penalty_weight': self.boundary_penalty_weight,
            'movement_scale': self.movement_scale,
            'cost_coefficient': self.cost_coefficient,
            'cost_power': self.cost_power,
            'distance_penalty_weight': self.distance_penalty_weight,
            'consensus_improvement_weight': self.consensus_improvement_weight,
            'fairness_weight': self.fairness_weight,
            'mission_bonus_value': self.mission_bonus_value,
            'time_penalty': self.time_penalty,
            'reward_clip': self.reward_clip,
            'opinion_range': self.opinion_range,
            'interval_half_width_range': self.interval_half_width_range,
            'interval_representation': 'center_half_width',
            'interval_action_semantics': 'scalar_translation',
            'stubbornness_range': self.stubbornness_range,
            'cost_sensitivity_range': self.cost_sensitivity_range
        }
