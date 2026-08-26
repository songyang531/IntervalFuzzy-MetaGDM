import numpy as np


CONTEXT_REWARD_SCALE = 10.0


def build_context_step(suggestions, actual_movements, reward, reward_scale=CONTEXT_REWARD_SCALE):
    suggestions = np.asarray(suggestions, dtype=np.float32)
    actual_movements = np.asarray(actual_movements, dtype=np.float32)
    reward = np.asarray(reward, dtype=np.float32)

    if reward.ndim == 0:
        reward = np.full(suggestions.shape, float(reward), dtype=np.float32)

    if suggestions.shape != actual_movements.shape or suggestions.shape != reward.shape:
        raise ValueError(
            "Context arrays must have matching shapes: "
            f"suggestions={suggestions.shape}, movements={actual_movements.shape}, reward={reward.shape}"
        )

    context_step = np.zeros((suggestions.shape[0], 3), dtype=np.float32)
    context_step[:, 0] = suggestions
    context_step[:, 1] = actual_movements
    context_step[:, 2] = reward / reward_scale
    return context_step


def append_context_step(context, suggestions, actual_movements, reward, reward_scale=CONTEXT_REWARD_SCALE):
    context = np.roll(context, shift=-1, axis=1)
    context[:, -1, :] = build_context_step(suggestions, actual_movements, reward, reward_scale)
    return context
