"""Pure NumPy metrics for frozen-policy interval audits.

The reference interval extension stores one scalar interval per group member,
not a pairwise interval-valued preference matrix.  Consequently, reciprocity
and additive-transitivity indices are deliberately outside this module.
"""

from __future__ import annotations

import itertools
from typing import Dict

import numpy as np


def _consensus_from_dispersion(dispersion: float, decay: float = 5.0) -> float:
    dispersion = float(dispersion)
    if dispersion < 0.01:
        return 1.0
    return float(np.exp(-decay * dispersion))


def interval_snapshot_metrics(
    centres: np.ndarray,
    half_widths: np.ndarray,
    *,
    decay: float = 5.0,
    threshold: float = 0.90,
) -> Dict[str, float]:
    """Return center/width/endpoint consensus and interval proximity metrics."""
    centres = np.asarray(centres, dtype=np.float64)
    half_widths = np.asarray(half_widths, dtype=np.float64)
    if centres.ndim != 1 or half_widths.ndim != 1 or centres.shape != half_widths.shape:
        raise ValueError("centres and half_widths must be equal-length one-dimensional arrays")
    if centres.size < 2:
        raise ValueError("at least two members are required")
    if not np.all(np.isfinite(centres)) or not np.all(np.isfinite(half_widths)):
        raise ValueError("interval arrays must be finite")
    if np.any(half_widths < 0):
        raise ValueError("half widths must be nonnegative")

    lower = centres - half_widths
    upper = centres + half_widths
    center_std = float(np.std(centres))
    width_std = float(np.std(half_widths))
    lower_std = float(np.std(lower))
    upper_std = float(np.std(upper))
    endpoint_rms_dispersion = float(np.sqrt(0.5 * (lower_std**2 + upper_std**2)))

    group_lower = float(np.mean(lower))
    group_upper = float(np.mean(upper))
    member_group_hausdorff = np.maximum(
        np.abs(lower - group_lower), np.abs(upper - group_upper)
    )
    pairs = list(itertools.combinations(range(centres.size), 2))
    pairwise_hausdorff = np.asarray(
        [max(abs(lower[i] - lower[j]), abs(upper[i] - upper[j])) for i, j in pairs],
        dtype=np.float64,
    )

    intersection = max(0.0, float(np.min(upper) - np.max(lower)))
    union = max(0.0, float(np.max(upper) - np.min(lower)))
    common_overlap_ratio = intersection / union if union > 0 else 1.0

    width_ceiling = _consensus_from_dispersion(width_std, decay)
    return {
        "center_std": center_std,
        "half_width_std": width_std,
        "lower_endpoint_std": lower_std,
        "upper_endpoint_std": upper_std,
        "endpoint_rms_dispersion": endpoint_rms_dispersion,
        "center_consensus": _consensus_from_dispersion(center_std, decay),
        "width_consensus": width_ceiling,
        "lower_endpoint_consensus": _consensus_from_dispersion(lower_std, decay),
        "upper_endpoint_consensus": _consensus_from_dispersion(upper_std, decay),
        "endpoint_consensus": _consensus_from_dispersion(endpoint_rms_dispersion, decay),
        "width_limited_consensus_ceiling": width_ceiling,
        "width_ceiling_reaches_threshold": float(width_ceiling >= threshold),
        "mean_member_group_hausdorff": float(np.mean(member_group_hausdorff)),
        "max_member_group_hausdorff": float(np.max(member_group_hausdorff)),
        "mean_pairwise_hausdorff": float(np.mean(pairwise_hausdorff)),
        "max_pairwise_hausdorff": float(np.max(pairwise_hausdorff)),
        "common_overlap_ratio": float(common_overlap_ratio),
        "all_intervals_valid": float(
            np.all(lower <= upper) and np.all(lower >= 0.0) and np.all(upper <= 1.0)
        ),
    }


def interval_information_distortion(
    initial_centres: np.ndarray,
    initial_half_widths: np.ndarray,
    final_centres: np.ndarray,
    final_half_widths: np.ndarray,
) -> Dict[str, float]:
    """Measure per-member endpoint Hausdorff and center/width distortion."""
    c0 = np.asarray(initial_centres, dtype=np.float64)
    h0 = np.asarray(initial_half_widths, dtype=np.float64)
    c1 = np.asarray(final_centres, dtype=np.float64)
    h1 = np.asarray(final_half_widths, dtype=np.float64)
    if not (c0.shape == h0.shape == c1.shape == h1.shape) or c0.ndim != 1:
        raise ValueError("all interval arrays must be equal-length one-dimensional arrays")
    l0, u0 = c0 - h0, c0 + h0
    l1, u1 = c1 - h1, c1 + h1
    hausdorff = np.maximum(np.abs(l1 - l0), np.abs(u1 - u0))
    center_shift = np.abs(c1 - c0)
    width_shift = np.abs(h1 - h0)
    return {
        "mean_hausdorff_distortion": float(np.mean(hausdorff)),
        "max_hausdorff_distortion": float(np.max(hausdorff)),
        "total_hausdorff_distortion": float(np.sum(hausdorff)),
        "mean_center_shift": float(np.mean(center_shift)),
        "max_center_shift": float(np.max(center_shift)),
        "mean_half_width_shift": float(np.mean(width_shift)),
        "max_half_width_shift": float(np.max(width_shift)),
        "translation_identity_max_error": float(np.max(np.abs(hausdorff - center_shift))),
    }
