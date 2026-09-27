"""Constructed interval-aware quadratic minimum-movement reference.

This module is NOT a reproduction of a named publication. It projects observed
centres onto a common endpoint-consensus target while keeping half-widths fixed:

    min_y sum_i (y_i - c_i)^2
    s.t. Var(y) + Var(h) <= (-log(threshold) / decay)^2,
         safe_low + h_i <= y_i <= safe_high - h_i.

Costs are uniform; neither hidden response types nor hidden cost coefficients
are available to this reference. The caller still executes bounded suggestions
through the original environment and charges that environment's actual costs.
The threshold here is a TARGET parameter, not the experiment's stopping rule.
There is no silent margin: callers may explicitly request, e.g., threshold=.905.

Run this file directly for numerical and independent SLSQP verification tests.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import brentq, minimize


def _kkt_residual(y, c, lower, upper, multiplier):
    gradient = y - c + multiplier * (y - y.mean())
    at_lower = y <= lower + 1e-10
    at_upper = y >= upper - 1e-10
    residual = np.abs(gradient)
    residual[at_lower] = np.maximum(-gradient[at_lower], 0.0)
    residual[at_upper] = np.maximum(gradient[at_upper], 0.0)
    residual[at_lower & at_upper] = 0.0
    return float(residual.max(initial=0.0))


def interval_qp_target(
    centres,
    widths,
    threshold=.9,
    decay=5.,
    safe_low=.1,
    safe_high=.9,
):
    """Return ``(target_centres, diagnostics)`` without changing either input.

    ``widths`` means HALF-widths. For valid, feasible inputs the target is the
    unique optimum of the stated strictly convex quadratic projection problem.
    Infeasible half-width dispersion is reported, not hidden by shrinking h.
    If any half-width cannot fit within the safe range, the returned constant
    midpoint target minimizes centre dispersion, but is explicitly unsafe.
    """
    c = np.asarray(centres, dtype=float)
    h = np.asarray(widths, dtype=float)
    if c.ndim != 1 or h.ndim != 1 or c.shape != h.shape or c.size == 0:
        raise ValueError("centres and half-widths must be matching nonempty 1-D arrays")
    if not np.isfinite(c).all() or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("centres and half-widths must be finite; half-widths nonnegative")
    if not (np.isfinite(threshold) and 0 < threshold <= 1):
        raise ValueError("threshold must belong to (0, 1]")
    if not (np.isfinite(decay) and decay > 0):
        raise ValueError("decay must be finite and positive")
    if not (np.isfinite(safe_low) and np.isfinite(safe_high) and safe_low < safe_high):
        raise ValueError("safe_low and safe_high must be finite and ordered")

    lower, upper = safe_low + h, safe_high - h
    dispersion_budget = float((-np.log(threshold) / decay) ** 2)
    width_variance = float(np.var(h))
    centre_budget = dispersion_budget - width_variance
    safe_box_exists = bool(np.all(lower <= upper))
    common_low, common_high = float(lower.max()), float(upper.min())
    midpoint = .5 * (safe_low + safe_high)

    def diagnostics(y, feasible, solver, multiplier=None, reason=None):
        centre_variance = float(np.var(y))
        value = {
            "reference_kind": "constructed_interval_quadratic_projection",
            "feasible": bool(feasible),
            "solver": solver,
            "reason": reason,
            "target_threshold": float(threshold),
            "decay": float(decay),
            "half_width_variance": width_variance,
            "centre_variance_budget": float(centre_budget),
            "centre_variance": centre_variance,
            "endpoint_dispersion_squared": centre_variance + width_variance,
            "variance_constraint_excess": float(max(centre_variance - centre_budget, 0)),
            "safe_bound_excess": float(max(np.max(lower - y), np.max(y - upper), 0)),
            "endpoint_factor_exp": float(np.exp(-decay * np.sqrt(centre_variance + width_variance))),
            "squared_target_movement": float(np.sum((y - c) ** 2)),
            "dual_multiplier": None if multiplier is None else float(multiplier),
        }
        if multiplier is not None:
            value["kkt_stationarity_residual"] = _kkt_residual(y, c, lower, upper, multiplier)
            value["kkt_complementarity_residual"] = float(
                abs(multiplier * c.size * (centre_variance - centre_budget))
            )
        return value

    if not safe_box_exists:
        y = np.full_like(c, midpoint)
        return y, diagnostics(y, False, "infeasible_midpoint", reason="half_width_exceeds_safe_half_range")

    if centre_budget < -1e-15:
        # All safe boxes share the same midpoint. A constant vector therefore
        # attains the smallest possible endpoint dispersion without changing h.
        y = np.full_like(c, np.clip(c.mean(), common_low, common_high))
        return y, diagnostics(y, False, "infeasible_closest_constant", reason="half_width_variance_exceeds_budget")

    centre_budget = max(centre_budget, 0.0)
    if centre_budget <= 1e-24:
        y = np.full_like(c, np.clip(c.mean(), common_low, common_high))
        return y, diagnostics(y, True, "zero_variance_closest_constant")

    clipped = np.clip(c, lower, upper)
    if float(np.var(clipped)) <= centre_budget + 1e-16:
        return clipped, diagnostics(clipped, True, "box_projection", multiplier=0.0)

    variance_c = float(np.var(c))
    if variance_c > centre_budget:
        shrink = np.sqrt(centre_budget / variance_c)
        y = c.mean() + shrink * (c - c.mean())
        if np.all(y >= lower - 1e-14) and np.all(y <= upper + 1e-14):
            multiplier = 1. / shrink - 1.
            return y, diagnostics(y, True, "closed_form_variance_projection", multiplier)

    # The dual minimizer for a multiplier lambda has
    # y_i = clip((c_i + lambda * mean(y)) / (1 + lambda), l_i, u_i).
    # Its mean is found with a scalar monotone root, followed by an outer root
    # on Var(y). Strict convexity gives a unique primal optimum.
    def dual_target(multiplier):
        def mean_equation(mean_y):
            y = np.clip((c + multiplier * mean_y) / (1. + multiplier), lower, upper)
            return float(mean_y - y.mean())

        mean_y = brentq(mean_equation, float(lower.min()), float(upper.max()), xtol=2e-14, rtol=1e-14)
        return np.clip((c + multiplier * mean_y) / (1. + multiplier), lower, upper)

    def variance_equation(multiplier):
        return float(np.var(dual_target(multiplier)) - centre_budget)

    upper_multiplier = 1.
    while variance_equation(upper_multiplier) > 0:
        upper_multiplier *= 2.
        if upper_multiplier > 1e14:
            raise RuntimeError("dual multiplier failed to bracket the feasible target")
    multiplier = brentq(variance_equation, 0., upper_multiplier, xtol=1e-12, rtol=1e-12)
    y = dual_target(multiplier)
    return y, diagnostics(y, True, "box_variance_dual_projection", multiplier)


def _self_test():
    rng = np.random.default_rng(20260926)
    solver_counts = {}
    objective_errors = []
    cases = []
    for n in (10, 40, 100):
        for _ in range(8):
            h = rng.uniform(.02, .08, n)
            c = rng.uniform(.1 + h, .9 - h)
            cases.append((c, h, .9))
        # Outside-safe and highly skewed cases exercise box-active dual roots.
        for _ in range(4):
            h = rng.uniform(.02, .08, n)
            c = rng.uniform(.78, 1.05, n)
            cases.append((c, h, .9))
    cases.extend([
        (np.array([-.2, .8, 1.3]), np.array([.1, .11, .12]), .99),
        (np.array([.05, .06, .07]), np.array([.02, .08, .05]), .7),
    ])
    start = time.perf_counter()
    for c, h, threshold in cases:
        original_c, original_h = c.copy(), h.copy()
        y, info = interval_qp_target(c, h, threshold)
        assert np.array_equal(c, original_c) and np.array_equal(h, original_h)
        solver_counts[info["solver"]] = solver_counts.get(info["solver"], 0) + 1
        if not info["feasible"]:
            assert np.var(h) > (-np.log(threshold) / 5) ** 2
            continue
        assert info["variance_constraint_excess"] < 1e-11, info
        assert info["safe_bound_excess"] < 1e-11, info
        assert info.get("kkt_stationarity_residual", 0) < 2e-9, info
        budget = (-np.log(threshold) / 5) ** 2 - np.var(h)
        # Independent optimizer check from the constant feasible midpoint.
        answer = minimize(
            lambda v: .5 * np.sum((v - c) ** 2),
            np.full_like(c, .5),
            jac=lambda v: v - c,
            method="SLSQP",
            bounds=list(zip(.1 + h, .9 - h)),
            constraints=[{
                "type": "ineq",
                "fun": lambda v: budget - np.var(v),
                "jac": lambda v: -2 * (v - v.mean()) / v.size,
            }],
            options={"ftol": 1e-11, "maxiter": 400},
        )
        assert answer.success, answer.message
        error = abs(np.sum((answer.x - c) ** 2) - np.sum((y - c) ** 2))
        objective_errors.append(error)
        assert error < 1e-7, (error, info)
    infeasible_h = np.array([.01, .15])
    y, info = interval_qp_target([.3, .7], infeasible_h)
    assert not info["feasible"] and np.var(y) < 1e-25
    y, info = interval_qp_target([.3, .7], [.41, .1])
    assert not info["feasible"] and info["safe_bound_excess"] > 0
    y, info = interval_qp_target([.2, .8], [.05, .05], threshold=1.)
    assert info["feasible"] and np.allclose(y, [.5, .5])
    y, info = interval_qp_target([.3], [.05])
    assert info["feasible"] and np.allclose(y, [.3])
    print(json.dumps({
        "status": "passed",
        "random_and_edge_cases": len(cases) + 4,
        "independent_slsqp_comparisons": len(objective_errors),
        "max_objective_difference": max(objective_errors),
        "solver_counts": solver_counts,
        "seconds": time.perf_counter() - start,
    }, indent=2))


if __name__ == "__main__":
    _self_test()
