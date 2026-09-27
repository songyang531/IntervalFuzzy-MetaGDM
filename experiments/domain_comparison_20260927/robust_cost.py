"""Li et al. (EJOR 2024) robust two-stage cost consensus baseline.

The native solver implements the paper's finite-scenario robust two-stage
model by eliminating economically one-sided second-stage recourse.  A second,
independent LP that retains both adjustment directions verifies that this
elimination reaches the same native optima; Benders decomposition changes the
solution procedure, not those optima.

The online ``Controller`` is deliberately labelled a restricted adaptation.
    The evaluation environment reveals one current interval state, not the
    paper's ex-ante catalogue of future decision environments.  We therefore use
    the observable lower/centre/upper interval points as a declared three-point
    sensitivity construction and the public q envelope [0.1, 0.9] as cost
    uncertainty.  Equal scenario weights are an adapter choice, not a
    probability distribution implied by a fuzzy interval or an empirical
    forecast. Hidden willingness is never supplied to, or reinterpreted as a
    cost by, the optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linprog, minimize_scalar


SOURCE_DOI = "10.1016/j.ejor.2024.04.020"
NATIVE_METHOD_LABEL = "Li2024 RTS-MCCM-U-DC direct deterministic equivalent"
METHOD_LABEL = "Li2024 RTS-box interval-scenario restricted adaptation"
METHOD_LABEL_FEEDBACK = METHOD_LABEL + " + observed-gain adapter"


def _weighted_median_midpoint(values: Any, weights: Any) -> float:
    """Return the midpoint of the weighted-L1 minimizer interval.

    Choosing the midpoint makes the analytic specialization agree with
    ``numpy.median`` in the even, equally weighted case while remaining a
    valid minimizer for arbitrary nonnegative scenario weights.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("values and weights must be nonempty vectors of equal length")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("weighted median inputs must be finite with positive total weight")
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    half = 0.5 * float(cumulative[-1])
    lower_index = int(np.searchsorted(cumulative, half, side="left"))
    if abs(float(cumulative[lower_index]) - half) <= 1e-14 * max(1.0, half):
        upper_index = min(lower_index + 1, len(values) - 1)
        return float(0.5 * (values[lower_index] + values[upper_index]))
    return float(values[lower_index])


def robust_support(exposure: np.ndarray, uncertainty: str, radius: float) -> float:
    """Support function of the paper's three perturbation sets.

    ``exposure[l]`` is the coefficient multiplying perturbation ``zeta_l``.
    The sets are Box_gamma, Polyhedral_tau (an l1 ball), and
    Intersection_sigma (l-infinity <= 1 and l1 <= sigma); see PDF pp. 984--986,
    Theorems 4.2, 4.4, and 4.5.
    """
    values = np.abs(np.asarray(exposure, dtype=float).reshape(-1))
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("exposure must be a nonempty finite vector")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("uncertainty radius must be finite and nonnegative")
    key = str(uncertainty).strip().lower()
    if key == "box":
        return float(radius * values.sum())
    if key in {"poly", "polyhedral"}:
        return float(radius * values.max())
    if key == "intersection":
        budget = min(float(radius), float(values.size))
        ordered = np.sort(values)[::-1]
        whole = int(np.floor(budget + 1e-14))
        fraction = budget - whole
        result = float(ordered[:whole].sum())
        if whole < values.size and fraction > 1e-14:
            result += float(fraction * ordered[whole])
        return result
    raise ValueError("uncertainty must be box, polyhedral, or intersection")


def _scenario_array(values: Any, scenarios: int, agents: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (agents,):
        array = np.broadcast_to(array, (scenarios, agents)).copy()
    if array.shape != (scenarios, agents) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({scenarios}, {agents})")
    return array


def _shift_array(values: Any, scenarios: int, agents: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == agents:
        array = np.broadcast_to(array, (scenarios,) + array.shape).copy()
    if array.ndim != 3 or array.shape[0] != scenarios or array.shape[2] != agents:
        raise ValueError(
            f"{name} must have shape (L, {agents}) or ({scenarios}, L, {agents})"
        )
    if array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite perturbation rows")
    return array


def _validated_problem(
    opinions: Any,
    upward_costs: Any,
    downward_costs: Any,
    upward_shifts: Any,
    downward_shifts: Any,
    probabilities: Any | None,
    tolerance: Any | None,
) -> tuple[np.ndarray, ...]:
    opinions = np.asarray(opinions, dtype=float)
    if opinions.ndim != 2 or min(opinions.shape) <= 0 or not np.all(np.isfinite(opinions)):
        raise ValueError("opinions must be a nonempty finite [scenarios, agents] array")
    scenarios, agents = opinions.shape
    upward = _scenario_array(upward_costs, scenarios, agents, "upward_costs")
    downward = _scenario_array(downward_costs, scenarios, agents, "downward_costs")
    if np.any(upward < 0) or np.any(downward < 0):
        raise ValueError("nominal adjustment costs must be nonnegative")
    up_shift = _shift_array(upward_shifts, scenarios, agents, "upward_shifts")
    down_shift = _shift_array(downward_shifts, scenarios, agents, "downward_shifts")
    if up_shift.shape[1] != down_shift.shape[1]:
        raise ValueError("upward and downward shifts must use the same L")
    if probabilities is None:
        probability = np.full(scenarios, 1.0 / scenarios)
    else:
        probability = np.asarray(probabilities, dtype=float).reshape(-1)
    if (
        probability.shape != (scenarios,)
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0)
        or probability.sum() <= 0
    ):
        raise ValueError("probabilities must be finite nonnegative scenario weights")
    probability = probability / probability.sum()
    if tolerance is None:
        free = np.zeros_like(opinions)
    else:
        free = _scenario_array(tolerance, scenarios, agents, "tolerance")
        if np.any(free < 0):
            raise ValueError("tolerance must be nonnegative")
    return opinions, upward, downward, up_shift, down_shift, probability, free


def robust_two_stage_objective(
    target: float,
    opinions: Any,
    upward_costs: Any,
    downward_costs: Any,
    upward_shifts: Any,
    downward_shifts: Any,
    *,
    probabilities: Any | None = None,
    tolerance: Any | None = None,
    uncertainty: str = "box",
    radius: float = 1.0,
    return_details: bool = False,
) -> float | tuple[float, list[dict[str, Any]]]:
    """Evaluate the finite-scenario robust two-stage objective.

    For a fixed first-stage target, economically one-sided recourse is
    ``max(opinion-target-tolerance, 0)`` downward and
    ``max(target-opinion-tolerance, 0)`` upward.  Substitution into the
    uncertainty-set support function yields the direct deterministic
    deterministic form of the paper's nested min/sup formulation.  With
    arbitrary user-supplied shift matrices, simultaneous opposing recourse
    could in principle alter the optimum; the native data are therefore also
    checked with ``_full_recourse_lp`` rather than relying on this assertion.
    """
    (
        opinions,
        upward,
        downward,
        up_shift,
        down_shift,
        probability,
        free,
    ) = _validated_problem(
        opinions,
        upward_costs,
        downward_costs,
        upward_shifts,
        downward_shifts,
        probabilities,
        tolerance,
    )
    target = float(target)
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    details: list[dict[str, Any]] = []
    total = 0.0
    for scenario in range(opinions.shape[0]):
        downward_recourse = np.maximum(
            opinions[scenario] - target - free[scenario], 0.0
        )
        upward_recourse = np.maximum(
            target - opinions[scenario] - free[scenario], 0.0
        )
        nominal = float(
            downward[scenario] @ downward_recourse
            + upward[scenario] @ upward_recourse
        )
        exposure = (
            down_shift[scenario] @ downward_recourse
            + up_shift[scenario] @ upward_recourse
        )
        protection = robust_support(exposure, uncertainty, radius)
        scenario_cost = nominal + protection
        total += float(probability[scenario] * scenario_cost)
        if return_details:
            details.append(
                {
                    "scenario": scenario,
                    "probability": float(probability[scenario]),
                    "nominal_cost": nominal,
                    "robust_support": protection,
                    "scenario_cost": scenario_cost,
                    "downward_recourse": downward_recourse.tolist(),
                    "upward_recourse": upward_recourse.tolist(),
                    "exposure": exposure.tolist(),
                }
            )
    if return_details:
        return float(total), details
    return float(total)


def solve_robust_two_stage(
    opinions: Any,
    upward_costs: Any,
    downward_costs: Any,
    upward_shifts: Any,
    downward_shifts: Any,
    *,
    probabilities: Any | None = None,
    tolerance: Any | None = None,
    uncertainty: str = "box",
    radius: float = 1.0,
    target_bounds: tuple[float, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Solve the scalar first stage and closed-form scenario recourse."""
    validated = _validated_problem(
        opinions,
        upward_costs,
        downward_costs,
        upward_shifts,
        downward_shifts,
        probabilities,
        tolerance,
    )
    opinions_v, upward, downward, up_shift, down_shift, probability, free = validated
    if target_bounds is None:
        lower, upper = float(opinions_v.min()), float(opinions_v.max())
    else:
        lower, upper = map(float, target_bounds)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        raise ValueError("target_bounds must be a finite ordered pair")

    evaluations = 0

    def objective(value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        total = 0.0
        for scenario in range(opinions_v.shape[0]):
            down_rec = np.maximum(
                opinions_v[scenario] - value - free[scenario], 0.0
            )
            up_rec = np.maximum(
                value - opinions_v[scenario] - free[scenario], 0.0
            )
            nominal = float(
                downward[scenario] @ down_rec + upward[scenario] @ up_rec
            )
            exposure = down_shift[scenario] @ down_rec + up_shift[scenario] @ up_rec
            total += float(
                probability[scenario]
                * (nominal + robust_support(exposure, uncertainty, radius))
            )
        return float(total)

    if upper - lower <= 1e-15:
        target = lower
        success = True
        message = "singleton target interval"
    else:
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 1000},
        )
        # Explicitly check all recourse breakpoints, interval endpoints, and
        # the ordinary median.  The latter gives deterministic tie-breaking
        # when a symmetric L1 objective has a flat minimizer interval.
        raw_breaks = np.concatenate(
            [
                opinions_v.ravel(),
                (opinions_v - free).ravel(),
                (opinions_v + free).ravel(),
                np.asarray([lower, upper, np.median(opinions_v)]),
                np.asarray([result.x]) if np.isfinite(result.x) else np.empty(0),
            ]
        )
        candidates = np.unique(np.clip(raw_breaks[np.isfinite(raw_breaks)], lower, upper))
        values = np.asarray([objective(float(value)) for value in candidates])
        best = float(values.min())
        near = candidates[values <= best + 2e-10 * max(1.0, abs(best))]
        anchor = float(np.clip(np.median(opinions_v), lower, upper))
        target = float(near[np.argmin(np.abs(near - anchor))])
        success = bool(result.success and np.isfinite(best))
        message = str(result.message)

    value, details = robust_two_stage_objective(
        target,
        opinions_v,
        upward,
        downward,
        up_shift,
        down_shift,
        probabilities=probability,
        tolerance=free,
        uncertainty=uncertainty,
        radius=radius,
        return_details=True,
    )
    probe = max(1e-8, 1e-6 * max(1.0, upper - lower))
    neighbours = [
        objective(float(np.clip(target - probe, lower, upper))),
        objective(float(np.clip(target + probe, lower, upper))),
    ]
    local_optimality_residual = max(0.0, value - min(neighbours))
    diagnostic = {
        "success": success,
        "message": message,
        "target": target,
        "objective": value,
        "target_lower": lower,
        "target_upper": upper,
        "objective_evaluations": evaluations,
        "local_optimality_residual": float(local_optimality_residual),
        "uncertainty": str(uncertainty),
        "radius": float(radius),
        "scenario_count": int(opinions_v.shape[0]),
        "agent_count": int(opinions_v.shape[1]),
        "recourse": details,
    }
    return target, diagnostic


def native_taihu_data() -> dict[str, np.ndarray]:
    """Table 1 and the two 5-by-5 perturbation matrices on PDF p. 991."""
    opinions = np.asarray(
        [
            [3.5, 4.0, 4.5, 5.0, 6.0],
            [4.2, 4.8, 5.4, 6.0, 7.2],
            [2.8, 3.2, 3.6, 4.0, 4.8],
        ]
    )
    upward = np.asarray(
        [
            [5.0, 4.0, 2.0, 5.0, 6.0],
            [5.5, 4.4, 2.2, 5.5, 6.6],
            [4.5, 3.6, 1.8, 4.5, 5.4],
        ]
    )
    downward = np.asarray(
        [
            [3.0, 2.0, 3.0, 2.0, 3.0],
            [3.3, 2.3, 3.3, 2.2, 3.3],
            [2.7, 1.8, 2.7, 1.8, 2.7],
        ]
    )
    tolerance = np.asarray(
        [
            [0.30, 0.10, 0.20, 0.20, 0.40],
            [0.33, 0.11, 0.22, 0.22, 0.44],
            [0.27, 0.09, 0.18, 0.18, 0.36],
        ]
    )
    # These are stored exactly as printed.  Equations use U_i=[u_i^1,...,u_i^L],
    # so the displayed 5-by-5 matrices admit two literal readings: rows can be
    # perturbation dimensions (used by the primary run below) or experts (the
    # transpose comparison).  The primary reading is nearer Table 2, but its
    # costs are still not bit-exact; both readings are reported transparently.
    upward_shifts = np.asarray(
        [
            [-0.20] * 5,
            [-0.05] * 5,
            [-0.03] * 5,
            [-0.10] * 5,
            [-0.15] * 5,
        ]
    )
    downward_shifts = np.asarray(
        [
            [0.10] * 5,
            [0.20] * 5,
            [0.40] * 5,
            [0.05] * 5,
            [0.25] * 5,
        ]
    )
    return {
        "opinions": opinions,
        "upward": upward,
        "downward": downward,
        "tolerance": tolerance,
        "upward_shifts": upward_shifts,
        "downward_shifts": downward_shifts,
        "probabilities": np.full(3, 1.0 / 3.0),
    }


def solve_native_table2(
    shift_orientation: str = "perturbation_rows",
    *,
    enforce_primary_near_agreement: bool = True,
) -> list[dict[str, Any]]:
    """Compare the six direct-equivalent results with PDF Table 2.

    ``perturbation_rows`` treats displayed matrix rows as perturbation
    dimensions and columns as experts. ``expert_rows`` applies the transpose.
    The notation on PDF p. 991 is not sufficiently explicit to suppress this
    comparison.
    """
    data = native_taihu_data()
    if shift_orientation == "perturbation_rows":
        upward_shifts = data["upward_shifts"]
        downward_shifts = data["downward_shifts"]
    elif shift_orientation == "expert_rows":
        upward_shifts = data["upward_shifts"].T
        downward_shifts = data["downward_shifts"].T
    else:
        raise ValueError("shift_orientation must be perturbation_rows or expert_rows")
    published = {
        (False, "box"): (4.48, 21.52),
        (False, "polyhedral"): (4.20, 17.85),
        (False, "intersection"): (4.20, 17.25),
        (True, "box"): (4.37, 16.72),
        (True, "polyhedral"): (4.30, 13.68),
        (True, "intersection"): (4.30, 13.25),
    }
    rows: list[dict[str, Any]] = []
    for with_tolerance in (False, True):
        for uncertainty in ("box", "polyhedral", "intersection"):
            target, diagnostic = solve_robust_two_stage(
                data["opinions"],
                data["upward"],
                data["downward"],
                upward_shifts,
                downward_shifts,
                probabilities=data["probabilities"],
                tolerance=data["tolerance"] if with_tolerance else None,
                uncertainty=uncertainty,
                radius=2.0,
            )
            published_target, published_cost = published[(with_tolerance, uncertainty)]
            row = {
                "model": "TB-RTS-MCCM-U-DC" if with_tolerance else "RTS-MCCM-U-DC",
                "uncertainty": uncertainty,
                "shift_orientation": shift_orientation,
                "target": target,
                "objective": diagnostic["objective"],
                "published_target": published_target,
                "published_cost": published_cost,
                "target_rounding_match": bool(round(target, 2) == published_target),
                "absolute_cost_error": abs(diagnostic["objective"] - published_cost),
                "solver_success": bool(diagnostic["success"]),
                "objective_evaluations": diagnostic["objective_evaluations"],
                "local_optimality_residual": diagnostic["local_optimality_residual"],
            }
            if (
                enforce_primary_near_agreement
                and (
                    shift_orientation != "perturbation_rows"
                    or not row["target_rounding_match"]
                    or row["absolute_cost_error"] > 0.03
                )
            ):
                raise AssertionError(f"native Table 2 disagreement: {row}")
            rows.append(row)
    return rows


def _support_lp(exposure: np.ndarray, uncertainty: str, radius: float) -> float:
    """Independent LP used only to test the closed-form support functions."""
    exposure = np.asarray(exposure, dtype=float).reshape(-1)
    length = exposure.size
    key = uncertainty.lower()
    if key == "box":
        result = linprog(-exposure, bounds=[(-radius, radius)] * length, method="highs")
    else:
        # Variables are zeta (free or box-bounded) and absolute epigraph t>=0.
        eye = np.eye(length)
        a_ub = np.vstack(
            [
                np.c_[eye, -eye],
                np.c_[-eye, -eye],
                np.r_[np.zeros(length), np.ones(length)][None, :],
            ]
        )
        b_ub = np.r_[np.zeros(2 * length), radius]
        zeta_bounds: Iterable[tuple[float | None, float | None]]
        if key in {"poly", "polyhedral"}:
            zeta_bounds = [(None, None)] * length
        elif key == "intersection":
            zeta_bounds = [(-1.0, 1.0)] * length
        else:
            raise ValueError(key)
        result = linprog(
            np.r_[-exposure, np.zeros(length)],
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=list(zeta_bounds) + [(0.0, None)] * length,
            method="highs",
        )
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


def _full_recourse_lp(
    opinions: Any,
    upward_costs: Any,
    downward_costs: Any,
    upward_shifts: Any,
    downward_shifts: Any,
    *,
    probabilities: Any | None = None,
    tolerance: Any | None = None,
    uncertainty: str = "box",
    radius: float = 1.0,
    target_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Independent deterministic LP retaining both recourse directions.

    This audit formulation does *not* assume positive-part recourse.  It is
    used to check that allowing simultaneous upward/downward variables cannot
    improve the native solutions obtained after analytic recourse elimination.
    """
    (
        opinions,
        upward,
        downward,
        up_shift,
        down_shift,
        probability,
        free,
    ) = _validated_problem(
        opinions,
        upward_costs,
        downward_costs,
        upward_shifts,
        downward_shifts,
        probabilities,
        tolerance,
    )
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    key = uncertainty.lower()
    if key == "poly":
        key = "polyhedral"
    if key not in {"box", "polyhedral", "intersection"}:
        raise ValueError("unsupported uncertainty set")
    scenarios, agents = opinions.shape
    perturbations = up_shift.shape[1]
    recourse_size = scenarios * agents
    down_offset = 1
    up_offset = down_offset + recourse_size
    protection_offset = up_offset + recourse_size
    if key == "box":
        protection_size = scenarios * perturbations
    elif key == "polyhedral":
        protection_size = scenarios
    else:
        protection_size = scenarios * (1 + perturbations)
    variable_count = protection_offset + protection_size
    objective = np.zeros(variable_count)

    def down_index(scenario: int, agent: int) -> int:
        return down_offset + scenario * agents + agent

    def up_index(scenario: int, agent: int) -> int:
        return up_offset + scenario * agents + agent

    for scenario in range(scenarios):
        for agent in range(agents):
            objective[down_index(scenario, agent)] = (
                probability[scenario] * downward[scenario, agent]
            )
            objective[up_index(scenario, agent)] = (
                probability[scenario] * upward[scenario, agent]
            )
        if key == "box":
            start = protection_offset + scenario * perturbations
            objective[start : start + perturbations] = probability[scenario] * radius
        elif key == "polyhedral":
            objective[protection_offset + scenario] = probability[scenario] * radius
        else:
            start = protection_offset + scenario * (1 + perturbations)
            objective[start] = probability[scenario] * radius
            objective[start + 1 : start + 1 + perturbations] = probability[scenario]

    rows: list[np.ndarray] = []
    bounds_rhs: list[float] = []
    # The modified opinion must lie within +/- tolerance of the common target.
    # At zero tolerance the pair of inequalities is the recourse equality.
    for scenario in range(scenarios):
        for agent in range(agents):
            row = np.zeros(variable_count)
            row[0] = -1.0
            row[down_index(scenario, agent)] = -1.0
            row[up_index(scenario, agent)] = 1.0
            rows.append(row)
            bounds_rhs.append(float(free[scenario, agent] - opinions[scenario, agent]))
            rows.append(-row)
            bounds_rhs.append(float(free[scenario, agent] + opinions[scenario, agent]))

    for scenario in range(scenarios):
        for perturbation in range(perturbations):
            exposure = np.zeros(variable_count)
            for agent in range(agents):
                exposure[down_index(scenario, agent)] = down_shift[
                    scenario, perturbation, agent
                ]
                exposure[up_index(scenario, agent)] = up_shift[
                    scenario, perturbation, agent
                ]
            if key == "box":
                auxiliary = protection_offset + scenario * perturbations + perturbation
                exposure[auxiliary] = -1.0
                rows.append(exposure)
                bounds_rhs.append(0.0)
                opposite = -exposure
                opposite[auxiliary] = -1.0
                rows.append(opposite)
                bounds_rhs.append(0.0)
            elif key == "polyhedral":
                auxiliary = protection_offset + scenario
                exposure[auxiliary] = -1.0
                rows.append(exposure)
                bounds_rhs.append(0.0)
                opposite = -exposure
                opposite[auxiliary] = -1.0
                rows.append(opposite)
                bounds_rhs.append(0.0)
            else:
                start = protection_offset + scenario * (1 + perturbations)
                alpha, beta = start, start + 1 + perturbation
                exposure[alpha] = -1.0
                exposure[beta] = -1.0
                rows.append(exposure)
                bounds_rhs.append(0.0)
                opposite = -exposure
                opposite[alpha] = -1.0
                opposite[beta] = -1.0
                rows.append(opposite)
                bounds_rhs.append(0.0)

    if target_bounds is None:
        lower, upper = float(opinions.min()), float(opinions.max())
    else:
        lower, upper = map(float, target_bounds)
    lp_bounds = [(lower, upper)] + [(0.0, None)] * (variable_count - 1)
    a_ub = np.vstack(rows)
    b_ub = np.asarray(bounds_rhs)
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=lp_bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"independent recourse LP failed: {result.message}")
    down_values = result.x[down_offset:up_offset].reshape(scenarios, agents)
    up_values = result.x[up_offset:protection_offset].reshape(scenarios, agents)
    return {
        "success": True,
        "target": float(result.x[0]),
        "objective": float(result.fun),
        "max_inequality_violation": float(max(0.0, np.max(a_ub @ result.x - b_ub))),
        "max_simultaneous_recourse": float(np.max(np.minimum(down_values, up_values))),
        "downward_recourse": down_values.tolist(),
        "upward_recourse": up_values.tolist(),
    }


class Controller:
    """State/context-only restricted interval adaptation of Li et al. (2024).

    The default box set uses nominal q=0.5 and diagonal shifts 0.4 I, hence it
    exactly spans the public per-member envelope q in [0.1, 0.9].  Equal-weight
    lower/centre/upper interval scenarios provide genuine scenario recourse.
    With symmetric costs the box protection is a common multiplier, so the
    target degenerates to the safe-bound-clipped median of the 3N scenario
    points.  This fact is intentional, tested, and must not be presented as a
    nontrivial reproduction of the full paper.

    ``feedback=True`` adds the shared observed-gain execution adapter already
    used by the domain pilot.  It models response execution, not uncertain
    cost, and is not part of the cited method.
    """

    def __init__(
        self,
        feedback: bool = False,
        uncertainty: str = "box",
        robustness: float = 1.0,
        scenario_probabilities: Iterable[float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> None:
        self.feedback = bool(feedback)
        self.uncertainty = str(uncertainty).lower()
        if self.uncertainty not in {"box", "poly", "polyhedral", "intersection"}:
            raise ValueError("unsupported uncertainty set")
        self.robustness = float(robustness)
        if (
            not np.isfinite(self.robustness)
            or self.robustness < 0
            or self.robustness > 1
        ):
            raise ValueError(
                "controller robustness must be in [0, 1] to remain inside public q support"
            )
        self.scenario_probabilities = np.asarray(
            tuple(scenario_probabilities), dtype=float
        )
        if self.scenario_probabilities.shape != (3,):
            raise ValueError("three interval-scenario probabilities are required")
        if (
            not np.all(np.isfinite(self.scenario_probabilities))
            or np.any(self.scenario_probabilities < 0)
            or self.scenario_probabilities.sum() <= 0
        ):
            raise ValueError("scenario probabilities must be nonnegative")
        self.scenario_probabilities /= self.scenario_probabilities.sum()
        self.stats: dict[str, float | int | bool | str] = {
            "calls": 0,
            "solver_failures": 0,
            "fallback_median": 0,
            "infeasible_safe_bounds": 0,
            "action_clip_count": 0,
            "objective_evaluations": 0,
            "max_local_optimality_residual": 0.0,
            "max_target_bound_violation": 0.0,
            "last_target": 0.0,
            "gain_low": 0,
            "gain_high": 0,
            "gain_no_history": 0,
            "ignored_context_calls": 0,
            "analytic_box_calls": 0,
            "box_median_degeneracy_confirmed": False,
            "method_label": (
                METHOD_LABEL_FEEDBACK if self.feedback else METHOD_LABEL
            )
            if self.uncertainty == "box"
            else f"Li2024 RTS-{self.uncertainty} interval-scenario restricted adaptation"
            + (" + observed-gain adapter" if self.feedback else ""),
        }

    def action(self, state: Any, context: Any) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        context = np.asarray(context, dtype=float)
        if state.ndim != 2 or state.shape[1] != 5 or state.shape[0] == 0:
            raise ValueError("state must have shape [N, 5]")
        agents = state.shape[0]
        if context.shape != (agents, 10, 3):
            raise ValueError("context must have shape [N, 10, 3]")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(context)):
            raise ValueError("state/context must be finite")
        centers, widths = state[:, 0], state[:, 1]
        if np.any(widths < 0):
            raise ValueError("half-widths must be nonnegative")
        lower_bound = float(np.max(0.1 + widths))
        upper_bound = float(np.min(0.9 - widths))
        self.stats["calls"] = int(self.stats["calls"]) + 1
        if lower_bound > upper_bound + 1e-12:
            self.stats["infeasible_safe_bounds"] = int(
                self.stats["infeasible_safe_bounds"]
            ) + 1
            # No common safe consensus can exist; move toward the least-bad
            # domain midpoint and record the declared infeasibility.
            target = float(np.clip(np.median(centers), 0.1, 0.9))
            self.stats["fallback_median"] = int(self.stats["fallback_median"]) + 1
        else:
            opinions = np.stack([centers - widths, centers, centers + widths])
            if self.uncertainty == "box":
                # With nominal symmetric costs 0.5 and diagonal shifts 0.4 I,
                # box robustness multiplies every absolute deviation by the
                # same positive factor.  The exact optimizer is therefore a
                # scenario-weighted median; use it directly in the hot path.
                point_weights = np.repeat(self.scenario_probabilities, agents)
                target = float(
                    np.clip(
                        _weighted_median_midpoint(opinions.ravel(), point_weights),
                        lower_bound,
                        upper_bound,
                    )
                )
                self.stats["analytic_box_calls"] = int(
                    self.stats["analytic_box_calls"]
                ) + 1
                self.stats["box_median_degeneracy_confirmed"] = True
            else:
                nominal = np.full((3, agents), 0.5)
                shifts = 0.4 * np.eye(agents)
                target, diagnostic = solve_robust_two_stage(
                    opinions,
                    nominal,
                    nominal,
                    shifts,
                    shifts,
                    probabilities=self.scenario_probabilities,
                    uncertainty=self.uncertainty,
                    radius=self.robustness,
                    target_bounds=(lower_bound, upper_bound),
                )
                if not diagnostic["success"]:
                    self.stats["solver_failures"] = int(
                        self.stats["solver_failures"]
                    ) + 1
                    raise RuntimeError(
                        "restricted robust controller solve failed: "
                        + str(diagnostic["message"])
                    )
                self.stats["objective_evaluations"] = int(
                    self.stats["objective_evaluations"]
                ) + int(diagnostic["objective_evaluations"])
                self.stats["max_local_optimality_residual"] = max(
                    float(self.stats["max_local_optimality_residual"]),
                    float(diagnostic["local_optimality_residual"]),
                )
                violation = max(lower_bound - target, target - upper_bound, 0.0)
                self.stats["max_target_bound_violation"] = max(
                    float(self.stats["max_target_bound_violation"]), float(violation)
                )

            violation = max(lower_bound - target, target - upper_bound, 0.0)
            self.stats["max_target_bound_violation"] = max(
                float(self.stats["max_target_bound_violation"]), float(violation)
            )

        self.stats["last_target"] = float(target)

        direct = (target - centers) / 0.1
        if self.feedback:
            actions, movements = context[:, :, 0], context[:, :, 1]
            denominator = 0.1 * np.sum(actions * actions, axis=1)
            gain = np.ones(agents)
            np.divide(
                np.sum(actions * movements, axis=1),
                denominator,
                out=gain,
                where=denominator > 1e-10,
            )
            self.stats["gain_no_history"] = int(self.stats["gain_no_history"]) + int(
                np.sum(denominator <= 1e-10)
            )
            self.stats["gain_low"] = int(self.stats["gain_low"]) + int(np.sum(gain < 0.05))
            self.stats["gain_high"] = int(self.stats["gain_high"]) + int(np.sum(gain > 1.0))
            direct = direct / np.clip(gain, 0.05, 1.0)
        else:
            self.stats["ignored_context_calls"] = int(
                self.stats["ignored_context_calls"]
            ) + 1
        self.stats["action_clip_count"] = int(self.stats["action_clip_count"]) + int(
            np.sum(np.abs(direct) > 1.0)
        )
        return np.clip(direct, -1.0, 1.0)


def self_test() -> dict[str, Any]:
    native = solve_native_table2()
    alternate_orientation = solve_native_table2(
        "expert_rows", enforce_primary_near_agreement=False
    )
    native_data = native_taihu_data()
    native_full_lp_checks: list[dict[str, Any]] = []
    native_lookup = {
        (row["model"] == "TB-RTS-MCCM-U-DC", row["uncertainty"]): row
        for row in native
    }
    for with_tolerance in (False, True):
        for uncertainty in ("box", "polyhedral", "intersection"):
            retained = _full_recourse_lp(
                native_data["opinions"],
                native_data["upward"],
                native_data["downward"],
                native_data["upward_shifts"],
                native_data["downward_shifts"],
                probabilities=native_data["probabilities"],
                tolerance=native_data["tolerance"] if with_tolerance else None,
                uncertainty=uncertainty,
                radius=2.0,
            )
            eliminated = native_lookup[(with_tolerance, uncertainty)]
            objective_error = abs(retained["objective"] - eliminated["objective"])
            target_error = abs(retained["target"] - eliminated["target"])
            if objective_error > 2e-7 or target_error > 2e-7:
                raise AssertionError(
                    (with_tolerance, uncertainty, retained, eliminated)
                )
            native_full_lp_checks.append(
                {
                    "model": eliminated["model"],
                    "uncertainty": uncertainty,
                    "eliminated_target": eliminated["target"],
                    "retained_recourse_lp_target": retained["target"],
                    "absolute_target_error": target_error,
                    "eliminated_objective": eliminated["objective"],
                    "retained_recourse_lp_objective": retained["objective"],
                    "absolute_objective_error": objective_error,
                    "max_simultaneous_recourse": retained[
                        "max_simultaneous_recourse"
                    ],
                    "max_inequality_violation": retained[
                        "max_inequality_violation"
                    ],
                }
            )
    rng = np.random.default_rng(20260926)
    support_checks = []
    for uncertainty, radius in (("box", 1.3), ("polyhedral", 1.7), ("intersection", 2.4)):
        for case in range(5):
            exposure = rng.normal(size=6)
            closed = robust_support(exposure, uncertainty, radius)
            lp_value = _support_lp(exposure, uncertainty, radius)
            error = abs(closed - lp_value)
            if error > 2e-9:
                raise AssertionError((uncertainty, closed, lp_value))
            support_checks.append(
                {
                    "uncertainty": uncertainty,
                    "case": case,
                    "closed_form": closed,
                    "lp_value": lp_value,
                    "absolute_error": error,
                }
            )

    controller = Controller()
    centers = np.asarray([0.22, 0.31, 0.48, 0.63, 0.78, 0.55])
    widths = np.asarray([0.02, 0.04, 0.03, 0.06, 0.02, 0.05])
    state = np.column_stack(
        [
            centers,
            widths,
            centers - centers.mean(),
            widths - widths.mean(),
            np.ones_like(centers),
        ]
    )
    context = np.zeros((len(centers), 10, 3))
    action = controller.action(state, context)
    scenario_points = np.concatenate([centers - widths, centers, centers + widths])
    safe = (float(np.max(0.1 + widths)), float(np.min(0.9 - widths)))
    expected_target = float(np.clip(np.median(scenario_points), *safe))
    inferred_targets = centers + 0.1 * action
    active = np.abs(action) < 1.0 - 1e-12
    if not np.all(np.isfinite(action)) or np.max(np.abs(action)) > 1.0 + 1e-12:
        raise AssertionError("controller action contract failed")
    if np.any(active) and np.max(np.abs(inferred_targets[active] - expected_target)) > 2e-7:
        raise AssertionError("box-to-median degeneracy check failed")

    def independently_solve_box_median(
        c: np.ndarray,
        h: np.ndarray,
        probabilities: np.ndarray,
    ) -> tuple[float, float]:
        points = np.stack([c - h, c, c + h])
        bounds = (float(np.max(0.1 + h)), float(np.min(0.9 - h)))
        nominal = np.full(points.shape, 0.5)
        shifts = 0.4 * np.eye(len(c))
        solved, _ = solve_robust_two_stage(
            points,
            nominal,
            nominal,
            shifts,
            shifts,
            probabilities=probabilities,
            uncertainty="box",
            radius=1.0,
            target_bounds=bounds,
        )
        point_weights = np.repeat(probabilities / probabilities.sum(), len(c))
        independent = float(
            np.clip(
                _weighted_median_midpoint(points.ravel(), point_weights), *bounds
            )
        )
        return solved, independent

    median_checks = []
    median_cases = [
        ("interior", centers, widths, np.asarray([1 / 3, 1 / 3, 1 / 3])),
        (
            "lower_safe_boundary",
            np.asarray([0.50, 0.12, 0.13, 0.14]),
            np.asarray([0.08, 0.02, 0.02, 0.02]),
            np.asarray([1 / 3, 1 / 3, 1 / 3]),
        ),
        (
            "upper_safe_boundary",
            np.asarray([0.50, 0.88, 0.87, 0.86]),
            np.asarray([0.08, 0.02, 0.02, 0.02]),
            np.asarray([1 / 3, 1 / 3, 1 / 3]),
        ),
        (
            "unequal_scenario_weights",
            centers,
            widths,
            np.asarray([0.60, 0.30, 0.10]),
        ),
    ]
    for name, c_case, h_case, probabilities in median_cases:
        solved, independent = independently_solve_box_median(
            c_case, h_case, probabilities
        )
        order = rng.permutation(len(c_case))
        permuted, permuted_independent = independently_solve_box_median(
            c_case[order], h_case[order], probabilities
        )
        residual = max(abs(solved - independent), abs(permuted - independent))
        if residual > 2e-7 or abs(permuted_independent - independent) > 1e-15:
            raise AssertionError((name, solved, independent, permuted))
        median_checks.append(
            {
                "case": name,
                "solver_target": solved,
                "independent_clipped_median": independent,
                "permuted_solver_target": permuted,
                "scenario_probabilities": probabilities.tolist(),
                "permutation_invariant": True,
                "absolute_residual": residual,
            }
        )

    feedback = Controller(feedback=True)
    feedback_action = feedback.action(state, context)
    if not np.all(np.isfinite(feedback_action)) or np.max(np.abs(feedback_action)) > 1.0:
        raise AssertionError("feedback controller action contract failed")
    return {
        "source_doi": SOURCE_DOI,
        "native_method_label": NATIVE_METHOD_LABEL,
        "controller_method_label": METHOD_LABEL,
        "native_case": "Taihu Lake, PDF Table 1 -> Table 2",
        "native_rows": native,
        "native_alternate_orientation_rows": alternate_orientation,
        "native_orientation_note": (
            "Primary results read displayed rows as perturbation dimensions. "
            "Because p. 991 also writes U_i=[u_i^1,...,u_i^5], the transpose "
            "expert-row reading is reported too. Primary results are nearer "
            "Table 2 but remain near matches, not bit-exact reproduction proof."
        ),
        "native_all_targets_match_published_rounding": all(
            row["target_rounding_match"] for row in native
        ),
        "native_max_absolute_cost_error": max(
            row["absolute_cost_error"] for row in native
        ),
        "native_retained_recourse_lp_checks": native_full_lp_checks,
        "native_retained_recourse_lp_max_objective_error": max(
            row["absolute_objective_error"] for row in native_full_lp_checks
        ),
        "native_retained_recourse_lp_max_target_error": max(
            row["absolute_target_error"] for row in native_full_lp_checks
        ),
        "support_function_lp_checks": support_checks,
        "support_function_max_absolute_error": max(
            row["absolute_error"] for row in support_checks
        ),
        "controller_contract": {
            "shape": list(action.shape),
            "max_abs_action": float(np.max(np.abs(action))),
            "safe_bounds": list(safe),
            "expected_clipped_median_target": expected_target,
            "box_median_degeneracy_confirmed": True,
            "independent_median_checks": median_checks,
            "stats": controller.stats,
        },
        "feedback_contract": {
            "max_abs_action": float(np.max(np.abs(feedback_action))),
            "stats": feedback.stats,
        },
        "scope_warning": (
            "The controller is a restricted three-point interval-scenario, "
            "cost-only specialization. Equal lower/centre/upper weights are a "
            "declared sensitivity construction, not an interval-implied probability "
            "distribution or empirical forecast. It is not the paper's full ex-ante "
            "multi-environment model, and hidden willingness is not cost uncertainty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-output", type=Path, required=True)
    args = parser.parse_args()
    args.test_output.mkdir(parents=True, exist_ok=False)
    result = self_test()
    (args.test_output / "native_and_solver_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = result["native_rows"]
    header = list(rows[0])
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(json.dumps(row[key], ensure_ascii=False) for key in header)
        )
    (args.test_output / "native_table2_comparison.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
