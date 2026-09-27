"""García-Zamora et al. (2023) FZZ-MCC: native check and interval instantiation.

This paper is a FRAMEWORK, not a fixed interval-policy algorithm.  The native
P-FZZ-MCC hiring case and the adapted endpoint-L1 controller are deliberately
separate.  The latter is our declared instantiation of Definitions 8, 10, 11,
not a claim to reproduce the paper's interval algorithm (there is no such one).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog, minimize

SOURCE_DOI = "10.1016/j.cie.2023.109295"
METHOD_LABEL = "FZZ-MCC2023 endpoint-L1 instantiation"


def solve_native_hiring(desired: int) -> dict:
    """Section 4.2, pp8--9, original P-FZZ-MCC and Table 1 (no environment).

    Four DMs, four candidates, reciprocal FPRs, equal costs and weights;
    epsilon_1=epsilon_2=.2. Optimize the 24 independent upper-triangle ratings.
    Desired is zero-based. Returned cost is the ORIGINAL normalized L1 cost,
    not the manuscript's nonlinear realized-movement cost.
    """
    if desired not in range(4):
        raise ValueError("desired candidate must be in range(4)")
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
    initial = np.array([[.5, .4, .8, .4, .8, .85],
                        [.95, 1., 1., .92, .94, .58],
                        [.6, .33, 1., .25, 1., 1.],
                        [.57, .72, .58, .67, .51, .34]])
    m, k = initial.shape
    nvar = m*k
    eye = np.eye(nvar)
    # Columns: adjusted preference x; absolute change d; absolute disagreement e.
    aggregate = np.kron(np.ones((m, m))/m, np.eye(k))
    deviation = eye-aggregate
    z = np.zeros_like(eye)
    matrix = [np.c_[eye, -eye, z], np.c_[-eye, -eye, z],
              np.c_[deviation, z, -eye], np.c_[-deviation, z, -eye]]
    rhs = [initial.ravel(), -initial.ravel(), np.zeros(2*nvar)]
    # Kappa2 is the average over DMs and six distinct alternative pairs.
    matrix.append(np.r_[np.zeros(2*nvar), np.ones(nvar)/nvar][None, :])
    rhs.append(np.array([.2]))
    # eta_i = mean_j mean_k P_k(i,j); reciprocal entries use 1-x.
    eta_coeff = np.zeros((4, nvar))
    eta_constant = np.full(4, .5/4)
    for dm in range(m):
        for p, (i, j) in enumerate(pairs):
            eta_coeff[i, dm*k+p] += 1/(4*m)
            eta_coeff[j, dm*k+p] -= 1/(4*m)
            eta_constant[j] += 1/(4*m)
    for other in range(4):
        if other != desired:
            matrix.append(np.r_[eta_coeff[other]-eta_coeff[desired],
                                np.zeros(2*nvar)][None, :])
            rhs.append(np.array([eta_constant[desired]-eta_constant[other]]))
    a_ub, b_ub = np.vstack(matrix), np.concatenate(rhs)
    objective = np.r_[np.zeros(nvar), np.ones(nvar)/nvar, np.zeros(nvar)]
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub,
                     bounds=[(0, 1)]*nvar+[(0, None)]*nvar+[(0, .2)]*nvar,
                     method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x[:nvar]
    ratings = np.full((m, 4, 4), .5)
    for dm in range(m):
        for p, (i, j) in enumerate(pairs):
            ratings[dm, i, j] = x[dm*k+p]
            ratings[dm, j, i] = 1-x[dm*k+p]
    costs = [.048, .072, .081, .173]
    measured_cost = float(np.mean(np.abs(x-initial.ravel())))
    aggregate_rating = ratings.mean(axis=0)
    ranking_scores = aggregate_rating.mean(axis=1)
    assert measured_cost == result.fun or abs(measured_cost-result.fun)<1e-9
    assert np.max(np.abs(ratings-aggregate_rating)) <= .2+1e-9
    assert ranking_scores[desired] >= ranking_scores.max()-1e-9
    return dict(candidate=desired+1, cost=measured_cost,
                published_cost_3dp=costs[desired],
                abs_error_to_published=abs(measured_cost-costs[desired]),
                # The fourth exact optimum is .1725, a rounding tie. Table 1
                # uses .173; Python's binary/banker's round can produce .172.
                matches_published_rounding=bool(abs(measured_cost-costs[desired])<=.0005+1e-12),
                scores=ranking_scores.tolist(), adjusted_fprs=ratings.tolist(),
                max_constraint_residual=float(max(0, np.max(a_ub@result.x-b_ub))))


def interval_l1_target(centers, widths, threshold=.905):
    """Uniform-price L1 endpoint movement under endpoint RMS consensus.

    min mean(abs(y-c)) s.t. Var(y)+Var(h)<=[-log(threshold)/5]^2,
    .1+h <= y <= .9-h.  Width preservation makes average absolute endpoint
    displacement exactly abs(y-c).  This is NOT the prior quadratic QP.

    Convex KKT solution: for a positive Lagrange multiplier, y_i is the
    box-projection of clip(c_i, mu-t, mu+t), with mu=mean(y).  t is selected
    so that Var(y) meets the residual radius.  Two scalar monotone roots
    avoid an N-dimensional nonlinear optimizer in every environment step.
    """
    c, h = np.asarray(centers, float), np.asarray(widths, float)
    if c.ndim != 1 or h.shape != c.shape or not len(c):
        raise ValueError("centers and widths must be equal-length vectors")
    if not np.isfinite(c).all() or not np.isfinite(h).all() or np.any(h<0):
        raise ValueError("invalid finite interval inputs")
    if not .01 < -np.log(threshold)/5 < 1:
        raise ValueError("target must use the exponential consensus branch")
    lower, upper = .1+h, .9-h
    if np.any(lower>upper):
        raise ValueError("an interval is too wide to fit the safe domain")
    radius2 = (-np.log(threshold)/5)**2 - float(np.var(h))
    feasible = radius2 >= 0
    # All domain intervals share a nonempty center interval for h<=.4.
    common_low, common_high = float(np.max(lower)), float(np.min(upper))
    if radius2 <= 1e-16:
        # If width-only disagreement exceeds target, this is a declared
        # minimum-dispersion fallback, not a feasible target claim.
        center = float(np.clip(np.median(c), common_low, common_high))
        y = np.full_like(c, center)
        return y, dict(feasible=feasible, objective=float(np.mean(abs(y-c))),
                       radius2=radius2, residual=max(0., -radius2), roots=0)
    y0 = np.clip(c, lower, upper)
    if np.var(y0) <= radius2+1e-15:
        return y0, dict(feasible=True, objective=float(np.mean(abs(y0-c))),
                        radius2=radius2, residual=0., roots=0)
    roots = 0

    def at_t(t):
        nonlocal roots
        roots += 1
        def yy(mu):
            return np.clip(np.clip(c, mu-t, mu+t), lower, upper)
        def fixed_point(mu):
            return float(yy(mu).mean()-mu)
        mu = brentq(fixed_point, float(lower.min())-1., float(upper.max())+1.,
                    xtol=2e-13, rtol=1e-13)
        return yy(mu)

    # For t>=2 all c in [0,1] pass through; use a bracket independent of data.
    # The endpoint at t=0 is feasible but degenerate, so use a tiny positive t.
    t = brentq(lambda t: float(np.var(at_t(t))-radius2), 1e-12, 2.,
               xtol=2e-12, rtol=1e-12)
    y = at_t(t)
    residual = float(max(0., np.var(y)-radius2,
                         np.max(lower-y), np.max(y-upper)))
    if residual > 2e-8:
        raise RuntimeError(f"FZZ interval L1 constraint residual {residual}")
    return y, dict(feasible=True, objective=float(np.mean(abs(y-c))),
                    radius2=radius2, residual=residual, roots=roots)


class Controller:
    """State/context-only interval FZZ-MCC instantiation, no hidden cost access.

    feedback=True adds the SAME observational compensation used for prior
    domain optimizers. It is our adapter, not part of the cited FZZ paper.
    """
    def __init__(self, feedback=False, threshold=.905):
        self.feedback, self.threshold = bool(feedback), float(threshold)
        self.stats = dict(calls=0, infeasible_width_calls=0, root_calls=0,
                          max_constraint_residual=0., gain_low=0, gain_high=0)

    def action(self, state, context):
        state, context = np.asarray(state), np.asarray(context)
        target, diagnostic = interval_l1_target(state[:, 0], state[:, 1], self.threshold)
        self.stats["calls"] += 1
        self.stats["infeasible_width_calls"] += int(not diagnostic["feasible"])
        self.stats["root_calls"] += diagnostic["roots"]
        if diagnostic["feasible"]:
            self.stats["max_constraint_residual"] = max(
                self.stats["max_constraint_residual"], diagnostic["residual"])
        direct = (target-state[:, 0])/.1
        if self.feedback:
            a, m = context[:, :, 0], context[:, :, 1]
            denominator = .1*np.sum(a*a, axis=1)
            gain = np.ones(len(state))
            np.divide(np.sum(a*m, axis=1), denominator, out=gain,
                      where=denominator>1e-10)
            self.stats["gain_low"] += int(np.sum(gain<.05))
            self.stats["gain_high"] += int(np.sum(gain>1.))
            direct /= np.clip(gain, .05, 1.)
        return np.clip(direct, -1., 1.)


def self_test():
    """Native Table 1 plus independent SLSQP check of interval instantiation."""
    native = [solve_native_hiring(i) for i in range(4)]
    rng = np.random.default_rng(20260926)
    tests = []
    for n in [4, 10, 20, 40]:
        for case in range(2):
            h = rng.uniform(.02, .08, n)
            c = rng.uniform(.1+h, .9-h)
            y, diagnostic = interval_l1_target(c, h)
            radius2 = diagnostic["radius2"]
            if not diagnostic["feasible"]:
                continue
            eye = np.eye(n)
            # SLSQP epigraph objective and all analytic derivatives.
            objective = np.r_[np.zeros(n), np.ones(n)/n]
            a = np.vstack([np.c_[eye, -eye], np.c_[-eye, -eye]])
            b = np.r_[c, -c]
            def variance_constraint(v):
                return radius2-np.var(v[:n])
            def variance_jac(v):
                return np.r_[-2*(v[:n]-v[:n].mean())/n, np.zeros(n)]
            start_y = np.full(n, .5)
            result = minimize(lambda v: objective@v,
                              np.r_[start_y, abs(start_y-c)+1e-7],
                              jac=lambda v: objective, method="SLSQP",
                              bounds=list(zip(.1+h, .9-h))+[(0, None)]*n,
                              constraints=[dict(type="ineq", fun=lambda v: b-a@v,
                                                jac=lambda v: -a),
                                           dict(type="ineq", fun=variance_constraint,
                                                jac=variance_jac)],
                              options=dict(ftol=1e-11, maxiter=500))
            gap = diagnostic["objective"]-float(result.fun)
            tests.append(dict(n=n, case=case, solver_success=bool(result.success),
                              analytical_objective=diagnostic["objective"],
                              slsqp_objective=float(result.fun), objective_gap=gap,
                              max_residual=diagnostic["residual"]))
            if not result.success or abs(gap)>2e-7:
                raise AssertionError(tests[-1])
    # Uniform-width analytic two-member case: target spread exactly 2*radius.
    y, d = interval_l1_target(np.array([.2, .8]), np.array([.04, .04]))
    r = -np.log(.905)/5
    assert abs(np.mean(abs(y-[.2,.8]))-(.3-r))<1e-9
    return dict(source_doi=SOURCE_DOI,
                native_case="Section 4.2 P-FZZ-MCC / Table 1", native=native,
                native_rounding_matches=sum(x["matches_published_rounding"] for x in native),
                interval_instantiation_tests=tests,
                note="Native L1 costs must not be compared numerically with environment nonlinear costs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-output", type=Path, required=True)
    args = parser.parse_args()
    args.test_output.mkdir(parents=True, exist_ok=True)
    result = self_test()
    (args.test_output/"native_and_solver_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
