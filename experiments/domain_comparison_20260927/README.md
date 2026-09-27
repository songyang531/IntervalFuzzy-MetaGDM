# Literature-informed paired comparison (2026-09-27)

This directory reproduces the **common-task adaptations**, not the original papers' complete native algorithms. The Meta-GDM checkpoint is the archived seed-20260507 paper model; it is never retrained here. The comparison uses 20 paired cases for each of two response conditions and group sizes 10, 40, and 100. Every method sees the same initial intervals and hidden response realization in a case, but the controllers cannot read the hidden parameters.

| Source | What is run in this repository | Important boundary |
| --- | --- | --- |
| Guo et al., *Information Fusion* 104 (2024), 102185, [DOI](https://doi.org/10.1016/j.inffus.2023.102185) | Dynamic minimum-cost feedback adapted to bounded interval-center suggestions | The original point-opinion/payment task is not reproduced in full. |
| García-Zamora et al., *Computers & Industrial Engineering* 181 (2023), 109295, [DOI](https://doi.org/10.1016/j.cie.2023.109295) | Endpoint-L1 instance of the FZZ-MCC framework | A chosen framework instance, not an algorithm claimed to be specified by the source paper. |
| Li et al., *European Journal of Operational Research* 317 (2024), 977--1002, [DOI](https://doi.org/10.1016/j.ejor.2024.04.020) | Restricted three-point interval-scenario adaptation | Not the full robust two-stage formulation; the symmetric cost box can reduce to a scenario-median target. |

Each adaptation is evaluated both directly and with the **same external observed-gain adapter**. The adapter uses the previous ten rounds' suggested and realized movements; it is not a component of any cited method. The original 0.9 endpoint-consensus target, 120-round horizon, fixed half-width dynamics, nonlinear physical cost of realized movements, and standard cumulative member-cost Gini are unchanged. Failures remain in all means and contribute 120 rounds. A low cost from failure is not evidence of success.

The code is split into a frozen-environment evaluator (`run_pilot.py`), the three control implementations (`fzz_mcc.py`, `robust_cost.py`, and `run_pilot.py`'s `MinimumCostLP`), a shared gain adapter (`run_robustness.py`), a joint runner (`run_all.py`), and an independent audit (`audit.py`). `interval_qp.py` is a support module for the copied evaluator; its constructed QP reference is **not** counted as a fourth paper baseline. The older standalone `main()` entry points in support modules are retained for traceability; use the commands below for the released seven-method comparison.

From the repository root, in a Python environment with packages in `requirements.txt`:

```powershell
python experiments/domain_comparison_20260927/run_all.py --episodes 20 --out results/domain_comparison_rerun
python experiments/domain_comparison_20260927/audit.py --run results/domain_comparison_rerun
```

The published record is `results/domain_comparison_20260927_release/`: `episodes.csv` has 840 method-case records, `trajectories.csv` has round-level records, `summary.csv` holds all 42 cells, `manifest.json` pins source/model hashes and protocol, and `audit.json` verifies all 120 paired initial cases, all failures, physical costs, standard Gini, rewards, AUC, and safety counts. Six safe-interval member-step excesses occur in the Meta-GDM hidden-response `N=10` cell; physical-boundary clipping is zero throughout this comparison. These are separate safety concepts.

The original native numerical examples were checked in the research workspace, but their supplied PDFs are not redistributed. Shen et al.'s payment-action RL consensus model is discussed as related work, not entered in the performance table: this environment has no payment or reweighting actions. The evaluation has one Meta-GDM training seed and 20 scenarios per cell, so sample rates such as 20/20 are **not** population guarantees or cross-seed stability evidence. Linear-response cost comparisons and every failure case are retained; this release must not be described as universal dominance over the published algorithms.
