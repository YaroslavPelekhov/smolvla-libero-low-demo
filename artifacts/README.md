# Evidence index

The repository separates confirmatory evidence from diagnostics. Nothing under `audit/` contributes
to a reported denominator.

| Location | Status | Contents |
|---|---|---|
| `protocol_manifest.json` | frozen protocol | Exact prompt join and nested 5/10/25-demo episode prefixes |
| `preregistration_v2.json` | frozen before phase I | Hash of the SAN-FT predictions |
| `preregistration_v3.json` | frozen before phase II | Hash of the CSWI coefficient, threshold, and stopping rule |
| `preregistration_v4.json` | frozen before phase III | Hash of the CSTVI construction and seed-2000 stopping rule |
| `preregistration_v5.json` | frozen before phase IV | Hash of the BDAM gate and seed-3000 stopping rule |
| `preregistration_ceiling.json` | frozen before ceiling extension | Hash of the 1/2-demo hypotheses and seed-4000 stopping rule |
| `results_v3/` | registered primary phase | Baseline vs CSWI plus controls: 48 cells, 960 outcomes |
| `results_v4/` | independent robustness phase | Fresh baseline vs CSTVI: 36 cells, 720 outcomes |
| `results_v5/` | independent robustness phase | Fresh baseline vs BDAM: 36 cells, 720 outcomes |
| `results_ceiling/` | assignment ceiling extension | Baseline vs unchanged CSWI: 72 cells, 1,440 outcomes on seed 4000 |
| `results/` | valid ablation | Baseline vs rejected SAN-FT and the Bonus B critic audit |
| `audit/` | excluded | Leakage-affected pilots and invalid gripper evaluations |

## Evidence contract

Validation must report 48/36/36/72 unique cells for v3/v4/v5/ceiling, each with 20 episodes. `per_run.csv` is
the source of truth: every row retains its 20-value Boolean success vector. `cost_curve.csv` and
`paired_comparison.json` are derived views. Internal method stages are `stabilized` (CSWI),
`task_vector` (CSTVI), and `adaptive_merge` (BDAM). The four matrices must not be pooled as if they
shared reset states; the report presents their paired deltas separately.

The completed ceiling matrix gives baseline 64/95/118/115/111 and CSWI 66/92/115/111/110 successes
out of 120 at budgets 1/2/5/10/25. Thus the frozen all-budget superiority criterion is rejected.

The three numbered PNGs in `results_v3/failures/` are contact sheets extracted from actual failed
rollouts. Their catalog retains the originating task, budget, training seed, episode, and absolute
video path from the reference WSL run.

`results/reward_optimization.json` is derived from the 42-cell Bonus B bank. Five recorded videos
per candidate determine the proxy score and policy selection; the simulator audit excludes those
indices and uses the other 15 Boolean outcomes. Thus reward maximization and true-success audit do
not reuse episodes.

Run `vla-verify` from the repository root to validate hashes, matrix coverage, paired statistics,
failure provenance, Bonus B coverage, and the delivery PDF signature.
