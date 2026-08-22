# Technical appendix

## Audit trail

The corrected phase-I matrix contains 38 training units and 48 evaluation units. Each evaluation contributes
20 simulator outcomes, giving 960 Bernoulli observations. Coverage validation requires the full
Cartesian product: two seen seeds; zero-shot and cyclic wrong-prompt controls; three target tasks; and
baseline/SAN-FT checkpoints for budgets 5, 10, and 25. The separately registered v3 follow-up
adds 18 CSWI evaluation units (360 outcomes); the primary baseline/CSWI matrix reuses the same
zero/language/baseline evidence and is independently validated as 48 units and 960 outcomes.

The runner verifies the manifest digest and the SHA-256 of the prediction file named inside
`artifacts/preregistration_v2.json` before every stage. The preregistration itself is immutable.
Aggregation rejects missing, unexpected, or duplicate experiment keys and exports the exact
20-element Boolean success vector for every cell alongside its numerator and denominator.

## Why the first matrix was invalid

LeRobot's `dataset.episodes` option limits sampled trajectories but does not recompute
`meta/stats.json`. The first 5k/500 engineering matrix therefore normalized a selected task with the
statistics of all 52,042 `libero_goal` frames and task indices 0--9. This both disclosed held-out task
data and changed the physical action scale. The matrix was moved to
`artifacts/audit/invalid_5k500_results/`; it is excluded regardless of whether any cell succeeded.

The corrected overlay computes weighted first and second moments from per-episode v3 statistics. For
task 0 at budget 10 it contains 1,359 frames and one task index. Action/state means and standard
deviations were independently matched against a direct read of the selected parquet frames.

## Evaluation convention incident

The dataset action channel is binary with `1=open, 0=closed`. This is verified temporally: in a
task-1 demonstration the state gripper remains near `|q|=0.040` while the action is 1, then closes
toward `|q|<0.015` after the action switches to 0. Native LIBERO was measured independently:
command -1 opens toward `|q|=0.039`, command +1 closes toward `|q|=0.001` after 15 steps. Therefore
the required bridge is `g_env = 1 - 2 g_data`.

The first corrected-training evaluation matrix forwarded `0/1` unchanged and produced 0/960. Its
videos showed purposeful reaching without reliable grasping. With only the bridge changed, the
preselected method/25/task-1/seed-17 diagnostic changed from 0/8 to 8/8. This intervention touches
evaluation only; the registered full rerun of the same checkpoint produced 19/20. Thus all 38
completed training units remain valid, while every one of the 48 evaluation
units is rerun. The discarded outcomes remain auditable but are excluded from every CSV and plot.

## Implementation decisions

- Target demonstrations are joined to simulator tasks by canonical prompt because simulator ids and
  NVIDIA conversion indices differ.
- `smolvla_base` stores a three-camera, 6-D state/action schema. Training clears serialized feature
  maps so LeRobot infers LIBERO's two-camera, 8-D state, 7-D action schema from dataset metadata.
- Every adapted checkpoint begins from the seen checkpoint with the same seed. Baseline and SAN-FT
  use identical update counts, parameter count, target frames, and episode prefixes.
- Baseline uses statistics from only the selected target episodes. SAN-FT uses the allowed seen
  state/action statistics to preserve the coordinate system in which the action expert was trained.
- The language control holds environment task and reset seed fixed and changes only the instruction.
- TorchCodec 0.5 is pinned for compatibility with PyTorch 2.7.

## Confirmatory versus diagnostic evidence

The 500-step target-only run and a 1,000-step task-0 full-expert run are diagnostic pilots, not points
on the cost curve. A rank-16 LoRA plus mild image-jitter pilot updated 742,656 parameters versus
99,880,992 in the full action expert and achieved 0/8. It was rejected before the confirmatory matrix
because it tested a capacity bottleneck and augmentation simultaneously, obscuring the normalization
hypothesis.

A post-audit replanning diagnostic then executed the same budget-5 baseline checkpoints with
`n_action_steps=5` instead of 10. On paired initial states it matched, rather than improved, the
registered controller: 7/8 on task 0 and 8/8 on task 2. Because the developmental test supplied no
positive signal, a new 18-cell curve was not launched and the confirmatory controller stayed at 10.

## Registered CSWI follow-up

The complete v2 matrix rejected seen-anchored normalization but exposed complementary failures
between training seeds. Before evaluating any interpolated policy, `predictions_v3.yaml` froze a
partner weight of 0.25, a minimum expected gain of two successes at budget 5, and a no-degradation
prediction at budgets 10 and 25. For every task/budget cell the method constructs
`0.75*w_anchor + 0.25*w_partner` in both directions. It uses the same demonstrations and target
normalizer, performs no gradient update, and never reads simulator reward. Each generated checkpoint
has a `soup_manifest.json` containing source/output SHA-256 digests and parameter counts.

All 18 registered cells were evaluated. CSWI reached 117/120, 114/120, and 117/120 at budgets
5/10/25 versus baseline 110/120, 114/120, and 114/120. At budget 5 the paired discordance is
10 wins/3 losses (exact McNemar p=0.092); six of seven net successes come from task 2, which rises
from 82.5% to 97.5%. The direction and preregistered effect threshold are satisfied, but the report
does not call the individual p=0.092 comparison statistically significant.

## Video retention and failures

LeRobot records the complete success vector in `eval_info.json`. Its batched writer may retain fewer
MP4 files than evaluated episodes, so quantitative results always use the JSON outcomes. Failure
extraction parses the video episode index and selects only episodes explicitly marked unsuccessful;
the catalog records the source run, task, budget, seed, and source path.

## Statistical interpretation

The primary estimand is pooled Bernoulli success across three target tasks and two training seeds.
Wilson intervals describe rollout uncertainty; `per_run.csv` preserves task/seed heterogeneity. Two
training seeds meet the assignment minimum but do not justify small-effect or optimizer-variance
claims. A measured 0/20 is reported as such with a nonzero upper confidence limit, not interpreted as
proof that success is impossible.

## Failure-hypothesis template

Each selected failure is described by a visible mechanism, a competing explanation, and an
intervention that distinguishes them. Candidate interventions include changing action-chunk length
while holding the checkpoint and reset fixed, pairing correct and wrong prompts on identical initial
states, masking the putative target object, and comparing oracle termination with one-step replanning.
The report text is finalized only after visual inspection of the corrected rollout frames.

## Rejected approaches

1. Full VLM fine-tuning: 5--25 demonstrations cannot identify hundreds of millions of additional
   parameters under the available 16 GB hardware and two-seed budget.
2. Mixing demonstrations from other held-out target tasks: likely to improve the headline number but
   violates the strict target-demo budget.
3. Selecting visually successful demonstrations: violates the fixed-prefix rule and introduces
   unreported manual supervision.
4. Public 40-task LIBERO fine-tune: its training universe can overlap the held-out goal tasks.
5. Global `libero_goal` normalization: numerically convenient but directly violates the held-out
   protocol.

## Reward-optimization audit

The small critic predicts normalized temporal position from seen RGB frames and never reads actions.
It can reward pauses, reversible motion, or a terminal-looking pose. Robometer can also be
miscalibrated outside its training distribution. The measured audit treats each proxy as a
zeroth-order optimizer over four frozen candidate policies for every task/budget group. Five videos
per candidate produce the score; all five corresponding Boolean outcomes are excluded from the
true-success audit, leaving 15 held-out simulator episodes per candidate.

In the measured 42-cell ranking, the small critic's outputs span only 0.0013 and correlate negatively
with simulator success (Spearman rho -0.323), consistent with near-constant collapse. Robometer spans
0.298 and reaches rho 0.252. It is the better ranker here, but the modest coefficient and several
high scores on imperfect policies retain the reward-hacking concern.

Maximizing the small critic selects policies with 85.9% held-out success versus a 93.3% baseline
reference and misses the simulator oracle in 5/9 groups. Robometer selects 91.1% and misses the oracle
in 4/9 groups. Thus proxy maximization makes the collapsed critic materially worse, while Robometer
reduces but does not remove optimizer's-curse regret. This finite-bank intervention answers the
reward-hacking question without claiming gradient-based adaptation of SmolVLA.

## Preregistered ceiling extension

Because the 5/10/25 naive baseline is already at 92--98%, the assignment explicitly permits a lower
data regime. `predictions_ceiling.yaml` therefore freezes strict nested budgets 1/2/5/10/25, unchanged
10k-update baseline and 75/25 CSWI, training seeds 17/29, reset seed 4000, and a complete stopping
rule before any one- or two-demo training. The new phase reuses only immutable seen and 5/10/25
checkpoints; newly trained low-demo checkpoints live under a separate output root. Results are
aggregated only after all 72 registered evaluation cells complete.

All 72 cells completed, contributing 1,440 Boolean outcomes. For budgets 1/2/5/10/25, baseline
obtained 64/95/118/115/111 successes out of 120; CSWI obtained 66/92/115/111/110. The paired changes
were +1.7/-2.5/-2.5/-3.3/-0.8 percentage points, with exact McNemar p-values
0.791/0.678/0.375/0.344/1.000. The preregistered all-budget superiority criterion therefore fails
at budgets 2, 5, 10, and 25. At one demonstration the two-success aggregate gain is localized to
task 2 (+3/40), while task 0 remains at the floor (0/40 for both methods).

An operational disk-exhaustion incident interrupted WSL and left zero-length serialization metadata
in two otherwise complete task-0/seed-17 checkpoints. The affected model tensors and soup manifests
were independently opened and hash-checked. Deterministic tokenizer and processor definitions were
restored from the corresponding completed peer; normalization tensors came from the same immutable
task/budget demonstration subset. No weight, example, update, seed, reset state, or registered
stopping rule changed. Both repaired policies then completed the original 20-episode evaluation;
the machine-readable incident record is `artifacts/ceiling_runtime_incident.json`.
