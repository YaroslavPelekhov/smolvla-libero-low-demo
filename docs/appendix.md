# Technical appendix

## Audit trail

The final matrix contains 38 completed training units and 48 completed evaluation units. Each
evaluation unit contributes 20 simulator outcomes, giving 960 total rollouts. Aggregation validates the
full Cartesian product before writing any curve: two seen seeds; zero-shot and wrong-prompt controls;
three target tasks; and baseline/method checkpoints for budgets 5, 10, and 25.

The protocol manifest digest is `82ddbb8c6248486c444c8ffd0a6e4328097441c833884332385e4cdc5d9e742b`.
The runner also verifies the preregistered prediction-file SHA-256 before every stage.

## Implementation decisions

- Target demonstrations are joined to simulator tasks by canonical prompt because task ids differ
  between LIBERO and the NVIDIA conversion.
- `smolvla_base` stores a three-camera, 6-D state/action schema. The training command clears serialized
  feature maps so LeRobot infers LIBERO's two-camera, 8-D state, 7-D action schema from metadata.
- TorchCodec 0.5 is pinned for compatibility with PyTorch 2.7.
- Every adapted checkpoint begins from the seen checkpoint with the same seed. Baseline and method use
  identical update counts and data prefixes.
- The language control holds the environment task and reset seed fixed and changes only the instruction.
- Three late LoRA evaluations encountered a transient CUDA driver error under four concurrent jobs.
  Partial outputs were archived and the same commands completed serially. No outcomes from failed
  attempts enter aggregation.

## Video retention

LeRobot v0.6.2 records all 20 simulator outcomes but the batched MP4 writer retained ten files per run,
for 480 videos across 48 evaluations. The result CSVs use all 960 outcomes. The three reported failures
come from proposed-method checkpoints whose complete outcome vectors are zero, so every retained MP4 is
unambiguously a failure. Bonus B uses a deterministic five-video subset per checkpoint and evaluates
rank correlation against the complete 20-episode checkpoint success rate.

## Statistical interpretation

The primary estimand is pooled Bernoulli success over three target tasks and two training seeds. Wilson
intervals quantify rollout uncertainty; `per_run.csv` preserves task/seed heterogeneity. Two training
seeds satisfy the requested minimum but do not support small-effect claims.

The only nonzero cell is naive fine-tuning at ten demonstrations: 8/120 pooled, entirely from task 0
(5/20 for seed 17 and 3/20 for seed 29). This isolated peak does not establish a monotone cost curve.
All LoRA cells are zero, so the proposed regularization hypothesis is rejected under this compute
budget. Correct- and wrong-prompt zero-shot runs are both zero, leaving the language control
uninformative rather than proving language invariance.

## Failure hypotheses

1. At five demonstrations on task 0, the end effector moves across the scene without reaching the
   drawer. Change only target texture/location while holding state and prompt fixed; systematic motion
   to a distractor indicates grounding failure, whereas correct approach followed by a miss indicates
   low-level control failure.
2. At ten demonstrations on task 1, the end effector approaches the stove region but does not complete
   placement. Re-run identical resets with `n_action_steps` in {1, 5, 10}; contact-phase improvement
   restricted to shorter execution supports action-chunk overshoot.
3. At 25 demonstrations on task 2, the arm moves toward the scene boundary rather than the bottle.
   Compare full instructions with explicit remaining-step prompts from states immediately before and
   after subgoal boundaries; selective recovery supports planning/language failure.

These are proposed discriminating interventions, not post-hoc claims that the experiments were run.

## Rejected approaches

1. Full VLM fine-tuning: 5-25 demonstrations cannot identify hundreds of millions of parameters, and a
   matched two-seed comparison does not fit the 16 GB hardware budget.
2. Mixing demonstrations from other held-out tasks: likely to improve the headline average, but direct
   target-universe leakage.
3. Selecting visually successful demonstrations: violates the fixed-prefix budget and introduces
   unreported manual supervision.
4. Public 40-task LIBERO checkpoint: its training universe overlaps the held-out goal tasks.

## Reward-hacking guard

The small critic predicts normalized temporal position from seen RGB frames and never reads actions.
It can reward pauses, reversible motion, or terminal-looking poses. Robometer can also be miscalibrated
outside its training distribution. Neither is optimized against. Before direct policy optimization,
freeze a simulator audit set and predeclare critic-score improvement accompanied by lower simulator
success as reward hacking.
