# Invalidated engineering run

`invalid_5k500_results/` preserves the first 5k/500-step experiment for audit only. It is not
part of the submitted metrics because LeRobot filtered target frames with `dataset.episodes` while
retaining target-suite-global `meta/stats.json` (52,042 action frames across all ten target tasks).
The directory also records the underpowered LoRA pilot that motivated the confirmatory 100k/10k
protocol. Final tables and figures are generated only from `/home/root/outputs/vla-benchmark-final`.

`predictions_invalid_v1.yaml` and `preregistration_invalid_v1.json` retain the matching preregistered
hypotheses. Their recorded relative path names the original repository location; the files were moved
here only after invalidation and are never read by the corrected runner.

`invalid_native_gripper_results_20260819/` preserves the aggregate artifacts from a later full
evaluation that used the corrected training checkpoints but forwarded the dataset's binary gripper
channel directly into native LIBERO. Those 960 outcomes are also invalid: `1=open, 0=closed` must be
bridged to LIBERO's `-1=open, +1=closed`. The raw invalid run tree is retained outside the repository
under the benchmark output's `audit/invalid_native_gripper_eval_20260819/`; the portable convention
probe and causal before/after evidence are in `gripper_convention.json`.
