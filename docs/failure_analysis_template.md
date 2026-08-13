# Failure analysis protocol

Run `vla-failures` only after recorded evaluation. The three generated contact sheets are evidence,
not diagnoses. Watch the full corresponding videos and assign each case to the closest mechanism;
do not force one case into every category.

1. **Grounding failure.** Hypothesis: the policy attends to the wrong object, rather than failing at
   low-level control. Discriminating experiment: keep the robot state and instruction fixed and swap
   only the target object's texture/location across matched LIBERO initial states. A systematic target
   switch supports grounding failure; correct approach followed by a missed grasp supports control
   error instead.

2. **Contact/control failure.** Hypothesis: visual intent is correct but action chunking overshoots at
   contact. Discriminating experiment: replay matched initial states with `n_action_steps` in {1, 5,
   10}, without retraining. A contact-localized improvement at shorter execution horizons supports the
   chunking account; unchanged wrong-object trajectories support a perception alternative.

3. **Stage/termination failure.** Hypothesis: the model completes an early subgoal but cannot detect
   progress or transition to the next stage. Discriminating experiment: evaluate from simulator states
   immediately before and after the suspected boundary, with the same full instruction and with an
   explicit remaining-subtask instruction. Improvement only with the latter supports language/planning
   failure; failure from both states suggests missing motor competence.

For every case record task id, checkpoint, training seed, evaluation seed, episode index, observed
event time, primary hypothesis, alternative, manipulated variable, and predicted outcomes under both.

