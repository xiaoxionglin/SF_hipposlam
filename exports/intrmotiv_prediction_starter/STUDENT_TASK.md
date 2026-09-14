# First assignment and project boundary

**Question:** can CA3 plus action information predict future DG events well
enough to reduce new environment experience required after reward relocation?

## Milestone 1: prediction only

1. Run the kit's tests and synthetic example.
2. Keep the original visual/DG/CA3/actor baseline fixed.
3. Measure DG-positive timesteps, true silent-gap lengths, activation bursts,
   simultaneous activations, and recurrence-boundary censoring.
4. Integrate the shadow target-hit/time head using INTEGRATION.md.
5. Compare held-out predictions with a simple target-frequency baseline.

Deliver a short result: baseline behavior unchanged, number of usable labels,
hit metrics with class baseline, timing error, and representative failures.
No transfer-performance claim follows from a lower training loss.

## Milestone 2: the student's new world model

The supplied heads are scaffolds. Define and implement:

- **Starting information:** current CA3, any existing bypass observations needed
  for control, and optional causal action history.
- **Intervention:** candidate first action, separate from previous action.
- **Continuation:** initially a fixed baseline policy; record which one.
- **Outcome:** next DG vector/event and duration, plus termination/no-event and
  any reward within the interval. Identity-only classification does not predict
  DG amplitudes or simultaneous activations.
- **Event definition:** next nonzero vector vs onset vs first distinct landmark.
  These are different scientific models; do not silently interchange them.

For the minimal exact CA3 skip, every omitted input must be zero. If persistent
or other DG inputs occur before the selected landmark event, either represent
their cumulative contribution or abandon the exact-skip claim. The predictor
must also represent any extra state consumed by a future critic; predicting
CA3 alone does not reconstruct visual/depth bypass or action-history state.

Candidate predictions require support in collected behavior. For an observed
state only the executed branch supplies a factual target; unexecuted actions
are not supervised counterfactual ground truth. Compare with a state-only model
and an action-only/class-frequency baseline on held-out data. Evaluate whether
action conditioning adds information before asking it to control the agent.

## Milestone 3: planning during learning

Use validated predictions to evaluate alternative actions and train a separate
auxiliary actor objective; keep real PPO training explicit. Test reward changes
with the transition model retained and reward/value evaluation updated. Count
new environment decisions and internal model computation separately.

The strong composition test is A->B and B->C in separate experiences, then
A->C without a complete training route. This requires compatible intermediate
states and executable continuations; predictive accuracy alone does not prove it.

## Keep out of the initial project

Do not import IntrMotiv graph managers, landmark recruitment/retirement,
intrinsic reward shaping, goal-conditioned worker, or contextual DG feedback
by default. Do not change the Lin et al. backbone just to match IntrMotiv.

## Handoff lessons

- Authoritative source plus focused tests was more reliable than older local
  checkouts and proposal notes; snapshot actual files, not only Git HEAD.
- The useful unit of reuse is a component with explicit inputs, labels,
  gradients, and masks, not the entire custom learner.
- Long-horizon supervision exposed a concrete failure: incomplete windows can
  masquerade as negatives. The kit contains a tested censor-aware replacement;
  the upstream runtime was not changed by this handoff.
- Before more model complexity, check state/replay consistency, event semantics,
  and usable-label counts. These checks should be reused for the next predictor.
