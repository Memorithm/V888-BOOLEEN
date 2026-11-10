# B888-4 — eligibility traces + reward-prediction error for scalar credit (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md). "Eligibility",
"reward" and "reward-prediction error" name parts of an update rule on a
Boolean graph; nothing here is a claim about dopamine, biology, insect memory
or BANC physiology.

Module: `src/eligibility_trace.rs`. CLI: `bench-eligibility-traces`.

## Motivation

[REWARD_LIMITS.md](REWARD_LIMITS.md) (#34) found the reward-modulated arm
breaks on credit assignment: one scalar per episode (+1 iff every predicted
spike was right) drops A-test balanced accuracy from 0.870 to 0.559, and the
previous episode's scalar to 0.502 (chance). Mechanism read from counters:
mostly negative scalars decrement correct spikes too, a silent target has no
eligibility and is never raised again, learning stalls at ≈ 250–480 updates.

## Remedy under test (parameters fixed here before any run)

Two new arms act only on the weights of **existing** edges (topology and
`V888PLS1` checkpoint format unchanged). They receive **exactly the same
scalar** as the #34 reward arm (same definition, same delay); they are never
told which target was right.

`trace` arm, per phase (state is phase-local and starts empty):

* **Eligibility trace** per original synapse, bounded 4-bit counter
  `e ∈ 0..=15`. Each episode: every `e` decays by 1 (saturating at 0); then
  every edge from an active source to a target that fired gets `e = 8`
  (pre/post co-activity tag). A tag therefore lasts 8 episodes.
* **Reward-prediction error**: a phase-local baseline `b ∈ 0..=1024`
  (fixed-point success rate, init 512). When a scalar `r ∈ {0, 1}` is
  consumed: `rpe = 1024·r − b`, then `b += (1024·r − b) / 16` (integer
  division toward zero).
* **Bounded credit accumulator** per synapse `c ∈ [−2048, 2048]`:
  `c += rpe · e / 8` (integer, toward zero) for every edge with `e > 0`; when
  `|c| ≥ 1024` the edge's 4-bit weight takes one saturating step in the sign
  of `c` and `c −= sign(c)·1024`.
* **Timing**: in episode k the arm decays, predicts, tags, then consumes the
  scalar that arrives in episode k (the scalar of episode k − d for delay d).
  Scalars still in flight at phase end are dropped, exactly as for the #34
  `global-delayed-no-trace` arm (≤ d per phase).

`trace-react` arm = `trace` + **reactivation of silent targets**: a
per-target counter counts consecutive episodes where the target was reached
by an active input but did not fire; at 8 the lowest-index edge from an
active source to that target whose weight is 0 is raised to 1 (one
saturating step), and the counter resets (it also resets whenever the target
fires). Reactivations are counted.

These constants (tag 8, decay 1, baseline 512 and 1/16, threshold 1024,
clamp 2048, reactivation after 8) are chosen a priori and **not tuned** on
the results.

## Protocol

* Graph, task, pattern sets, episode orders, init weight 1, budgets and
  metrics: `bench-generalization` / `bench-reward-limits` defaults (64 inputs,
  24 targets, fan-in 6; 4 active; 32 A-train / 32 B-train / 64 TEST; 8 rounds
  per phase = 256 episodes per phase; flip 50 %; balanced accuracy on clean
  labels; forgetting = bal(A test after A) − bal(A test after B)). Seeds
  1..10.
* Credit conditions (all scalar, one per episode):
  `global` (d = 0, scalar per episode), `global-delay-1` (previous-episode
  scalar), `global-delay-4` (delayed by 4 episodes).
* Arms per (condition, seed), equal episode budget and identical order:
  1. `supervised-matched` — clean supervised control (credit modes do not
     apply to a teacher; identical to the #34 control there);
  2. `reward-modulated` — the #34 arm (`Credit::Global` /
     `Credit::GlobalDelayedNoTrace(d)`), bit-identical code path;
  3. `trace`; 4. `trace-react`.
  Plus one reference row per seed: the #34 reward arm with clean per-target
  reward (expected 0.870 mean, reproducing #34).
* Instrumentation per phase: episodes, scalars emitted and consumed, weight
  steps attempted, weights changed (up / down), reactivations.
* **Pre-stated reading** (margin 0.05, #27 verdict rule: mean paired delta
  ≥ 0.05 and better in ≥ 80 % of seeds):
  * the remedy **rescues** a condition if `trace` (or `trace-react`) beats
    the #34 reward arm on A-test balanced accuracy;
  * it **reaches the control** if its paired A-test delta vs the supervised
    control is > −0.05 (otherwise: still worse than the control);
  * reactivation **helps** if `trace-react` beats `trace`;
  * forgetting differences are reported with the B-test guard (B must not
    drop by more than 0.05) and not claimed otherwise.
  Negative results are reported as produced.
* Determinism: the grid is run twice, CSV must be byte-identical.
  Checkpoint/resume of weights between A and B is bit-exact (trace state is
  phase-local; tested).

```bash
cargo +1.89 run --release -- bench-eligibility-traces --csv results/eligibility-traces-v1.csv
```

## Results

_Not run yet at the time this protocol was committed._
