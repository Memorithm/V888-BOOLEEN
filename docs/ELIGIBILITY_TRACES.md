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

## Results (pre-registered grid; CSV `results/eligibility-traces-v1.csv`)

CSV: 130 rows (10 seeds × (1 clean reference + 3 conditions × 4 arms)),
sha256 `58d1eb83cffb6c771e1f10e3ca08d81eb6283f2c4898ceb65f717da0f27edb89`; a
second run gave a byte-identical CSV and identical stdout. Run time ≈ 1.5 s
(release, CPU). `topology_unchanged=true` for every row. The clean per-target
reference reproduces #34 (A test 0.870±0.039, forgetting 0.142, B 0.716) and
the `reward-modulated` rows of `global` / `global-delay-1` reproduce the #34
`global` / `global-delayed-no-trace-1` rows (0.559, 0.502; bit-identity of
the code path is also tested). Mean ± sample sd over 10 seeds.

| condition | arm | A train | A test | forgetting | B test | updates (A+B) | weights changed (A+B) | reactivations |
|---|---|---|---|---|---|---|---|---|
| per-target clean | #34 reward (reference) | 0.945 | 0.870±0.039 | 0.142 | 0.716 | 2724.0 | 1978.9 | 0 |
| any | supervised control | 0.945 | 0.870±0.039 | 0.253 | 0.803 | 512.0 | 2198.1 | 0 |
| global (d = 0) | #34 reward | 0.547 | 0.559±0.030 | 0.030 | 0.531 | 476.3 | 467.8 | 0 |
| global (d = 0) | trace | 0.582 | 0.588±0.031 | 0.026 | 0.548 | 425.8 | 390.6 | 0 |
| global (d = 0) | trace-react | 0.628 | 0.603±0.039 | 0.038 | 0.569 | 819.8 | 862.1 | 121.4 |
| previous episode (d = 1) | #34 reward | 0.501 | 0.502±0.032 | 0.000 | 0.508 | 244.3 | 280.6 | 0 |
| previous episode (d = 1) | trace | 0.564 | 0.570±0.036 | 0.011 | 0.540 | 369.2 | 338.2 | 0 |
| previous episode (d = 1) | trace-react | 0.595 | 0.580±0.048 | 0.031 | 0.576 | 804.7 | 851.9 | 134.8 |
| delayed (d = 4) | #34 reward | 0.517 | 0.523±0.041 | 0.020 | 0.503 | 230.7 | 262.4 | 0 |
| delayed (d = 4) | trace | 0.536 | 0.548±0.032 | 0.007 | 0.534 | 363.6 | 327.3 | 0 |
| delayed (d = 4) | trace-react | 0.556 | 0.554±0.041 | 0.009 | 0.587 | 782.2 | 821.6 | 125.3 |

Pre-registered paired comparisons on A-test balanced accuracy (b/t/w =
better/tie/worse seeds; verdict = #27 rule, margin 0.05, ≥ 8/10 seeds):

| condition | x vs y | Δ A test | b/t/w | verdict | reaches control? |
|---|---|---|---|---|---|
| global | trace vs #34 reward | +0.030±0.020 | 10/0/0 | no clear difference | — |
| global | trace-react vs #34 reward | +0.045±0.045 | 9/0/1 | no clear difference | — |
| global | trace-react vs trace | +0.015±0.043 | 8/0/2 | no clear difference | — |
| global | trace vs control | −0.282±0.040 | 0/0/10 | worse | no |
| global | trace-react vs control | −0.267±0.047 | 0/0/10 | worse | no |
| d = 1 | trace vs #34 reward | +0.068±0.042 | 9/0/1 | **x beats y (rescue)** | — |
| d = 1 | trace-react vs #34 reward | +0.078±0.071 | 9/0/1 | **x beats y (rescue)** | — |
| d = 1 | trace-react vs trace | +0.009±0.048 | 7/0/3 | no clear difference | — |
| d = 1 | trace vs control | −0.300±0.036 | 0/0/10 | worse | no |
| d = 1 | trace-react vs control | −0.290±0.051 | 0/0/10 | worse | no |
| d = 4 | trace vs #34 reward | +0.025±0.027 | 8/0/2 | no clear difference | — |
| d = 4 | trace-react vs #34 reward | +0.030±0.043 | 7/0/3 | no clear difference | — |
| d = 4 | trace-react vs trace | +0.005±0.035 | 7/0/3 | no clear difference | — |
| d = 4 | trace vs control | −0.322±0.036 | 0/0/10 | worse | no |
| d = 4 | trace-react vs control | −0.316±0.021 | 0/0/10 | worse | no |

No forgetting verdict passes: every trace arm forgets much less than the
control (+0.215 to +0.246, 10/10) but with B test 0.216–0.269 lower (B guard
fails), i.e. it learned little; versus the #34 reward arm forgetting
differences are −0.031 to +0.013 (no clear difference).

Instrumentation (mean per seed, A+B, from the CSV): scalars emitted /
consumed — trace: global 338.2 / 338.2, d = 1 314.1 / 312.8, d = 4 295.0 /
290.9; trace-react: 465.7 / 465.7, 467.1 / 465.1, 459.7 / 452.6 (out of 512
episodes: in the rest no reached target fired, so no scalar exists). Trace
steps up / down — trace: 171.7 / 218.9 (global), 136.9 / 201.3 (d = 1),
129.7 / 197.6 (d = 4); trace-react: 300.7 / 440.0, 283.2 / 433.9, 272.6 /
423.7.

## Findings, including negative ones

* **The remedy does not fix scalar credit assignment.** Under the
  pre-registered constants, eligibility traces + reward-prediction error
  stay 0.28–0.32 below the supervised control in every condition (10/10
  seeds worse). Best A test is 0.603 (trace-react, global) vs 0.870.
* **Partial rescue only for the previous-episode scalar** (d = 1): +0.068
  (trace) and +0.078 (trace-react) over the #34 arm, 9/10 seeds — the #34
  arm is at chance (0.502) there because it applies credit to the wrong
  episode's eligibility; the trace keeps the right episode tagged. Lifting
  it from chance to 0.570–0.580 passes the rescue rule but is still far from
  useful.
* **Scalar per episode (d = 0) and delay 4: no rescue by the pre-stated
  rule.** Gains of +0.025 to +0.045 are consistent in sign (7–10/10 seeds)
  but below the 0.05 margin. At d = 4 the target episode's tag (4) is weaker
  than the tags of the 4 later episodes (5–8), so most credit lands on the
  wrong episodes; only RPE averaging separates them.
* **Silence is still the bottleneck.** Even with reactivation, trace steps
  are mostly downward (1.3–1.6 down per up) and 9–42 % of episodes yield no
  scalar at all. Reactivation (≈ 121–135 per seed) raises A train
  (0.582 → 0.628 at d = 0) but its A-test effect over plain trace
  (+0.005 to +0.015) is below the margin: **reactivation does not help by
  the pre-stated rule.**
* The binary "all predicted spikes right" scalar carries very little
  information when several targets fire per episode; the remedy cannot
  create information the signal does not carry. This is the main negative
  result: on this task, a per-target signal (or a teacher) is required for
  useful learning; the remedy recovers only a small part of the gap.
* Constants were not tuned; a tuned variant might do better, but any such
  run would be a new pre-registered cycle, not a reading of this one.

## Caveats

Synthetic, small (144 edges), bipartite generated graph (no recurrence),
artificial OR-rule labels, threshold 1, init weight 1, 10 seeds, one set of
a-priori constants, trace state phase-local (reset between A and B), ≤ d
in-flight scalars dropped per phase (as for the #34 arm). After the first
run, the `steps_up` counter was redefined to exclude reactivations (which
have their own column); this changed no metric and the CSV above is from the
final code. Not evidence about BANC or biology.
