# B888-4 — consolidation and bounded rehearsal on the graded-reward arms: can forgetting drop without losing learning? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md).
"Consolidation", "rehearsal" and "reward" name parts of an update rule on a
Boolean graph; nothing here is a claim about sleep, replay, dopamine,
biology, insect memory or BANC physiology.

Module: `src/consolidated_graded.rs`. CLI: `bench-consolidated-graded`.

## Motivation

[GRADED_REWARD.md](GRADED_REWARD.md) (#38) showed that A-test generalisation
rises with reward information (group-8 0.839, per-target 0.908 vs supervised
control 0.870), but forgetting rises along the ladder (0.026 → 0.122;
control 0.253) and B test stays 0.11–0.26 below the control (0.803). This
cycle asks whether two concrete mechanisms, both acting **only during task
B** and only on weights of existing edges, cut forgetting of the strong
learners without costing A or B.

## Pre-registration (this section is committed before any run)

### Learners (unchanged code paths)

* `supervised-matched` — the clean supervised control of #34/#36/#38.
* `trace-group-8` — #38 graded reward, group = target index mod 8, #36
  trace + RPE machinery, delay 0.
* `trace-per-target` — #38 graded reward, one bit per firing target, delay 0.

With mechanism `none`, each learner must be **bit-identical** to its #38
row (tested).

### Mechanisms (constants fixed a priori, not tuned)

* `consolidate` (adapted from #25, [CONSOLIDATION.md](CONSOLIDATION.md)).
  After task A, an edge is *protected* iff its 4-bit weight after A differs
  from the initial weight (it was moved by A learning). During task B, every
  update attempt on a protected edge increments a per-edge attempt counter
  and is applied only when `attempts % PROTECT_PERIOD == 0`,
  `PROTECT_PERIOD = 4` (1 of 4 attempts pass). An *attempt* is what each
  rule already calls one: for the supervised rule, an edge offered by
  `apply_update_gated` (#25 semantics, refused edge neither visited nor
  changed); for the trace rule, a credit threshold crossing (the credit is
  consumed whether or not the step is allowed, so a blocked step is lost,
  not retried). Unprotected edges are unaffected. Memory overhead: one
  protected bit and one u32 counter per edge.
* `rehearse` (bounded rehearsal inside the equal budget). Task B keeps its
  256-episode budget and its #38 episode order; every episode at position
  `k` with `k % REHEARSAL_PERIOD == REHEARSAL_PERIOD − 1`,
  `REHEARSAL_PERIOD = 8`, is **replaced** by a task-A training episode (A
  pattern, A valence/labels) taken in order from a fixed deterministic
  shuffle of A-train (`episode_order(32, 8, seed ^ 0x5EAE_A55E)`). So B
  gets 224 B episodes + 32 A-rehearsal episodes (12.5 %), total 256, equal
  to every other arm. The learner applies its own rule to rehearsal
  episodes (supervised update with A labels; graded reward judged against A
  labels). No test pattern is ever rehearsed.
* `both` — `consolidate` and `rehearse` together (exploratory; reported
  with the same rule but not part of H1/H2).

### Protocol

`bench-graded-reward` defaults: generated graph 64 inputs / 24 targets /
fan-in 6; 4 active; 32 A-train / 32 B-train / 64 TEST patterns; 8 rounds per
phase (256 episodes per phase, equal for every arm); flip 50 %; init weight
1; threshold 1; balanced accuracy; forgetting = bal(A test after A) −
bal(A test after B); seeds 1..10. Grid: 3 learners × 4 mechanisms = 12 rows
per seed, 120 rows. Deterministic: the grid is run twice and the CSV must
be byte-identical. Only weights of existing edges change (topology checked
per row). Both mechanisms act only in B, so A test after A is identical to
the `none` arm by construction (asserted and reported).

### Pre-stated success rule (margin 0.05; #27 rule: ≥ 8/10 seeds)

For each learner L and mechanism M ∈ {consolidate, rehearse}, paired by
seed against (L, none), M **passes for L** iff all of:

1. forgetting reduction (none − M) has mean ≥ 0.05 and is strictly
   positive in ≥ 8/10 seeds;
2. A test after A: mean drop ≤ 0.05 (identical by construction);
3. B test after B: mean drop (none − M) ≤ 0.05.

* **H1 (consolidation)**: `consolidate` passes for both graded learners.
* **H2 (rehearsal)**: `rehearse` passes for both graded learners.
* Passes on `supervised-matched` are reported separately (does the
  mechanism also help the control?).
* **H3 (trade-off broken vs the control)**: some graded learner L with some
  M ∈ {consolidate, rehearse} that passes for L also, paired against
  (`supervised-matched`, none): A-test delta > −0.05, B-test delta > −0.05,
  and forgetting lower by ≥ 0.05 in ≥ 8/10 seeds. Otherwise H3 is not
  supported.
* Retained A (A test after B) is reported for every arm with paired deltas.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-consolidated-graded --csv results/consolidated-graded-v1.csv
```

## Results

_Filled in only after the run, numbers exactly as produced._
