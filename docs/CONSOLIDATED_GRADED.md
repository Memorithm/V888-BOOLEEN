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

## Results (pre-registered grid; CSV `results/consolidated-graded-v1.csv`)

CSV: 120 rows (10 seeds × 3 learners × 4 mechanisms), sha256
`98db675e1c71207fc487c74b443b2c60500178f0748dfe7f233f855443f2de86`; a
second run gave a byte-identical CSV and identical stdout. Run time ≈ 3.1 s
wall (release, CPU, including cargo startup). `topology_unchanged=true` for every
row. The `none` rows reproduce #38 exactly (control 0.870 / forgetting 0.253
/ B 0.803; group-8 0.839 / 0.118 / 0.681; per-target 0.908 / 0.122 / 0.695;
bit-identity also tested). A test after A is identical across mechanisms
(both act only in B). Mean ± sample sd over 10 seeds; counters are means.

| learner | mechanism | A test (after A) | A test after B (retained) | forgetting | B test | protected edges | rehearsal eps | weights changed (A+B) | B protected attempts | B blocked |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised control | none | 0.870±0.039 | 0.617±0.036 | 0.253±0.070 | 0.803±0.027 | 0 | 0 | 2198.1 | 0 | 0 |
| supervised control | consolidate | 0.870±0.039 | 0.699±0.029 | 0.171±0.057 | 0.688±0.044 | 126.3 | 0 | 1637.5 | 2276.0 | 1707.0 |
| supervised control | rehearse | 0.870±0.039 | 0.657±0.042 | 0.213±0.073 | 0.756±0.033 | 0 | 32 | 2317.6 | 0 | 0 |
| supervised control | both | 0.870±0.039 | 0.709±0.035 | 0.161±0.050 | 0.681±0.043 | 126.3 | 32 | 1605.5 | 2312.0 | 1778.6 |
| trace-group-8 | none | 0.839±0.033 | 0.721±0.022 | 0.118±0.043 | 0.681±0.054 | 0 | 0 | 735.0 | 0 | 0 |
| trace-group-8 | consolidate | 0.839±0.033 | 0.810±0.036 | 0.028±0.036 | 0.574±0.059 | 125.0 | 0 | 499.1 | 352.7 | 288.9 |
| trace-group-8 | rehearse | 0.839±0.033 | 0.726±0.017 | 0.113±0.038 | 0.669±0.052 | 0 | 32 | 732.9 | 0 | 0 |
| trace-group-8 | both | 0.839±0.033 | 0.820±0.033 | 0.018±0.032 | 0.574±0.060 | 125.0 | 32 | 484.3 | 335.0 | 278.2 |
| trace-per-target | none | 0.908±0.037 | 0.786±0.031 | 0.122±0.050 | 0.695±0.048 | 0 | 0 | 793.5 | 0 | 0 |
| trace-per-target | consolidate | 0.908±0.037 | 0.889±0.028 | 0.019±0.025 | 0.576±0.060 | 127.1 | 0 | 510.3 | 368.8 | 306.9 |
| trace-per-target | rehearse | 0.908±0.037 | 0.793±0.036 | 0.115±0.053 | 0.675±0.054 | 0 | 32 | 784.0 | 0 | 0 |
| trace-per-target | both | 0.908±0.037 | 0.892±0.030 | 0.017±0.026 | 0.581±0.057 | 127.1 | 32 | 501.3 | 359.9 | 301.5 |

Pre-registered paired comparisons (Δ forgetting + = x forgets less; b/t/w =
seeds where x forgets strictly less / tie / more; Δ retained = x − y on A
test after B; Δ B = x − y on B test; "rule" = pre-stated pass rule):

| x | vs y | Δ A test | Δ forgetting | b/t/w | Δ retained A | Δ B | rule | trade-off vs control (H3) |
|---|---|---|---|---|---|---|---|---|
| control + consolidate | control | +0.000 | +0.083±0.026 | 10/0/0 | +0.083 | −0.115±0.036 | **fails (B guard)** | — |
| control + rehearse | control | +0.000 | +0.041±0.016 | 10/0/0 | +0.041 | −0.047±0.021 | **fails (below margin)** | — |
| control + both | control | +0.000 | +0.092±0.038 | 10/0/0 | +0.092 | −0.122±0.036 | fails (B guard) | — |
| group-8 + consolidate | group-8 | +0.000 | +0.090±0.029 | 10/0/0 | +0.090 | −0.107±0.032 | **fails (B guard)** | — |
| group-8 + rehearse | group-8 | +0.000 | +0.005±0.019 | 4/0/6 | +0.005 | −0.012±0.026 | **fails (no effect)** | — |
| group-8 + both | group-8 | +0.000 | +0.099±0.030 | 10/0/0 | +0.099 | −0.106±0.033 | fails (B guard) | — |
| per-target + consolidate | per-target | +0.000 | +0.103±0.030 | 10/0/0 | +0.103 | −0.119±0.034 | **fails (B guard)** | — |
| per-target + rehearse | per-target | +0.000 | +0.007±0.015 | 6/0/4 | +0.007 | −0.020±0.020 | **fails (no effect)** | — |
| per-target + both | per-target | +0.000 | +0.106±0.034 | 10/0/0 | +0.106 | −0.114±0.033 | fails (B guard) | — |
| group-8 none | control | −0.031±0.024 | +0.136±0.035 | 10/0/0 | +0.104 | −0.123±0.051 | — | no (B) |
| group-8 + consolidate | control | −0.031±0.024 | +0.225±0.046 | 10/0/0 | +0.194 | −0.230±0.057 | — | no (B) |
| group-8 + rehearse | control | −0.031±0.024 | +0.140±0.045 | 10/0/0 | +0.109 | −0.135±0.042 | — | no (B) |
| group-8 + both | control | −0.031±0.024 | +0.235±0.052 | 10/0/0 | +0.203 | −0.229±0.057 | — | no (B) |
| per-target none | control | +0.038±0.017 | +0.131±0.036 | 10/0/0 | +0.169 | −0.109±0.038 | — | no (B) |
| per-target + consolidate | control | +0.038±0.017 | +0.235±0.054 | 10/0/0 | +0.273 | −0.227±0.050 | — | no (B) |
| per-target + rehearse | control | +0.038±0.017 | +0.138±0.047 | 10/0/0 | +0.177 | −0.128±0.040 | — | no (B) |
| per-target + both | control | +0.038±0.017 | +0.237±0.059 | 10/0/0 | +0.275 | −0.223±0.047 | — | no (B) |

## Pre-stated readings

* **H1 (consolidation) not supported.** Consolidation cuts forgetting of
  both graded learners well beyond the margin (group-8 0.118 → 0.028,
  +0.090, 10/10; per-target 0.122 → 0.019, +0.103, 10/10) and lifts
  retained A to 0.810 / 0.889, but B test drops by 0.107 / 0.119, so the
  B guard fails. Same pattern on the control (+0.083 forgetting, −0.115 B).
* **H2 (bounded rehearsal) not supported.** 1-in-8 A rehearsal has almost no
  effect on the graded learners (Δ forgetting +0.005, 4/10; +0.007, 6/10)
  and costs little B (−0.012 / −0.020). On the supervised control it lowers
  forgetting by +0.041 in 10/10 seeds with B −0.047 — consistent, inside the
  B guard, but below the 0.05 margin, so it also fails the rule.
* **H3 (trade-off broken vs the control) not supported.** No passing
  mechanism exists; every graded arm, with or without a mechanism, stays
  0.11–0.23 below the control on B test.
* `both` (exploratory) behaves like `consolidate` (forgetting 0.017–0.018 on
  graded learners, B −0.106 / −0.114): rehearsal adds ≈ +0.01 on top.

## Findings, including negative ones

* **The trade-off moved, it did not break.** Consolidation with a fixed
  1-of-4 gate on A-moved edges trades B learning for A retention at roughly
  1:1 (Δ retained A ≈ +0.09–0.10, Δ B ≈ −0.11–0.12) on all three learners.
  It is a stability/plasticity dial, not a free reduction of forgetting.
* With the gate, per-target retains A at 0.889 after B (vs control 0.617)
  — the best retained-A number of the series — but B (0.576) is far below
  the control (0.803).
* Almost every edge is protected (≈ 125–127 of the generated graph's edges
  move during A for every learner), so this protection rule is not
  selective on this small graph: it is close to a global 1/4 learning-rate
  cut during B. A selective rule (e.g. higher weight threshold, #25's
  `min_weight`) was **not** run; constants were fixed before the run.
* Rehearsal through the reward rule is nearly inert for the graded learners:
  32 A episodes judged against A labels are outweighed by 224 B episodes and
  move only ≈ 2–10 weights fewer overall. The supervised rule uses
  rehearsal better (+0.041) but still below margin.
* The B-test gap of the graded learners vs the control (0.11–0.13 without
  any mechanism) is the remaining blocker for any forgetting claim; neither
  mechanism addresses it, and consolidation widens it.
* Constants were not tuned; no other configuration was run.

## Limitations

Synthetic generated graph only (not BANC; DOI 10.7910/DVN/7WTH1N not built
on the box); artificial OR-rule task with 50 % label flip between A and B;
one graph size; one protection period and one rehearsal rate; delay-0 graded
learners only; balanced accuracy on 64 test patterns. No biological claims.
