# B888-4 cycle 14: equal weight-write budget across strategies, flip sweep (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124). Nothing here is a claim about synapses,
biology, insect memory or BANC physiology.

Module: `src/write_budget.rs`. CLI: `bench-write-budget`.

## Motivation (what #124 left open)

1. **Budget in episodes, not writes.** In #124 every arm saw 256 episodes per
   task, but the arms did not write the same number of weights: over four
   tasks at flip 50, `E/E-cue@16` changed 445 weights, `error-gated` 209 and
   `supervised` 3594. The +0.061 all-pairs cue gain could be partly bought by
   the extra writes.
2. **Flip dependence only probed as a secondary reading.** The CN gain of
   the cue was −0.009 / +0.018 / +0.110 at flips 25 / 50 / 100. No hypothesis
   tested it, and flip 75 was not run.

## Design (pre-registered)

* **Graph, task, pool, cues:** exactly the #124 fixed-pool design
  (`cue_reach::DESIGN`): generated graph per seed, 128 inputs / 32 targets /
  fan-in 8, threshold 1, init weight 1, 4-bit weights; four tasks T1→T4,
  patterns of 6 active inputs, 32 training patterns per task, one shared TEST
  set of 128 patterns; pair classes seen / CN / ICN / UE as in #123; four
  disjoint greedy cues of up to 16 lines, all removed from the pattern pool.
  The cued arm uses the full cue (level 16, full reach).
* **Strategies** (same episode order, 8 rounds × 32 patterns = 256 episodes
  per task, no rehearsal, original edges only, topology checked per row):
  * `supervised` (S): teacher step on every edge of every active line;
  * `error-gated` (E): teacher step only into reached targets it gets wrong;
  * `E/E-cue` (EE): error gate on pattern and cue edges, task cue on;
  * `reward` (R): reward-modulated with a clean per-target reward (the
    `PerTarget` reference of `reward_limits`): for each reached target that
    fires, the edges from the active lines into it are reinforced if the
    firing was right and weakened if it was wrong. Uncued.
  With no limit, S / E / EE are bit-identical to `cue_gating::step` and R to
  per-target `RewardModulated` updates (tested).
* **Write:** a 4-bit step that actually changes a stored weight. Steps at
  the 0/15 plateaus change nothing, cost nothing and are never refused.
* **Budgets** (per (flip, seed, task), as a fraction of the writes of
  uncapped `error-gated` on the same task, floor):
  * `free`: no limit (= the #124 equal-episode protocol, new seeds);
  * **`paced100` (primary):** after every episode e an arm may have made at
    most as many writes as uncapped `error-gated` had made by episode e. All
    arms get `error-gated`'s write schedule. `error-gated` itself is
    unchanged (bit-identical, no refusal; tested and gated);
  * `total100` (secondary): one cap on the task total, spent in episode
    order (front-loaded);
  * `paced50` (secondary): half of `error-gated`'s schedule (binds
    `error-gated` too).
  A refused step is counted (`refused`, `first_refusal`); the arm keeps
  seeing the remaining episodes. Edges are visited source-major inside an
  episode, so a partially funded episode updates the lowest-index active
  lines first.
* **Rows:** per (flip, seed) 4 strategies × 4 budgets = 16 rows.
* **Seeds:** 161..200 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25, **50 (primary)**, 75, 100 (high conflict).
* **Gains:** x − `error-gated` (uncapped), paired by seed.

Metrics (ACC, LEARN, FGT, LAST at k = 2..4 on all / seen / CN / ICN / UE
pairs), the balanced accuracy, the bootstrap (95 % percentile, 2000
resamples, seed 0xB0075EED), **strong** (mean ≥ 0.05, > 0 in ≥ 32 of 40
seeds, CI lower bound > 0) and the **sequence rule** (Δ ACC strong,
Δ LAST ≥ −0.05, Δ LEARN ≥ −0.05) are as in #123/#124.

### Pilot write census (seeds 1..5, flip 50; writes only, no accuracy computed)

Seeds 1..5 are not used by any hypothesis. Total writes over 4 tasks:

| strategy | free | paced100 | total100 | first refusal (episode of 256) |
|---|---|---|---|---|
| supervised | 3176–3816 | 192–208 (= E) | 192–208 | paced: 1; total: 5–8 |
| error-gated | 192–208 | 192–208, never refused | same | — |
| E/E-cue | 429–462 | 192–208 | 192–208 | paced: 1–8; total: 7–15 |
| reward | 2161–2902 | 192–208 | 192–208 | paced: 1–6; total: 5–9 |

The paced budget binds every non-E arm from the first episodes; the total
budget is exhausted after 5–15 episodes of 256 (front-loaded). The grid
(2560 rows) takes about 8 min on CPU.

## Design gates (reported first)

* **G1 (budget integrity):** every row keeps its topology; every task's
  writes ≤ its allowance; `error-gated` at `paced100` and `total100` is
  bit-identical to uncapped `error-gated` with zero refusals.
* **G2 (budget binds):** at flip 50, `E/E-cue@paced100` refuses at least one
  write in ≥ 50 % of (seed, task).
* **G3 (novel pairs):** median CN pairs per (seed, task) at flip 50 ≥ 20.

If any gate fails, H1–H7 are declared **not evaluable**.

## Hypotheses (k = 4, seeds 161..200; `EE` = `E/E-cue@paced100`, base = uncapped `error-gated`)

* **H1 (cue gain survives equal writes, all pairs, flip 50):** `EE` vs
  `error-gated` passes the sequence rule on all pairs.
* **H2 (most of the gain is not bought by writes, flip 50):** mean
  Δ ACC_all(`EE` − E) ≥ 0.5 × mean Δ ACC_all(`E/E-cue@free` − E), **and**
  its CI lower bound > 0.
* **H3 (CN gain rises with conflict at equal writes):** the mean CN gain of
  `EE` over E never drops by more than 0.005 from flip 25 → 50 → 75 → 100,
  **and** the per-seed difference gain@100 − gain@25 has CI lower bound > 0.
* **H4 (novel pairs at high conflict, flip 100):** `EE` vs E passes the
  sequence rule on CN pairs.
* **H5 (novel-pair forgetting at high conflict, flip 100):**
  FGT_CN(E) − FGT_CN(`EE`) is strong.
* **H6 (error gate vs supervised at equal writes, flip 50):** E vs
  `supervised@paced100` passes the sequence rule on CN pairs.
* **H7 (reward-modulated at equal writes, flip 50, non-inferiority):**
  Δ ACC_all(`reward@paced100` − E) has CI lower bound > −0.05.

**Secondary (reported, no pass/fail):** all strategies at every budget and
flip; `free` (replication of #124 on new seeds), `total100` and `paced50`;
k = 2 and 3; seen, ICN and UE pairs; consecutive flip contrasts; writes,
refusals, first-refusal episode and saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-write-budget --csv results/write-budget-v1.csv > results/write-budget-report-v1.txt
```

## Results

Pending (pre-registration commit first; no learning run on seeds 161..200
before it is pushed).

## Limitations (known before the run)

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1. Nothing is measured on the real BANC graph
  (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- The budget is matched to `error-gated`, so it is the reference arm's
  natural write count, not a free parameter swept over a wide range.
- A paced or total budget changes *which* writes an arm makes (the first
  edges in source-major order inside a partially funded episode, the first
  episodes under `total`), not only how many. Other allocation rules (random
  subsampling, per-target quotas) are not tested.
- The reward arm is uncued, per-target and clean; it cannot raise a silent
  target that should fire.
- CN pairs share TEST patterns with seen pairs. The cue is assumed known at
  test time. 40 seeds. No multiplicity correction across H1–H7.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.
