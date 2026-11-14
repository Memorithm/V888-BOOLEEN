# B888-4 cycle 12: a task-sequence design where generalisation to never-seen input combinations can be measured (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task from
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md), run as the
four-task sequence of [TASK_SEQUENCE.md](TASK_SEQUENCE.md) with one context
cue per task. Nothing here is a claim about synapses, biology, insect memory
or BANC physiology.

Module: `src/novel_split.rs`. CLI: `bench-novel-split` (`--census` for the
design-only pair count). Helper: `context_cue::choose_n_cues_capped`. With
the cap at 12 it is exactly `choose_n_cues` (tested).

## Motivation (what #122 left open)

#122 (cycle 11) found that `E/E-cue` beats `error-gated` and `supervised` by
+0.21 to +0.34 ACC on T1→T4 sequences (40/0/0 seeds). It could **not**
evaluate generalisation to never-seen input combinations. Four reserved
cues of up to 12 lines left ~25 pattern inputs. Only 1.9 % of TEST pairs
were novel, with a median of 6 per task against the pre-registered 20.
`error-gated` reached 1.000 on TEST, which is memorisation. The small pool
also inflated the effect sizes.

This cycle builds a design in which the novel-pair count is large by
construction. It checks that count first, through a design-validity gate.
Then it compares the three arms on novel pairs.

## Design (pre-registered)

* **Graph:** generated per seed, **128 inputs / 32 targets / fan-in 8**,
  threshold 1, init weight 1, 4-bit weights. The #117–#122 graph was
  64 / 24 / 6.
* **Cues:** four disjoint greedy-cover cues with a **cap of 4 lines each**
  (#122 used 12), from `context_cue::choose_n_cues_capped`. All 16 cue lines
  are excluded from the pattern pool for every arm. Cue j is active while
  training task j and while evaluating task j. Uncued arms never activate a
  cue line. A 4-line cue reaches only part of the targets (the pilot
  minimum is 15 of 32). That is a cost of the wider pool, and it is reported
  per row (`cue_reach_*`).
* **Tasks:** T1→T2→T3→T4 with the #122 generator
  (`task_sequence::build_sequence`). In T1 each edge is positive with
  probability 50 %. Each later task flips every edge label of the previous
  task with the flip rate. Patterns have **6 active inputs**. Each task
  gets 32 training patterns, and the four training sets are disjoint. One
  shared held-out TEST set of **128** patterns is disjoint from every
  training set.
* **Target rule (structured):** a reached target should fire iff at least
  one of its active pattern edges is positive (the OR of per-edge
  valences). The label of any configuration is therefore a function of
  per-edge quantities that can be trained in other configurations.
  Generalisation to unseen combinations is possible in principle. A pure
  lookup table of training configurations scores exactly chance (0.5
  balanced) on novel configurations.
* **Budget (equal for every arm):** 8 rounds × 32 patterns = 256 episodes
  per task. Every arm of a seed uses the same episode order. There is no
  rehearsal, and updates touch original edges only. Topology is checked on
  every row.
* **Seeds:** 81..120 (40 seeds), never used by an earlier B888-4 cycle.
  **Flips:** 25, **50 (primary)**, 100.
* **Arms (4):** `supervised` (control), `error-gated`, `S/S-cue` (supervised
  on pattern and cue edges), and `E/E-cue` (error gate on pattern and cue
  edges, the #121/#122 winner).

### Pair classes (per task j, w.r.t. task j's training set)

Each (TEST pattern, reached target) pair has a configuration: the sorted
set of active usable pattern edges into that target.

* **seen:** that (target, configuration) occurred in task j's training set.
* **compositional novel (CN):** the configuration never occurred, but every
  edge in it was active in some training configuration of task j.
* **unseen edge (UE):** at least one edge was never active in task j's
  training. Its label depends on an untrained valence. This class is
  reported, but no hypothesis uses it.
* **inferable CN (ICN):** the subset of CN whose label is logically implied
  by task j's training labels under the OR rule. An edge is known negative
  if it sits in a negative training configuration. It is known positive if
  it is the only edge not known negative in a positive training
  configuration. A negative CN pair is inferable if all its edges are known
  negative. A positive CN pair is inferable if one of its edges is known
  positive.

### Metrics

R_c[i][j] is the balanced accuracy of task j on the TEST pairs of class c,
measured after training task i. For k = 2..4 (prefixes of the 4-task run):

* ACC_c,k = mean over j < k of R_c[k−1][j], the final accuracy.
* LEARN_c,k = mean over j < k of R_c[j][j], the just-learned accuracy.
* FGT_c,k = mean over j < k−1 of R_c[j][j] − R_c[k−1][j], the forgetting.
* LAST_c,k = R_c[k−1][k−1].

Pairing is by seed. CIs are 95 % percentile bootstrap intervals (2000
resamples, seed 0xB0075EED). **Strong** = mean ≥ 0.05, > 0 in ≥ 32 of 40
seeds, and CI lower bound > 0. **Sequence rule** (x vs base on class c at k):
Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

### Pilot (design only, no learning, seeds 1..5)

The design was chosen from a pair census (`census`) on pilot seeds 1..5 at
flip 50. The census needs no training and gives no outcome. No learning run
was looked at before this file was committed.

| design (in/tgt/fan-in, active, cue cap) | pool | min cue reach | seen frac | CN min / median per task | ICN min / median |
|---|---|---|---|---|---|
| 64/24/6, 4, 12 (≈ #122) | 18 | 23 | 0.975 | 1 / 9 | 1 / 9 |
| **128/32/8, 6, 4 (primary)** | 93 | 15 | 0.699 | 169 / 230 | 127 / 174 |
| 128/32/8, 4, 4 | 93 | 15 | 0.634 | 54 / 92 | 41 / 58 |
| 96/32/12, 6, 4 | 77 | 21 | 0.593 | 471 / 662 | 402 / 513 |
| 256/48/8, 8, 4 | 186 | 13 | 0.606 | 141 / 202 | 87 / 143 |
| 128/32/12, 6, 4 | 103 | 19 | 0.579 | 305 / 426 | 214 / 279 |

The primary row is the smallest step from the old shape (fan-in 6 → 8) that
clears the gate by a wide margin. Seeds 1..5 are not used for any
hypothesis.

## Design-validity gate (reported first)

**G:** on seeds 81..120 at flip 50, the median number of CN TEST pairs per
(seed, task) must be ≥ **20** (`MIN_NOVEL_PAIRS`). If G fails, H1–H5 are
declared **not evaluable** and no effect is claimed. The ICN analyses
(secondary) use the same gate on the ICN count. The census and the gate are
reported before any accuracy.

## Hypotheses (flip 50, k = 4, seeds 81..120, CN pairs)

* **H1 (generalisation exists):** `E/E-cue` LEARN_CN,4 − 0.5 is strong. A
  lookup memoriser scores 0.5 by construction.
* **H2:** `E/E-cue` vs `error-gated` passes the sequence rule on CN at k = 4.
* **H3:** `E/E-cue` vs `supervised` passes the sequence rule on CN at k = 4.
* **H4 (forgetting):** `E/E-cue` vs `error-gated`, with
  Δ FGT_CN,4 = FGT(error-gated) − FGT(E/E-cue), is strong.
* **H5 (small generalisation gap):** for `E/E-cue`, the 95 % CI upper bound
  of ACC_seen,4 − ACC_CN,4 is ≤ 0.10.

**Secondary (reported, no pass/fail):** the same comparisons at k = 2 and 3,
on seen, ICN, UE and all pairs, `S/S-cue` vs `supervised`, `E/E-cue` vs
`S/S-cue`, `error-gated` vs `supervised`, flips 25 and 100, weights changed
and saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run.

## Command

```bash
cargo +1.89 run --release -- bench-novel-split --census > results/novel-split-census-v1.txt
cargo +1.89 run --release -- bench-novel-split --csv results/novel-split-v1.csv
```

## Results

Run after the pre-registration commit 2dbc9f8 had been pushed. The grid
(480 rows, 3 flips × 40 seeds × 4 arms) takes about 35 s on CPU. Two runs
produce a byte-identical CSV (sha256 `28dbaa20…28f5a2`) and an identical
report. Every row keeps the original topology (`topology_unchanged_all=true`).
Files: `results/novel-split-census-v1.txt` and `results/novel-split-v1.csv`.

### 1. Design-validity gate (reported first)

| flip | TEST pairs per task | seen | **CN min / median / max** | ICN min / median | UE median | pool | cue lines / reach |
|---|---|---|---|---|---|---|---|
| 50 | 1301 (median) | 911 (69.8 %) | **130 / 208 / 317** | 99 / 156 | 179 | 91–103 | 4 / 14–22 of 32 |

The CN counts do not depend on the flip. **G passes**: the median of 208
CN pairs per (seed, task) is ≥ 20, and the minimum is 130. #122 had a
median of 6. The ICN gate also passes (median 156). H1–H5 are evaluable.

### 2. Arms (flip 50, k = 4, mean over 40 seeds)

| arm | ACC_all | ACC_seen | **ACC_CN** | ACC_ICN | ACC_UE | LEARN_seen | **LEARN_CN** | LEARN_ICN | **FGT_CN** | T1 CN at end | weights changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised | 0.615 | 0.630 | 0.587 | 0.618 | 0.538 | 0.879 | 0.706 | 0.802 | 0.158 | 0.546 | 5160 |
| error-gated | 0.637 | 0.650 | 0.637 | 0.682 | 0.545 | 1.000 | 0.861 | 1.000 | 0.299 | 0.542 | 330 |
| S/S-cue | 0.611 | 0.625 | 0.585 | 0.615 | 0.533 | 0.851 | 0.695 | 0.786 | 0.147 | 0.545 | 10963 |
| **E/E-cue** | 0.649 | 0.666 | **0.640** | 0.689 | 0.545 | 1.000 | **0.861** | 1.000 | **0.295** | 0.548 | 485 |

### 3. Hypotheses (flip 50, k = 4, CN pairs, paired by seed, 95 % bootstrap CI, seeds better / tied / worse)

| | statistic | value | result |
|---|---|---|---|
| G | median CN pairs per (seed, task) ≥ 20 | 208 (min 130) | **pass** |
| H1 | `E/E-cue` LEARN_CN,4 − 0.5 | +0.361 ± 0.037 [+0.349, +0.373], 40/0/0 | **pass** |
| H2 | `E/E-cue` − `error-gated` Δ ACC_CN,4 (Δ LEARN, Δ LAST) | +0.003 ± 0.013 [−0.001, +0.007], 25/3/12 (+0.000, −0.000) | **fail** |
| H3 | `E/E-cue` − `supervised` Δ ACC_CN,4 (Δ LEARN, Δ LAST) | +0.053 ± 0.032 [+0.043, +0.063], 37/0/3 (+0.155, +0.183) | **pass** (narrowly: mean just over 0.05) |
| H4 | FGT(`error-gated`) − FGT(`E/E-cue`) on CN, k = 4 | +0.004 ± 0.016 [−0.001, +0.009], 25/3/12 | **fail** |
| H5 | `E/E-cue` ACC_seen,4 − ACC_CN,4, CI upper bound ≤ 0.10 | +0.026 ± 0.041 [+0.013, +0.038] | **pass** |

### 4. Secondary (no pass/fail)

* **H3 belongs to the error gate, not the cue.** `error-gated` −
  `supervised` on CN at k = 4 is +0.050 [+0.040, +0.059], 38/0/2. That
  misses "strong" only because the mean is just under 0.05. `E/E-cue` −
  `error-gated` is +0.003. The supervised control loses on CN mainly
  because it learns each task less well (LEARN_CN 0.706 vs 0.861). It
  forgets less in absolute terms (0.158 vs 0.299) because it has less to
  forget.
* **The #122 cue benefit does not carry over to this design.** On all
  pairs, `E/E-cue` − `error-gated` is +0.011 [+0.009, +0.013] at k = 4
  (+0.010 at k = 2, +0.011 at k = 3). #122 measured +0.215. At flip 25 it
  is +0.006, and at flip 100 it is +0.020 (CN +0.013). T1 at the end of the
  sequence is near chance for every arm (CN 0.542–0.548). The cue no longer
  protects the first task.
* **Generalisation is real but has limits.** Both error-gated arms score
  1.000 on seen pairs and 1.000 on **inferable** novel pairs (ICN) right
  after each task, at every flip. So whenever the training labels logically
  fix a novel label, the error-gated rule gets it right. On all CN pairs
  they reach 0.861, and on unseen-edge pairs 0.550, near chance as
  expected. Supervised arms reach 0.802 on ICN. Their update reinforces
  negative edges that co-occur with positive ones under the OR teacher.
* **`S/S-cue` vs `supervised`:** the cue costs supervised learning a little
  on every slice (CN Δ ACC −0.003, Δ LAST −0.009 at k = 4; all pairs Δ LAST
  −0.021), as #120 found.
* **Gap at other flips:** for `E/E-cue`, seen − CN is +0.033 at flip 25 and
  −0.005 at flip 100. H5 holds at every flip.
* **Write cost:** `E/E-cue` changes 485 weights over the 4 tasks,
  `error-gated` 330, `supervised` 5160 and `S/S-cue` 10963.

### Reading

The design question from #122 is answered. With a wider pattern pool and
4-line cues, every task has hundreds of never-seen input combinations whose
labels follow from trained per-edge quantities. On those the error-gated
rule generalises well above lookup chance (+0.36 balanced accuracy), and it
is perfect on the logically implied subset. That makes it better than
supervised training on novel pairs (H3, narrow).

The cue result of #121/#122 does **not** replicate in this design. The
error gate on cue edges adds nothing measurable on novel pairs (H2 fail)
and does not reduce forgetting (H4 fail). The likely cause, **not tested
here**, is that a 4-line cue reaches only 14–22 of 32 targets, while
#122's 12-line cues reached every target. Many targets therefore have no
context signal to separate the tasks. A follow-up should vary cue cap and
reach at a fixed pool before drawing any conclusion about cues and
generalisation.

## Limitations

- Synthetic graph and artificial OR-rule task only. One primary design
  (128/32/8, active 6, cue cap 4), threshold 1, init 1. Nothing is measured
  on the real BANC graph (DOI 10.7910/DVN/7WTH1N), which is not built on
  this machine.
- The design was chosen from a label-free census on pilot seeds 1..5. Only
  the pair counts were looked at, never an accuracy.
- Cue reach is confounded with pool width: fewer lines per cue widens the
  pool **and** shrinks cue reach. This cycle cannot separate the two
  effects.
- CN pairs share TEST patterns with seen pairs, so they are not
  independent samples. The pair count is per (seed, task), not per
  pattern.
- The cue is assumed known at test time. 40 seeds. No multiplicity
  correction across H1–H5. H3 passes by a narrow margin (mean +0.053
  against the 0.05 threshold).
- The equal budget is in episodes, not weight writes.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas)
are untouched. No NVIDIA dependency, no secrets. CPU only.
