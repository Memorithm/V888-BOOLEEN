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

(pending — filled in after the run, below this line only)
