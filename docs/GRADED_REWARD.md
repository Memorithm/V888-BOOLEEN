# B888-4 — graded reward ladder: does reward information per episode explain the gap? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md). "Reward",
"eligibility" and "group" name parts of an update rule on a Boolean graph;
nothing here is a claim about dopamine, biology, insect memory or BANC
physiology.

Module: `src/graded_reward.rs`. CLI: `bench-graded-reward`.

## Motivation

[ELIGIBILITY_TRACES.md](ELIGIBILITY_TRACES.md) (#36) left the trace arm
0.27–0.32 below the supervised control (0.870) and diagnosed that a single
"all predicted spikes right" bit per episode carries too little information;
at delay 4 the right episode's tag (4) is weaker than the tags of the 4
later episodes. This cycle tests both points directly.

## Pre-registration (this section is committed before any run)

### Information-content ladder (delay 0)

Every ladder arm uses the **#36 trace machinery unchanged** (4-bit trace,
tag 8, decay 1 per episode, reward-prediction error with baseline init 512
and rate 1/16, bounded credit ±2048, step threshold 1024, no reactivation).
Only the reward signal differs. A *signal* is a pair (group g, value
v ∈ 0..=1024); each group has its own phase-local baseline (init 512), and a
signal for group g updates only the credit of tagged edges whose target is in
group g, with `rpe = v − b_g`, `b_g += (v − b_g)/16`,
`c += rpe·e/8` (integer arithmetic toward zero, as in #36). Signals exist only
for groups containing at least one firing reached target ("firing set").

| arm | signals per episode | value |
|---|---|---|
| `bit` | 1 (group = all targets) | 1024 iff every firing target is right, else 0 — **identical to the #36 `trace` arm at d = 0** (tested bit-identical) |
| `fraction` | 1 (group = all targets) | `1024·right/firing` (integer division) |
| `group-2` / `group-4` / `group-8` | one per group with a firing target; group = target index mod k | 1024 iff every firing target of the group is right |
| `per-target` | one per firing target (group = the target) | 1024 iff the target is right |

Group partition: target neuron index mod k (on the generated graph the 24
targets are indices 64..87, so k ∈ {2, 4, 8} gives equal groups of 12, 6, 3).

Reward information per episode is reported as counted from the run:
signals emitted per episode, and a nominal bit count `Σ log2(levels)` over
emitted signals (binary signal = 1 bit; `fraction` signal = log2(firing+1)).

### Episode-stamped tag arm (delay fix, separate)

`stamped` (bit reward, d ∈ {1, 4}): instead of the decaying trace, the arm
remembers the exact set of edges tagged (active pre, firing post) in each
episode; when episode k − d's scalar arrives in episode k, credit is applied
with full eligibility (e = 8) to **that episode's** tagged edges only.
Baseline, RPE, credit, threshold and clamp as in #36. Compared with the #36
`trace` arm at the same delay (same scalar, same budget). Scalars still in
flight at phase end are dropped (≤ d per phase), as in #34/#36.

### References (same seeds, same budget, same order)

`supervised-matched` (clean supervised control), the #34
`reward-clean-per-target-ref` (per-target reward, #34 update rule), the #34
`reward-modulated` global arm (d = 0), and the #36 `trace` arm at d = 1 and
d = 4. All reuse the merged code paths.

### Protocol

`bench-eligibility-traces` defaults: generated graph 64 inputs / 24 targets /
fan-in 6; 4 active; 32 A-train / 32 B-train / 64 TEST patterns; 8 rounds per
phase (256 episodes per phase, equal for every arm); flip 50 %; init weight
1; threshold 1; balanced accuracy; forgetting = bal(A test after A) −
bal(A test after B); seeds 1..10. Per seed: 3 references + 6 ladder arms +
2 trace + 2 stamped = 13 rows, 130 rows total. Deterministic: the grid is run
twice and the CSV must be byte-identical. Only weights of existing edges
change (topology checked per row).

### Pre-stated reading (margin 0.05; #27 verdict rule: mean paired delta ≥ 0.05 and better in ≥ 8/10 seeds)

* **H1 (information content)** is *supported* iff `per-target` beats `bit`
  on A-test balanced accuracy **and** the mean A test is non-decreasing along
  the partition-refinement chain `bit → group-2 → group-4 → group-8 →
  per-target`. If `per-target` beats `bit` but the chain is not monotone:
  *partially supported*. Otherwise: *not supported*.
* **H2**: `fraction` beats `bit` (graded value without spatial credit).
* **H3**: `per-target` (trace rule) *reaches the control* iff its paired
  A-test delta vs supervised control is > −0.05.
* **H4 (delay fix)**: `stamped` beats the #36 `trace` arm at d = 4; the same
  comparison at d = 1 is reported, not part of H4.
* Forgetting differences are reported with the B-test guard (B must not drop
  by more than 0.05) and not claimed otherwise.
* All constants above are fixed a priori and are **not tuned** on results.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-graded-reward --csv results/graded-reward-v1.csv
```

## Results (pre-registered grid; CSV `results/graded-reward-v1.csv`)

CSV: 130 rows (10 seeds × 13 arms), sha256
`9eeddaf52682d028d6d93f5f1a06fdcd14621be0fbc3e362aba3405d489ce85f`; a second
run gave a byte-identical CSV and identical stdout. Run time ≈ 1.8 s
(release, CPU). `topology_unchanged=true` for every row. References reproduce
#34/#36 exactly (control 0.870, #34 global 0.559, #36 trace 0.588 / 0.570 /
0.548 at d = 0 / 1 / 4; `trace-bit` is bit-identical to the #36 `trace` arm,
also tested). Mean ± sample sd over 10 seeds. "signals/ep" and "bits/ep" are
counted over the 512 A+B episodes; for the #34 reward reference they are its
emitted scalars, for the control and the clean reference they are not
applicable (0 in the CSV); step up/down counters are only defined for the
trace-machinery arms.

| arm | d | A train | A test | forgetting | B test | signals/ep | bits/ep | episodes with signal (of 512) | weights changed | steps up / down |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised control | – | 0.945 | 0.870±0.039 | 0.253 | 0.803 | – | – | – | 2198.1 | – |
| #34 reward, clean per-target (ref) | – | 0.945 | 0.870±0.039 | 0.142 | 0.716 | – | – | – | 1978.9 | – |
| #34 reward, global (ref) | 0 | 0.547 | 0.559±0.030 | 0.030 | 0.531 | 0.443 | 0.443 | 226.7 | 467.8 | – |
| trace-bit (= #36 trace) | 0 | 0.582 | 0.588±0.031 | 0.026 | 0.548 | 0.661 | 0.661 | 338.2 | 390.6 | 171.7 / 218.9 |
| trace-fraction | 0 | 0.651 | 0.612±0.028 | 0.014 | 0.579 | 0.988 | 2.934 | 505.7 | 298.5 | 209.3 / 89.2 |
| trace-group-2 | 0 | 0.657 | 0.652±0.041 | 0.026 | 0.572 | 1.289 | 1.289 | 446.3 | 561.0 | 302.3 / 258.7 |
| trace-group-4 | 0 | 0.787 | 0.746±0.028 | 0.083 | 0.633 | 2.359 | 2.359 | 497.9 | 669.6 | 400.0 / 269.6 |
| trace-group-8 | 0 | 0.893 | 0.839±0.033 | 0.118 | 0.681 | 3.699 | 3.699 | 510.3 | 735.0 | 480.8 / 254.2 |
| trace-per-target | 0 | 0.979 | 0.908±0.037 | 0.122 | 0.695 | 5.361 | 5.361 | 511.8 | 793.5 | 579.9 / 213.6 |
| #36 trace (ref) | 1 | 0.564 | 0.570±0.036 | 0.011 | 0.540 | 0.613 | 0.613 | 314.1 | 338.2 | 136.9 / 201.3 |
| stamped-bit | 1 | 0.608 | 0.616±0.046 | 0.049 | 0.557 | 0.718 | 0.718 | 367.5 | 286.8 | 111.0 / 175.8 |
| #36 trace (ref) | 4 | 0.536 | 0.548±0.032 | 0.007 | 0.534 | 0.576 | 0.576 | 295.0 | 327.3 | 129.7 / 197.6 |
| stamped-bit | 4 | 0.605 | 0.611±0.045 | 0.043 | 0.554 | 0.727 | 0.727 | 372.0 | 293.9 | 115.9 / 178.0 |

Pre-registered paired comparisons on A-test balanced accuracy (b/t/w =
better/tie/worse seeds; verdict = #27 rule, margin 0.05, ≥ 8/10 seeds;
Δ forgetting + = x forgets less; Δ B = x − y):

| x vs y | Δ A test | b/t/w | verdict | Δ forgetting | Δ B | forgetting verdict | reaches control? |
|---|---|---|---|---|---|---|---|
| fraction vs bit | +0.024±0.038 | 7/0/3 | no clear difference | +0.013 | +0.031 | no clear difference | — |
| group-2 vs bit | +0.064±0.037 | 10/0/0 | **x beats y** | +0.000 | +0.024 | no clear difference | — |
| group-4 vs bit | +0.158±0.046 | 10/0/0 | **x beats y** | −0.057 | +0.086 | x worse (forgets more) | — |
| group-8 vs bit | +0.250±0.035 | 10/0/0 | **x beats y** | −0.091 | +0.133 | x worse (forgets more) | — |
| per-target vs bit | +0.320±0.036 | 10/0/0 | **x beats y** | −0.096 | +0.147 | x worse (forgets more) | — |
| bit vs control | −0.282±0.040 | 0/0/10 | worse | +0.227 | −0.256 | no clear difference (B guard) | no |
| fraction vs control | −0.258±0.038 | 0/0/10 | worse | +0.240 | −0.224 | no clear difference (B guard) | no |
| group-2 vs control | −0.218±0.027 | 0/0/10 | worse | +0.227 | −0.231 | no clear difference (B guard) | no |
| group-4 vs control | −0.124±0.033 | 0/0/10 | worse | +0.170 | −0.170 | no clear difference (B guard) | no |
| group-8 vs control | −0.031±0.024 | 2/0/8 | no clear difference | +0.136 | −0.123 | no clear difference (B guard) | **yes** |
| per-target vs control | +0.038±0.017 | 10/0/0 | no clear difference (below margin) | +0.131 | −0.109 | no clear difference (B guard) | **yes** |
| per-target (trace rule) vs #34 clean per-target | +0.038±0.017 | 10/0/0 | no clear difference | +0.020 | −0.021 | no clear difference | — |
| stamped vs #36 trace, d = 1 | +0.046±0.047 | 8/0/2 | no clear difference | −0.039 | +0.018 | no clear difference | — |
| stamped vs control, d = 1 | −0.254±0.064 | 0/0/10 | worse | +0.204 | −0.246 | no clear difference (B guard) | no |
| stamped vs #36 trace, d = 4 | +0.063±0.051 | 9/0/1 | **x beats y** | −0.035 | +0.019 | no clear difference | — |
| stamped vs control, d = 4 | −0.259±0.064 | 0/0/10 | worse | +0.211 | −0.250 | no clear difference (B guard) | no |

H1 chain (mean A test): bit 0.588 → group-2 0.652 → group-4 0.746 → group-8
0.839 → per-target 0.908 (monotone).

## Pre-stated readings

* **H1 supported.** `per-target` beats `bit` (+0.320, 10/10) and the
  refinement chain is monotone. With the #36 trace rule unchanged, A-test
  generalisation rises with the number of independent reward bits per
  episode: 0.66 → 1.29 → 2.36 → 3.70 → 5.36 bits/ep gives 0.588 → 0.652 →
  0.746 → 0.839 → 0.908.
* **H2 not supported.** `fraction` vs `bit` +0.024 (7/10), below margin,
  although its nominal information (2.93 bits/ep) exceeds group-2 and group-4.
  A graded value without *spatial* attribution does not help much: the bits
  that matter are those that say *which* targets were right.
* **H3: per-target reaches the control** (+0.038 vs 0.870, 10/10 better,
  below the 0.05 margin so not "beats"). group-8 (3 targets per group) also
  reaches it (−0.031). group-4 and coarser remain worse.
* **H4 supported (narrowly).** Episode-stamped tags beat the #36 decaying
  trace at d = 4 (+0.063, 9/10), lifting 0.548 → 0.611; at d = 1 +0.046
  (8/10) is below margin. Stamped bit reward is still ≈ 0.25 below the
  control at both delays: fixing *when* credit lands does not fix the
  information deficit of a single bit.

## Findings, including negative ones

* The #36 diagnosis is confirmed on this synthetic task: the gap to the
  supervised control is mainly an information gap, not an update-rule gap.
  The same trace + RPE rule closes it once the reward identifies groups of
  ≤ 3 targets (group-8) or single targets.
* Generalisation and forgetting trade off along the ladder. Finer reward
  learns more and therefore forgets more: forgetting 0.026 (bit) → 0.083
  (group-4) → 0.118 (group-8) → 0.122 (per-target), "x worse" vs bit for
  group-4/8/per-target. All trace arms still forget less than the control
  (0.253), but always with B test 0.11–0.26 lower, so no forgetting claim
  passes the B guard.
* Fine-grained arms also hit every episode: episodes with a signal rise from
  338 (bit) to 512 (per-target) of 512, since a fine-grained signal exists
  whenever any target fires; steps shift from mostly down (bit 172 up / 219
  down) to mostly up (per-target 580 / 214).
* Negative: graded *value* without spatial credit (`fraction`) is not a
  remedy; stamping fixes timing but not information; nothing here beats the
  control by the pre-stated margin.
* Nominal "bits per episode" is an upper-bound count of signal levels, not
  a measured mutual information; the fraction arm shows it is not the right
  axis by itself.
* Constants were not tuned; no other configuration was run.

## Caveats

Synthetic, small, bipartite generated graph (no recurrence), artificial
OR-rule labels, threshold 1, init weight 1, 10 seeds, one set of a-priori
constants, group partition by target index mod k (arbitrary, not anatomical
regions), trace state and baselines phase-local, ≤ d in-flight scalars
dropped per phase. Not evidence about BANC or biology.
