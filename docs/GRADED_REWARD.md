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

## Results

(Filled in after the run, numbers exactly as produced.)
