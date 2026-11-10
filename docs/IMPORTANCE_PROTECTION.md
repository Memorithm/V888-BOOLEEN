# B888-4 — importance-weighted protection and sparse task-gated plasticity: a different handle on the learning-vs-forgetting trade-off (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md).
"Importance", "protection" and "gating" name parts of an update rule on a
Boolean graph with 4-bit edge weights; nothing here is a claim about
synapses, biology, insect memory or BANC physiology. The importance rule is
an engineering analogue of EWC-style penalties (Kirkpatrick et al. 2017,
PNAS 114:3521) for discrete weights, not a reimplementation of EWC.

Module: `src/importance_protection.rs`. CLI: `bench-importance-protection`.

## Motivation

[CONSOLIDATED_GRADED.md](CONSOLIDATED_GRADED.md) (#39): protecting every
edge moved by task A (≈ 125 of 144 edges) with a 1-of-4 gate cut forgetting
by ≈ 0.09–0.10 but cost ≈ 0.11–0.12 of B test on all three learners — a
≈ 1:1 trade, effectively a global learning-rate cut. Bounded rehearsal had
≈ no effect. This cycle tests two **selective** alternatives that leave most
of the graph fully plastic during B:

1. protect only the edges that carried the most task-A credit;
2. split the original edges into two disjoint per-task plastic subsets.

## Pre-registration (this section is committed before any run)

### Learners (unchanged code paths, from #38/#39)

* `supervised-matched` — clean supervised control (#34/#36/#38).
* `trace-group-8` — #38 graded reward, group = target index mod 8, delay 0.
* `trace-per-target` — #38 graded reward, one bit per firing target, delay 0.

### Mechanisms (constants fixed a priori, not tuned)

Only weights of existing (original) edges ever change; no edge is added or
removed. Every arm uses the same 256 A episodes and the same 256 B episodes
in the #38 orders (equal budget, no rehearsal).

* `none` — must be bit-identical to the #39 `none` row (tested).
* `consolidate-39` — the #39 `consolidate` rule reproduced through the same
  gate (protected = weight after A ≠ init, 1 of 4 B attempts pass); must be
  bit-identical to the #39 row (tested). Reference arm.
* `importance-25` / `importance-50` — **importance from task-A credit**.
  During A, every update *attempt* on edge e (the existing gate call: an edge
  offered to the supervised update, or a credit-threshold crossing of the
  trace rule) increments `credit[e]`. After A, importance(e) =
  `credit[e]` if the weight of e moved during A (weight after A ≠ init),
  else 0. The edges with importance > 0 are ranked by importance
  (descending, ties by edge index ascending) and the top
  `ceil(q · edges)` (q = 25 % / 50 % of **all** edges, capped by the number
  with importance > 0) are **frozen** during B: every B attempt on them is
  refused (counted, not retried; same "lost step" semantics as #39). All
  other edges are fully plastic in B. Memory: one u32 counter + one bit per
  edge.
* `task-gated` — **sparse task-gated plasticity**. A fixed deterministic
  per-edge mask `m(e) = mix(e ⊕ seed ⊕ GATE_SALT) & 1`
  (`GATE_SALT = 0x6A7E_D00D`) splits the edges ≈ 50/50. During A only edges
  with `m = 0` are plastic, during B only edges with `m = 1`. A refused
  attempt is counted. This arm changes A learning too, so its A test after A
  is guarded like everything else.

### Protocol

`bench-graded-reward` / `bench-consolidated-graded` defaults: generated
graph 64 inputs / 24 targets / fan-in 6; 4 active; 32 A-train / 32 B-train /
64 TEST patterns; 8 rounds per phase (256 episodes per phase); flip 50 %;
init weight 1; threshold 1; balanced accuracy on held-out TEST patterns;
forgetting = bal(A test after A) − bal(A test after B); **seeds 1..10**.
Grid: 3 learners × 5 mechanisms = 15 rows per seed, 150 rows. The grid is
run twice; the CSV must be byte-identical. Topology checked per row.
Checkpoint/resume: saving the circuit after A (`.v888pls`) plus the
importance mask, reloading and running B must give a bit-identical result
(tested).

### Pre-stated rule (margin 0.05; ≥ 8/10 seeds), paired by seed vs (L, none)

M **passes for L** iff (1) forgetting reduction (none − M) mean ≥ 0.05 and
strictly positive in ≥ 8/10 seeds; (2) A test after A mean drop ≤ 0.05;
(3) B test after B mean drop ≤ 0.05.

* **H1 (importance protection)**: `importance-25` passes for both graded
  learners.
* **H2 (task-gated plasticity)**: `task-gated` passes for both graded
  learners.
* **H3 (better exchange rate than #39)**: for some graded learner L and some
  M ∈ {importance-25, importance-50, task-gated}: the *trade sum*
  Δretained-A + ΔB-test (x − (L, none)) has mean ≥ +0.05 and is > 0 in
  ≥ 8/10 seeds, **and** forgetting reduction mean ≥ 0.05. (For
  `consolidate-39` this sum was ≈ −0.02, i.e. ≈ 1:1.)
* **H4 (trade-off vs the control)**: some graded (L, M) that passes for L
  also has, vs (`supervised-matched`, none): A-test Δ > −0.05, B-test Δ >
  −0.05, forgetting lower by ≥ 0.05 in ≥ 8/10 seeds.
* `importance-50` is reported under the same rule but is not part of H1.
  Passes on `supervised-matched` are reported separately.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-importance-protection --csv results/importance-protection-v1.csv
```

## Results (pre-registered grid; CSV `results/importance-protection-v1.csv`)

CSV: 150 rows (10 seeds × 3 learners × 5 mechanisms), sha256
`7eb30eb6b718b65e4deb7bb00d43eeb00f38d1f1b1b5cde54c92c30cbad7b691`. A
second run produced a byte-identical CSV and identical stdout. Run time is
about 1.9 s (release build, CPU). `topology_unchanged=true` for every row.
The `none` and `consolidate-39` rows reproduce #39 exactly (also tested
bit-for-bit). The graph has 144 edges, so `importance-25` freezes 36 and
`importance-50` freezes 72; on average `task-gated` leaves 77 edges plastic
in A and 67 in B. Values are mean ± sample sd over 10 seeds; counters are
means.

| learner | mechanism | A test (after A) | A test after B (retained) | forgetting | B test | protected edges | weights changed (A+B) | A blocked | B blocked |
|---|---|---|---|---|---|---|---|---|---|
| supervised control | none | 0.870±0.039 | 0.617±0.036 | 0.253±0.070 | 0.803±0.027 | 0 | 2198.1 | 0 | 0 |
| supervised control | consolidate-39 | 0.870±0.039 | 0.699±0.029 | 0.171±0.057 | 0.688±0.044 | 126.3 | 1637.5 | 0 | 1707.0 |
| supervised control | importance-25 | 0.870±0.039 | 0.687±0.038 | 0.183±0.066 | 0.735±0.033 | 36 | 1964.3 | 0 | 620.0 |
| supervised control | importance-50 | 0.870±0.039 | 0.763±0.030 | 0.107±0.053 | 0.668±0.028 | 72 | 1700.0 | 0 | 1310.4 |
| supervised control | task-gated | 0.671±0.029 | 0.688±0.024 | −0.017±0.038 | 0.703±0.021 | 67.0 | 1160.8 | 1413.6 | 1217.6 |
| trace-group-8 | none | 0.839±0.033 | 0.721±0.022 | 0.118±0.043 | 0.681±0.054 | 0 | 735.0 | 0 | 0 |
| trace-group-8 | consolidate-39 | 0.839±0.033 | 0.810±0.036 | 0.028±0.036 | 0.574±0.059 | 125.0 | 499.1 | 0 | 288.9 |
| trace-group-8 | importance-25 | 0.839±0.033 | 0.767±0.017 | 0.072±0.032 | 0.628±0.056 | 36 | 594.1 | 0 | 166.4 |
| trace-group-8 | importance-50 | 0.839±0.033 | 0.810±0.030 | 0.028±0.031 | 0.586±0.050 | 72 | 494.5 | 0 | 289.7 |
| trace-group-8 | task-gated | 0.629±0.032 | 0.642±0.037 | −0.013±0.046 | 0.696±0.036 | 67.0 | 439.7 | 480.3 | 200.4 |
| trace-per-target | none | 0.908±0.037 | 0.786±0.031 | 0.122±0.050 | 0.695±0.048 | 0 | 793.5 | 0 | 0 |
| trace-per-target | consolidate-39 | 0.908±0.037 | 0.889±0.028 | 0.019±0.025 | 0.576±0.060 | 127.1 | 510.3 | 0 | 306.9 |
| trace-per-target | importance-25 | 0.908±0.037 | 0.818±0.029 | 0.090±0.044 | 0.660±0.057 | 36 | 647.9 | 0 | 155.4 |
| trace-per-target | importance-50 | 0.908±0.037 | 0.857±0.027 | 0.052±0.031 | 0.610±0.060 | 72 | 531.4 | 0 | 283.1 |
| trace-per-target | task-gated | 0.678±0.032 | 0.696±0.033 | −0.018±0.036 | 0.725±0.027 | 67.0 | 425.8 | 391.7 | 210.4 |

Paired comparisons against the same learner's `none` arm. Δ forgetting > 0
means x forgets less. b/t/w counts seeds where x is better / tied / worse.
Trade sum = Δretained A + ΔB; +/t/− counts seeds where it is positive /
zero / negative.

| x | Δ A test | Δ forgetting | b/t/w | Δ retained A | Δ B | trade sum | +/t/− | rule | H3 |
|---|---|---|---|---|---|---|---|---|---|
| control + consolidate-39 | +0.000 | +0.083±0.026 | 10/0/0 | +0.083 | −0.115 | −0.033±0.025 | 1/0/9 | fails (B) | no |
| control + importance-25 | +0.000 | +0.071±0.037 | 10/0/0 | +0.071 | −0.068 | +0.003±0.022 | 4/0/6 | fails (B) | no |
| control + importance-50 | +0.000 | +0.146±0.040 | 10/0/0 | +0.146 | −0.136 | +0.011±0.040 | 7/0/3 | fails (B) | no |
| control + task-gated | −0.199 | +0.271±0.056 | 10/0/0 | +0.072 | −0.100 | −0.029±0.044 | 2/0/8 | fails (A, B) | no |
| group-8 + consolidate-39 | +0.000 | +0.090±0.029 | 10/0/0 | +0.090 | −0.107 | −0.017±0.035 | 3/0/7 | fails (B) | no |
| group-8 + importance-25 | +0.000 | +0.046±0.017 | 10/0/0 | +0.046 | −0.052 | −0.006±0.024 | 4/0/6 | **fails (margin, B)** | no |
| group-8 + importance-50 | +0.000 | +0.089±0.029 | 10/0/0 | +0.089 | −0.095 | −0.005±0.032 | 4/0/6 | fails (B) | no |
| group-8 + task-gated | −0.210 | +0.131±0.038 | 10/0/0 | −0.079 | +0.015 | −0.063±0.056 | 1/0/9 | **fails (A)** | no |
| per-target + consolidate-39 | +0.000 | +0.103±0.030 | 10/0/0 | +0.103 | −0.119 | −0.016±0.026 | 4/0/6 | fails (B) | no |
| per-target + importance-25 | +0.000 | +0.032±0.019 | 10/0/0 | +0.032 | −0.035 | −0.003±0.010 | 4/0/6 | **fails (margin)** | no |
| per-target + importance-50 | +0.000 | +0.071±0.027 | 10/0/0 | +0.071 | −0.085 | −0.014±0.019 | 3/0/7 | fails (B) | no |
| per-target + task-gated | −0.230 | +0.141±0.046 | 10/0/0 | −0.090 | +0.030 | −0.060±0.042 | 0/0/10 | **fails (A)** | no |

Against the supervised control (`none`), every graded arm stays at least
0.079 below it on B test. For per-target the gaps are: none −0.109,
importance-25 −0.143, importance-50 −0.193, and task-gated −0.079, the last
with A test also −0.192. H4 therefore fails for every arm (the full table is
in the CLI output).

## Pre-stated readings

* **H1 (importance-25): not supported.** Freezing the 36 top-credit edges
  cut forgetting in 10/10 seeds, but by less than the margin (group-8
  +0.046, per-target +0.032), and B fell by about the same amount (−0.052 /
  −0.035).
* **H2 (task-gated): not supported.** Forgetting disappears (−0.013 to
  −0.018, so A test after B ≥ A test after A) because B cannot touch A's
  edges. But with only half the edges, A test after A drops by 0.21–0.23,
  so the A guard fails for every learner. B test is not hurt (+0.015 /
  +0.030 on the graded learners).
* **H3 (better exchange rate than 1:1): not supported.** No arm reaches a
  trade sum of +0.05; they range from −0.063 to +0.011. Importance
  protection sits at about 0 (−0.003 to +0.011), so the trade is **still
  about 1:1**. That is only slightly better than #39 consolidation (−0.016
  to −0.033).
* **H4 (trade-off vs the control): not supported.** No arm passes its own
  rule, and every graded arm stays 0.08–0.23 below the control on B test.

## Findings, including negative ones

* Choosing the protected edges by task-A credit did not break the
  trade-off. Across frozen edges, the gain in retained A and the loss in B
  were almost equal, at both 25 % and 50 %. On group-8, `importance-50`
  matches #39 consolidation's forgetting (0.028) while protecting 72 edges
  instead of about 125, and loses slightly less B (−0.095 vs −0.107). On
  per-target it is weaker on both measures. It behaves like a dial again:
  at best a little more efficient than #39, never by the margin.
* One likely explanation is that on this graph (fan-in 6, 50 % label flip
  between A and B), the edges that carry A credit are largely the same
  edges B needs to flip, so B has to overwrite them. Any protection rule
  would then be stuck at a 1:1 trade. This is an interpretation, not a
  tested claim.
* Task-gated plasticity is the only arm that keeps A completely and leaves
  B learning intact for the graded learners. The cost comes earlier: each
  task gets half the capacity, and A generalisation falls to 0.63–0.68.
  Retained A after B (0.642 / 0.696) is still below the `none` arm
  (0.721 / 0.786).
* Not run (constants were fixed before the run): a random-freeze control
  with the same number of edges, so this run does not show that
  credit-based selection beats random selection; other freeze fractions;
  soft (partial) protection; gate splits other than 50/50; larger fan-in.

## Limitations

Synthetic generated graph only (not BANC; DOI 10.7910/DVN/7WTH1N is not
built on the box). The task is an artificial OR rule with a 50 % label flip
between A and B. One graph size (144 edges), two freeze fractions, one gate
split, delay-0 graded learners only, and balanced accuracy on 64 held-out
test patterns. No biological claims.
