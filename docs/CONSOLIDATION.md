# B888-4 — consolidation contre oubli (hypothèse exécutable)

Expérience : apprendre la tâche A, puis la tâche B qui entre en conflit sur une arête partagée, et mesurer ce qui reste de A. Les étiquettes sont artificielles ; aucune affirmation biologique. Seules les arêtes originales reçoivent un coefficient 4 bits ; l'anatomie n'est jamais modifiée (`topology_unchanged` est vérifié).

- Tâche A : source `a` doit activer la cible, source `b` ne doit pas.
- Tâche B : source `b` doit activer la cible, source `c` ne doit pas (conflit sur `b → cible`).
- Témoin : `protect_period=1`, règle supervisée seule, même budget (`rounds_a`, `rounds_b`).
- Consolidation : les arêtes visitées pendant A (`min_visits`) sont protégées pendant B ; une arête protégée n'accepte qu'une tentative de mise à jour sur `protect_period`.

API : `consolidation::{ConsolidationConfig, ConsolidationState, learn_task_a, learn_task_b_and_evaluate, run_experiment, sweep}`. Checkpoint `V888CON1` (poids, visites, compteurs ; jamais écrasé ; rejet si graphe/troncature différents) : reprise après A bit à bit identique à l'exécution continue (testé). CLI : `bench-consolidation [--graph] [--seed] [--rounds-a] [--rounds-b] [--periods 2,4,8,1000]`.

Instrumentation : arêtes visitées, poids effectivement changés, mises à jour bloquées, taux d'effet en B, activations observées, octets de poids et de surcoût (compteurs `u32` par arête), durée.

## Résultats mesurés (graphe synthétique `comparison_graph`, 1 cible, 3 sources)

| protect_period | A après A | A après B | B après B |
|---|---|---|---|
| 1 (témoin) | 2/2 | 1/2 | 2/2 |
| 2, 4, 8 (8 tours B) | 2/2 | 1/2 | 2/2 |
| 2…32 (64 tours B) | 2/2 | 1/2 | 2/2 |
| 1000 (gel de fait) | 2/2 | 2/2 | 1/2 |

**Résultat négatif.** La protection graduée n'améliore pas la rétention : un seul pas suffit à franchir le seuil, donc dès que la protection laisse passer une mise à jour, A est écrasée comme dans le témoin. Seul le gel quasi total conserve A, au prix exact de l'incapacité d'apprendre B (compromis stabilité/plasticité, sans gain gratuit). La graine ne change ici que l'ordre des épisodes. Un seul graphe synthétique minuscule : aucune généralisation revendiquée ; pas encore mesuré sur BANC.

## Strategy matrix and CSV export (follow-up)

`v888 bench-consolidation-matrix [--graph g.v888b] [--seeds 1,2,3] [--rounds-a 8] [--rounds-b 8] [--variants 2,4,8,1000,1000:1,1000:15] [--csv out.csv]`

Runs every rule (supervised control, associative-coactivation,
reward-modulated) through the same A-then-B protocol with identical budgets
(same rounds, same seeds, same episode order, same protection variants).
Protection is applied through a per-edge gate on the *same* update code each
strategy uses in `learning_strategies`, so the rules differ only in their own
semantics. Only original edges are touched; checkpoint/resume is bit-identical
for all three rules.

Variants are `period[:min_weight]`. `min_weight` is the threshold variant: only
edges visited during A **and** with a 4-bit weight >= `min_weight` after A are
protected. Period 1 is always run first as the unprotected control.

Scenarios: `conflict` (B reuses A's negative source as its positive one) and
`disjoint` (B uses two other sources of the same target; no label conflict, so
protection must be free).

CSV columns (`schema=b888-4-consolidation-matrix-v1`): scenario, strategy,
seed, protect_period, min_weight, protected_edges, a_after_a, a_after_b,
b_after_b, presentations, a/b visited/changed, b_blocked, weight_bytes,
overhead_bytes, topology_unchanged, elapsed_us (the only non-deterministic
column).

### Measured on the synthetic comparison graph (seeds 1; 8+8 rounds)

* All three strategies behave identically on A->B retention: control keeps A
  1/2 after B (B 2/2); freezing (period 1000) keeps A 2/2 but B falls to 1/2;
  periods 2/4/8 do **not** retain A (1/2), as in the supervised-only result.
* Threshold variant (negative): `1000:1` protects 1 edge, `1000:15` protects
  none, and both forget A like the control. The edge that must flip for B
  (source b -> target) still holds weight 0 after A, so a weight threshold
  never protects it, and protecting it would block B. The conflict is a single
  edge, so no graded rule can both keep A and learn B here.
* Disjoint scenario: every strategy and protection setting keeps A 2/2 and
  learns B 2/2, so protection costs nothing without conflict.
* Strategies differ only in bookkeeping (reward-modulated skips silent-target
  edges: 2 protected edges instead of 3). No strategy advantage is claimed.

## Scale experiment on a GENERATED graph (`bench-consolidation-scale`)

**Real-graph run: BLOCKED.** No BANC V888 graph (`.v888b`) exists in this
repository or on the box; `docs/SOURCES.md` states that no raw data is shipped
and acquisition (Harvard Dataverse DOI 10.7910/DVN/7WTH1N, BANC/FlyWire) needs a
user-side download + `v888 build`. Nothing was downloaded. Once a `.v888b`
exists the existing `bench-consolidation-matrix --graph` path accepts it. The
numbers below are therefore **synthetic** (generated graph, artificial labels);
BANC (Bates et al. 2026; BANC/FlyWire consortium) is credited for the real data
only and none of it is used here.

Protocol, fixed in `src/consolidation_scale.rs` before looking at results:
64 sources, 24 targets, fan-in 6 (≈144 edges, contact counts 1..3, many sources
projecting to several targets so episodes touch overlapping edges); task A per
target `in[0]` fires / `in[1]` silent; task B conflicts (`in[1]` fires /
`in[2]` silent) on 0 %, 50 % or 100 % of targets and otherwise uses `in[2]`/`in[3]`;
8+8 rounds, seeds 1..10 (seed varies graph and episode order), equal budgets,
three rules, unprotected control (period 1) run first for each seed. Success
rule: mean A-after-B gain ≥ 0.10 over the same-seed control while mean B-after-B
drops by < 0.10.

Results (means over 10 seeds; fractions of 2×24 presentations; CSV in
`results/consolidation-scale-v1.csv`):

| conflict | variant | A after B (control→variant) | B after B | rule met |
|---|---|---|---|---|
| 50 % | period 8 | 0.490 → 0.596 (+0.106) | 0.854 (−0.048) | yes (marginal) |
| 50 % | freeze, weight≥1 | 0.490 → 0.610 (+0.121) | 0.902 (±0) | yes |
| 50 % | full freeze | 0.490 → 0.806 | 0.633 (−0.269) | no (B lost) |
| 100 % | period 8 | 0.287 → 0.417 (+0.129) | 0.829 (−0.046) | yes |
| 100 % | freeze, weight≥1 | 0.287 → 0.431 (+0.144) | 0.875 (±0) | yes |
| 100 % | full freeze | 0.287 → 0.806 | 0.500 (−0.375) | no (B lost) |

Paired per-seed gains for supervised period-8 / threshold-1 are positive in
10/10 seeds (sd 0.04–0.07; min +0.02–0.04).

Negatives and caveats:
* Gains are modest (≈ +0.11…+0.14): A after B stays far below A after A (0.806);
  only full freezing restores it, at the cost of B (as in the single-edge case).
* Period 2 and the stronger threshold (weight≥2) do **not** meet the rule.
* Reward-modulated learns A perfectly (1.0) and no gated variant helps except
  full freeze (which loses B): the rule never meets the success criterion.
* Associative-coactivation is numerically identical to supervised here: with
  single-source episodes the two rules update the same edges. No strategy
  advantage is claimed.
* With 0 % conflict protection yields no rule-meeting gain (+0.03…+0.09 on A,
  nothing useful), and weight-0 freezes cost B.
* Synthetic, small (≈144 edges), artificial labels, 10 seeds, no confidence
  interval beyond the per-seed spread above. Not evidence about BANC.
