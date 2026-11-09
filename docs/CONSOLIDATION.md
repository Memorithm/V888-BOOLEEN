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
