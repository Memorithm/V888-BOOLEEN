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
