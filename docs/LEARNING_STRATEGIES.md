# B888-4 — stratégies d'apprentissage exécutables (hypothèses)

Ce module ajoute trois règles réellement exécutables sur les coefficients plastiques 4 bits déjà présents. Il ne prétend pas reproduire une plasticité biologique, une mémoire d'insecte ou la physiologie BANC. L'anatomie du graphe n'est jamais créée, inversée ou réécrite : seules les arêtes originales reçoivent un coefficient.

## Règles

1. **`supervised`** — témoin existant (`PlasticCircuit::train`) : un enseignant binaire indique quelles cibles doivent s'activer. Toutes les arêtes sortantes des sources actives sont renforcées ou diminuées.
2. **`associative-coactivation`** — règle locale non supervisée : pour chaque arête originale dont la source est active, le coefficient augmente si la destination est aussi active dans l'état observé, et diminue sinon. Aucun enseignant n'est consulté.
3. **`reward-modulated`** — éligibilité par co-activation, puis modulation par un signal de récompense externe explicite (`None`, `Positive`, `Negative`). `None` ne change aucun poids. Une récompense positive renforce uniquement les paires co-actives ; une récompense négative les diminue. Les arêtes dont la destination est inactive ne sont pas modifiées.

Les trois règles sont déterministes pour une même graine (l'ordre des épisodes est mélangé par un LCG fixé). La sauvegarde `V888PLS1` existante reste le format de reprise ; un checkpoint rechargé doit produire les mêmes coefficients, bit à bit, qu'une exécution continue.

## Comparaison à budget égal

`compare_strategies` et la commande `bench-strategies` entraînent les trois règles avec le même nombre d'épisodes :

- rétention des motifs vus, avant et après apprentissage ;
- généralisation sur deux sources jamais activées, déjà connectées à la même cible ;
- oubli par inversion des étiquettes des sources d'entraînement, puis mesure de l'ancienne et de la nouvelle tâche.

Les étiquettes sont artificielles. Un échec de transfert vers les sources tenues à l'écart est un résultat attendu de ces règles locales, pas une anomalie à dissimuler.

```bash
cargo run -- bench-strategies --seed 1 --rounds 8
cargo run -- bench-strategies --graph brain.v888b --seed 1 --rounds 8
```

## Attributions

Les données anatomiques BANC V888 restent attribuées selon [SOURCES.md](SOURCES.md) : Bates, A. S., Phelps, J. S., Kim, M. et collaborateurs (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature, https://doi.org/10.1038/s41586-026-10735-w ; consortium BANC/FlyWire, jeu BANC v888, https://doi.org/10.7910/DVN/7WTH1N ; documentation https://github.com/htem/BANC-project. Ce dépôt n'inclut pas les fichiers sources.

Cette tranche est indépendante de B888-3 (exploration de circuits et télémétrie régionale).
