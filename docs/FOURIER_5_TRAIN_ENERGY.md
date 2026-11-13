# B888-4 — Mesures physiques de l'énergie durant l'entraînement complet

Le binaire `fourier5-reward-full-vs-cache-bench` fournit un mode opt-in
`V888_TRAIN_POWER_OBSERVER=1`, réservé au workflow hôte de
`Memorithm/RemoteOps`. L'entraînement est **output-only**, sur les mêmes
256 propositions ordonnées que la comparaison TRAIN de PR #98.

## Mesures isolées dans le temps

Pour chacun des trois enseignants et jeux TRAIN/VALID/TEST synthétiques
(`0x9a770001`, `0x9a770002`, `0x9a770003`), deux phases
sont exécutées dans le même conteneur hors réseau :

- FULL : construction du modèle et recalcul intégral pour chaque proposition ;
  reconstruction complète après chaque modification acceptée.
- CACHED : construction du modèle et évaluation/acceptation incrémentale.

Les entrées et labels enseignants sont générés **avant** la fenêtre RAPL.
La fenêtre **inclut** la création du modèle élève, toutes les propositions,
l'entraînement et toutes les acceptations. VALID/TEST sont évalués **après**
la fenêtre ; les scores sont publiés et comparés dans les deux bras.

L'ordre est inversé d'une graine à l'autre. Les propositions acceptées, les
sorties à chacune des 256 étapes ainsi que les scores VALID/TEST doivent
coïncider exactement. En cas de divergence, l'exécution échoue.

Les marqueurs `V888_TRAIN_MARKER=S<graine>_<FULL|CACHED>_<BEGIN|END>`
sont horodatés par le **processus hôte de confiance**, jamais par le
conteneur non privilégié. Les domaines RAPL `package-0/1` et `dram-0/1`
ainsi que les compteurs CAS lecture/écriture `uncore_imc` disponibles
sont lus séparément en lecture seule. Les compteurs peuvent inclure
d'autres processus. Une variation mesurée n'est donc pas nécessairement
attribuable au seul moteur.

Le temps total comprend également la gestion des propositions et
l'initialisation ; il exclut la génération des données et les scores
held-out, par symétrie. Les capteurs IMC comptent les événements
sur l'hôte entier et ne permettent pas à eux seuls une attribution par
processus ni une équivalence universelle 1 événement CAS = 1 octet.

## Conditions de preuve

Publication des données brutes par graine et phase, sans choisir les seules
mesures favorables. Ne pas extrapoler ces résultats à l'entraînement de
synapses réelles, au connectome biologique BANC ou aux performances
d'apprentissage d'un modèle de langage. Le protocole porte sur
256 propositions au total par méthode et par graine, dont seules certaines
sont acceptées. Il reste à répéter avec davantage d'exemples et
d'autres topologies.

Fusion uniquement après Rust 1.89 rustfmt, Clippy strict, tests et
qualification RemoteOps SHA exact, puis CI native verte du commit fusionné.
