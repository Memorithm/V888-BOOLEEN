# Admission topologique par coûts mesurés — moteur asynchrone

**Statut : expérimental, option explicite.** Ce travail n'altère ni les
stimuli, ni les règles neuronales, ni les compteurs booléens, ni les données
anatomiques. La référence demeure `EventRuntime::run`.

## Deux modes d'admission

1. `run_disjoint_parallel_commits(max_events, workers)` conserve le
   profil adaptatif introduit dans la PR #157 : calibration locale de
   groupes, 12 catégories de topologies et conservation des bornes
   prudentes au démarrage. `run_disjoint_parallel_commits_with_profile`
   permet de conserver les observations d'un appel à l'autre.
2. `run_disjoint_parallel_commits_with_costs(max_events, workers, profile)`
   accepte un **profil de mesures externes** et estime si les deux passes
   de threads peuvent être amorties pour le groupe réellement rencontré.
   Si son enveloppe topologique n'est pas respectée, le groupe est traité
   séquentiellement. Les seuils du calibrage en ligne ne bloquent pas les profils externes
explicitement admis ; les calculs booléens restent identiques.

`MeasuredDisjointCosts` comporte le nombre de workers de la mesure, une
enveloppe minimale/maximale de producteurs, contacts effectifs et cibles
distinctes, cinq coefficients positifs en nanosecondes : surcoût fixe
total des deux passes, coûts séquentiels par contact/cible, coûts
parallèles par contact/cible. La valeur de `contacts` compte les paires
agrégées si le cache de contacts est actif, sinon les lignes CSR.
Le nombre de cibles est calculé à partir des écritures non nulles et
dédupliquées ; le nombre de producteurs exclut les drives sans effet.

Le modèle est délibérément borné : aucun coefficient ne se transfère
implicitement entre machines, nombres de workers ou configurations de
cache différents. Le caller est responsable de mesurer ses coefficients
dans des essais appariés, avec affinité CPU, cache, topologie, coût de
lancement des threads et variance documentés. Des coefficients
synthétiques dans les tests ne sont **pas** une qualification physique.

## Fonction de décision (heuristique, pas prédiction garantie)

Pour `C` unités de préparation, `T` compteurs distincts, `P`
producteurs et `W` workers admis :

```text
serial_ns   = C * serial_contact_ns + T * serial_target_ns
parallel_ns = two_pass_overhead_ns
            + ceil(C / min(W,P)) * parallel_contact_ns
            + ceil(T / min(W,T)) * parallel_target_ns
admettre si et seulement si parallel_ns < serial_ns
```

Tous les calculs sont contrôlés en `u128`. Un profil invalide renvoie
`EventError::InvalidCostProfile` **avant mutation**. Hors enveloppe ou
en cas de coût prédit défavorable, le moteur reprend l'exécution série
canonique. Les passes parallèles existantes conservent leurs préflights
atomiques, leurs frontières disjointes, et le repli en cas d'erreur.

Le résultat `DisjointCommitWork` expose `cost_profile_accepted` et
`cost_profile_rejected` : ce sont des **décisions de préadmission** par
préfixe candidat, pas des exécutions parallèles réussies. Le compteur
`disjoint_groups` reste la preuve qu'un groupe a effectivement atteint
le commit parallèle ; les mesures de temps et d'énergie restent externes.

Cette heuristique externe est indépendante de la calibration en ligne de
#157. Aucune observation n'est ajoutée au profil adaptatif par cette API.

Cette heuristique ignore les effets NUMA, caches partagés, bande passante
mémoire, création réelle de threads et variance temporelle non modélisés.
Les mesures d'énergie et les gains de performance restent à démontrer.
Un profil favorable peut être moins rapide en production ; sa validation
requiert des mesures appariées au SHA exact sur RemoteOps.

## Qualification et progression

```bash
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
cargo test --offline --all-targets
cargo run --offline --release --bin v888-event-bench -- --mode=serial --shape=heavy-disjoint --sources=256 --repetitions=8 --workers=4
cargo run --offline --release --bin v888-event-bench -- --mode=disjoint --shape=heavy-disjoint --sources=256 --repetitions=8 --workers=4
```

Ce benchmark existant **ne fournit pas automatiquement** les cinq
coefficients. Une qualification ultérieure doit acquérir les mesures
nécessaires pour construire un profil par configuration, vérifier les
résultats et les huit compteurs logiques contre `EventRuntime::run`
et documenter le coût total de préparation et la mémoire.
