# Frontières événementielles parallèles — phase expérimentale

Le moteur `EventRuntime` dispose d'une API opt-in
`run_parallel_frontiers(max_events, requested_workers)`. Elle fournit
une **préparation concurrente** des événements booléens en Rust standard,
sans dépendance NVIDIA, Tokio ou moteur matriciel FP.

## Garanties

- Le moteur de référence `run` demeure inchangé et contrôle l'égalité
  exacte des états binaires, compteurs d'entrée `u128`, file d'événements,
  dates de traitement et huit composantes `EventWork`.
- Le groupe de préparation ne franchit jamais la date logique courante,
  ni la frontière entre stimuli `DRIVE` et réévaluations `RECHECK`.
- Les stimuli de même date et de sources distinctes sont préparés
  simultanément. Les réévaluations ne rejoignent ce groupe que si aucune
  transition antérieure du groupe ne peut écrire leur entrée.
- Les mises à jour des cibles communes, la déduplication des réveils
  et le traitement des erreurs sont **committés séquentiellement** dans
  l'ordre immuable du moteur de référence.
- Une erreur sur un événement conserve l'atomicité de cet événement ;
  les événements antérieurs déjà acceptés restent commis.
- Le nombre de threads de préparation est plafonné à 32 et chaque lot
  contient au plus 32 événements par thread. Ce sont des bornes de
  préparation, pas une garantie de mémoire résidente indépendante de
  la taille des fanouts CSR.
- `requested_workers=1` reproduit exactement `run(max_events)` sans
  créer de threads. `requested_workers=0` est refusé avant mutation.

## Subtilité causale conservée

Une transition produit immédiatement une variation du compteur d'entrée
de ses cibles et programme leur réveil pour `t + 1`. En conséquence,
une réévaluation plus tardive **à la même date t** peut déjà lire cette
variation, lorsqu'un réveil antérieur l'avait aussi programmée.

Un ordonnanceur qui traiterait tous les `RECHECK` de t à partir d'un
seul instantané changerait la simulation. Le nouvel ordonnanceur refuse
donc de grouper un `RECHECK` si une transition antérieure du lot
pourrait écrire son entrée. Les groupes sont des préfixes contigus :
il ne déplace aucun événement vers l'avant.

Les sommes synaptiques locales sont préparées avec des `BTreeMap`.
Les sommes d'entrée finales sont, elles, calculées et contrôlées lors
du commit en série pour préserver l'ordre des additions et soustractions.

## Admission adaptative du travail

Les mesures pilotes du 18 novembre 2026 ont montré que la création de
threads pouvait coûter plus cher que la propagation de quelques contacts.
La préparation parallèle est désormais **refusée** si le graphe total
contient moins de 65 536 lignes CSR, ou si le préfixe causal candidat
comporte moins de quatre événements ou de 65 536 lignes de contacts
susceptibles d'être lues sur des transitions. Le préfixe est alors
consommé séquentiellement, sans relancer inutilement sa planification.
Le seuil de 65 536 contacts est une **hypothèse conservatrice** issue de
la campagne pilote du 18 novembre 2026 : 16 384 lignes CSR étaient
encore défavorables au parallélisme dans le cas `wide-disjoint`.
Ce n'est ni un optimum prouvé ni une calibration énergétique. Les contraintes causales et le budget de
contacts restent inchangés.

## Comptabilité et limites

`ParallelFrontierWork` contient le même `EventWork` que l'exécution
séquentielle, plus quatre compteurs de préparation :

- `parallel_groups` : groupes de taille supérieure à un ;
- `prepared_events` : événements effectivement committés après staging ;
- `threads_spawned` : nombre cumulé de threads créés pour ces groupes ;
- `max_group_width` : nombre maximal d'événements d'un groupe.

Un nombre positif de threads ne prouve **aucune** exécution plus rapide.
La préparation double parfois le parcours des contacts lors de la
détection des dépendances ; les coûts de synchronisation et d'allocation
ne sont pas contenus dans `EventWork`. Il faut mesurer le temps CPU et
la consommation réelle dans des conditions appariées via RemoteOps.
Le nombre de threads créés ne prouve pas davantage que les threads ont
effectivement tourné simultanément sur des cœurs distincts.

Le budget de contacts `run_with_contact_budget` reste une autre
politique d'admission ; il n'est pas combiné implicitement à cette
API, afin de préserver la spécification de ses limites.

## Qualification

```bash
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
cargo test --offline event_runtime
cargo run --offline --bin v888-event-parallel-demo
```

Les tests comparent l'ordonnanceur parallèle au témoin séquentiel pour
les stimuli simultanés, entrées convergentes, cycles, effets immédiats
entre réévaluations du même tick, préfixes d'exécution bornés, saturation
de file, dépassement du temps logique, absence de threads en mode
séquentiel et validation de paramètres.

Cette voie demeure un **prototype de moteur de recherche**, et non
une simulation biologiquement validée ni une optimisation énergétique
démontrée. Elle utilise de vrais threads CPU, mais le commit est
encore sériel : ce design vise l'exactitude, pas une vitesse théorique
garantie à l'échelle de tout le cerveau.
