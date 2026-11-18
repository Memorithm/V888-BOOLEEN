# Commit parallèle sans conflit — modèle événementiel V888

Statut : **implémentation expérimentale opt-in**. Le moteur
`EventRuntime::run` reste la référence indépendante.

## Différence technique avec la PR #144

`run_parallel_frontiers` utilisait des threads pour préparer les
contributions, puis commitait tous les compteurs en série.

`run_disjoint_parallel_commits(max_events, requested_threads)` applique
maintenant réellement les mises à jour de plusieurs compteurs d'entrée
`u128` **en parallèle**, en utilisant `std::thread::scope` et des
sous-tranches `&mut [u128]` de l'état. Aucun pointeur non sûr ni
variable globale mutable ne sont nécessaires. Les neurones et la file
BTreeMap ne sont cependant **pas** modifiés en parallèle.

Le moteur sélectionne un préfixe d'événements `DRIVE` d'une même date,
à condition que leurs contacts sortants non nuls visent des cibles
distinctes. Un conflit coupe le préfixe, sans reporter ni réordonner
l'événement. Les `RECHECK` sont traités séquentiellement car ils peuvent
se lire et s'écrire mutuellement à la même date logique.

Les contributions de chaque source sont préparées depuis un instantané
immuable. Chaque événement est ensuite prévalidé, dans son ordre
original, pour les dépassements `u128`, le temps logique et la
capacité de file. En cas d'erreur possible, le moteur reprend le chemin
séquentiel, un événement à la fois : les événements valides précédents
restent commis et le premier événement fautif reste intact.

Lorsqu'un groupe est validé, les destinations distinctes sont triées,
réparties entre des segments non chevauchants du vecteur d'entrée, puis
écrites par les threads. Les bits des sources, réveils futurs,
déduplications et compteurs de travail sont publiés par l'unique
propriétaire, dans l'ordre causal d'origine.

## Admission adaptative et repli sans threads

À la suite des mesures pilotes sur Dell, le mode de commit parallèle
passe immédiatement par le moteur séquentiel lorsque le graphe entier
comprend moins de **65 536 lignes CSR**. Sur un grand graphe, les groupes
de drives sans conflit sont eux-mêmes traités séquentiellement tant que
leurs transitions cumulées n'examineraient pas 65 536 lignes de contacts,
ou si moins de deux sources émettent des contributions réellement actives.
Cela évite une double création de threads pour des groupes de petite
taille. Les préfixes rejetés sont consommés en ordre, sans modifier la
sémantique et sans relancer inutilement leur préparation.

Ce seuil de recherche de 65 536 contacts est volontairement explicite,
**non calibré** en joules et non garanti optimal. Lors de la première
campagne du 18 novembre 2026, le graphe `wide-disjoint` à 16 384 lignes
CSR restait environ neuf fois plus lent avec les commits parallèles. Les groupes volumineux continuent à
utiliser de véritables écritures concurrentes sur des segments disjoints.

## Admission selon les compteurs réellement distincts

Une grande quantité de contacts CSR ne signifie pas que le groupe a beaucoup
de compteurs d'entrée différents à modifier. Sur les graphes à contacts très
répétés, créer des threads pour seulement huit destinations pouvait
multiplier le temps d'exécution malgré 65 536 contacts logiques.

Le préfixe causal suit déjà les cibles non nulles dans un `BTreeSet` pour
détecter les conflits. Le moteur réutilise maintenant la cardinalité exacte
de cet ensemble, **sans balayage supplémentaire du graphe**, pour refuser
l'exécution concurrente lorsque moins de 1 024 compteurs distincts seraient
écrits. Ce seuil conservateur s'ajoute aux conditions précédentes
(au moins deux producteurs et 65 536 unités de préparation effective).
Les contacts non mis en cache restent comptés comme lectures CSR ; les contacts
mis en cache comme paires de destinations agrégées.

Pour les sources sans cache, les runs adjacents visant la même cible non nulle
évitent aussi les recherches répétées dans l'ensemble des cibles. Les contacts
de poids nul et les destinations non triées restent sémantiquement identiques.

Le repli consomme les événements dans l'ordre de référence. Les états,
réveils, dates et huit compteurs logiques restent inchangés. Un test avec
1 024 destinations disjointes démontre que les écritures concurrentes sont
toujours réellement exercées lorsqu'elles peuvent être admises.

Ce seuil n'est pas une estimation d'énergie ni une garantie d'accélération :
des mesures appariées de temps CPU, mémoire et coût de synchronisation
restent nécessaires pour en apprécier la pertinence selon la topologie.

## Bornes et observabilité

- 32 threads au plus, 32 événements par thread et par groupe.
- `requested_threads=0` : rejet sans mutation.
- `requested_threads=1` : délégation directe au moteur séquentiel.
- `DisjointCommitWork.work` doit être exactement identique à
  `EventRuntime::run`.
- `disjoint_groups`, `grouped_events`, `target_counters_written`,
  `target_writer_threads`, `serial_events` et `max_group_width`
  décrivent le travail de ce chemin, non des mesures physiques.

Les contributions sont préparées à l'avance et des structures
temporaires sont allouées. La taille mémoire peut donc augmenter, même
si les écritures ciblées deviennent parallèles. L'existence de plusieurs
threads n'implique ni une accélération de l'exécution ni une réduction
de l'énergie. Les performances doivent être mesurées sur une charge
commune et des cœurs fixés via RemoteOps, avec coûts de création des
threads et synchronisation inclus.

## Qualification

```bash
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
cargo test --offline event_runtime
cargo run --offline --bin v888-event-disjoint-demo
```

Les tests couvrent les écritures vers plusieurs cibles, fronts
montants/descendants, collisions de destinations, capacités de file,
erreurs de date logique, groupes bornés, parcours cyclique et identité
exacte des résultats partiels. La qualification Rust complète et ses
benchmarks obligatoires passent uniquement par Memorithm/RemoteOps.
