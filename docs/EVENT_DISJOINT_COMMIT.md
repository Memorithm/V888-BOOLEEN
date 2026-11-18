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

## Garde-fous historiques et exploration adaptative

Le moteur a d'abord appliqué deux seuils fixes : au moins **65 536 unités
de préparation** et **1 024 compteurs cibles distincts**, sur un préfixe
comprenant au moins deux sources réellement actives. Ces valeurs avaient
été introduites après les mesures pilotes du 18 novembre 2026 :
un graphe `wide-disjoint` à 16 384 lignes CSR était alors environ neuf
fois plus lent sur le chemin de commits parallèles.

Ces valeurs restent un **point de départ conservateur en l'absence de
mesures**, pas une règle absolue. Le mode actuel étudie aussi les groupes
de plus petite taille pour apprendre le coût réel du chemin séquentiel.
Il peut ensuite sonder le chemin parallèle, sous des minima de sécurité,
selon la topologie effective et les observations de coût par classe.

Une grande quantité de contacts CSR ne signifie pas que le groupe a beaucoup
de compteurs d'entrée différents à modifier. Le préfixe causal utilise
déjà la cardinalité exacte des destinations non nulles du `BTreeSet`,
sans balayage supplémentaire pour ce chiffre. Les contacts sans poids ne
produisent pas de mises à jour ; les contributions mises en cache sont
comptées comme paires pondérées lors de la préparation.

Un groupe rejeté est commis séquentiellement, dans l'ordre d'origine.
Les bits, compteurs entrants, réveils, dates et huit compteurs logiques
conservent donc leur signification. L'adaptation ne prétend pas supprimer
les coûts du préfixe causal ou garantir une accélération.

## Admission mesurée par topologie (profil expérimental)

Le point d'entrée existant `run_disjoint_parallel_commits` utilise désormais
un profil local créé à chaque appel. Pour réutiliser les mesures entre plusieurs
tranches ou répétitions indépendantes, l'appelant peut conserver un
`DisjointAdmissionProfile` et appeler
`run_disjoint_parallel_commits_with_profile(max_events, threads, &mut profile)`.
Le profil appartient à l'appelant ; il n'existe ni état global mutable, ni
fichier de calibration, ni modification du graphe.

Chaque **préfixe causal DRIVE sans cible commune** est décrit par des données
du graphe réellement parcouru :

- nombre de sources dont le bit change et qui possèdent des contacts ;
- nombre total de lignes CSR de ces sources ;
- unités de préparation effectives : lignes CSR sans cache, couples
  (destination, poids) lorsque le cache est présent ;
- nombre exact de destinations non nulles distinctes, déjà déterminé par
  le contrôle d'absence de conflits.

Le profil utilise douze classes : trois bandes de cardinalité des cibles
(`<128`, `128..1023`, `>=1024`), deux classes de compression par cache
et deux classes de répétition des cibles. Les observations proviennent des
durées murales réelles `Instant` de groupes séquentiels et parallèles ; leur
moyenne exponentielle amortie est normalisée en nanosecondes pour 1 024 unités
de travail estimées :

\[
U=C_{\mathrm{eff}}+T+8S,\qquad
c=\frac{1024\,t_{\mathrm{observé}}}{\max(1,U)}.
\]

Ici `T` est le nombre de cibles distinctes et `S` celui des sources
actives ; le coefficient 8 est **un poids heuristique**, non une constante
physique mesurée. Pour une classe avec au moins deux observations de chaque
chemin, l'admission privilégie le parallèle seulement si
`100 × coût_parallèle < 85 × coût_séquentiel`. Un sondage du chemin temporairement défavorisé tous les 32 candidats
réactualise alternativement le coût séquentiel ou parallèle lorsque les conditions changent.
Les mesures de groupe n'incluent pas le calcul du préfixe, commun aux deux
chemins ; elles ne justifient pas à elles seules de déclarer un gain global.

Avant calibration, les groupes suffisamment larges conservent la condition
historique (65 536 unités et 1 024 cibles). Le profil collecte également
des groupes séquentiels comparables. Après deux observations séquentielles,
un sondage parallèle limité peut être tenté sous les anciens seuils, mais
**jamais** pour moins de 256 unités de préparation, 32 destinations distinctes
ou deux sources actives. Une fenêtre d'événements trop petite est toujours
traitée séquentiellement.

La variabilité des threads, l'ordonnancement hôte, les tailles de groupe et
les coûts du cache peuvent perturber les moyennes. Le profil n'est ni une
calibration énergétique ni une garantie de débit. Les seuils d'exploration,
le coefficient de normalisation et la marge de 15 % restent des hypothèses
révisables après mesures reproductibles contrôlées par RemoteOps.

**Invariant** : le profil ne modifie que le choix du chemin d'exécution.
Le contrôle des conflits, les prévalidations, les arithmétiques et les
publications restent inchangés. Les sondages parallèles passent par le même
protocole exact ; tous les RECHECK demeurent séquentiels.

## Bornes et observabilité

- 32 threads au plus, 32 événements par thread et par groupe.
- `requested_threads=0` : rejet sans mutation.
- `requested_threads=1` : délégation directe au moteur séquentiel.
- `DisjointCommitWork.work` doit être exactement identique à
  `EventRuntime::run`.
- `DisjointAdmissionProfile::observation_counts()` rapporte les observations
  en mémoire, par chemin ; les comptes sont bornés par classe.
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
