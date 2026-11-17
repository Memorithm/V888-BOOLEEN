# Moteur événementiel booléen expérimental — V888

Statut : **expérimental, opt-in**, distinct du moteur `BooleanEngine::step` et
de la simulation biologique. Le graphe `Connectome` conserve sa structure
CSR, son sens orienté, ses multiplicités et ses identifiants V888.

## Contrat sémantique

- Chaque neurone possède un état binaire et un compteur d'entrée `u128`.
- Les entrées sont la somme **exacte** des multiplicités des contacts
  sortants des seules sources actuellement actives.
- Un stimulus externe `schedule_drive(t, id, bit)` assigne son état à la
  date logique `t`. Une transition applique son delta à chaque cible
  concernée ; chaque cible est réévaluée à `t + 1`.
- Les événements sont ordonnés par (date logique, stimuli avant
  réévaluations, indice de neurone). Les réveils de même date et cible sont
  fusionnés. Les réévaluations lisent les compteurs au moment de leur
  exécution, pas au moment de leur programmation.
- Un changement de source n'examine que ses contacts sortants, et une
  réévaluation sans changement d'état ne propage rien. Il n'existe ni
  horloge synchrone globale à chaque neurone ni scan dense du connectome.
- La file est bornée par `max_pending` et le travail par `run(max_events)`.
  Dépasser le budget de file, dépasser le temps logique `u64` ou produire
  un compteur hors `u128` bloque **l'événement concerné avant mutation**.
  Les événements déjà validés dans une exécution restent commis.
- Les demandes contradictoires sur le même neurone à la même date sont
  rejetées ; les doublons identiques sont idempotents. Les stimuli ne
  peuvent pas être injectés dans le passé ni à la dernière date traitée.

Les stimuli ont une priorité déterministe avant les réévaluations pour une
date commune. Ils ne constituent pas des états forcés permanents : une
réévaluation ultérieure peut changer à nouveau un neurone stimulé.
Les circuits cycliques peuvent générer une suite illimitée d'événements ;
le budget d'exécution constitue une borne explicite, **pas** une preuve de
convergence.

## Comptabilité logique vers TDI-28

Le résultat `EventWork` renvoie six compteurs indépendants :

| Compteur | Signification |
|---|---|
| `events_consumed` | Événements retirés, y compris sans effet |
| `state_transitions` | Changements d'état booléen |
| `contacts_examined` | Lignes CSR lues lors de transitions réelles |
| `threshold_checks` | Comparaisons à un seuil sur événements de réveil |
| `wakeups_inserted` | Nouveaux événements de réévaluation |
| `wakeups_coalesced` | Événements de réévaluation dédoublonnés |

Ces variables peuvent alimenter le volet A de TDI-28 (vecteur de coûts
logiques) et comparer des politiques B (calculer, réutiliser ou différer)
**sans modifier le moteur de référence**. Les compteurs ne comptent pas
directement les accès mémoire en octets, les messages matériels, les joules
ni une consommation électrique. Il faut les calibrer et effectuer des
mesures physiques appariées avant de tirer des conclusions énergétiques.

## Expérience reproductible

```bash
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
cargo test --offline event_runtime
cargo run --offline --bin v888-event-demo
```

Les tests couvrent la chaîne d'activation et de désactivation, la
convergence de plusieurs stimuli, le seuil, l'ordre indépendant de
l'insertion, la file saturée atomiquement, les événements sans effet,
les poids nuls, les entrées invalides, un oracle CSR indépendant après chaque transition d'un cycle, deux contacts de multiplicité maximale et le dépassement atomique du temps logique.

La qualification des PR s'effectue sur le SHA exact via
[Memorithm/RemoteOps](https://github.com/Memorithm/RemoteOps)
en environnement Rust isolé. Tant que cette qualification et les checks
requis ne sont pas verts, **ne pas fusionner**.

## Limites scientifiques

L'événement logique `t + 1` est une convention de simulation, et non un
délai synaptique mesuré. Le moteur ne modélise ni neurotransmetteurs,
conduction, plasticité, stochastique, ni consommation physique. Le
résultat ne prouve ni efficacité énergétique, ni performance supérieure,
ni fidélité aux mécanismes du cerveau de drosophile.
