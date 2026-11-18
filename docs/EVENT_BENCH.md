# Mesures appariées du moteur asynchrone booléen

Ce banc de recherche n'utilise **aucune trace biologique réelle**. Il
compare trois exécuteurs Rust sur le **même graphe, les mêmes stimuli, le
même seuil et la même séquence causale**.

- `serial` : `EventRuntime::run` ;
- `frontier` : `EventRuntime::run_parallel_frontiers` ;
- `disjoint` : `EventRuntime::run_disjoint_parallel_commits`.

Quatre topologies contrôles sont disponibles :

- `disjoint` : N sources indépendantes alimentent N cibles distinctes ;
- `convergent` : N sources alimentent seulement quatre cibles ; des
  conflits limitent volontairement la parallélisation.
- `wide-disjoint` : N sources ciblent N destinations différentes, mais
  chacune possède **64 lignes CSR parallèles vers la même destination**.
  Cette topologie augmente les contacts traversés, pas le nombre de
  neurones ou de destinations distinctes.
- `heavy-disjoint` : même schéma mais **512 lignes CSR par source**,
  afin de vérifier la voie parallèle réelle après admission du seuil
  conservateur de 65 536 lignes CSR.

Les N sources sont activées à t=0 et désactivées à t=3. Tous les
`RECHECK` programmés par les transitions sont exécutés avant la
fin de la mesure. Les trois modes sont comparés **bit à bit** avec le
résultat du témoin séquentiel (états, compteurs `u128`, file d'attente,
horodatage final et huit compteurs `EventWork`). Chaque sortie impose
`exact_parity=true`, sinon le programme échoue.

## Exécution reproductible sur une machine de développement

Construire une fois le binaire optimisé :

```bash
cargo build --offline --release --bin v888-event-bench
```

Exécuter chacun des six cas dans un **processus distinct**, afin de
remettre à zéro la mémoire maximale (`VmHWM`) entre deux modes :

```bash
for shape in disjoint convergent wide-disjoint heavy-disjoint; do
  for mode in serial frontier disjoint; do
    ./target/release/v888-event-bench \
      --shape="$shape" --mode="$mode" \
      --sources=256 --repetitions=8 --workers=4
  done
done
```

Les paramètres sont strictement bornés : 8 à 4096 sources, 1 à 64
répétitions et 1 à 4 workers. Par défaut : 64, 3 et 4.
Le warmup et la création du graphe ne font **pas** partie des
durées `min_ns`, `median_ns`, `max_ns` et `total_ns` :
le chronométrage entoure strictement l'appel d'exécution événementiel.
Toutes les répétitions sont revérifiées sur l'oracle.

La sortie machine `V888_EVENT_BENCH` comprend :

- nombre exact d'événements, transitions, contacts explorés ;
- durée de chaque mode agrégée par min/médiane/max/total (nanosecondes) ;
- nombre cumulé de threads auteurs pour `disjoint` ;
- mémoire résidente courante `VmRSS` et maximum **du processus entier**
  `VmHWM`, en KiB, si `/proc/self/status` est disponible ;
- ticks CPU du processus (`/proc/self/stat`), sans conversion implicite
  en secondes ; ils incluent le travail d'oracle et d'administration
  entre les chronométrages et sont **différents** de la durée pure.

## Contrôle des facteurs de confusion

Le banc est volontairement borné et n'écrit aucun paramètre système.
Les mesures de temps restent sensibles au placement des threads,
aux caches, à la fréquence, à la contention du Dell et à l'environnement
Docker. Une seule exécution ne suffit pas à établir une amélioration.

Le RSS maximal dépend aussi des allocations de préparation et du
runtime Rust, et **ne mesure pas les octets de lecture RAM**.
Le banc ne lit pas les capteurs RAPL. Des mesures physiques appariées
nécessitent le plan de contrôle [Memorithm/RemoteOps](https://github.com/Memorithm/RemoteOps),
des capteurs Intel RAPL hôte en lecture seule et, si disponibles,
les compteurs IMC. Ces compteurs couvrent **tout le serveur partagé**,
pas uniquement le conteneur du programme. Les joules hôte ne peuvent
donc pas être attribués causalement au moteur sans contrôle de la
charge et du bruit.

Ne pas annoncer de gain en débit, de diminution de consommation ou de
transposition à V888 réel avant une campagne de résultats traçables,
comparés à une référence indépendante et répétée.
