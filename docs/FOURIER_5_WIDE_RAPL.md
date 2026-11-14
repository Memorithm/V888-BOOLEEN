# B888-4 — Contrat d'observation énergétique sur hôte Dell

Les capteurs matériels existent sur les deux sockets du Dell T430, d'après la qualification RemoteOps [#43503766081](https://github.com/Memorithm/RemoteOps/actions/runs/43503766081) du 13 novembre 2026 :
- package-0 et package-1 : énergie des paquets processeurs ;
- dram (sur les deux sockets) : énergie des sous-domaines mémoire.

La lecture des compteurs est accessible au processus de contrôle sur l'hôte avec les privilèges appropriés, **pas** au modèle Rust isolé dans Docker. La lecture des compteurs DRAM mesure une **estimation d'énergie par l'interface RAPL**, pas le nombre d'accès mémoire : le binaire continue donc de publier séparément les lectures logiques (`LARGE_MEMORY_ACCESS_ESTIMATE`), qui ne doivent pas être appelées « transactions DRAM ».

## Marqueurs flushés (opt-in)

La variable `V888_HOST_RAPL_OBSERVER=1` provoque quatre fenêtres :
`FULL_A`, `CACHED_A`, `CACHED_B`, `FULL_B`.
Chacune émet `V888_RAPL_MARKER=<PHASE>_BEGIN` et
`V888_RAPL_MARKER=<PHASE>_END` dans la sortie standard, immédiatement après
`flush`. Le plan de contrôle RemoteOps, **et non le code étudiant**, lit
`/sys/class/powercap/**/energy_uj` à l'arrivée de chaque marqueur.

Chaque fenêtre FULL répète 64 interventions à partir du même état entraîné.
Chaque fenêtre CACHED répète 8 192 fois les mêmes 64 interventions (= 524 288 propositions).
Les deux moteurs sont comparés **sur le même modèle**, et toutes les sorties
sont confrontées aux résultats de référence. Les propositions ne sont pas
commises dans ce microbenchmark. Ordre ABBA pour réduire la dérive temporelle,
non pour garantir une causalité énergétique.

Le protocole de calcul de l'énergie par intervention est :
`(E_FIN - E_DEBUT modulo MAX_ENERGY_RANGE) / nombre_interventions`.
Ne jamais additionner un paquet CPU et son sous-domaine DRAM.
La consommation RAPL reflète également les autres tâches du processeur,
les sollicitations de l'observateur et les consommations au repos. Présenter
la puissance moyenne et la durée de chaque phase, préciser toute charge
concurrente ; publier l'impossibilité de mesurer si les capteurs sont
inaccessibles. Un calcul ABBA ne suffit pas à lui seul à isoler l'énergie
marginale propre au code. Évaluer séparément l'énergie d'apprentissage.

La qualification de branche conserve les anciennes vérifications sans activer
les marqueurs ; **les mesures physiques** sont réalisées après fusion, sur le
SHA exact, par une action trusted-main RemoteOps. Pas de modification des
réglages RAPL, CPU, services, pare-feu ou charge des autres projets.
