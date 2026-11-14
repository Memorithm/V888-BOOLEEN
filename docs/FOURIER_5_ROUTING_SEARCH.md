# B888-4 — Sélection bornée du routage des cercles Fourier

**Statut : recherche expérimentale sur circuits booléens synthétiques.**
Cette tranche ne change ni les gains, ni les masques, ni la plasticité,
ni la topologie anatomique BANC. Le réseau récurrent précédent reste inchangé.

## Hypothèse testée

Une même architecture hiérarchique à cinq entrées par neurone ne produit
pas les mêmes fonctions lorsque les **sources des entrées** changent.
Après la démonstration de l'équivalence exacte entre Fourier scalaire,
table de vérité et exécution bit-sliced (PR #85), cette tranche apprend
*uniquement la permutation de feuilles* parmi une famille bornée
de propositions et vérifie sa qualité sur des exemples non consultés
lors du choix du routage.

La sélection des connexions n'est **pas** une plasticité synaptique :
un ensemble figé de modèles `FiveCircle` et leurs tables 32 bits est
conservé à l'identique. La recherche produit une autre bijection des
positions d'entrée et ne peut ni créer ni supprimer de neurones/contacts.

## Nouvelle API Rust

- `FiveCircleHierarchy::rerouted(routing)` valide une bijection complète,
  clone le circuit existant et **réutilise** les tables de vérité déjà
  compilées plutôt que de recalculer 32 états × 156 neurones par
  candidat. Les gains, modes de Fourier, biais et couches sont inchangés.
- `fourier5_routing_search::candidate_swaps(width, seed, K)` propose
  `K <= 64` échanges de deux feuilles sur des sous-arbres
  de niveau supérieur distincts, sans recevoir les données ou
  les labels. Les paires sont distinctes et reproductibles.
- `select_route(base, TRAIN, proposal_seed, K)` évalue
  exactement `K+1` topologies (dont la base sans modification).
  Sur TRAIN uniquement, un coût est mesuré par la somme des
  prédictions exactes. Une égalité conserve la proposition de plus
  petit index. L'évaluation utilise le même moteur booléen
  bit-sliced 64 voies que la PR #85.
- Le résultat contient le modèle routé, la paire éventuellement
  échangée, le nombre de modèles examinés, les bonnes prédictions
  TRAIN et le nombre exact d'inférences d'entraînement effectuées.
  Les entrées incorrectes, doublons de TRAIN, dépassements de budget
  et lots non multiples de 64 sont rejetés.

**Limites fixes :** largeur 25, 125 ou 625 bits,
au plus 1 024 exemples d'entraînement et 64 propositions d'échanges.
Aucune mise à jour de paramètres, aucun secret, aucune donnée biologique.

## Préenregistrement du banc synthétique

Pour chaque profondeur, le banc crée **à l'avance** les 64 échanges
candidats, indépendamment des exemples et des labels. Le générateur
d'étiquettes utilise l'une de ces propositions comme « enseignant »
synthétique caché du sélecteur ; la bonne solution est donc
**garantie d'appartenir à l'espace de recherche**. C'est un contrôle
d'identifiabilité de la connectivité, pas une démonstration de
découverte topologique générale ou de généralisation à de nouvelles
connectivités.

Les trois hiérarchies figées sont :

| Entrées | Profondeur | Cercles-neurones | Contacts orientés | Modes successifs |
|---:|---:|---:|---:|---|
| 25 | 2 | 6 | 30 | Majorité → Parité |
| 125 | 3 | 31 | 155 | Majorité → Parité → Parité |
| 625 | 4 | 156 | 780 | Majorité → Parité → Majorité → Parité |

Pour chaque profondeur, trois graines de génération
`0xb8880011`, `0xb8880022`, `0xb8880033`
avec indices d'enseignant respectifs 7, 31 et 53.

| Bloc | Nombre d'exemples | Distribution des bits | Usage |
|---|---:|---|---|
| TRAIN | 512 pour 25/125 bits ; 1 024 pour 625 bits | Bernoulli(0,5) | Seul bloc lu par la recherche |
| VALIDATION | 512 | Bernoulli(0,5) | Diagnostic sans ajustement |
| TEST | 1 024 | Bernoulli(0,5) | Évaluation finale |
| Décalage | 512 | Bernoulli(0,25) | Robustesse à densité différente |

Le générateur exige un **ensemble de motifs disjoints**, au sein
de chaque profil et entre TRAIN, VALIDATION, TEST et décalage :
si deux états entiers coïncident, il tire un nouvel état.
Les labels sont produits par un oracle booléen de couches,
indépendant de `FiveCircle::predict` et des tables compilées.

La comparaison porte sur quatre représentations du **même nombre
de neurones et de contacts** : réseau sans échange, un routage
préchoisi sans labels, routage sélectionné sur TRAIN, enseignant
construit artificiellement. Tous les bras appliquent la même
famille figée de règles Fourier ; seul le mapping des sources
change. Les témoins sans sélection n'emploient pas le
même budget *d'entraînement* : le surcoût `(K+1) × |TRAIN|`
inférences est explicitement publié.

## Critères et interprétation

Le protocole impose :
- 0 erreur TRAIN pour le gagnant, puisque l'enseignant appartient
  aux propositions ; c'est un test de bon fonctionnement, **pas**
  un score d'apprentissage hors tâche ouverte.
- Aucune modification des connexions/paramètres internes :
  nombres de neurones et de contacts identiques.
- Conformité de toutes les sorties de l'enseignant à l'oracle
  indépendant sur les trois jeux de contrôle.
- Publication sans filtre des bonnes prédictions sur VALIDATION,
  TEST et décalage, y compris si les écarts sont défavorables.
- Validation des limites, permutations et répétabilité par tests Rust.

Cette expérience ne démontre **ni** une bonne performance sur
n'importe quel routage inconnu, **ni** un apprentissage des
masques Fourier, **ni** une amélioration de l'architecture biologique,
**ni** un avantage énergétique. Les temps rapportés couvrent la
recherche sur les exemples pré-générés ; la génération du jeu
d'entraînement n'est pas incluse dans ce chronométrage.

## Qualification RemoteOps

Exécuter `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`, les six bancs précédents,
`cargo run --offline --bin fourier5-routing-bench` et
`cargo run --offline -- info` dans le conteneur Rust 1.89
RemoteOps verrouillé sur le SHA exact, non privilégié, sans réseau,
avec limite de 4 CPU/6 Gio. Ne fusionner qu'après preuve
d'exécution et vérifier ensuite la CI native du commit fusionné.
