# B888-4 — Plasticité différée par traces d'éligibilité (expérience opt-in)

Statut : **hypothèse informatique expérimentale**. Aucun modèle du
développement biologique ni simulation physiologique n'est revendiqué.
L'évolution standard du moteur V888 reste inchangée tant que l'appelant
n'active pas explicitement cette API.

## Question falsifiable

Un neurone booléen peut-il associer un événement local à une supervision
reçue **après plusieurs transitions récurrentes**, plutôt que d'apprendre
sur l'état présent au moment de la récompense ?

La distinction avec la précédente étude est cruciale :

- PR #82 : apprentissage individuel avant la propagation, puis poids figés ;
- cette étude : conservation explicite d'un *indice local* et mises à jour
  **pendant une séquence d'épisodes récurrents**, après le délai ;
- aucun mécanisme d'association autonome, ni restitution d'une information
  sans indice de lecture, n'est encore démontré.

## Articulation avec la PR #83 déjà fusionnée

La PR #83 apporte un premier mécanisme de trace **mono-neurone**, avec
contrôle de délai par compteur de ticks et stimulus contrôlé. Le présent
lot conserve cet API et le complète par une expérimentation **de population**
(1 024 et 8 192 neurones), un témoin « état au moment de la récompense »,
une supervision par neurone, un filtrage TRAIN/TEST exact et des tâches A/B.
Ces deux mécanismes sont indépendants et opt-in ; aucun code n'est écrasé.

## Mécanisme implémenté

`src/fourier5_population_eligibility.rs` :
`EligibilityBank::capture(network)` enregistre, pour chaque neurone, les
5 bits de données et les 5 bits contextuels lus à cet instant
(deux octets par neurone) **sans conserver le label**.

`tick(network)` fait avancer le réseau d'un tic en conservant ces
entrées. L'âge maximum des traces est borné à 128 tics.
`reinforce(network, labels)` n'est admis qu'après au moins un tic ;
les labels sont fournis tardivement et mis en correspondance avec
les signaux mémorisés au moment de l'indice, pas avec le nouvel état.
Un exemple supervisé par neurone et par épisode provoque au plus
un passage de la règle existante `train_context_gates`.

Cette règle modifie uniquement les gains contextuels de quatre portes
inhibitrices par neurone, par pas entier saturé. Le cinquième cercle
porte la parité d'ordre cinq. Les masques, phases, rayons permanents,
connectivités et IDs restent gelés. Aucune plasticité Hebb/STDP,
sélection de masque ou rétropropagation non documentée.

Validation des dimensions avant mutation, refus de trace concurrente,
refus d'un label sans trace ou sans délai, limite de durée, suppression
explicite des traces consommées, aucune opération de système.

`FiveCircleLayer` et `RecurrentFiveCircle` exposent des méthodes
opt-in de capture et de feedback ainsi qu'un remplacement contrôlé de
l'état synthétique. Les transitions standards restent identiques.

## Préenregistrement du banc

Binaire : `fourier5-online-bench`. Deux tailles seulement, pour
limiter le coût pendant cette étude préparatoire :

| Neurones | Contacts | Délai d'un épisode | Épisodes TRAIN par tâche | Episodes TEST |
|---:|---:|---:|---:|---:|
| 1 024 | 10 240 | 32 tics | 128 | 24 |
| 8 192 | 81 920 | 8 tics | 128 | 24 |

Graphe orienté synthétique, semences déterministes, mêmes connexions
et mêmes états initiaux pour les groupes appariés.

Chaque neurone i possède un enseignant artificiel différent :

`y_A = odd_parity(data_5) AND NOT(context[b_i])`

`y_B = odd_parity(data_5) AND NOT(context[(b_i + 1) mod 4])`.

`b_i` est calculé par graine + index, sans être fourni à l'élève.

**Trois témoins** : apprentissage sur la trace au moment de l'indice,
apprentissage sur la nouvelle activité au moment de la récompense
(label pourtant issu de l'indice), et modèle inchangé.
Un *snapshot* des poids après la tâche A est conservé pour mesurer
exactement l'interférence A→B à la fin de l'entraînement sur B.
La banque A est un témoin supplémentaire avec stockage de modèle
supplémentaire, pas une protection gratuite contre l'oubli.

**Anti-fuite** : pour chaque neurone, un tableau de 1024 bits (32×32)
enregistre toutes les paires `(données,contexte)` vues pendant TRAIN,
pour A **et** B. Les résultats TEST ne comptent que les paires jamais
vues sur ce neurone. Les épisodes de TEST utilisent une graine distincte.
Le tableau de provenance n'est consulté que pour l'évaluation et ne
modifie aucun modèle. Les paires test couvrent la même distribution
synthétique : ce n'est pas une généralisation OOD forte.

Deux lectures sont rapportées séparément :

1. **Rappel indicé** : représenter de nouveau l'indice à l'unité
   neuronale et comparer la prédiction du modèle à son label (ce qui
   mesure une association locale apprise).
2. **Rappel autonome sans réinjection de l'indice** : comparer la
   sortie du neurone après un délai sans indice réinjecté au label
   initial. Il n'y a aucune raison de supposer qu'il réussira.

Le résultat attendu n'est **pas** imposé par une assertion de supériorité
statistique : imprimer exactitudes, dénominateurs TEST, temps,
nombre d'updates et échec éventuel, même si le témoin fait mieux.

## Critères d'admission

- Aucune régression des tests classiques ou du moteur sans plasticité.
- Le retard est strictement exécuté entre l'indice et la livraison
  de la supervision, et son dépassement est rejeté.
- La phase de feedback ne change ni câblage ni état instantané,
  seulement les poids contextuels.
- Pas de fuite entre paires TRAIN et TEST pour un neurone.
- Trois modèles et lectures comparables, résultats négatifs conservés.

Qualification exclusivement via `Memorithm/RemoteOps`, exécution
Rust 1.89 CPU-only dans Docker isolé sans réseau, non privilégié,
4 CPU / 6 Gio, sans perturber les charges Dell/Thor/Mac.

Commandes : `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`,
`cargo run --offline --bin fourier5-bench`,
`cargo run --offline --bin fourier5-mask-bench`,
`cargo run --offline --bin fourier5-context-bench`,
`cargo run --offline --bin fourier5-online-bench`,
`cargo run --offline -- info`. Le banc récurrent long de la PR #82
est déjà qualifié sur son SHA ; il est délibérément omis ici pour
éviter de reconduire un exercice long et identique à chaque validation.
Aucune fusion sans CI verte sur le SHA exact.
