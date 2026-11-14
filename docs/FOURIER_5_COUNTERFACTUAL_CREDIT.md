# B888-5 — Attribution du crédit par interventions contrefactuelles

Statut : protocole expérimental **préenregistré avant mesure sur Dell**.
Hypothèse synthétique testée en Rust uniquement ; ne prétend pas à une
plasticité biologique, à la découverte du connectome BANC ni à une
méthode de rétropropagation sans gradients démontrée sur données réelles.

## Problème réel résolu par cette tranche

La PR #90 adapte les couches cachées en exposant les **sorties internes
de l'enseignant**. C'est une supervision privilégiée impossible à
supposer disponible dans un réseau autonome.

Le nouveau module `fourier5_counterfactual_credit.rs` prend comme
entrées **seulement** un circuit étudiant, des états TRAIN et un
label final booléen par état. Il ne prend ni modèle enseignant, ni
trace cachée, ni paire (entrée/sortie) d'un neurone interne.

Un coup d'apprentissage :
1. Générer une proposition selon une graine indépendante des labels.
   Elle modifie soit **un** paramètre d'un neurone interne
   (coefficient Walsh signé ou biais, de ±1, ±2 ou ±4 avec saturation
   [-32,+32]), soit échange deux sources distinctes traversant des
   groupes de cinq. Le circuit conserve ses masques, sa taille et
   ses dix contacts par couche de composition.
2. Évaluer le circuit entier de l'entrée à la réponse finale sur
   **TRAIN uniquement**, par le moteur bit-sliced 64 voies déjà
   qualifié. Le coût est le nombre de prédictions complètes,
   pas une estimation indirecte du coût neuronal.
3. Sur un budget explicite de propositions, conserver uniquement
   le candidat qui augmente strictement le nombre de bonnes réponses
   TRAIN (politique `Greedy`). En cas d'égalité ou de régression,
   les coefficients ne sont pas modifiés. Il n'y a **aucun
   ajustement par VALID/TEST/SHIFT**.
4. Répéter un nombre borné de cycles même en cas de plateau. La
   politique `Blind`, à **budget et opérations d'inférence égaux**,
   choisit à l'avance une proposition au hasard, sans considérer
   le score TRAIN pour la choisir. Ce bras peut régressser.

Les quatre scopes sont `HiddenParameters` (neurones cachés seuls),
`OutputParameters` (sortie seule), `LeafRouting` (connexions
seules) et `JointHiddenAndRouting` (paramètres internes et sources).
Les mises à jour sont opt-in. Le noyau récurrent et les expériences
antérieures de plasticité sont inchangés.

## Charge expérimentale préenregistrée

- Trois graines `0xc0880101`, `0xc0880202`, `0xc0880303`
  et trois hiérarchies : 25 / 125 / 625 bits, respectivement
  6 / 31 / 156 neurones et 30 / 155 / 780 contacts dirigés.
- Enseignant synthétique : Majorité sur les cinq bits de chaque
  neurone de la première couche, puis Parité sur toutes les couches
  supérieures. Son routage subit trois échanges produits avec
  une graine **distincte** de celle de l'apprenant ; aucun de
  ces échanges ou états internes n'est fourni au moteur.
- Étudiant : même circuit initial, mais chaque cinquième
  neurone de première couche utilise ET au lieu de Majorité
  (mêmes masques singleton, biais différent), et les sources
  restent contiguës. Les positions des paramètres à corriger
  sont connues uniquement du générateur de la tâche, **pas**
  du moteur qui prend des décisions.
- 512 exemples TRAIN, 256 VALID, 512 TEST, 256 SHIFT par
  graine/profondeur, **mutuellement disjoints par état complet**.
  Le label final vient d'un oracle booléen indépendant des
  tables Fourier, et l'exécution Fourier de l'enseignant doit
  concorder. TRAIN/VALID/TEST : bits Bernoulli(0,5) ;
  SHIFT : bits Bernoulli(0,25).
- **8 cycles de recherche × 64 propositions**, soit **512
  contrefactuels et 513 × 512 = 262 656 prédictions TRAIN
  de modèle complet par bras**, y compris le score initial,
  pour chaque profondeur et graine. Les temps par bras sont
  mesurés après préparation des données, sans inclure les
  oracles ni l'inférence sur les jeux retenus.
- Les six bras étudiés sont : étudiant initial, sortie seule,
  plasticité des couches cachées seules, connexions seules,
  optimisation conjointe et contrôle `Blind` de même
  budget que le conjoint. L'enseignant est un témoin
  de génération de labels, **non** un bras apprenant.
- Mesures : scores TRAIN/VALID/TEST/SHIFT, vrais positifs et
  vrais négatifs sur chaque partition, nombre de mises
  à jour des paramètres internes / échanges de sources,
  tours sans amélioration, prédictions TRAIN et temps CPU
  observé. Aucune sélection a posteriori sur VALID/TEST.

**Conditions d'acceptation de code :** chaque forme d'adaptation
garde les masques Walsh et le nombre de contacts. Toutes les
méthodes réalisent le même nombre de comparaisons TRAIN dans
ce banc. La recherche `Greedy` ne peut diminuer la précision
TRAIN ; les scores TEST/SHIFT peuvent régresser, et seront
publiés. Le résultat ne doit pas être présupposé parfait.

## Limites

Cette procédure effectue des évaluations d'intervention
**complètes**, ce qui coûte beaucoup plus cher qu'une mise à
jour biologique locale. L'« attribution » du crédit signifie
ici une **différence mesurée du score final** entre modèle
original et modèle contrefactuel, et non une cause biomédicale
démontrée. Une permutation des feuilles et une adaptation des
paramètres peuvent produire des fonctions équivalentes sans
que les mécanismes sous-jacents soient identifiables.

Le réseau enseignant utilise toujours une fonction synthétique
définie à l'avance. La densité de SHIFT ne modifie pas la tâche
de classification, seulement la distribution des états. Aucune
conclusion ne peut être transposée aux comportements réels de
l'insecte ni à la totalité du connectome sans qualification
séparée de la version V888, des données originales et de leurs
contrats.

## Qualification

Rust 1.89, CPU uniquement et RemoteOps comme plan de contrôle
exclusif : `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`, tous les précédents
bancs Fourier, `cargo run --offline --bin fourier5-credit-bench`,
puis `cargo run --offline -- info`. SHA exact publié depuis le
Dell dans un conteneur non root sans réseau ni SDK NVIDIA ;
quatre CPU maximum, 6 Gio et 25 minutes maximum.
Ne fusionner que si toute la qualification est verte ; vérifier
ensuite la CI native sur le commit exact de fusion.

## Prochain palier

Tester l'inférence de crédit sans devoir évaluer le réseau entier
pour chaque candidat : caches de sous-arbres, sensibilité
booléenne et scores d'impulsions localisées. Comparer les gains
à coût exact et mesurer la qualité de généralisation hors tâche,
sans recourir aux activations cachées d'un enseignant.
