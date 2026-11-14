# B888-4: apprendre cinq masques Fourier-Walsh (préenregistrement)

**Protocole fixé avant exécution Rust.** Dépôt expérimental V888-BOOLEEN,
sans modification des données anatomiques BANC ni simulation biologique validée.

## Hypothèses testables

- **H1 — Parité quatre entrées** : un mode Walsh choisi par les données
  identifie un masque non trivial, `0b01111`, et obtient 16/16 sur un test
  à cinquième bit tenu à l'écart. Le témoin à conjonctions pourrait ne pas
  représenter correctement la même fonction avec cinq modes ; publier son
  résultat, favorable ou non.
- **H2 — Conjonction quatre entrées** : le témoin conjonctif doit au moins
  égaler le modèle Walsh en qualité sur le même partage, mais peut utiliser
  moins de modes actifs. Ne pas supposer que Walsh gagne partout.
- **H3 — Généralisation** : une séparation entraînement/test sur un bit
  *non pertinent par construction* teste seulement l'absence de dépendance
  parasite à ce bit ; elle ne démontre aucune généralisation OOD au sens large.
  Évaluer aussi une fonction majorité quatre entrées afin de conserver
  et publier les échecs éventuels de la sélection gloutonne.

## Comparaison à capacité et entraînement proches

Même entrée de cinq bits, même ensemble de 31 masques non nuls, même
cinq emplacements pour les masques et poids signés entiers (plus biais),
même ordre d'exemples, même nombre maximal d'itérations de perceptron
et même règle gloutonne de choix des masques par erreurs puis marge hinge.

- **Fourier-Walsh** : fonction de mode
  `chi_m(x)=(-1)^popcount(x & m)`.
- **Conjonction** : fonction de mode
  `a_m(x)=1-2*[x & m == m]`.

Chaque famille dispose de 31 modes candidats pour cinq places. Les masques
sont **appris**, pas fournis par l'oracle. Les poids et le biais sont
réajustés sur les *seules données TRAIN*. Un masque nul désigne une
place inactive. Les ex aequo sont réglés par ordre numérique du masque.
L'algorithme glouton est déterministe mais **n'assure pas** le meilleur
sous-ensemble global de cinq modes. Les tailles de représentation et le
coût par mode ne sont pas présumés identiques en CPU : popcount/parité
et test de conjonction ont des coûts matériels distincts.

Les tables synthétiques fixées utilisent toutes les entrées x de 0 à 15
pour TRAIN et 16 à 31 pour TEST. L'étiquette ne dépend que de x & 15 :
parité quatre bits, conjonction quatre bits et majorité trois-sur-quatre.
Chaque moitié comporte 16 exemples disjoints, sans accès aux étiquettes
de TEST pendant l'apprentissage.

## Mesures de qualification

Commande prévue : `cargo run --offline --bin fourier5-mask-bench`.
Publier pour chaque fonction et chaque famille : train correct / 16,
test correct / 16, nombre de modes actifs, masques choisis, mises à
jour du perceptron. Ces chiffres ne sont pas encore des mesures.

Exiger via RemoteOps sur le SHA exact : Rust 1.89
`cargo fmt --all -- --check`, `cargo clippy --offline --all-targets
-- -D warnings`, `cargo test --offline --all-targets`, exécution
des deux benchmarks Fourier, et smoke CLI. Après fusion, CI `main`
verte au commit de fusion. Aucun accélérateur NVIDIA.

## Limites et suites

Le choix d'une famille de cinq modes n'est pas une preuve d'une
architecture neuronale efficace. Après ce lot : séparation temporelle,
séquences avec contexte, perte des souvenirs, comparaison à un petit
réseau booléen multicouche à budget paramétrique réellement apparié,
latence, mémoire et sensibilité aux perturbations. Une intégration
physiologique avec le connectome exige des hypothèses et des preuves
distinctes.

## Résultats après l'exécution du protocole

La qualification physique de la première implémentation a réussi sur le SHA
`ee2ef49db3b2f9cdc8afb254758f96c119d4cbe2` avec
[RemoteOps run 43435506198, tentative 3](https://github.com/Memorithm/RemoteOps/actions/runs/43435506198).
Les contrôles `fmt`, `clippy -D warnings`, `cargo test --all-targets`,
l'ancien banc Fourier, le nouveau banc Fourier-masques et le smoke CLI sont
passés. Les tests Rust incluent 154 tests de bibliothèque et deux tests du
nouveau binaire ; aucun benchmark microarchitectural n'a été exécuté.

| Cible | Famille | TRAIN | TEST | Modes actifs | Masques choisis | Updates du dernier refit |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| parity4 | Walsh | 16/16 | 16/16 | 1 | [15] | 2 |
| parity4 | Conjonction | 9/16 | 9/16 | 1 | [15] | 898 |
| and4 | Walsh | 15/16 | 15/16 | 1 | [1] | 129 |
| and4 | Conjonction | 16/16 | 16/16 | 1 | [15] | 2 |
| majority4 | Walsh | 15/16 | 11/16 | 5 | [1,17,4,7,13] | 388 |
| majority4 | Conjonction | 15/16 | 15/16 | 4 | [13,6,11,2] | 137 |

Interprétation : H1 est confirmée *dans cette petite tâche synthétique*
pour Walsh (un masque 15, 16/16), H2 observée (conjonction 16/16,
Walsh 15/16), et H3 révèle un cas de généralisation dégradée
pour Walsh sur majority4 (15/16 TRAIN et seulement 11/16 TEST).
La sélection gloutonne est donc imparfaite, y compris sur des fonctions
simples ; ce constat interdit toute conclusion de supériorité globale.
Le nombre d'updates ne couvre que le refit final du sous-ensemble retenu
et ne mesure pas le coût complet de recherche de masques. Les deux familles
ont les mêmes budgets *maximaux* mais des coûts d'exécution différents.

Les résultats s'appliquent uniquement à des tables synthétiques de
cinq bits. Aucune hypothèse sur le connectome réel BANC n'a été validée.
