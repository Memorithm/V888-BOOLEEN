# B888-4 — Coût physique apparié de l'entraînement, recalcul intégral contre cache

## Objet exact

Comparer **un vrai processus d'entraînement avec acceptation persistante**,
sur un circuit synthétique Fourier à 6 250 000 sources et 1 562 501 neurones
indépendamment paramétrables. Les deux bras reçoivent **le même flux**
de propositions, les mêmes 64 entrées TRAIN et le même enseignant
synthétique. Les seules étiquettes disponibles pour apprendre sont celles
de la sortie finale. Le modèle reste strictement identique entre les bras
après chaque décision, vérifiée bit à bit.

- Bras FULL : calcul intégral des 1 562 501 portes pour **chaque proposition**,
  puis second recalcul intégral sur toute la hiérarchie après chaque
  modification acceptée (`accept_full`).
- Bras CACHED : proposition contrefactuelle par chemin affecté,
  avec acceptation persistante des seuls mots intermédiaires modifiés.
- La construction initiale de chacun des deux circuits est **incluse**
  dans le temps total de son bras, mais la génération des entrées et
  de l'enseignant synthétique ainsi que les évaluations VALID/TEST
  sont exclues **symétriquement**. Les durées des 4 passages sont
  publiées séparément.
- Pour limiter le biais des caches chauds, l'ordre des deux bras
  est alterné à chaque intervention.

## Répétitions et observation

Trois graines : `0x9a770001`, `0x9a770002`, `0x9a770003`.
Par graine, quatre passages successifs de 64 propositions, donc 256
propositions dans chaque bras et jusqu'à 768 propositions par méthode
sur l'ensemble des trois graines. On relève à chaque passage le nombre
de propositions acceptées, les scores TRAIN/VALID/TEST, les temps des
deux méthodes, les portes et lectures logiques du cache.

Ne pas omettre les graines défavorables, les stagnations TRAIN et les
régressions VALID/TEST. Ce benchmark ne démontre ni apprentissage
synaptique sur le vrai BANC, ni capacité biologique ni généralisation
significative. L'étape mesure le **coût d'entraînement apparié**, pas
la performance finale d'un modèle utile.

## Limites

Les temps CPU comprennent les opérations de mesure logicielle, mais
**pas la consommation énergétique**. Une mesure RAPL dédiée et isolée
sera nécessaire pour chiffrer package et DRAM durant l'entraînement.
Aucun compteur CAS matériel n'est lu dans ce binaire ; ne pas assimiler
ses lectures logiques aux transferts DRAM physiques.

La méthode FULL est un comparateur volontairement direct de recalcul
complet, pas une implémentation optimisée de toutes les autres méthodes
d'entraînement possibles. Les deux modèles comprennent une structure
de cache de même taille à l'initialisation, mais FULL l'actualise
entièrement après chaque acceptation.

Qualification obligatoire : Rust 1.89, format, Clippy strict, tests,
et exécution physique via RemoteOps sur SHA exact, sans secrets
ni réseau dans le conteneur.
