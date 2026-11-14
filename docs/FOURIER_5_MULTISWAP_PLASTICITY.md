# B888-5 — Combinaisons de connexions et plasticité paramétrique Fourier

État initial : prototype expérimental, mesures physiques à réaliser. **Les deux axes sont explicitement autorisés.** Aucune modification du connectome BANC, aucun câblage biologique supposé. Pas de GPU/NVIDIA, ni de modification des services productifs.

## Deux mécanismes distincts et une expérience jointe

1. **Connectivité structurelle :** rechercher toutes les suites de 0, 1, 2 ou 3 échanges distincts dans une banque de 8 paires de feuilles. Une permutation de sources conserve exactement le nombre de neurones, le fan-in de cinq et les contacts dirigés ; elle ne change ni les masques Walsh, ni le biais, ni les rayons.
2. **Plasticité supervisée (opt-in) :** seul le **dernier neurone** d'une hiérarchie reçoit des mises à jour de ses cinq coefficients Walsh signés (rayons et phases) et de son seuil entier (biais). Les identités des masques, tous les étages intermédiaires et le routage sont figés au cours d'une adaptation locale. Un perceptron déterministe borne chaque coefficient et le biais entre -32 et +32 ; 1 à 16 passages TRAIN, puis arrêt possible sans erreur. L'apprentissage n'est **ni** plasticité synaptique biologique, **ni** STDP, ni apprentissage non supervisé.
3. **Couplage :** pour chaque routage candidat, construire les cinq bits présynaptiques du dernier neurone sur TRAIN, entraîner ses coefficients par le même perceptron et mesurer l'exactitude de son modèle ajusté. La meilleure solution est choisie uniquement avec TRAIN ; le candidat sans plasticité est également admissible, donc la combinaison ne peut être moins bonne sur TRAIN que le meilleur circuit gelé.

La recherche est **exhaustive uniquement dans une banque bornée**, pas dans toutes les permutations possibles de 625 bits. Avec 8 paires proposées et 3 échanges maximum, la combinatoire théorique est 1 + 8 + 28 + 56 = 93 propositions. Les routages identiques après permutations sont dédoublonnés ; nombre exact de modèles évalués, mises à jour et visites d'entraînement sont enregistrés.

Le générateur d'échanges ne lit aucune donnée ni étiquette. Les suites appliquent les indices de la banque dans un ordre croissant déterministe. Deux échanges peuvent partager un indice de source ; les bijections et budgets restent vérifiés. Une égalité des performances conserve un gagnant plus simple rencontré plus tôt (zéro, puis un, deux, trois échanges). L'espace de recherche ne garantit pas un optimum hors banque.

## Protocole expérimental préenregistré

Trois tailles : 25, 125, 625 bits, avec exactement 6, 31, 156 neurones Fourier et 30, 155, 780 contacts. Les couches internes sont des combinaisons gelées de fonctions **Majorité** et **Parité**. Le dernier neurone initial est un ET à cinq entrées ; le neurone-enseignant synthétique est une Majorité, avec les **mêmes cinq masques singleton**. Les gains, phases et seuils du readout doivent donc pouvoir s'adapter, sans modifier sa définition structurelle.

L'enseignant combine deux ou trois échanges de connexion dans la banque, avec trois graines indépendantes `0x58880101`, `0x58880202`, `0x58880303`. Le code vérifie que le routage-enseignant ne coïncide ni avec la connexion initiale ni avec une proposition à un seul échange. La vérité est définie par un **oracle booléen indépendant** des tables Fourier.

Pour chaque graine et taille :
- TRAIN : 512 motifs, bits Bernoulli(0,5) ; seul ensemble utilisé par les deux mécanismes.
- VALID : 256 motifs, même distribution ; diagnostic uniquement, sans sélection ni arrêt anticipé.
- TEST : 512 motifs, même distribution ; mesure finale.
- SHIFT : 256 motifs, bits Bernoulli(0,25) ; décalage de densité mesuré séparément.
- Les motifs entiers sont tous disjoints entre ensembles au sein d'une expérience ; aucune étiquette VALID/TEST/SHIFT n'entre dans la recherche, la plasticité ou les décisions de sélection.

Six bras sur chaque ensemble : circuit initial gelé ; meilleur échange simple gelé ; meilleure combinaison gelée ; circuit initial avec plasticité du dernier neurone ; routage + plasticité conjoints ; enseignant synthétique. Les quatre premières variantes ne peuvent pas cacher un échec par la valeur de la dernière. Les effectifs positifs/négatifs et les vrais positifs/vrais négatifs sont **obligatoirement** imprimés pour déceler les scores élevés dus au déséquilibre des classes. L'ensemble TEST n'est pas utilisé pour le choix des paramètres.

**Limite cruciale :** le graphe enseignant à plusieurs échanges figure **par construction** dans la banque de recherche conjointe, et son dernier neurone possède des masques apprenables. C'est une démonstration d'**identification conditionnelle** de routages et de paramètres, pas une découverte générale, ni une preuve OOD stricte. Le décalage de densité ne change pas la tâche ni les connexions de l'enseignant.

Les six bras partagent le **même budget de neurones et de contacts à l'inférence**. Leurs budgets d'entraînement diffèrent beaucoup : aucune revendication d'efficacité ne doit les assimiler. Le banc publie, outre les durées, le nombre de candidats, les modèles de readout ajustés, les corrections de perceptron, les présentations TRAIN au fit, et les classifications globales. Les coûts de préparation/transpose et de génération des exemples sont exclus des durées de sélection ; ne pas interpréter ces durées comme des temps de déploiement bout en bout.

## Critères d'admission

- Comparaisons indépendantes et sans fuite d'étiquettes, données dédoublonnées, plusieurs graines et classes surveillées.
- Chaque permutation doit conserver bijection et budget fixe ; aucune modification des poids des étages intermédiaires ni de l'identité des masques Fourier.
- Le meilleur modèle conjoint sur TRAIN doit être au moins égal aux solutions avec uniquement connexions, ou uniquement plasticité, puisque les deux appartiennent explicitement à l'espace de candidats.
- Les erreurs et performances TEST/SHIFT sont rapportées sans corriger a posteriori les seuils d'acceptation.
- Rust 1.89 hors réseau, `cargo fmt --all -- --check`, `cargo clippy --offline --all-targets -- -D warnings`, `cargo test --offline --all-targets`, tous les bancs Fourier antérieurs, `cargo run --offline --bin fourier5-joint-bench`, et CLI `cargo run --offline -- info`, **sur le HEAD exact**.
- Validation physique dans `Memorithm/RemoteOps` uniquement : conteneur non privilégié, sans réseau, <= 4 CPU et <= 6 Gio, timeout borné. Pas de service productif arrêté, pas de paramètres de swap ou secrets modifiés.
- Fusion uniquement après qualification complète verte du SHA exact puis contrôle de CI native du commit intégré.

## Limites et suite

La plasticité est limitée **au neurone de sortie** ; les étages internes restent figés. Il faut encore étudier le routage hors banque, les interactions avec la plasticité de plusieurs étages, les tâches séquentielles et la robustesse hors distribution sur des fonctions/graphes inconnus. Cette tranche ne prétend rien sur la connectivité anatomique réelle et ne modifie pas les mécanismes récurrents précédents.
