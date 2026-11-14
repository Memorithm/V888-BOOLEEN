# B888-5 — Enseignant hors banque et plasticité des couches cachées

Statut initial : préenregistrement **avant** qualification physique des résultats.
Expérience booléenne synthétique, exclusivement Rust, non biologique.
Deux interventions explicitement distinctes : changements de sources et
adaptation des paramètres internes. Aucune arête BANC n'est modifiée.

## L'hypothèse et la limite corrigée

Dans la PR #88, les cibles sont exactement contenues dans la banque de
93 routages et le neurone terminal enseignant est représentable par les
mêmes cinq masques. Un score de 100 % y mesure une identification
conditionnelle et non une découverte générale.

La présente tranche construit un enseignant à trois échanges de feuilles
issus d'une banque **différente**, avec une graine distincte de celle du
chercheur. Le programme vérifie exhaustivement que le routage enseignant
n'est égal à **aucun** des 93 routages candidats (0 à 3 permutations de
la banque de recherche). Cet oracle est généré indépendamment de la
recherche et des données de test. Le meilleur candidat peut être
imparfait : l'exactitude TEST/SHIFT n'est jamais forcée à 100 %.

## Nouvelle fonctionnalité : plasticité supervisée des couches internes

`FiveCircleHierarchy::layer_trace` expose pour un état d'entrée les
cinq bits présynaptiques et la sortie de chaque neurone à chaque couche.
`FiveCircleHierarchy::with_neuron_parameters` remplace un seul modèle
validé en préservant les cinq **identités de masques** Walsh, le nombre
de neurones et le routage ; sa table de vérité est recompilée pour
le seul neurone modifié.

`fit_hidden_with_teacher_taps(student, teacher, TRAIN, epochs)` ajuste
séquentiellement tous les neurones **internes**, niveau par niveau,
en utilisant les cinq entrées réellement présentes sur le circuit
étudiant et comme label la sortie du neurone enseignant à la même
position de couche. Chaque règle est un perceptron à poids entiers
signés bornés à [-32, 32], avec 8 époques dans le banc, **sans
modifier les masques, ni introduire de nouvelles arêtes**. Une fois
les couches cachées ajustées, la sortie du circuit est également
ajustée avec les labels externes de TRAIN et le mécanisme existant
`fit_output`. Les comptes de mises à jour, de neurones modifiés,
de présentations locales et de scores locaux et globaux sont exposés.

**Limite déterminante : les sorties intermédiaires de l'enseignant
ne sont normalement PAS connues.** Cette procédure utilise une
supervision locale privilégiée, réservée à l'expérience synthétique.
Elle n'est pas un apprentissage de bout en bout, ni la solution au
credit assignment d'un réseau réel. Pour ce banc, ces étiquettes
intermédiaires ne sont consultées que sur TRAIN, jamais sur VALID,
TEST ou SHIFT.

## Comparateurs et données, préenregistrés

Trois graines `0x58900101, 0x58900202, 0x58900303`.
Profondeurs 2, 3, 4 ; respectivement **25, 125, 625 bits**,
**6, 31, 156 neurones** et **30, 155, 780 contacts**.

Couches enseignant :
- Profondeur 2 : [majorité, majorité] ;
- Profondeur 3 : [majorité, parité, majorité] ;
- Profondeur 4 : [majorité, parité, majorité, majorité].

Couches étudiantes : remplacer chaque porte majorité par ET, qui
partage les mêmes cinq masques singleton mais possède un biais
différent ; les portes parité restent identiques. Cet objectif
sépare explicitement **plastification** et **connexion**.

Chaque profil comporte quatre ensembles d'états **disjoints,
dédoublonnés par état bit-packed complet** :
TRAIN 512 Bernoulli(0,5) ; VALID 256 Bernoulli(0,5) ; TEST 512
Bernoulli(0,5) ; SHIFT 256 Bernoulli(0,25).
Le générateur de labels est un oracle booléen indépendant des
tables et du modèle Fourier (`GateKind::oracle` couche par couche).

Sept bras prévus :
1. Étudiant gelé, routage original ;
2. Recherche 0–3 échanges seule, sans plasticité ;
3. Plasticité du dernier neurone seule, route originale ;
4. Recherche 0–3 échanges + plasticité du seul dernier neurone
   (algorithme existant PR #88, routage enseignant désormais hors banque) ;
5. Plasticité locale des neurones cachés puis du dernier neurone,
   **sans aucune recherche de connexion** ;
6. Recherche + plasticité du readout, puis plasticité cachée et
   réajustement du readout ;
7. Oracle synthétique, uniquement pour vérification de labels et
   estimation de l'écart irréductible dans l'espace de recherche.

Les sept bras ont le même **nombre de neurones et de contacts
à l'inférence**. Le coût d'entraînement est très différent :
le bras 6 utilise des étiquettes supplémentaires de tous les
neurones cachés et tout le budget du bras 4 ; il n'est donc **pas**
un comparateur à supervision égale. Les tableaux de résultats
incluent `TP/positifs` et `TN/négatifs` pour éviter les succès
triviaux sur les distributions de classes déséquilibrées.
Un modèle perdant ou un test SHIFT négatif doit être publié.

## Admission / sécurité

- Conformité des labels de tous les ensembles à l'oracle indépendant,
  et impossibilité vérifiée du routage enseignant dans la banque.
- Aucun label VALID/TEST/SHIFT lu dans la recherche, l'entraînement
  caché ni la mise à jour du readout.
- Même structure, contact count et masques après adaptation ; rejet
  atomique des indices et masques incorrects.
- Budgets Rust : 512 TRAIN, <=16 époques, <=1024 motifs par fonction,
  8 propositions d'échanges et <=3 échanges par candidat (93 au
  maximum) ; sans chargement de données biologiques.
- Préqualification `cargo fmt --all -- --check`, `cargo clippy
  --offline --all-targets -- -D warnings`, `cargo test --offline
  --all-targets`, bancs Fourier précédents + nouveau
  `fourier5-outbank-hidden-bench`, CLI smoke ;
  uniquement sous `Memorithm/RemoteOps` sur Dell, SHA exact,
  Rust 1.89, Docker non privilégié, sans réseau, 4 CPU et 6 Gio.
- Pas de fusion sans qualification complète verte, puis
  CI native verte sur le SHA de fusion.

## Prochaines étapes scientifiques

Le routage enseignant hors banque est un progrès méthodologique, mais
il reste tiré d'une autre banque prédéfinie et les fonctions
cibles sont construites ; ne pas parler de connectivité générale
découverte. Le prochain palier est de remplacer la supervision
des sorties cachées par une règle end-to-end ou locale sans «
teacher taps », et de comparer à coûts et signaux d'apprentissage
réellement appariés. Une évaluation sur vraies données BANC
nécessite le manifeste d'origine, SHA-256, validation de licences
et conversion réversible ; aucun résultat synthétique ne
remplace ces vérifications.
