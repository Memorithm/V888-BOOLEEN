# B888-5 — Mesure physique à 6 250 000 bits, ancien calcul intégral contre cache exact

**Protocole préenregistré AVANT l'exécution** : expérience synthétique de
calcul booléen, non connectome biologique BANC V888. Le vrai moteur
`FiveCircleHierarchy` est volontairement limité à **625 bits** ;
il serait incorrect de prétendre l'avoir exécuté tel quel avec
6 250 000 entrées. Cette tranche introduit un **nouvel évaluateur de
taille bornée** `WideFiveTree` qui réemploie les mêmes tables
Fourier-Walsh de 32 états et la même fonction exacte `shannon_64`.
Il reproduit côte à côte les **deux algorithmes d'évaluation** :
le calcul intégral « ancien » généralisé et le cache incrémental
« nouveau » généralisé, sur **le même circuit**, les mêmes entrées,
les mêmes interventions et les mêmes paramètres.

## Géométrie exacte de la taille demandée

**6 250 000 sources d'entrée réelles**, chacune représentée par
un `u64` pour **64 motifs parallèles** (400 000 000 bits logiques
d'entrée au total). Ce ne sont pas 6,25 millions de neurones :
le circuit comporte exactement **1 562 501 neurones artificiels**
à cinq emplacements d'entrée :

| Étage depuis les sources | Neurones |
|---|---:|
| 1 | 1 250 000 |
| 2 | 250 000 |
| 3 | 50 000 |
| 4 | 10 000 |
| 5 | 2 000 |
| 6 | 400 |
| 7 | 80 |
| 8 | 16 |
| 9 | 4 |
| 10 (sortie) | 1 |

Nombre total de connexions réelles orientées :
**7 812 500**, en comptant 6 250 000 sources → premier
étage et les connexions entre étages. Pour atteindre
une sortie unique dans un arbre quinaire irrégulier,
**cinq emplacements vides** sont ajoutés aux étages
supérieurs (quatre au niveau 9 et un au niveau 10).
Ces entrées sont **des constantes FAUX**, ne sont pas
comptabilisées comme données réelles ou arêtes. Aucun
nouveau neurone anatomique BANC n'est inventé.

Tous les neurones du premier étage sont des portes
**Majorité** représentées par cinq cercles Fourier-Walsh ;
les autres sont des portes **Parité** représentées par les
mêmes mécanismes. Ces paramètres sont identiques à
l'intérieur d'un même étage (circuit synthétique
homogène, pas 1,56 million de poids indépendants).
Le modèle utilise des représentations compactes :
**50 000 000 octets** pour les 64 jeux d'entrées
bit-sliced, **12 500 008 octets** de sorties intermédiaires
pour le cache, sans compter l'allocateur et le processus.
La mémoire maximale résidente est relevée dans
`/proc/self/status` (`VmHWM`) sur Linux.

## Ancien contre nouveau

L'évaluateur `WideFiveTree::evaluate_full` visite
**chaque neurone**, indépendamment de la portée
du changement. Il réutilise **exactement** la
fonction booléenne `shannon_64` de l'ancien modèle.
C'est une **extension fidèle du principe de
recalcul complet**, et NON une exécution de l'ancienne
API `infer_bitplanes` qui rejette cette largeur.

`WideFiveCache::new` construit les sorties du
circuit **une seule fois** pour les 64 exemples.
`evaluate_delta` recalcule uniquement le neurone
dont les paramètres ont changé et ses ancêtres,
ou les deux neurones de premier étage touchés
par l'échange de sources et leurs ancêtres
uniques. Une propagation s'arrête dès que
la sortie à 64 voies ne change plus. **Le coût
de construction du cache est compté dans le
temps du nouveau moteur**, afin d'éviter
de présenter un gain artificiel.

Les paramètres Wiener-Walsh et les masques
conservés sont vérifiés sur chaque intervention.
Les interventions sont proposées sans optimiser
aucune métrique d'entraînement : ce banc mesure
la **vitesse et l'exactitude de l'inférence
contrefactuelle**, pas un apprentissage supervisé,
la précision d'un classifieur ni une mémoire
biologique.

Chaque profil teste quatre interventions
indépendantes depuis la même référence :

1. Remplacer les paramètres du premier neurone
   Majorité par ceux d'une porte ET (mêmes masques).
2. Échanger deux sources très éloignées
   (positions 0 et `largeur/2`).
3. Inverser la phase du premier masque Walsh
   d'un neurone de l'étage médian (identité
   des masques inchangée).
4. Inverser la phase de sortie du neurone racine.

Pour chaque modification, les **64 bits
de sortie** doivent être identiques entre
le recalcul complet et le cache. L'état initial
est vérifié contre une **référence indépendante**
sans compilation Fourier : la majorité de 5 bits
par disjonction des dix triplets, puis le XOR
de toutes les majorités (équivalent des étages
Parité, y compris avec les cinq constantes FAUX).
Aucun des deux évaluateurs n'est utilisé
pour établir ce résultat de référence.

## Budget de mesure fixé à l'avance

Trois graines de génération de 64 motifs aléatoires
`0x62500001`, `0x62500002`, `0x62500003`,
à **625 bits** (témoin de taille initiale) et
à **6 250 000 bits** (taille demandée), soit
6 profils et 24 interventions appariées.
Pour chaque profil : 1 évaluation intégrale
initiale + 4 recalculs intégraux contre
1 construction de cache + 4 interventions
incrémentales. Les deux versions produisent
les mêmes sorties sur chacune des 64 voies,
pour **toutes** les interventions et graines.
L'ordre de mesure des deux évaluateurs est
alterné pour les interventions successives.

Le temps total **inclut l'initialisation
du cache de référence** mais **exclut**
la génération des données d'entrée (son coût
est publié séparément) et le calcul de l'oracle
indépendant (lui aussi publié séparément).
Les coûts de chaque proposition, le temps
initial, le temps total, le nombre de portes
évaluées et le pic mémoire physique
sont publiés sans filtrage des régressions.
Les durées sont les temps réels dans
un conteneur Linux **Release** optimisé,
à CPU borné ; elles ne sont pas des
mesures d'énergie ou de trafic mémoire.

## Conditions de qualification et sécurité

- Fonctionne en **Rust** sans dépendance
  NVIDIA, sans modification du moteur BANC,
  du swap, des secrets ou d'un service de l'hôte.
- `MAX=6_250_000`, longueurs et indices
  des interventions contrôlés avant calcul.
  Aucun accès aux bits en dehors du tableau
  de sources ; les seuls zéros supplémentaires
  sont les cinq emplacements vides explicites.
- Tests des deux évaluateurs avec différentes
  tailles quinaires et **non** puissances de
  cinq (dont 1 000 / 6 250 / 62 500),
  plusieurs graines, phase et biais,
  échanges de sources distants / proches.
  Les erreurs d'indices et de masques
  doivent être rejetées sans corruption
  du cache.
- Qualification sur `Memorithm/RemoteOps`
  uniquement : Rust 1.89, `cargo fmt --all
  -- --check`, Clippy strict, tous tests
  et anciens bancs, puis **`cargo run
  --offline --release --bin fourier5-wide-bench`**
  et smoke CLI. Docker sans réseau, non
  privilégié, limites hôte inchangées :
  **4 CPU maximum et 6 Gio** de RAM ; le
  binaire est conçu pour occuper bien
  moins que cette limite. Timeout
  RemoteOps borné à 720 secondes.
- Ne fusionner que sur SHA exact
  entièrement qualifié, puis vérifier
  CI native sur le SHA de fusion.

## Portée et prochaines mesures

Cette étude est une **extrapolation
d'architecture** démontrée sur un
réseau synthétique à 6 250 000 bits.
Elle ne simule pas 6,25 millions de
neurones du connectome de mouche,
ne prédit aucune performance d'apprentissage
ou de mémoire, et ne traite pas
la connectivité récurrente du BANC.

En cas de résultats positifs, tester
ensuite la généralisation, les
coûts de mise à jour des caches
après plusieurs interventions
acceptées, et le comportement
sur des modèles non homogènes.
