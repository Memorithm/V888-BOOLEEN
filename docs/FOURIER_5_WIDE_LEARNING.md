# B888-4 — Plasticité persistante et apprentissage booléen à 6 250 000 entrées

## Objectif mesurable

Évaluer la possibilité d'entraîner des paramètres Fourier **distincts pour chacun
des 1 562 501 neurones** d'un arbre synthétique à cinq entrées, et distinguer
explicitement cette capacité d'un apprentissage end-to-end fondé sur la seule
sortie. Ce programme n'est **pas** une simulation fidèle du cerveau BANC.

`WidePlasticCircuit` contient un `FiveCircle` et une table de vérité
indépendants par neurone ; il maintient les 64 sorties de chaque neurone.
`propose` calcule une intervention locale sans mutation et `accept` conserve
les paramètres retenus ainsi que le nouveau cache le long du chemin modifié.
Chaque proposition est liée à une génération : impossible de réutiliser une
proposition obsolète après un commit. Rejet = modèle inchangé.

## Deux protocoles d'apprentissage volontairement séparés

**A — Supervision locale privilégiée (capacité fonctionnelle, pas end-to-end).**
Un enseignant synthétique reçoit, pour chaque neurone, une fonction déterministe
tirée de cinq variantes Fourier : majorité, parité, ET, OU et\ncomplément de la parité (phase de Walsh inversée). Les étages\nsupérieurs utilisent parité/complement pour éviter l'effondrement\nvers une sortie constante. L'étiquette cible de sortie doit\ncomporter de 16 à 48 positifs sur les 64 voies de chaque split. Ce modèle de
référence produit les sorties cachées de chaque couche sur **64 motifs TRAIN**.
L'élève ajuste indépendamment les 1 562 501 neurones : pour chaque noeud,
il choisit les paramètres correspondant à la fonction candidate qui minimise
les erreurs sur les **64 labels cachés observés**. Une seule passe est effectuée
de l'entrée vers la sortie. Les labels TEST/VALID ne servent jamais au choix.
Le score des neurones (somme des bits exacts sur chaque couche) et le score
de la sortie sont publiés sur TRAIN et sur **64 motifs VALID disjoints**.
Les modèles de l'enseignant ne sont **pas copiés** dans l'élève.

IMPORTANT : les activations cachées de l'enseignant sont des signaux supervisés
privilégiés, artificiellement disponibles dans ce banc. Un bon score de
validation **ne prouve pas** qu'un modèle apprendra sans accès à ces activations
ou que l'apprentissage s'appliquera aux données biologiques BANC.

**B — Seuls labels de sortie (apprentissage end-to-end borné).**
Un autre réseau, initialisé sans labels cachés, effectue 4 096 propositions
de nouveaux paramètres neuronaux, sur plusieurs couches. Une proposition
est retenue seulement si elle augmente strictement l'exactitude des
64 sorties TRAIN. Le modèle conserve les changements acceptés entre essais.
Les 64 labels VALID ne sont utilisés qu'après l'entraînement. Publier
exactitude TRAIN avant/après, exactitude VALID avant/après, propositions,
commits et latence. Ne pas masquer les stagnations et régressions VALID.

Ces expériences n'optimisent pas encore les connexions synaptiques ni une
récompense différée. Les cinq variantes Fourier candidates ne constituent
pas une recherche libre de tous les paramètres entiers possibles.

## Comparateur de performance après apprentissage

Sur le même modèle entraîné, 64 modifications indépendantes sont évaluées
par parcours complet et par propositions exactes dans le cache. Vérification
de toutes les sorties sur 64 voies **bit pour bit**.

Publier séparément :

- `full_us` et `cached_us` : durées des 64 propositions sur cache déjà
  construit ; le coût de construction/entraînement est donc **exclu** de
  ce sous-banc, à la différence de B888-5 PR #94 ;
- `full_logical_child_words` et `cached_logical_child_words` : comptes
  déterministes des lectures logiques de mots `u64` du programme. Ce ne
  sont **ni** des transactions DRAM **ni** des compteurs cache matériels ;
- `cached_changed_words` : nombre de mots de cache **qui changeraient** si
  les propositions étaient acceptées ; aucune écriture n'est commise dans
  ce sous-banc (ne pas confondre avec les écritures physiques du processeur) ;
- `VmHWM` : pic RSS du processus complet, pas la consommation attribuable
  à un seul moteur ;
- `LARGE_RAPL_PACKAGE_ENERGY` : lecture d'énergie du paquet CPU `energy_uj`
  si, et **seulement si**, le noyau expose des compteurs Intel RAPL accessibles.
  Si non accessibles, publier `unavailable`, **ne pas inventer de joules**.
  Même accessibles, ces compteurs incluent les autres charges du paquet CPU
  et les batches de durées très différentes ; ils ne suffisent pas pour
  conclure à un gain énergétique propre au moteur. Limites de résolution et
  enroulement des compteurs à surveiller.

Une vraie mesure de trafic DRAM et une comparaison énergétique isolée
nécessiteront, dans un protocole physique séparé et approuvé, des compteurs
matériels compatibles (par exemple via `perf` ou les contrôleurs mémoire),
des répétitions appariées et la caractérisation des charges concurrentes.
Ne jamais présenter des lectures logiques comme des octets bus mémoire.

## Ressources et exécution

Commande : `cargo run --offline --release --bin fourier5-wide-learning-bench`.

Aucune dépendance NVIDIA, pas de poids tiers, pas de réseau à l'exécution.
Exécution Dell uniquement via l'isolement RemoteOps existant, jusqu'à quatre
CPU et six Gio RAM, dans le budget de temps défini par le workflow.
Le fait que le modèle ait 1 562 501 *emplacements de paramètres indépendants*
est distinct du nombre réel de modèles modifiés ou appris : publier
`models_changed` et les exactitudes des deux procédures sans les confondre.

**Critères de qualification :** format Rust 1.89, Clippy strict, tous les tests
d'unités, benchmark physique 6 250 000 bits et absence de divergence.
Ne fusionner que sur le SHA qualifié. Répliquer sur plusieurs graines,
fournir intervalles de dispersion, et comparer des tâches non privilégiées
avant toute revendication d'apprentissage général ou biologique.
