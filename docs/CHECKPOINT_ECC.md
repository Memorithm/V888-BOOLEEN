# V888 — snapshots d'état booléen protégés par SECDED 128

## Périmètre

Le moteur V888 conserve **un bit d'activation par neurone**. Le module Rust
`checkpoint_ecc` ajoute un mécanisme **opt-in** pour capturer puis restaurer
une copie d'état, sans intervenir dans les mises à jour des synapses,
les décisions de sélection, l'inférence ou la mémoire résidente.

Chaque bloc de 128 bits est protégé par un code de Hamming étendu
raccourci : huit bits de syndrome et un bit de parité globale, stockés
séparément. Dans le format en mémoire actuel, chaque tag occupe un `u16` :
**16/128 = 12,5 %** de surcoût de métadonnées par bloc complet. Avec un
empaquetage bit à bit, la limite théorique serait **9/128 = 7,03125 %**.
Les blocs incomplets sont complétés par des zéros vérifiés à la lecture.

Ce mécanisme peut corriger **un bit erroné par bloc protégé** (y compris
dans le tag) et détecter **deux bits erronés dans un même codeword**.
Trois erreurs ou plus ne sont pas couvertes et peuvent être mal corrigées
ou ne pas être détectées. Les erreurs dans des blocs différents sont
traitées indépendamment.

## Interface

```rust
use v888_booleen::{checkpoint_ecc::ProtectedBooleanSnapshot, BooleanState};

let mut live = BooleanState::new(129);
live.set(0, true).unwrap();
live.set(128, true).unwrap();

let snapshot = ProtectedBooleanSnapshot::capture(&live);
let (len, blocks, tags) = snapshot.into_parts();
let stored = ProtectedBooleanSnapshot::from_parts(len, blocks, tags).unwrap();
let report = stored.recover().unwrap();

assert_eq!(report.state, live);
assert!(report.repairs.is_empty());
```

`capture` et `recover` n'altèrent pas l'état du moteur. Une erreur
non corrigeable empêche de renvoyer un état partiellement réparé.

Pour un chargement externe, `recover_bounded(max_neurons)` impose un plafond
explicite au nombre de neurones **avant l'allocation** de l'état reconstruit.
Si `len > max_neurons`, la restauration échoue sans état partiel avec
`NeuronBudgetExceeded`. Ce plafond ne limite pas à lui seul la mémoire
déjà consommée par l'encodage : la taille de l'entrée et son cadrage doivent
être contrôlés en amont.
`into_parts` fournit les données nécessaires à un futur format durable,
mais **n'écrit aucun fichier**.

## Garanties, limites et critères de qualification

- Tests : longueur vide, frontières des mots (63/64, 127/128), 188 508
  positions ; tous les 137 défauts simples et toutes les 9 316 paires
  de défauts par mot de code pour trois contenus ; défauts par blocs ;
  tags invalides, dimensions incohérentes et bits de remplissage.
- `len`, le nombre de blocs, les frontières et un éventuel conteneur
  disque **ne sont pas authentifiés ni protégés par ce code**. Pour une
  sauvegarde durable, ils nécessiteront un format atomique et une
  protection indépendante du cadrage avant de prétendre à la résistance
  aux corruptions de fichier.
- Aucune garantie contre les altérations malveillantes, aucun calcul
  quantique, aucune promesse d'accélération de l'inférence.
- Implémentation autonome en Rust pour préserver la qualification
  `cargo --offline` de V888. L'interface est compatible dans son principe
  avec le code SECDED128 de `scirust-algebra`, PR SciRust #1738, mais
  aucun lien logiciel ni interopérabilité binaire n'est revendiqué
  sans tests inter-projets.
- Les tests sur la PR doivent être exécutés sur le SHA exact via
  `Memorithm/RemoteOps`, conteneur isolé sans réseau, avant fusion.
