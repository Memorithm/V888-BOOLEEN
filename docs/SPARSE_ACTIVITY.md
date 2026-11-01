# V888 — accélération des circuits booléens à activation parcimonieuse

Le graphe BANC V888 GCS-v2 contient **188 508 neurones** et **11 752 828 connexions agrégées**. Le premier mécanisme à coefficients de 4 bits scannait chaque neurone source, même lorsque son état était nul.

La méthode `BooleanState::active_indices()` parcourt directement les bits à 1 des mots `u64` par `trailing_zeros` et retrait du bit le moins significatif. La méthode ne fait aucune allocation de tableau d'indices et conserve l'ordre croissant. `PlasticCircuit::train` et `predict` visitent ensuite uniquement les arêtes sortantes de ces neurones actifs. La fonction de plasticité, la quantification 4 bits et les seuils restent strictement inchangés.

Les tests comparent le parcours parcimonieux au parcours exhaustif pour différentes tailles (jusqu'aux 188 508 indices), ainsi que les coefficients entraînés et les états résultants sur un graphe de 130 neurones franchissant plusieurs frontières de mots, pour cinq seuils. Une comparaison de performance sur le Dell devra être effectuée sur **la même architecture, le même graphe SHA-pinné et le même protocole** avant toute déclaration de gain de temps.

Cette optimisation est potentiellement transférable aux états booléens de SML-GENIUS, Scirust et NNIS, sous réserve d'une PR et de tests propres à chaque dépôt. La représentation des 30 attributs anatomiques et le protocole scientifique de généralisation restent indépendants de cette optimisation.
