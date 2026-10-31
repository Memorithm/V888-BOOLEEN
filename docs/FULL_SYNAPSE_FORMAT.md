# V888C30A — contrat de représentation exacte des 30 champs individuels

L'audit Parquet réel du snapshot BANC V888 GCS-v2 (SHA-256 `e75d1430089a55ef0e2eb40ba88803d0ddc59175c98af03ba78ef58d26832092`) a établi 155 247 357 lignes avec 30 colonnes scalaires, dont 10 champs de texte, 19 colonnes `double` et une étiquette `int32`.

Ce format Rust est **un codec expérimental de toutes ces valeurs**, distinct de la projection de deux identifiants `V888SYB1`. Le tableau constant `COLUMNS` fige le nom, l'ordre et la nature de chaque colonne observée ; toute variation de schéma nécessite une version différente.

En-tête : huit octets ASCII `V888C30A`, version `u64` en little-endian égale à 1 et nombre de lignes `u64`. Pour chaque ligne, les 30 bits de présence sont inscrits dans un masque `u32` ; chaque champ présent est ensuite stocké dans l'ordre source. Les champs `string` sont des octets UTF-8 avec taille `u32`; les `double` sont les **64 bits bruts IEEE-754**, sans aucun calcul flottant, ni arrondi, ni quantification (y compris les NaN/payload et -0.0) ; `label` est un entier signé `i32` sur quatre octets. Le codec vérifie les types, l'UTF-8, les tailles et les masques inconnus.

Les champs NULL restent explicitement NULL, sans être transformés en zéro ou en chaîne vide. Le codec préserve l'ordre et les duplications des lignes, et ne suppose aucune explication biologique. Il procède par flux borné ; il ne charge pas toutes les synapses en RAM.

**État : code et tests du codec.** La conversion à grande échelle des 155 247 357 lignes vers ce nouveau format n'est PAS encore effectuée. Il reste à ajouter un adaptateur PyArrow garantissant **le transfert exact des buffers de bits** sans conversion via les flottants Python, mesurer le débit et vérifier un roundtrip intégral contre le Parquet original. Pour tous ces contrôles, le fichier original reste conservé à l'identique hors GitHub.

Cette représentation n'est pas une modélisation physiologique du cerveau : seule la fidélité des données peut être prouvée au niveau du codec.
