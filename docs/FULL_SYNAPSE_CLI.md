# V888 — commandes du codec intégral de 30 champs

Ces commandes Rust ne nécessitent aucune bibliothèque externe, aucun processeur graphique et ne conservent qu'une ligne synaptique en RAM :

```bash
cargo run --release -- pack-all --expected-rows 155247357 --output all.v888all < exact-records.bin
cargo run --release -- all-info --file all.v888all
cargo run --release -- validate-all --file all.v888all
```

Le flux de `pack-all` doit respecter strictement le schéma défini dans `docs/FULL_SYNAPSE_FORMAT.md` ; il contient les 30 champs du Parquet dans leur ordre scientifique. Tous les champs NULL sont explicitement codés, les doubles sont leurs **bits IEEE-754 inchangés**, les chaînes sont des octets UTF-8 exacts et l'entier est signé.

`validate-all` décode tous les enregistrements, rejette les troncatures, valeurs mal formées et octets résiduels. Le binaire n'est jamais écrasé : `create_new(true)` empêche la réutilisation d'un nom déjà présent et une conversion qui échoue supprime son fichier incomplet.

**Ce contrat n'autorise pas à déclarer la conversion du connectome complet déjà terminée** : il faut qualifier un adaptateur qui restitue les vrais buffers de données Arrow de façon bit-exacte, puis mesurer la capacité disque et effectuer un retour inverse indépendant. L'exécution biologique ne se déduit pas de la conservation des données anatomiques.
