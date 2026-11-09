# B888-3 — Exploration sélective et télémétrie des circuits (prototype Rust)

Cette fonctionnalité exécute plusieurs hypothèses de **routage**, sans modifier le graphe BANC V888 ni prétendre reproduire la physiologie de la drosophile.

## Politiques de connexion

- `directed` : toute arête orientée originale est disponible, avec son nombre de contacts inchangé, y compris les doublons et les comptes nuls (signal nul).
- `strongest-k` : index construit une seule fois en conservant, par source, les `K` arêtes aux nombres de contacts les plus élevés. Les égalités sont départagées par l'indice cible puis l'ordre d'origine. **La construction lit toutes les arêtes**, mais une propagation n'examine que les arêtes présélectionnées des neurones actifs.
- `symmetric-hypothesis` : active aussi le trajet inverse des arêtes, en mémoire uniquement. **Ces arêtes inverses ne sont pas des connexions anatomiques attestées.** Les boucles ne sont pas doublées. Ce mode exige au plus 30 millions d'arêtes en entrée pour borner son index supplémentaire.

Les trois stratégies utilisent une règle synchrone, positive et expérimentale : chaque neurone actif envoie un signal égal au `count` de chaque arête retenue ; un neurone cible s'active si la somme de ses signaux atteint le seuil. Les additions et comparaisons passent par les portes booléennes existantes. Ni inhibition, ni délai, ni signe neurotransmetteur n'est déduit des données.

## Instrumentation

À chaque cycle, le CLI publie : nombre de neurones actifs, nouvelles activations, désactivations, sources parcourues, arêtes transmises, somme des contacts transmis, distribution des activations par région et échantillon borné à 16 identifiants originaux. La distribution régionale exige **une table d'annotations explicite** ; les neurones non annotés restent `UNMAPPED`. Aucune zone biologique n'est devinée.

La table CSV facultative respecte exactement ce format :

```csv
neuron_id,region
100,A
200,B
```

Les identifiants doivent réellement appartenir au graphe et rester uniques. Les noms de région sont alphanumériques (tirets et underscores acceptés), de longueur maximale 64 octets.

## Commandes

```bash
cargo run --release -- explore-circuits --graph data/processed/brain.v888b --seed 123456 --policy directed --steps 3
cargo run --release -- explore-circuits --graph data/processed/brain.v888b --seed 123456 --policy strongest-k --fanout 8 --threshold 2 --steps 3 --regions zones.csv
cargo run --release -- explore-circuits --graph data/processed/brain.v888b --seed 123456 --policy symmetric-hypothesis --steps 3
```

Remplacer `123456` par un ID effectivement présent dans le graphe, sans en inventer un. Exemple de région ci-dessus : **schéma uniquement**, pas des annotations V888 vérifiées.

## Qualification expérimentale attendue

Comparer sur le **même SHA de connectome, les mêmes graines et le même budget CPU** les trois politiques : taux d'activation, couverture de régions annotées, stabilité au fil des cycles, temps de préparation, temps par cycle, mémoire additionnelle et nombre d'arêtes réellement lues. Confronter les résultats aux mesures de récupération associative et de plasticité livrées séparément. Aucune métrique physique n'est prétendue avant mesure par RemoteOps sur Dell/Thor.

Cette tranche B888-3 est indépendante du chantier de plasticité B888-4 attribué à Grok.
