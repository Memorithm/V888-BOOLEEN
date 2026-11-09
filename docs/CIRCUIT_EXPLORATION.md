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

## Rappel associatif multi-indices (structurel, non appris)

Le module `CircuitExplorer::recall` et la commande `recall-circuits` exploitent plusieurs indices neuronaux distincts, sur au plus trois sauts. Pour chaque indice, le score d'un chemin est son minimum de contacts ; pour une cible, le chemin le plus fort de cet indice est conservé. Une cible peut être classée selon le nombre d'indices distincts qui l'atteignent, puis la somme des forces de chemins. Le rapport inclut les preuves individuelles `cue_id`, `strongest_path` et `hops`, ainsi que le nombre d'arêtes examinées.

```bash
cargo run --release -- recall-circuits --graph data/processed/brain.v888b --cues 123456,789012 --policy strongest-k --fanout 8 --hops 2 --min-cues 2 --top-k 10 --edge-budget 100000 --regions zones.csv
```

Les identifiants sont **des espaces réservés** à remplacer par des IDs réellement présents dans le graphe. La commande échoue explicitement si la limite globale d'arêtes est dépassée, si les indices sont inconnus ou dupliqués, ou si le front dépasse 100 000 cibles. `--min-cues 2` exige au moins deux indices distincts et ne fonctionne donc pas avec un seul indice (utiliser `--min-cues 1` dans ce cas). Le résultat est une **association structurelle déterministe**, pas une mémoire apprise ni une probabilité. Aucun score de qualité ni gain de performance n'est revendiqué sans comparaison sur un graphe et des objectifs préenregistrés.
