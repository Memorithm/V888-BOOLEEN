# Agrégation par runs CSR du moteur événementiel

**Portée :** optimisation du noyau commun de propagation asynchrone,
sans nouvelle règle neuronale, aucune suppression de ligne anatomique,
aucune modification des fichiers sources V888.

## Problème mesuré

La campagne physique [RemoteOps #44258420442](https://github.com/Memorithm/RemoteOps/actions/runs/44258420442)
a comparé trois ordonnanceurs sur le Dell. Sur `heavy-disjoint`
(512 sources, 512 contacts chacune), les médianes release étaient
d'environ 4,4 ms pour `serial` et 24,3 ms pour `disjoint` en
mode à quatre workers. Cette campagne a obtenu l'égalité fonctionnelle,
mais ne prouve pas les gains énergétiques (capteurs hôte partagé).

Les contacts CSR consécutifs ayant la même destination provoquaient
auparavant autant de recherches dans `BTreeMap<target, u128>` que de
contacts, même si tous visaient un seul neurone. Le format importé par
`Connectome::import` trie explicitement les lignes par
`(source, destination, count)`, donc ces runs peuvent aussi exister
dans les données scientifiques sans que leur fréquence soit connue.

## Algorithme exact

`src/event_runtime/contacts.rs::aggregate_outgoing` parcourt
chaque ligne de contact. Il accumule les poids `u64` non nuls d'un
même run de destination dans un accumulateur `u128` temporaire,
puis effectue **une seule insertion/recherche BTreeMap par run**
au lieu d'une insertion par contact.

Lorsque les destinations alternent ou ne sont pas triées, le code
conserve une accumulation ordonnée par destination (BTreeMap) et
la même somme totale ; une destination peut donc apparaître dans
plusieurs runs et ses sommes se combinent. Les poids nuls ne créent
aucun réveil. Les additions restent vérifiées par `checked_add`.

La sortie `Vec<(target, weight)>` est dans l'ordre croissant des
destinations comme auparavant. Les budgets, l'ordre des événements,
les compteurs `EventWork.contacts_examined`, la coalescence des
réveils et les critères de dépassement arithmétique sont conservés.

Ce noyau unique est utilisé par `EventRuntime::consume_one` et par
le préflight parallèle `prepare_one` ; le commit parallèle utilise
déjà ce même préflight et en bénéficie donc indirectement.

## Tests et qualification

Un oracle indépendant somme chaque ligne avec la stratégie
historique. Des tests couvrent les destinations triées et non triées,
les répétitions, les poids nuls, les valeurs `u64::MAX` et les
répétitions déterministes. Un test de bout en bout compare les trois
moteurs sur des drives successifs avec contacts répétés et alternés.

Qualification à exiger sur le SHA exact :
`cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`,
et la matrice de benchmarks release de RemoteOps.

**Aucune accélération n'est présumée.** Le changement évite des
opérations logicielles BTreeMap sur des runs, mais n'élimine ni la
lecture des contacts CSR, ni les accès mémoire, ni la contention du
serveur Dell. L'énergie hôte RAPL n'est pas attribuable uniquement à
ce processus. Les implications pour FLAT-ATTENTION et SML-GENIUS
restent des pistes à qualifier séparément.
