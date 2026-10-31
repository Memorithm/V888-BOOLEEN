# Conversion intégrale compressée V888 — flux Rust standard

Le résultat de l'audit RemoteOps #41238883869 établit qu'un fichier binaire non compressé couvrant les 30 colonnes du snapshot GCS-v2 est estimé à **46 012 002 798 octets** (heuristique), alors que le Dell dispose de **54 390 063 104 octets** libres et doit conserver une réserve de **10 737 418 240 octets**. La sortie non compressée n'est donc pas admise sur ce volume.

Le module Rust accepte désormais `pack-all --output -` : écrit exclusivement les **octets binaires V888C30A** sur stdout (messages opératoires sur stderr), permettant une compression sans perte en flux :

```bash
v888-booleen pack-all --expected-rows 155247357 --output - < exact-records.bin | gzip -1 > brain-all30.v888all.gz
gzip -dc brain-all30.v888all.gz | v888-booleen validate-all --file -
```

Il n'est pas nécessaire de créer un fichier binaire complet intermédiaire. Les bits IEEE-754 et les octets UTF-8 ne sont pas modifiés par gzip ; le décodage Rust valide les 30 colonnes. La compression est une transformation d'archivage et ne constitue pas un calcul de réseau neuronal.

**Mesure préalable obligatoire :** fixer une limite stricte de taille de fichier sur la sortie compressée, proportionnelle à l'espace libre au moment de l'exécution, et refuser si la réserve de 10 Gio ne peut plus être préservée. Un score de compression supposé ne suffit pas à garantir que le fichier pourra être achevé.

Une archive compressée ne prouve pas à elle seule l'exactitude face au Parquet : il faut encore comparer l'empreinte du corps V888C30A décompressé à celle des 30 champs extraits bit-à-bit des buffers Arrow, et contrôler le total de 155 247 357 lignes sans anomalie.
