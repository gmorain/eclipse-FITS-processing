# Détails

Le README couvre l'objectif, les prérequis, l'installation et les exemples. Ce
fichier documente la chaîne étape par étape, les options de chaque type de
sortie et la structure du dépôt.

## Chaîne complète

```
uv run python -m eclipse --delta 8
```

Calibration si `calibration.json` manque, mesure, courbes de contrôle, sélection
au pas de `--delta` (écart visé entre deux trames retenues, mesuré sur DATE-OBS), classement par phase, planches, **rendu du timelapse**, puis
rapport HTML. La vidéo est la sortie principale de la commande, elle est écrite
dans `out/timelapse/`. `--skip-render` s'arrête avant elle et ne produit que
l'analyse, `--recalibrate` refait la calibration. `--size`, `--tint`, `--couleur`
et `--nettete` sont passés au rendu. `--delta 8` cherche une trame toutes les 8 s d'observation, en tolérant de sauter un ou plusieurs intervalles quand un nuage a tout mangé.

Étapes individuelles :

```
uv run python -m eclipse.measure --workers 8    # mesure géométrique et photométrique
uv run python -m eclipse.diagnose               # courbes de contrôle
uv run python -m eclipse.select --delta 8       # sélection temporelle, pas visé en secondes
uv run python -m eclipse.rank                   # classement par phase, listes d'empilement
uv run python -m eclipse.render --tint 0.5      # rendu couleur, teinte 0 = disque blanc, 1 = jaune solaire
uv run python -m eclipse.html                   # rapport autonome, --leger pour publier
```

Le chemin de la session vient de `ECLIPSE_SESSION_DIR`, lu dans `.env` à la
racine du dépôt ou dans l'environnement du processus, qui prime. Absent ou
invalide, la chaîne s'arrête sur un message unique, avant toute lecture. Les
seuils, les constantes de site et de matériel restent dans `eclipse/config.py`.

## Sorties à l'unité

Trois cas distincts, sortie dans `out/single/`.

### Prérequis

Un tirage ne relit pas la séance, il reprend ce que la passe d'analyse a écrit
dans `out/analysis/`. Il faut donc l'avoir jouée une fois.

| Sortie | Exige | Reprend si présent |
|---|---|---|
| `render --at` | `calibration.json`, `metrics.csv` | `render.json`, `white_balance.json` |
| `compose` | les mêmes, plus `selection_phase.csv` | `render.json`, `white_balance.json` |
| `recover` | rien | rien |

Le plus court chemin, qui produit tout :

```
uv run python -m eclipse --skip-render
```

`calibration.json` n'a pas d'entrée en ligne de commande à lui : il est écrit par
`python -m eclipse` quand il manque, ou réécrit avec `--recalibrate`.
`selection_phase.csv` vient de `python -m eclipse.rank`, qui n'exige que
`metrics.csv`. Un fichier exigé qui manque arrête la commande sur une trace
`FileNotFoundError` nommant le fichier, il n'y a pas de contrôle préalable.

Les deux fichiers facultatifs ne bloquent rien mais changent le résultat.
`render.json` porte la normalisation, la balance, la teinte et la netteté
établies sur les 1350 trames : sans lui le tirage part des valeurs par défaut de
la ligne de commande et ne partage plus la cohérence de la vidéo.
`white_balance.json` est recalculé au vol s'il manque, ce qui relit quelques
trames sur le disque de la session.

### Tirage d'une trame de la série

Export centré et calibré pour finition dans Photoshop, DxO Photolab ou Nik Collection : 
histogramme, colorimétrie, traitement du bruit et des détails. 
La trame est désignée par son heure UTC ou par un fragment de nom de fichier.

```
uv run python -m eclipse.render --at 182548
uv run python -m eclipse.render --fichier _0021 --nom essai --tint 0 --nettete 4.0
uv run python -m eclipse.render --at 180010 --toile 3456x2234
uv run python -m eclipse.render --at 180010 --legende --etiquettes
uv run python -m eclipse.render --at 180010 --legende "12 août 2026\nMontastruc (65)"
```

`--legende` occupe le coin supérieur gauche, un saut de ligne y passant à la
ligne. Sans valeur, c'est « 12 août 2026 » puis « Montastruc (65) ». Un `\n`
tapé en ligne de commande fait le saut.

`--etiquettes` pose les circonstances de la trame dans le coin supérieur droit,
calées à droite et en plus petit : heure UTC, obscuration et élévation. Les deux
blocs partagent une ligne de base. La légende est éditoriale, les étiquettes
sont mesurées : d'où deux réglages et non un, tous deux optionnels.

Hérite de la série ce qui fait sa cohérence : normalisation d'exposition mesurée
sur le disque, balance et teinte cible, déconvolution calée sur la PSF de la
trame. Ces grandeurs sont établies sur les 1350 trames, elles sont reprises de
`out/analysis/render.json`, écrit par le dernier rendu de série. Les arguments passés ici
surchargent.

N'hérite pas des compromis de la vidéo :

- **échelle native, aucune mise à l'échelle.** Le disque garde ses 1265 px de
  diamètre, à 1,4955 arcsec/px. `--toile LxH` donne un cadre de format libre,
  centré sur le disque par défaut, soit le même alignement que le timelapse.
  `--pos fx,fy` déplace le centre du disque en fractions du cadre. Ce que le
  capteur ne couvre pas est rempli de noir, qui est la valeur mesurée du ciel
  derrière l'OD 3,8 au-delà de 2,6 R, pas un bouchon ;
- **aucun écrêtage.** Le niveau blanc passe de 1,05 à 1,75. Mesuré sur 14 trames
  de classe A, le pixel le plus brillant de la séance atteint 1,61 fois le
  niveau du disque : la vidéo en écrête jusqu'à 7 % ;
- **16 bits, courbe et profil sRGB embarqués.** Le TIFF est lu tel quel par DxO
  et Nik, l'export JPEG se fait là. Le PNG à côté n'est qu'un aperçu de tri.

L'heure est celle de `DATE-OBS`, en UTC, pas celle du nom de fichier, en avance
de deux heures. Une trame noire ou sans centre récupérable est refusée plutôt
que rendue de travers, `--centre x,y` force la position en pixels pleine
résolution. Sous 3,8° d'élévation le modèle géométrique ne tient plus : la trame
sort, avec un avertissement.

### Une trame hors chaîne (non couvert par les mesures)

Le fit paramétrique y échoue, le disque étant tronqué par la végétation, des nuages ou 
la déformation du disque solaire trop bas sur l'horizon. Aucun
centrage, aucun réglage partagé avec la série : normalisation sur le canal
dominant de la trame, débruitage à préservation de bords.

```
uv run python -m eclipse.recover chemin/vers/trame.fit
uv run python -m eclipse.recover chemin/vers/trame.fit --nom coucher_21h04
```

`--nom` suit la convention des autres modules : sortie dans `out/single/`, un
`.tif` 16 bits à profil sRGB embarqué, un `.png` d'aperçu et un `.json` de
réglages. Sans lui, le nom est celui de la trame.

Champ entier par défaut, 3840 x 2160, soit 49,8 Mo de TIFF non compressé.
`--crop x0,y0,x1,y1` découpe en pixels pleine résolution, sans mise à l'échelle.

La courbe de tonalité de `--gamma` porte déjà les valeurs dans le domaine
d'affichage : le profil sRGB embarqué ne déplace aucun pixel, il rend explicite
l'hypothèse sous laquelle le réglage a été fait, tout visualiseur lisant un
fichier sans profil comme du sRGB.

### Une composition

Disque plein au centre et couronne des phases autour, tirée de
`selection_phase.csv`.

```
uv run python -m eclipse.compose
uv run python -m eclipse.compose --branche toutes --echelle 0.28 --sans-etiquettes
uv run python -m eclipse.compose --toile 3456x2234 --echelle-centre 0.5 --echelle 0.28
uv run python -m eclipse.compose --legende
uv run python -m eclipse.compose --legende "Autre titre"
```

`--legende` pose un titre dans les deux coins supérieurs : la première ligne à
gauche, la suivante à droite, calée sur le bord. Sans valeur, c'est « Eclipse
solaire du 12 août 2026 » et « Montastruc (65) ». En un seul bloc, le titre
viendrait toucher les étiquettes de la vignette de midi. Un `\n` tapé en ligne
de commande fait la séparation.

Branche montante par défaut : elle est complète, neuf tranches sans trou, de 0 à
90,8 % d'obscuration. La descente est amputée de cinq tranches par le nuage et
s'arrête à 0,27° d'élévation. Le disque central est la meilleure trame non
occultée de la séance, pas celle de la tranche 0.

Espacement angulaire régulier et non proportionnel au temps : les tranches sont
uniformes en obscuration, les intervalles réels vont de 42 s à 11 min.

Deux échelles, toutes deux en fraction de la découpe native : `--echelle` pour
les vignettes, `--echelle-centre` pour le disque central, 1,0 par défaut. Elles
sont indépendantes, réduire le centre rapproche la couronne sans toucher à la
taille des vignettes. C'est le seul endroit de la chaîne où le soleil est mis à
l'échelle.

Couleur unifiée par défaut, quel que soit le dernier rendu de série : la
composition juxtapose des trames prises entre 16,4° et 6,7° d'élévation, où
l'extinction différentielle fait glisser le rapport rouge sur vert de 1,00 à
1,55. Le halo de diffusion est fondu au noir à partir de 1,10 rayon solaire,
sans quoi le carré de la découpe se lit sur le fond. `--sans-fondu` le conserve.

Le disque central reçoit sa propre teinte, `--tint-centre`, 1,5 par défaut
contre 1,0 pour les vignettes, dans les deux modes de couleur. La normalisation
cale la médiane du disque sur la teinte cible : sur un disque plein tout
l'intérieur est plus lumineux que cette médiane, et la courbe sRGB étant
concave, un même rapport linéaire y rend moins de saturation. Sans ce réglage le
centre sort crème là où un croissant sort jaune. Au-delà de 1,75 le bleu tombe
sous 0,16 et la couleur sonne faux.

`--toile LxH` et `--pos fx,fy` fonctionnent comme pour un tirage : la
composition n'est pas remise à l'échelle pour entrer dans le cadre, celui-ci est
complété en noir et ce qui déborde est rogné, avec un avertissement. Le rayon de
la couronne est imposé par le disque central : à `--echelle-centre 1`, il fait
1644 px et un cadre plus court qu'environ 2900 px ne suffit pas. Un fond d'écran
3456 x 2234 tient avec `--echelle-centre 0.5 --echelle 0.28`.

## ffmpeg hors du PATH

La vidéo n'est assemblée que si `ffmpeg` est présent dans le `PATH` du processus
Python, testé par `shutil.which` au moment du rendu. Un ffmpeg installé mais
absent de ce `PATH`, cas d'un lancement depuis un service ou un cron à
environnement réduit, revient au même qu'un ffmpeg absent : le rendu de série va
jusqu'au bout, laisse les trames dans `out/timelapse/frames` et avertit qu'il
n'a produit
aucun mp4. Les autres sorties n'en dépendent pas.

## Modules

| | |
|---|---|
| `io.py` | lecture FITS, canal R, piédestal, rafales, réparation d'horodatage |
| `astro.py` | éphémérides, réfraction de Bennett, aplatissement, masse d'air |
| `calibrate.py` | rayon solaire et verticale locale, mesurés sur les données |
| `limb.py` | profils radiaux, points de limbe, fit à rayon figé, métriques |
| `measure.py` | passe complète, une rafale par tâche, cohérence de trajectoire |
| `select.py` | classes A/B/C, interpolation des centres, sélection temporelle |
| `rank.py` | phase, score composite, top-k par tranche, listes d'empilement |
| `render.py` | dématriçage, translation, normalisation, vidéo, tirage à l'unité |
| `compose.py` | composition, disque plein central et couronne des phases |
| `texte.py` | titres et étiquettes, contours de police remplis par OpenCV |
| `tiff.py` | écriture TIFF 16 bits, courbe et profil sRGB embarqués |
| `diagnose.py` | courbes de contrôle |
| `report.py` | vignettes annotées et planches |
| `html.py` | rapport autonome, images en base64 |
| `recover.py` | développement d'une trame hors chaîne, sans mesure ni centrage |

## Sorties, dans `out/`

Quatre dossiers, rien à la racine. `ECLIPSE_OUT_DIR` déplace l'ensemble.

```
out/
  analysis/   metrics.csv                 une ligne par trame
              timelapse.csv               sélection et translations
              selection_phase.csv         trame élue par tranche de phase
              empilement_par_rafale.txt   listes pour AutoStakkert
              anomalies.txt
              calibration.json            rayon solaire et verticale locale
              white_balance.json          balance mesurée sur les classes A
              render.json                 réglages du dernier rendu de série
  timelapse/  <nom>.mp4                   `--nom` renomme la vidéo
              frames/                     trames intermédiaires, purgées à chaque rendu
  single/     tirages, compositions et développements hors chaîne
  report/     rapport.html                rapport autonome
              planche_*.png, controle*.png, annotation_*.jpg
```

Les images de `report/` sont produites séparément puis embarquées en base64 dans
le rapport. Elles y restent lisibles seules.
