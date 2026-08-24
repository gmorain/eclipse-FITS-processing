# Pipeline d'alignement et timelapse, éclipse partielle du 12 août 2026

Traitement d'une série de FITS d'éclipse solaire partielle : centrage sous-pixel,
sélection temporelle, classement par phase, rendu timelapse couleur, rapport.

Ce document décrit le pipeline **tel que construit et mesuré**. La première
version de la spec reposait sur des hypothèses prises avant lecture des données ;
celles qui n'ont pas tenu sont signalées explicitement, avec la mesure qui les a
contredites. Ne pas les réintroduire.

## Contrainte de rendu

**Vue observateur terrestre. Aucune dérotation de champ.**

La monture est alt/az, l'orientation du capteur par rapport à l'horizon est donc
fixe pour toute la session. La transformation finale appliquée aux images est une
**translation pure**.

L'aplatissement du disque par la réfraction est un phénomène réel que
l'observateur voit. Il est conservé dans le rendu. Il n'est corrigé que
temporairement, pendant le fit géométrique, pour ne pas biaiser le centre.

## Site

**Montastruc, Hautes-Pyrénées, 43,1683 N / 0,3872 E, 485 m.** Bord de champ entre
Castelbajac et Houeydets, au nord de Lannemezan.

Le nom du dossier de session et les en-têtes FITS désignent un site mémorisé
**38 km au sud**. Les deux sont faux : l'ASIAIR a gardé une position enregistrée
au lieu de relever la sienne. L'écart déplace le maximum de 34 s et
l'obscuration maximale de 0,36 point. Les coordonnées d'en-tête ne sont pas dans
le dépôt, elles ne servaient qu'à afficher l'erreur.

Les coordonnées retenues reproduisent les circonstances calculées par Eclipsefan
(maximum 18:26:58 UTC, obscuration 98,8 %, élévation 5,9°, azimut 284,8°) à 6 s
et 0,1 point près. Les coordonnées d'en-tête ne le font pas. Captures dans
`05 - Annotations`, embarquées dans le rapport.

## Matériel et géométrie mesurée

| Paramètre | Valeur |
|---|---|
| Lunette | SkyOptic 66/400, f/6 |
| Caméra | ZWO ASI 585 MC Air, IMX585, 3840 x 2160, 2,9 µm, RGGB |
| Monture | Celestron NexStar SLT, alt/az, `ROTATOR` constant à 357 |
| Filtre | Astrosolar OD 3,8 en avant d'ouverture, plus IR-cut |
| Échelle **mesurée** | 2,9950 arcsec/px plan R, 1,4975 pleine résolution |
| Focale équivalente | 399,4 mm |
| Rayon solaire **calibré** | 316,22 px plan R, dispersion 0,01 px |
| Verticale locale **calibrée** | -178,25°, dispersion 0,52° |

Le plan R occupe le coin (0,0) de chaque cellule 2x2, vérifié sur les données.
Le code de dématriçage OpenCV correspondant est `COLOR_BayerBG2RGB`, nommé selon
une autre convention que le mot-clé FITS : vérifié, pas déduit.

## Ce que contient la session

1350 trames retenues sur 1372 (7 trames de test à 11:27 UTC avec une mise au
point différente, 15 trames au filtre L-Ultimate). Les fichiers AppleDouble `._*`
doublent le compte apparent, les écarter.

- **17:28:14 à 19:05:37 UTC**, 31 rafales de 3 à 200 trames, cadence interne
  1,3 à 4,1 s, écarts entre rafales de 30 s à 11 min.
- **Élévation 16,4° à -0,8°**, masse d'air 3,5 à 37,9. Maximum d'éclipse à 6,0°.
- Pose de 32 µs à 700 ms, gain 0 / 120 / 200 / 260, 17 valeurs de pose.
- Données 12 bits décalées de 4 bits : pas de quantification **16 ADU**,
  écrêtage observé à 64512.
- Obscuration maximale 98,85 %.

### Pièges relevés dans les données

- **Une trame porte un `DATE-OBS` décalé de +2 h exactement** (trame 31 de la
  rafale de 17:55:49). Réparée par la cadence de sa rafale.
- **La MAD des coins vaut souvent zéro.** Le fond derrière l'OD 3,8 est si noir
  que tout seuil calculé en multiples de sigma s'ouvre sur le bruit. Plancher
  obligatoire au pas de quantification.
- **Aucun recadrage manuel.** Le centre balaie 120 x 131 px sur la séance, avec
  une dérive continue de 0,31 px/s et des repositionnements entre rafales. La
  segmentation prévue sur les sauts manuels se réduit donc au découpage par
  rafale. En revanche la monture encaisse des à-coups : un saut de 45 px puis
  retour en 20 s a été observé à 18:16.
- **`EGAIN` est présent par trame** et vaut exactement 10^(gain/200) fois la
  valeur de base. Utiliser la mesure d'en-tête plutôt que la formule.
- **Nuage massif.** 320 trames sans aucun signal. La rafale du maximum
  (18:25:48, 50 trames) plafonne à un pas de quantification au-dessus du fond :
  rien à récupérer. Cause visible sur la capture VentuSky.

## Étapes

### 1. Lecture et piédestal (`io.py`)

Refroidissement désactivé, capteur en dérive lente. Ni darks ni bias : la durée
de pose décide, pas la température, et le piédestal mesuré par trame sur les
coins absorbe à la fois les changements de gain et la dérive thermique. Lire
`CCD-TEMP` pour le diagnostic.

Le champ ne fait que 53,8' de haut pour un disque de 32' : un anneau à 1,6 R ne
tient pas verticalement dès que le disque est décentré. **Utiliser les coins.**

Le bruit de fond est estimé sur les différences de pixels voisins, avec plancher
à `QUANT_ADU`. Ne jamais utiliser une MAD brute ici.

### 2. Réfraction (`astro.py`)

**Modèle de Bennett, pas la loi en cot h.** `b/a = 1 + dR/dh`, dérivée
numériquement, corrigée de la pression et de la température du site.

La formule `1 - 2.83e-4/sin²h` de la spec initiale surestime l'aplatissement de
36 % à 5° et diverge sous 1,7°. Inutilisable sur la fin de séance.

Également fournis : masse d'air de Kasten-Young, rayons solaire et lunaire,
obscuration analytique, angle de position de la lune, fraction d'arc solaire
encore visible.

### 3. Calibration géométrique (`calibrate.py`)

**Rayon** par fit d'ellipse libre sur les trames non occultées, puis figé.

**Verticale locale par l'angle de position de la lune**, pas par l'aplatissement.
La méthode de la spec initiale exige une basse élévation, or dans cette session
les basses élévations sont aussi les croissants les plus fins, où le fit
d'ellipse libre est mal conditionné. L'angle de position lunaire est connu par
éphéméride, lisible dans l'image, indépendant de la réfraction, et disponible sur
toute trame partiellement occultée.

### 4. Détection de limbe et fit (`limb.py`)

Le long de chaque rayon, ne retenir que le passage de bord **le plus externe**.
Le limbe lunaire est toujours interne, son bord contre le ciel est invisible.
Aucune classification solaire/lunaire n'est nécessaire.

**Seuil à 50 % d'une référence locale au bord, en deux passes.** Un anneau fixe à
0,85-0,95 R tombe dans l'ombre lunaire près du maximum, où le croissant ne fait
plus qu'une dizaine de pixels. Passe 1 : seuil global tiré du percentile 99,9 de
l'image entière, pas de la fraction de rayons éclairés. Passe 2 : référence
locale prise juste en dedans du bord détecté.

Modèle d'ellipse à rayon et forme figés, centre libre : deux paramètres. Le
« stretch » de la spec initiale est équivalent et inutile, on fit directement
l'ellipse de forme connue.

`limb_width` = **0,9394 A / max|dI/ds|**, FWHM d'un bord franc convolué par une
gaussienne. La mesure 10-90 % de la spec initiale traverse l'assombrissement
centre-bord et donne 60 arcsec là où la PSF en fait 6,6. Le second moment du
gradient dépend trop de la fenêtre à cause des ailes de diffusion.

`snr_disk` est estimé sur les différences de pixels voisins : l'écart-type brut
du disque mesurerait l'assombrissement centre-bord, pas le bruit.

**`sigma_center`**, incertitude 1 sigma sur le centre issue de la matrice normale
du fit, dans sa pire direction. C'est la grandeur de classement utile : elle
intègre le nombre de points, leur étalement azimutal et la dispersion des
résidus. Optimiste, les rayons voisins partageant la même turbulence.

### 5. Passe de mesure (`measure.py`)

**L'unité de travail est la rafale, pas la trame.** Le centre dérive de 0,3 px/s
et les trames sont espacées de 2 à 4 s : la trame précédente est le meilleur
point de départ possible. Sur croissant fin le vote de Hough échoue, l'amorce
temporelle non.

Passe avant amorcée, puis reprise des échecs depuis le voisin fiable le plus
proche. Une amorce n'est propagée que si le saut reste sous 3 px/s.

**Contrôle de cohérence de trajectoire.** Un fit à rayon figé sur un arc court se
cale n'importe où le long de la normale à l'arc avec un rms excellent : le rms ne
suffit pas. Trajectoire affine locale sur fenêtre glissante de 45 s, rejet des
écarts, puis reprise amorcée sur la position attendue.

**Mais un fit très bien contraint prime sur la trajectoire** (`sigma_center` <
0,05 px, `rms` < 0,6 px, arc > 120°). La monture encaisse des rafales de vent et
revient, ce qu'un ajustement affine ne sait pas représenter. Sans cette
exception, deux trames parfaites étaient remplacées par une interpolation fausse.

### 6. Classification et sélection (`select.py`)

**Classer sur `sigma_center`, pas sur N.** N est borné par la géométrie : à 94 %
d'obscuration le limbe solaire n'offre plus que 44 % de sa circonférence et une
trame parfaite n'en rend qu'une quarantaine de points. Le `N >= 200` de la spec
initiale déclassait tout le maximum.

| Classe | Critère |
|---|---|
| A | `sigma_center` < 0,10 px et `rms` < 0,6 px |
| B | `sigma_center` < 0,60 px et `rms` < 1,5 px |
| C | reste, centre interpolé dans la rafale |

**Le voile ne participe pas à la classe.** Une trame aux trois quarts voilée dont
l'arc visible donne un centre à 0,08 px reste une excellente ancre de position.
Drapeau séparé, utilisé par `rank.py`.

Détecteur de voile : **`obsc_gap`**, écart entre obscuration mesurée sur l'image
et éphéméride, c'est-à-dire la part du disque cachée par autre chose que la lune.
La transparence brute ne convient pas, elle est dominée par l'extinction
atmosphérique sur quatre ordres de grandeur.

**Programmation dynamique** pour la sélection temporelle, avec deux écarts à la
spec initiale imposés par la structure de la séance :

- **Pénalité de saut plafonnée.** Avec `μ(k-1)` non borné et un pas de 4 s,
  franchir une attente de dix minutes coûte 150 μ, davantage que tout ce que la
  suite rapporte. La version non plafonnée s'arrêtait avant le maximum.
- **Chemin reconstruit depuis la dernière trame candidate**, pas depuis le
  meilleur score cumulé, qui est toujours atteint juste avant le premier trou.

Le balayage de Δ ne tranche pas : l'objectif croît mécaniquement avec le nombre
de trames. La qualité moyenne des trames élues reste plate de 3 s à 60 s, le
choix de Δ ne fixe donc que la durée de la vidéo.

### 7. Classement par phase (`rank.py`)

Phase prise sur les éphémérides, exacte, l'obscuration mesurée servant de
contrôle et de détecteur de voile.

Tranches uniformes en obscuration jusqu'à 90 %, puis en `log10(1 - obsc)`,
branches montante et descendante séparées.

Score composite normalisé **à l'intérieur de chaque rafale**. Normaliser
globalement reviendrait à garder toutes les trames de la meilleure demi-heure.
Écart temporel minimal imposé entre trames retenues d'une même tranche.

Deux sorties, même classement : la meilleure trame par tranche, et les 15 %
meilleures de chaque rafale pour empilement AutoStakkert. Une rafale de 25 à 50
trames est un jeu de lucky imaging, pas une redondance.

### 8. Rendu (`render.py`)

Dématriçage couleur pleine résolution. **Translation et découpe carrée de 2,6 R
fusionnées en une seule passe Lanczos 4** sur l'image d'origine non étirée. Le
dématriçage ajoute un second rééchantillonnage, assumé et documenté.

Le pixel R (i, j) du plan sous-échantillonné est le photosite (2i, 2j) de la
trame complète : facteur deux, et décalage d'un demi-pixel à traiter
explicitement pour tout tracé superposé.

**Coupure du timelapse à 3,8° d'élévation.** Au-delà le disque réfracté s'écarte
de plus de 0,5 px de la meilleure ellipse et le modèle géométrique ne tient plus.
Les trames sous la coupure restent classées et exportées pour traitement
individuel.

**Normalisation d'intensité**, trois modes. `disque` par défaut : niveau
photosphérique mesuré. `instrument` garde l'extinction lisible mais noircit la
fin de séance. `aucune` est inexploitable, la pose variant d'un facteur 7000.

**Couleur unifiée.** Une balance figée ne suffit pas : l'extinction
différentielle fait glisser le rapport rouge sur vert du disque de 1,00 à 1,55
sur la séquence. En mode `unifiee`, chaque canal est ramené à son propre niveau
photosphérique mesuré sur la trame, puis la teinte cible est imposée. Sous
400 ADU par canal ou 300 pixels éclairés, retour à la balance figée : en dessous
les rapports entre canaux ne mesurent plus que le bruit.

**Accentuation** par déconvolution de Wiener calée sur la PSF mesurée :
`W = Gc Gt / (Gc² + K)`, FWHM courante prise sur le limbe de la trame, `K` tiré
du `snr_disk`. Appliquée à la luminance seule, le gain reporté sur les trois
canaux comme un rapport, ce qui conserve la teinte et évite tout liseré coloré.
Le ciel est protégé sous 10 % du niveau de disque.

Trois faits mesurés justifient le modèle : anisotropie du limbe à 6 % de la FWHM,
donc noyau gaussien isotrope ; FWHM tenant dans 6,46 à 6,70 arcsec entre déciles,
donc turbulence peu dérivante ; FWHM de 4,4 px pleine résolution contre 1,4 px de
plancher instrumental, donc marge réelle avant l'échantillonnage.

Le coût se lit en bruit, pas en rebonds : la sous-oscillation reste sous le seuil
de visibilité, le ciel étant à zéro. Cible par défaut 4,5 arcsec, soit 1,56 fois
le bruit d'origine.

### 8 bis. Tirage à l'unité (`render.py --at`, `tiff.py`)

Une trame élue de `selection_phase.csv` est un tirage, pas une image de vidéo.
Elle hérite de la séquence ce qui fait sa cohérence : normalisation
d'exposition, balance, teinte cible, déconvolution calée sur sa PSF. Elle
n'hérite d'aucun de ses compromis.

**Échelle native, jamais de mise à l'échelle.** Le disque garde 1265 px de
diamètre. Un cadre de format libre est complété en noir : le fond derrière
l'OD 3,8 est mesuré à **0 ADU exactement** au-delà de 2,6 R, à toute élévation.
Le halo de diffusion vaut 0,22 % du niveau de disque à 1,5 R et 0,06 % à 2,3 R,
il tient donc entièrement dans la découpe. Le remplissage est la valeur vraie.

**Niveau blanc 1,75, pas 1,05.** Le 1,05 de la vidéo écrête jusqu'à 7 % des
pixels : la référence de normalisation est la médiane du disque, or le centre
la dépasse. Mesuré sur 14 trames de classe A, le maximum de la séance atteint
1,58 en balance figée et 1,61 en couleur unifiée.

**Courbe sRGB et profil ICC embarqué.** Le linéaire 16 bits dépense sa
résolution dans les hautes lumières et sort noir dans un logiciel qui suppose
du sRGB. OpenCV n'embarque aucun profil et Pillow n'écrit pas de RGB 48 bits,
d'où l'écriture TIFF de base dans `tiff.py`. Vérifié relu par OpenCV, Pillow et
ColorSync.

Le seuil de protection du ciel de l'accentuation est une fraction du niveau de
disque, lequel vaut `1 / white` : il est divisé par `white`, sinon il triple
quand la dynamique s'ouvre.

### 9. Contrôles et rapport (`diagnose.py`, `report.py`, `html.py`)

Rapport HTML autonome, images en base64, aucune dépendance externe. `--leger`
recode en JPEG les images que cela allège, 2,6 Mo au lieu de 7,0 : les planches
de trames y gagnent un facteur trois, les courbes de contrôle doublent et
restent donc en PNG. La version publiée sur GitHub Pages est dans `docs/`.
Sections :
résumé, site et conditions, configuration, courbes, planches, netteté, couleur,
sélection par tranche, couverture par rafale, anomalies, limites.

Planches : sélection par tranche de phase, croissants fins au-delà de 85 %,
contrôle des pires rms et trames écrêtées, modes de couleur, réglages de teinte,
effet de l'accentuation.

Sur les vignettes, le cercle ajusté est retracé dans le repère capteur d'origine
et apparaît donc comme une **ellipse**. C'est le contrôle visuel du modèle de
réfraction. Les points rejetés par sigma-clip sont le diagnostic le plus utile :
un paquet d'un seul côté signale un biais systématique, pas du bruit.

Les captures de `05 - Annotations` sont réduites et embarquées en JPEG.

## Structure

```
config.py    constantes de session, seuils, chemins lus dans .env
io.py        lecture FITS, canal R, piédestal, rafales, réparation d'horodatage
astro.py     éphémérides, Bennett, aplatissement, masse d'air, arc visible
calibrate.py rayon solaire et verticale locale, mesurés sur les données
limb.py      profils radiaux, points de limbe, fit à rayon figé, métriques
measure.py   passe complète, une rafale par tâche, cohérence de trajectoire
select.py    classes A/B/C, interpolation des centres, sélection temporelle
rank.py      phase, score composite, top-k par tranche, listes d'empilement
render.py    dématriçage, translation, normalisation, couleur, accentuation,
             tirage à l'unité sur cadre libre
tiff.py      écriture TIFF 16 bits, courbe et profil sRGB embarqués
texte.py     titres et étiquettes, contours de police remplis par OpenCV
compose.py   composition, disque central et couronne des phases
diagnose.py  courbes de contrôle
report.py    vignettes annotées, planches, annotations de session
html.py      rapport autonome
recover.py   developpement d'une trame unique, hors chaine
__main__.py  chaîne complète
```

Dépendances : `numpy`, `scipy`, `astropy`, `matplotlib`, `opencv-python`. `uv`
pour l'environnement, `ruff` pour lint et format.

### 10. Trame isolee (`recover.py`)

Une trame de coucher derriere un premier plan n'est pas une trame de mesure et
sort de la chaine. `render --at` la refuse : la passe de mesure la classe noire. Le fit parametrique y echoue, le disque etant tronque par la
vegetation : 6 rayons valides sur 720.

**Une seule pose, pas d'empilement.** La rafale gagne bien la racine du nombre
de trames sur le bruit du disque, verifie a 2,44 fois pour six trames, mais le
soleil descend de 13 arcsec/s derriere un premier plan fixe au sol. Aligner sur
le soleil brouille la haie, aligner sur la haie brouille le limbe : les deux
sujets ne peuvent pas etre nets dans un meme empilement.

Tout le travail porte donc sur le bruit, ce que le contenu autorise :

- le disque a 38 masses d'air n'a aucun contenu haute frequence, seules la haie
  et la morsure lunaire en ont : un lissage a preservation de bords y est quasi
  sans perte ;
- la chrominance ne porte aucune information spatiale, 5 % de bleu pour 100 %
  du bruit : elle se lisse largement, en rapports au signal pour traverser la
  courbe de tonalite sans deriver ;
- la luminance optimale n'est pas la luminance video mais la somme ponderee par
  le signal de chaque canal, le bruit de lecture etant identique sur les trois.
  Mesuree ici a 0,74 / 0,21 / 0,05.

Normaliser sur le canal dominant et non sur cette luminance : a 38 masses d'air
le rouge vaut 1,24 fois la luminance ponderee, et normaliser dessus ecrete le
disque en aplat rouge, sans assombrissement centre-bord ni gradient
d'extinction.

### 11. Composition (`compose.py`)

Disque plein au centre, couronne des trames de `selection_phase.csv` autour.

**Branche montante seule.** Elle est complète, neuf tranches sans trou, de 0 à
90,8 % d'obscuration, sur 16,4° à 6,7° d'élévation. La descente est amputée de
cinq tranches par le nuage et s'arrête à 0,27° : elle déséquilibrerait la
couronne sans rien apporter.

**Espacement angulaire régulier.** Les tranches sont uniformes en obscuration,
pas en temps : les intervalles réels vont de 42 s à 11 min. Un espacement
proportionnel serait fidèle et illisible.

**Disque central pris sur `metrics.csv`, pas sur la sélection.** Cinq trames
non occultées de classe A, toutes de la rafale de 17:28. La meilleure au limbe
donne 6,80 arcsec contre 6,87 pour celle qu'élit `rank`, dont le score composite
sert un autre but.

**Couleur unifiée imposée**, quel que soit le dernier rendu de série. La
composition juxtapose des trames prises entre 16,4° et 6,7° d'élévation : c'est
exactement le cas pour lequel ce mode existe.

**Halo fondu au noir de 1,10 à 1,28 rayon solaire.** Le halo de diffusion vaut
2 % de l'échelle au bord de la découpe, qui s'arrête à 1,30 R. Réel, mais sur
fond noir il ne se lit pas comme un halo, il se lit comme le carré de la
découpe. `--sans-fondu` le conserve.

**Teinte propre au disque central**, 1,5 contre 1,0 pour les vignettes. Elle
passe par deux chemins selon le mode : cible imposée à chaque canal en couleur
unifiée, balance elle-même en balance figée, où `target` n'est pas lu. Les deux
sont fournis, sans quoi le réglage reste sans effet en mode `fixe`. La
normalisation unifiée cale la médiane du disque sur la cible : sur un disque
plein, tout l'intérieur est plus lumineux que cette médiane, et la courbe sRGB
étant concave, un même rapport linéaire y rend moins de saturation. Le centre
sortait crème là où un croissant sort jaune. Ce n'est pas un défaut de mesure,
c'est la cible qui est définie sur une médiane.

**Deux échelles indépendantes**, rapportées à la découpe native et non l'une à
l'autre : réduire le disque central rapproche la couronne sans toucher à la
taille des vignettes.

**Cadre libre**, comme pour un tirage : la composition n'est pas redimensionnée
pour y entrer, le cadre est complété en noir et rogné avec avertissement. Le
rayon de la couronne est imposé par le disque central, 1644 px à l'échelle
pleine. Un cadre 3456 x 2234 tient à 0,5 de centre et 0,28 de vignette.

Texte rastérisé depuis les contours de la police matplotlib, remplis par
OpenCV en suréchantillonnage 4x (`texte.py`) : pas de moteur de rendu de texte
ajouté à la chaîne, et des sorties entièrement 16 bits, ce qu'un détour par une
figure matplotlib interdirait. Deux blocs de texte ne s'alignent que par leur
ligne de base, leur boîte englobante dépendant des accents et des jambages
qu'ils portent.

La légende occupe les deux coins supérieurs, première ligne à gauche et suite à
droite. En un seul bloc elle touchait les étiquettes de la vignette de midi.

Le tirage à l'unité partage le vocabulaire mais pas le partage : la légende y
tient entièrement à gauche, un saut de ligne passant à la ligne, et les
étiquettes occupent le coin droit. Une composition a la largeur pour porter un
titre étalé sur les deux coins, un tirage carré non. Les deux sont optionnels.

## Usage

```bash
uv run python -m eclipse --delta 8
uv run python -m eclipse.render --nettete 4.5 --tint 1.0 --couleur unifiee
```

## État actuel

| | |
|---|---|
| Trames mesurées | 793 sur 1350 |
| Échecs | 320 trames noires, 119 fits divergents, 115 centres incohérents, 3 limbes insuffisants |
| `rms` | médiane 0,253 px, p90 0,599 px |
| `sigma_center` | médiane 0,025 px |
| `limb_width` | médiane 6,63 arcsec |
| Classes sur la plage timelapse | A 589, B 150, C 433 |
| Trame exploitable la plus profonde | 97,12 % d'obscuration |
| Timelapse | 268 trames, 10,7 s à 25 im/s |

## Contrôles à tracer systématiquement

- `N(t)` avec la courbe de fond géométrique : les chutes brutales sont les nuages
- `sigma_center(t)` par classe : c'est le critère de classement
- `limb_width(t)` brut et normalisé, avec l'élévation en second axe
- Trajectoire du centre en x/y par rafale
- Obscuration mesurée contre éphéméride : l'écart est le voile
- Écarts à la grille temporelle des trames retenues
- `transparency(t)` et `CCD-TEMP(t)`, pour le diagnostic seulement
