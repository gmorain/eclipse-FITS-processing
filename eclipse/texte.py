"""Rendu de texte sur les images, partage par les sorties a l'unite.

Le texte est rasterise depuis les contours de la police matplotlib, remplis par
OpenCV a `SS` fois la taille finale puis reduits : l'antialiasing vient de la
reduction. Aucun moteur de rendu de texte n'est ajoute a la chaine, et la
composition reste entierement en 16 bits, ce qu'un detour par une figure
matplotlib interdirait.
"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

SS = 4  # surechantillonnage avant reduction
INTERLIGNE = 1.45
POLICE = FontProperties(family="DejaVu Sans")

# Reperes de mise en page, en fraction du petit cote de l'image.
MARGE_FRAC = 0.035
LEGENDE_FRAC = 0.022
ETIQUETTE_FRAC = 0.016


def _polys(txt: str, px: float, dy: float) -> list[np.ndarray]:
    """Contours d'une ligne de texte, en coordonnees police, ligne de base a dy."""
    # TextPath lache sur une chaine vide, qu'une ligne blanche dans un titre
    # produit naturellement
    if not txt.strip():
        return []
    tp = TextPath((0.0, dy), txt, size=px, prop=POLICE)
    return [p for p in tp.to_polygons() if len(p) >= 3]


def bloc(lignes: list[str], px: float, align: str = "gauche") -> tuple[np.ndarray, float]:
    """Masque alpha d'un bloc de texte, entre 0 et 1, et sa ligne de base.

    Les lignes sont posees sur une ligne de base commune et non sur leur boite
    englobante, sinon une virgule descendante decalerait toute une ligne. La
    ligne de base renvoyee, comptee depuis le haut du masque, sert a aligner
    deux blocs differents : leurs boites englobantes dependent des accents et
    des jambages qu'ils portent, s'aligner par le haut les decalerait.
    """
    par_ligne = [_polys(t, px * SS, -k * px * SS * INTERLIGNE) for k, t in enumerate(lignes)]
    if not any(par_ligne):
        return np.zeros((1, 1), np.float32), 0.0

    if align == "droite":
        fin = max(max(p[:, 0].max() for p in ps) for ps in par_ligne if ps)
        par_ligne = [
            [p + np.array([fin - max(q[:, 0].max() for q in ps), 0.0]) for p in ps] if ps else []
            for ps in par_ligne
        ]

    tous = [p for ps in par_ligne for p in ps]
    pts = np.concatenate(tous)
    x0 = pts[:, 0].min()
    x1, y1 = pts.max(axis=0)
    w = int(np.ceil(x1 - x0)) + 2
    h = int(np.ceil(y1 - pts[:, 1].min())) + 2
    m = np.zeros((h, w), np.uint8)
    # la police va vers le haut, l'image vers le bas
    cv2.fillPoly(
        m,
        [np.round(np.column_stack([p[:, 0] - x0, y1 - p[:, 1]])).astype(np.int32) for p in tous],
        255,
    )
    petit = cv2.resize(m, (max(w // SS, 1), max(h // SS, 1)), interpolation=cv2.INTER_AREA)
    return petit.astype(np.float32) / 255.0, float(y1 * petit.shape[0] / h)


def masque(lignes: list[str], px: float, align: str = "gauche") -> np.ndarray:
    return bloc(lignes, px, align)[0]


def ecrire(fond: np.ndarray, m: np.ndarray, x: float, y: float, ax: float, ay: float) -> None:
    """Compose un masque de texte en blanc, ancre a la fraction (ax, ay) de sa boite."""
    h, w = m.shape
    x0, y0 = int(round(x - ax * w)), int(round(y - ay * h))
    xs, ys = (
        slice(max(x0, 0), min(x0 + w, fond.shape[1])),
        slice(max(y0, 0), min(y0 + h, fond.shape[0])),
    )
    if xs.stop <= xs.start or ys.stop <= ys.start:
        return
    a = m[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0, None]
    fond[ys, xs] = fond[ys, xs] * (1.0 - a) + a


def coins(
    img: np.ndarray,
    gauche: list[str] | None,
    droite: list[str] | None,
    px_gauche: float,
    px_droite: float,
    marge: float,
) -> None:
    """Pose deux blocs dans les coins superieurs, alignes sur une ligne de base commune."""
    g = bloc(gauche, px_gauche) if gauche else None
    d = bloc(droite, px_droite, align="droite") if droite else None
    ref = max(g[1] if g else 0.0, d[1] if d else 0.0)
    if g:
        ecrire(img, g[0], marge, marge + ref - g[1], 0.0, 0.0)
    if d:
        ecrire(img, d[0], img.shape[1] - marge, marge + ref - d[1], 1.0, 0.0)
