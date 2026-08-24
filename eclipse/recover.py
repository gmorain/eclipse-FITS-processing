"""Developpement d'une trame unique, independant de la chaine timelapse.

Une trame de coucher derriere un premier plan n'est pas une trame de mesure.
Elle porte deux sujets dont les mouvements sont incompatibles : le soleil
descend de 13 arcsec/s derriere une haie fixe au sol. Empiler la rafale gagne
bien la racine du nombre de trames sur le bruit du disque, mais brouille l'un
ou l'autre selon ce sur quoi on aligne. On developpe donc **une seule pose**,
et tout le travail porte sur le bruit.

Ce que le contenu autorise :

- le disque a 38 masses d'air n'a aucun contenu haute frequence, seules la haie
  et la morsure lunaire en ont. Un lissage a preservation de bords y est donc
  quasi sans perte ;
- la chrominance ne porte aucune information spatiale a ce niveau de signal,
  5 % de bleu pour 100 % du bruit. Elle se lisse largement ;
- la luminance optimale n'est pas la luminance video mais la somme ponderee par
  le signal de chaque canal, le bruit de lecture etant identique sur les trois.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

from . import config, io, render, tiff


@dataclass
class Develop:
    """Mesures et reglages du developpement, ecrits a cote de l'image."""

    fichier: str
    pedestal: float
    sigma_adu: float
    niveaux_rvb: tuple[float, float, float]
    poids_luminance: tuple[float, float, float]
    snr_pixel: float
    remplissage_puits: float  # fraction de la pleine echelle atteinte
    fwhm_arcsec: float
    denoise: float
    gamma: float
    noir: float
    blanc: float


def channel_levels(rgb: np.ndarray, thr_frac: float = 0.5) -> np.ndarray:
    """Niveau median du sujet eclaire, par canal."""
    r = rgb[..., 0]
    lit = r > thr_frac * np.percentile(r, 99.9)
    return np.median(rgb[lit], axis=0)


def denoise(
    rgb: np.ndarray, sigma: float, levels: np.ndarray, strength: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Lissage a preservation de bords sur la luminance, large sur la chrominance.

    La chrominance est traitee en rapports au signal et non en differences :
    elle traverse ainsi la courbe de tonalite sans deriver.
    """
    w = np.asarray(levels, np.float32)
    w = w / w.sum()  # ponderation par le signal, optimale a bruit egal
    lum = rgb @ w
    eps = max(sigma, 1.0)
    ratio = rgb / np.maximum(lum, eps)[..., None]

    # la chrominance ne porte aucun detail a ce niveau de signal
    ratio = cv2.GaussianBlur(ratio, (0, 0), 8.0 * strength)

    scale = float(np.max(levels))
    l_n = lum / scale
    l_d = cv2.bilateralFilter(
        l_n, d=0, sigmaColor=float(2.0 * strength * sigma / scale), sigmaSpace=7.0
    )
    return (l_d[..., None] * ratio * scale).astype(np.float32), w


def tone(v: np.ndarray, noir: float, blanc: float, gamma: float) -> np.ndarray:
    out = np.clip((v - noir) / max(blanc - noir, 1e-6), 0.0, 1.0).astype(np.float32)
    return out ** np.float32(1.0 / gamma) if gamma != 1.0 else out


def develop(
    path: Path,
    nom: str | None = None,
    out_dir: Path | None = None,
    fwhm_arcsec: float = 5.5,
    nettete: float = 4.0,
    denoise_strength: float = 1.0,
    gamma: float = 1.4,
    noir_sigma: float = 2.0,
    blanc: float = 1.45,
    crop: tuple[int, int, int, int] | None = None,
) -> Develop:
    """Developpe une trame et ecrit un TIFF 16 bits plus un PNG de visualisation."""
    path = Path(path)
    out_dir = out_dir or (config.SINGLE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(path, memmap=False) as h:
        raw = h[0].data
    ped, sigma = io.pedestal(raw[0::2, 0::2])
    rgb = cv2.cvtColor(raw, render.BAYER_CODE).astype(np.float32) - ped

    lev = channel_levels(rgb)
    clean, w = denoise(rgb, sigma, lev, denoise_strength)

    # Normalisation sur le canal le plus fort et non sur la luminance ponderee :
    # a 38 masses d'air le rouge vaut 1,24 fois cette luminance, et normaliser
    # dessus ecreterait le disque en aplat rouge, sans modele ni assombrissement
    # centre-bord. `blanc` est donc le niveau du canal dominant qui atteint le
    # blanc, superieur a 1 pour garder les parties les plus brillantes.
    scale = float(np.max(lev))
    v = clean / scale
    snr = float(np.max(lev) / sigma)
    if nettete > 0:
        # la luminance de ponderation est celle du sujet, pas la luminance video
        v = render.sharpen(
            v,
            fwhm_arcsec,
            nettete,
            snr * float(denoise_strength + 1.0),
            weights=tuple(float(x) for x in w),
        )

    noir = noir_sigma * sigma / scale
    out = tone(v, noir, blanc, gamma)
    if crop:
        x0, y0, x1, y1 = crop
        out = out[y0:y1, x0:x1]

    # concatenation et non with_suffix : un nom de trame contient des points
    # (50.0ms, 32.7C) que with_suffix prendrait pour une extension
    stem = str(out_dir / (nom or path.stem))

    def _p(ext: str) -> str:
        return stem + ext

    info = Develop(
        fichier=path.name,
        pedestal=float(ped),
        sigma_adu=float(sigma),
        niveaux_rvb=tuple(float(x) for x in lev),
        poids_luminance=tuple(float(x) for x in w),
        snr_pixel=snr,
        # percentile et non maximum : un pixel chaud isole donnerait 100 %
        remplissage_puits=float(np.percentile(rgb[..., 0], 99.99) / (config.SATURATION_ADU / 0.98)),
        fwhm_arcsec=fwhm_arcsec,
        denoise=denoise_strength,
        gamma=gamma,
        noir=float(noir),
        blanc=blanc,
    )
    reglages = asdict(info)
    Path(_p(".json")).write_text(json.dumps(reglages, indent=2))

    # `tone` a deja porte les valeurs dans le domaine d'affichage : c'est sous
    # cette hypothese que `gamma` a ete regle a l'oeil, tout visualiseur lisant
    # un fichier sans profil comme du sRGB. Embarquer le profil ne change aucun
    # pixel, il rend l'hypothese explicite et transportable.
    tiff.write_rgb16(
        Path(_p(".tif")),
        (out * 65535.0 + 0.5).astype(np.uint16),
        tiff.srgb_icc(),
        description=json.dumps(reglages, separators=(",", ":")),
        software="eclipse recover",
    )
    # apercu 8 bits, meme encodage : sert au tri, pas au tirage
    apercu = cv2.cvtColor((out * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(_p(".png"), apercu)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fits", type=Path)
    ap.add_argument("--nom", default=None, help="nom de sortie, par defaut celui de la trame")
    ap.add_argument("--fwhm", type=float, default=5.5, help="FWHM mesuree de la trame, arcsec")
    ap.add_argument("--nettete", type=float, default=4.0, help="FWHM cible, 0 pour aucune")
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument(
        "--blanc", type=float, default=1.45, help="niveau du canal dominant qui atteint le blanc"
    )
    ap.add_argument(
        "--crop",
        default=None,
        metavar="x0,y0,x1,y1",
        help="decoupe en pixels pleine resolution, par defaut le champ entier",
    )
    a = ap.parse_args()
    inf = develop(
        a.fits,
        a.nom,
        fwhm_arcsec=a.fwhm,
        nettete=a.nettete,
        denoise_strength=a.denoise,
        gamma=a.gamma,
        blanc=a.blanc,
        crop=tuple(int(x) for x in a.crop.split(",")) if a.crop else None,
    )
    print(f"piedestal {inf.pedestal:.0f} ADU, bruit {inf.sigma_adu:.1f} ADU")
    print(
        f"niveaux R/V/B {inf.niveaux_rvb[0]:.0f} / {inf.niveaux_rvb[1]:.0f} / "
        f"{inf.niveaux_rvb[2]:.0f}, poids de luminance "
        f"{inf.poids_luminance[0]:.2f} / {inf.poids_luminance[1]:.2f} / "
        f"{inf.poids_luminance[2]:.2f}"
    )
    print(f"snr par pixel {inf.snr_pixel:.1f}, puits rempli a {100 * inf.remplissage_puits:.1f} %")
