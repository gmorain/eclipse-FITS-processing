"""Constantes de session et parametres de traitement."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _lire_dotenv(chemin: Path) -> dict[str, str]:
    """Paires `cle=valeur` d'un fichier .env.

    Volontairement minimal : pas d'interpolation, pas de `export`, une paire par
    ligne, guillemets exterieurs otes. L'environnement du processus prime.
    """
    valeurs: dict[str, str] = {}
    if not chemin.is_file():
        return valeurs
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, _, val = ligne.partition("=")
        valeurs[cle.strip()] = val.strip().strip("\"'")
    return valeurs


_ENV = _lire_dotenv(RACINE / ".env")


def _chemin(cle: str) -> Path | None:
    val = os.environ.get(cle) or _ENV.get(cle)
    return Path(val).expanduser() if val else None


# Dossier des FITS de la session, hors du depot. Defini dans .env, voir
# .env.example. Passer par config.session_dir() plutot que par cette variable :
# elle vaut None tant que rien n'est configure.
SESSION_DIR = _chemin("ECLIPSE_SESSION_DIR")
OUT_DIR = _chemin("ECLIPSE_OUT_DIR") or RACINE / "out"
# Sorties rangees par nature : tables et mesures, video et ses trames, tirages a
# l'unite, rapport et les images qu'il embarque. Rien ne reste a la racine.
ANALYSIS_DIR = OUT_DIR / "analysis"
TIMELAPSE_DIR = OUT_DIR / "timelapse"
SINGLE_DIR = OUT_DIR / "single"
REPORT_DIR = OUT_DIR / "report"
# Captures d'annotation embarquees dans le rapport. Par defaut "05 - Annotations"
# trois niveaux au-dessus des captures, la ou l'ASIAIR range la session.
ANNOTATIONS_DIR = _chemin("ECLIPSE_ANNOTATIONS_DIR") or (
    SESSION_DIR.parent.parent.parent / "05 - Annotations" if SESSION_DIR else None
)


def exige_analyse(*fichiers: str) -> None:
    """Arrete la commande si la passe d'analyse n'a pas produit ces fichiers.

    Sans ce controle, un tirage lance avant la mesure sort une trace
    FileNotFoundError sur le premier fichier manquant, qui ne dit pas quoi
    lancer.
    """
    manquants = [n for n in fichiers if not (ANALYSIS_DIR / n).is_file()]
    if manquants:
        raise SystemExit(
            f"passe d'analyse incomplete dans {ANALYSIS_DIR}, il manque "
            + ", ".join(manquants)
            + "\n  jouer d'abord : uv run python -m eclipse --skip-render"
        )


def session_dir() -> Path:
    """Dossier des captures, exige. Un seul message pour toute la chaine."""
    if SESSION_DIR is None:
        raise SystemExit(
            "ECLIPSE_SESSION_DIR n'est pas defini : copier .env.example en .env et y "
            "porter le chemin du dossier de captures."
        )
    if not SESSION_DIR.is_dir():
        raise SystemExit(f"ECLIPSE_SESSION_DIR ne designe aucun dossier : {SESSION_DIR}")
    return SESSION_DIR


# Site reel : bord de champ a Montastruc, Hautes-Pyrenees, entre Castelbajac et
# Houeydets, au nord de Lannemezan.
#
# Les en-tetes FITS portent un site memorise a 38 km au sud : l'ASIAIR a garde
# une position enregistree au lieu de relever la sienne. Le nom du dossier de
# session vient de la meme erreur. Les coordonnees ci-dessous reproduisent les
# circonstances calculees par Eclipsefan pour le lieu (maximum a 18:26:58 UTC,
# obscuration 98,8 %, elevation 5,9 deg, azimut 284,8 deg) a 6 secondes et
# 0,1 point pres, ce que les coordonnees d'en-tete ne font pas.
SITE_NAME = "Montastruc (65)"
SITE_LAT = 43.1683
SITE_LON = 0.3872
SITE_ALT_M = 485.0

# Erreur des en-tetes, conservee pour le rapport. Seul l'ordre de grandeur sert
# au diagnostic, les coordonnees memorisees ne designent pas le lieu d'observation.
HEADER_SITE_ERROR_KM = 38
PRESSURE_HPA = 880.0
TEMP_C = 15.0

# Capteur IMX585, 3840 x 2160, 2,9 um, lunette 400 mm f/6.
FULL_SHAPE = (2160, 3840)
ARCSEC_PER_PX_FULL = 1.4955
ARCSEC_PER_PX_R = 2.0 * ARCSEC_PER_PX_FULL  # plan R = un pixel sur deux
BAYER_PATTERN = "RGGB"
R_PLANE_OFFSET = (0, 0)  # position du pixel R dans la cellule 2x2, verifiee sur les donnees

# Rayon solaire de la session, constant a 0,01 % pres (947,0 arcsec le 2026-08-12).
R_SUN_ARCSEC = 947.0
R_SUN_PX_R = R_SUN_ARCSEC / ARCSEC_PER_PX_R  # ~316,5 px dans le plan R

# Niveau d'ecretage. Donnees 12 bits decalees a gauche de 4 bits ; plafond
# observe sur la session a 64512 ADU.
SATURATION_ADU = 64224  # 0,98 x 65535

# Les donnees sont du 12 bits decale de 4 bits : le pas de quantification vaut
# 16 ADU. Le fond derriere l'OD 3,8 est si noir que la MAD des coins tombe a
# zero, ce qui rendrait tout seuil calcule dessus absurde. Plancher obligatoire.
QUANT_ADU = 16.0

# Rejets. Les trames de 11:27 UTC ont une mise au point differente (FOCUSPOS
# 17634 contre 15490) et le filtre LUlt est une bande etroite non comparable.
SESSION_START_UTC = dt.datetime(2026, 8, 12, 17, 0)
KEEP_FILTERS = ("IRCt",)

# Coupure du timelapse : au-dela, le disque refracte s'ecarte de plus de 0,5 px
# de la meilleure ellipse et le modele geometrique ne tient plus.
MIN_ALT_DEG_TIMELAPSE = 3.8

# Detection de rafale : ecart entre trames consecutives au-dela duquel on
# considere une nouvelle rafale (cadence interne mesuree 1,3 a 4,1 s).
BURST_GAP_S = 20.0


@dataclass(frozen=True)
class LimbParams:
    """Parametres de la detection de limbe et du fit."""

    n_rays: int = 720
    r_in: float = 0.80  # debut de la fenetre de recherche, en unites de rho(phi)
    r_out: float = 1.12  # fin de la fenetre
    step_px: float = 0.25
    ref_lo: float = 0.85  # anneau de reference locale
    ref_hi: float = 0.95
    min_contrast: float = 0.15  # contraste minimal exige sur le rayon
    clip_sigma: float = 3.0
    clip_iters: int = 5
    min_points: int = 30


@dataclass(frozen=True)
class Geometry:
    """Repere local : direction de la verticale dans le repere capteur.

    `vert_angle_deg` est l'angle du petit axe (verticale locale) mesure depuis
    l'axe +y du plan R, dans le sens trigonometrique. Zero tant que
    l'auto-calibration n'a pas tourne.
    """

    vert_angle_deg: float = 0.0
    r_sun_px: float = R_SUN_PX_R
    calibrated: bool = False
    n_frames_used: int = 0
    scatter_deg: float = float("nan")


DEFAULT_LIMB = LimbParams()
