"""Composition : disque plein au centre, couronne des phases autour.

Troisieme sortie de la chaine, apres le timelapse et le tirage a l'unite. Elle
lit `selection_phase.csv`, la meilleure trame de chaque tranche de phase, et
pose ces trames en couronne autour du disque non occulte.

**Branche montante seule par defaut.** Elle est complete, neuf tranches sans
trou, de 0 a 90,8 % d'obscuration, sur 16,4° a 6,7° d'elevation. La descente
est amputee de cinq tranches par le nuage et s'arrete a 0,27° d'elevation :
elle desequilibrerait la couronne sans rien apporter.

**Espacement angulaire regulier, pas proportionnel au temps.** Les tranches
sont uniformes en obscuration, pas en temps : les intervalles reels vont de
42 s a 11 min. Un espacement proportionnel serait fidele et illisible.

Le disque central garde l'echelle native. Les vignettes partagent une echelle
commune, plus petite, pour que la morsure lunaire se compare de l'une a
l'autre. C'est le seul endroit de la chaine ou le soleil est mis a l'echelle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import config, diagnose, io, render, select, texte, tiff
from .calibrate import Calibration

VIGNETTE_FRAC = 0.34  # cote d'une vignette, en fraction de la decoupe native
FONDU = (1.10, 1.28)  # fondu radial vers le noir, en rayons solaires

# Teinte du disque central. En mode unifie, la normalisation cale la mediane du
# disque sur la cible : sur un disque plein, tout l'interieur est plus lumineux
# que cette mediane, et la courbe sRGB etant concave, un meme rapport lineaire
# y rend moins de saturation. Le centre sort donc creme la ou un croissant sort
# jaune. Au-dela de 1,75 le bleu tombe sous 0,16 et la couleur sonne faux.
TINT_CENTRE = 1.5
ECART = 0.16  # jeu entre disque central et vignettes, en fraction du cote d'une vignette
TEXTE_FRAC = 0.09  # hauteur de police, en fraction du cote d'une vignette
LEGENDE_FRAC = 1.6  # taille de la legende, en multiples de celle des etiquettes
# texte visible et non commentaire : les accents y sont, la police les porte
LEGENDE = "Eclipse solaire du 12 août 2026\nMontastruc (65)"


def _fondu(im: np.ndarray, r_sun_px: float) -> np.ndarray:
    """Ramene la decoupe au noir hors d'un disque, en fondu.

    Le halo de diffusion du filtre vaut 0,22 % du niveau de disque a 1,5 R et
    reste mesurable jusqu'au bord de la decoupe, a 1,30 R. Sur une composition
    posee sur fond noir, il ne se lit pas comme un halo mais comme le carre de
    la decoupe. Il est reel, il est neanmoins retire ici : la composition est
    une image de presentation, pas une mesure. `--sans-fondu` le conserve.
    """
    h, w = im.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - (w - 1) / 2.0, y - (h - 1) / 2.0) / r_sun_px
    r0, r1 = FONDU
    a = np.clip((r1 - r) / (r1 - r0), 0.0, 1.0).astype(np.float32)
    return im * a[..., None]


def _poser(fond: np.ndarray, im: np.ndarray, cx: float, cy: float) -> None:
    """Pose une image carree centree en (cx, cy), par maximum.

    Le maximum plutot que l'affectation : tout le contenu est sur fond noir
    mesure, deux vignettes qui se toucheraient s'unissent au lieu de se
    decouper au carre.
    """
    h, w = im.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xs, ys = (
        slice(max(x0, 0), min(x0 + w, fond.shape[1])),
        slice(max(y0, 0), min(y0 + h, fond.shape[0])),
    )
    sx, sy = slice(xs.start - x0, xs.stop - x0), slice(ys.start - y0, ys.stop - y0)
    np.maximum(fond[ys, xs], im[sy, sx], out=fond[ys, xs])


def _etiquette(ligne: dict) -> list[str]:
    obsc = float(ligne["obsc"])
    return [
        f"{ligne['t'][11:19]} UTC",
        f"{obsc:.1f} %".replace(".", ",") + f"   {float(ligne['alt']):.1f}°".replace(".", ","),
    ]


def _centre_ideal(d: dict) -> int:
    """Meilleure trame non occultee : limbe le plus fin, puis centre le mieux contraint."""
    m = (
        (d["obsc_eph"] <= 0.0)
        & (select.classify(d) == "A")
        & np.isfinite(d["cx"])
        & ~select.veiled(d)
    )
    if not m.any():
        raise SystemExit("aucune trame non occultee de classe A")
    i = np.nonzero(m)[0]
    return int(i[np.lexsort((d["sigma_center"][i], d["limb_width"][i]))][0])


def compose(
    branche: str = "montante",
    frac: float = VIGNETTE_FRAC,
    frac_centre: float = 1.0,
    etiquettes: bool = True,
    fondu: bool = True,
    tint_centre: float = TINT_CENTRE,
    legende: str | None = None,
    toile: tuple[int, int] | None = None,
    ancre: tuple[float, float] = (0.5, 0.5),
    centre_at: str | None = None,
    nom: str = "composition",
    out_dir: Path | None = None,
    **over,
) -> dict:
    """Assemble la composition et l'ecrit en TIFF 16 bits profile sRGB."""
    config.exige_analyse("calibration.json", "metrics.csv", "selection_phase.csv")
    out_dir = out_dir or (config.SINGLE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = diagnose.load()
    ic = select.interpolate_centers(d)
    # La composition juxtapose des trames prises entre 16,4 et 6,7 degres
    # d'elevation : l'extinction differentielle fait glisser le rapport rouge
    # sur vert du disque de 1,00 a 1,55 sur cette plage. C'est le cas meme pour
    # lequel le mode unifie existe, il est donc le defaut ici, quel que soit le
    # dernier rendu de serie.
    if over.get("couleur") is None:
        over["couleur"] = "unifiee"
    cfg = render.tirage_reglages(**over)
    rows = list(csv.DictReader((config.ANALYSIS_DIR / "metrics.csv").open()))
    norms = render._norm_factors(rows, cfg["mode"])
    cal = Calibration.load()
    chemins = {f.name: f.path for f in io.discover().frames}
    par_nom = {str(f): k for k, f in enumerate(d["file"])}

    i_centre = render._find_frame(d, centre_at, None, 30.0) if centre_at else _centre_ideal(d)

    phases = [r for r in csv.DictReader((config.ANALYSIS_DIR / "selection_phase.csv").open())]
    if branche != "toutes":
        phases = [r for r in phases if r["branche"] == branche]
    # la tranche a 0 % est deja le disque central
    phases = [r for r in phases if float(r["obsc"]) > 0.5]
    phases.sort(key=lambda r: r["t"])
    if not phases:
        raise SystemExit(f"aucune trame de phase pour la branche {branche!r}")

    # Les deux echelles se rapportent a la decoupe native, pas l'une a l'autre :
    # reduire le disque central rapproche la couronne sans toucher a la taille
    # des vignettes, ce qui est le but.
    crop_px = int(round(cfg["crop_r"] * cal.r_sun_px * 2))
    c_px = int(round(crop_px * frac_centre))
    v_px = int(round(crop_px * frac))
    n = len(phases)

    # La teinte du centre passe par deux chemins selon le mode : en couleur
    # unifiee elle est la cible imposee a chaque canal, en balance figee elle
    # est dans la balance elle-meme, `target` n'y etant pas lu. Les deux sont
    # donc fournis, sans quoi --tint-centre reste sans effet en mode fixe.
    cible_centre = render.apply_tint((1.0, 1.0, 1.0), tint_centre)
    wb_centre = render.apply_tint(render.base_white_balance(), tint_centre)

    def rendu(i: int, taille: int, target=None, wb=None) -> np.ndarray:
        fwhm = float(d["limb_width"][i])
        if not np.isfinite(fwhm):
            fwhm = float(np.nanmedian(d["limb_width"]))
        im = render.render_array(
            chemins[d["file"][i]],
            float(ic["cx"][i]) * 2.0,
            float(ic["cy"][i]) * 2.0,
            float(norms[i]),
            wb if wb is not None else cfg["wb"],
            target if target is not None else cfg["target"],
            cfg["couleur"],
            crop_px,
            taille,
            cal.r_sun_px * 2.0,
            cfg["white"],
            1.0,
            fwhm,
            cfg["nettete"],
            float(d["snr_disk"][i]),
            render.SHARPEN_PROTECT / cfg["white"],
        )
        return _fondu(im, cal.r_sun_px * 2.0 * taille / crop_px) if fondu else im

    # Rayon de la couronne : le plus contraignant des deux, ne pas chevaucher
    # le voisin sur le cercle, et laisser un jeu avec le disque central.
    jeu = ECART * v_px
    rayon = max((v_px + jeu) / (2.0 * np.sin(np.pi / n)), (c_px + v_px) / 2.0 + jeu)

    # angles depuis le haut, dans le sens horaire, ordre chronologique
    ang = [-np.pi / 2 + 2 * np.pi * k / n for k in range(n)]
    pos = [(rayon * np.cos(a), rayon * np.sin(a)) for a in ang]

    blocs = []
    if etiquettes:
        for a, (x, y), r in zip(ang, pos, phases, strict=True):
            m = texte.masque(_etiquette(r), TEXTE_FRAC * v_px)
            ux, uy = np.cos(a), np.sin(a)
            bx, by = x + ux * (v_px / 2 + jeu / 2), y + uy * (v_px / 2 + jeu / 2)
            blocs.append((m, bx, by, (1 - ux) / 2, (1 - uy) / 2))

    # Le cadre est deduit du contenu : les etiquettes des diagonales portent
    # plus loin que celles des axes, une taille fixe rognerait.
    ext = [c_px / 2] + [max(abs(x), abs(y)) + v_px / 2 for x, y in pos]
    for m, bx, by, ax, ay in blocs:
        h, w = m.shape
        ext.append(max(abs(bx - ax * w), abs(bx + (1 - ax) * w)))
        ext.append(max(abs(by - ay * h), abs(by + (1 - ay) * h)))
    cote = int(round(2 * (max(ext) + 0.12 * v_px)))

    # Le cadre libre ne met rien a l'echelle : la composition est posee telle
    # quelle et ce qui deborde est rogne, comme pour un tirage a l'unite.
    large, haut = toile or (cote, cote)
    mid_x, mid_y = large * ancre[0], haut * ancre[1]
    if toile and (large < cote or haut < cote):
        print(
            f"attention : la composition fait {cote} x {cote} px et sera rognee. "
            f"Le disque central occupant {c_px} px, un cadre plus court se remplit en "
            f"reduisant --echelle, sans quoi le rayon de la couronne reste impose par lui"
        )

    lin = np.zeros((haut, large, 3), np.float32)
    _poser(lin, rendu(i_centre, c_px, cible_centre, wb_centre), mid_x, mid_y)
    for (x, y), r in zip(pos, phases, strict=True):
        _poser(lin, rendu(par_nom[r["file"]], v_px), mid_x + x, mid_y + y)

    # le texte se compose apres la courbe de tonalite : c'est un graphisme, il
    # doit rester blanc pur et d'opacite constante
    img = tiff.srgb_encode(lin)
    for m, bx, by, ax, ay in blocs:
        texte.ecrire(img, m, mid_x + bx, mid_y + by, ax, ay)
    if legende:
        # La legende se lit en deux blocs et non en deux lignes empilees : la
        # premiere ligne en haut a gauche, le reste cale a droite. Une legende
        # d'un seul tenant occuperait la largeur du cadre et viendrait toucher
        # les etiquettes de la vignette de midi.
        px = TEXTE_FRAC * LEGENDE_FRAC * v_px
        # un \n tape en ligne de commande arrive en deux caracteres
        lignes = legende.replace("\\n", "\n").split("\n")
        texte.coins(
            img, lignes[:1], lignes[1:] or None, px, px, texte.MARGE_FRAC * min(large, haut)
        )

    info = {
        "branche": branche,
        "centre": {
            "file": str(d["file"][i_centre]),
            "t": d["t"][i_centre].isoformat(),
            "limb_width": float(d["limb_width"][i_centre]),
        },
        "vignettes": [
            {"t": r["t"], "obsc": float(r["obsc"]), "alt": float(r["alt"]), "file": r["file"]}
            for r in phases
        ],
        "taille": [large, haut],
        "cote_naturel": cote,
        "ancre": [float(ancre[0]), float(ancre[1])],
        "tint_centre": tint_centre,
        "disque_central_px": 4.0 * cal.r_sun_px * frac_centre,
        "echelle_vignette": frac,
        "echelle_centre": frac_centre,
        "rayon_couronne_px": float(rayon),
        "etiquettes": etiquettes,
        "legende": legende,
        "fondu": list(FONDU) if fondu else None,
        "transfert": "sRGB IEC61966-2.1",
        "reglages": {k: (list(x) if isinstance(x, tuple) else x) for k, x in cfg.items()},
    }

    stem = str(out_dir / nom)
    tiff.write_rgb16(
        Path(stem + ".tif"),
        (img * 65535.0 + 0.5).astype(np.uint16),
        tiff.srgb_icc(),
        description=json.dumps(info, separators=(",", ":")),
        software="eclipse compose",
    )
    ech = 1600 / max(large, haut)
    apercu = cv2.resize(img, (round(large * ech), round(haut * ech)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        stem + ".png", cv2.cvtColor((apercu * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    Path(stem + ".json").write_text(json.dumps(info, indent=2))
    return {"stem": Path(stem), **info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--branche",
        choices=("montante", "descendante", "toutes"),
        default="montante",
        help="branche de l'eclipse portee par la couronne",
    )
    ap.add_argument(
        "--echelle",
        type=float,
        default=VIGNETTE_FRAC,
        help="cote d'une vignette, en fraction de la decoupe a l'echelle native",
    )
    ap.add_argument(
        "--echelle-centre",
        type=float,
        default=1.0,
        help="cote du disque central, en fraction de la decoupe a l'echelle native. "
        "Le reduire rapproche la couronne et laisse plus de place aux vignettes",
    )
    ap.add_argument("--sans-etiquettes", action="store_true")
    ap.add_argument(
        "--legende",
        nargs="?",
        const=LEGENDE,
        default=None,
        help="legende. La premiere ligne va dans le coin superieur gauche, la suite "
        "dans le coin superieur droit, calee a droite. Sans valeur : "
        f"{LEGENDE!r}",
    )
    ap.add_argument(
        "--sans-fondu",
        action="store_true",
        help="conserve le halo de diffusion jusqu'au bord carre de la decoupe",
    )
    ap.add_argument("--centre", help="trame du disque central, heure UTC HHMMSS")
    ap.add_argument(
        "--tint-centre",
        type=float,
        default=TINT_CENTRE,
        help="teinte du seul disque central, 1 = celle des vignettes, 1,5 = jaune solaire franc",
    )
    ap.add_argument(
        "--toile",
        type=lambda v: tuple(int(x) for x in v.lower().replace("x", " ").split()),
        help="cadre de sortie, LxH en pixels. Rien n'est mis a l'echelle, le cadre est "
        "complete en noir et ce qui deborde est rogne",
    )
    ap.add_argument(
        "--pos",
        type=lambda v: tuple(float(x) for x in v.replace(",", " ").split()),
        default=(0.5, 0.5),
        help="position du centre de la composition dans le cadre, en fractions",
    )
    ap.add_argument("--nom", default="composition")
    ap.add_argument("--mode", choices=("disque", "instrument", "aucune"), default=None)
    ap.add_argument("--white", type=float, default=None)
    ap.add_argument("--tint", type=float, default=None)
    ap.add_argument("--couleur", choices=("fixe", "unifiee"), default=None)
    ap.add_argument("--nettete", type=float, default=None)
    a = ap.parse_args()

    r = compose(
        branche=a.branche,
        frac=a.echelle,
        frac_centre=a.echelle_centre,
        etiquettes=not a.sans_etiquettes,
        fondu=not a.sans_fondu,
        tint_centre=a.tint_centre,
        legende=a.legende,
        toile=a.toile,
        ancre=a.pos,
        centre_at=a.centre,
        nom=a.nom,
        mode=a.mode,
        white=a.white,
        tint=a.tint,
        couleur=a.couleur,
        nettete=a.nettete,
    )
    print(
        f"centre {r['centre']['t'][11:19]} UTC, limbe {r['centre']['limb_width']:.2f} arcsec, "
        f"{len(r['vignettes'])} vignettes de la branche {r['branche']}"
    )
    print(
        f"{r['taille'][0]} x {r['taille'][1]} px, disque central {r['disque_central_px']:.0f} px, "
        f"centre a {100 * r['echelle_centre']:.0f} %, vignettes a "
        f"{100 * r['echelle_vignette']:.0f} %, teinte du centre {r['tint_centre']:g}"
    )
    print("->", str(r["stem"]) + ".tif")
