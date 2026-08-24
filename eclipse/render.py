"""Rendu : dematricage pleine resolution, translation pure, normalisation.

Vue observateur terrestre, aucune derotation de champ. La monture est alt/az,
l'orientation du capteur par rapport a l'horizon est fixe pour toute la
session : la transformation appliquee aux images est une translation pure.
L'aplatissement du disque par la refraction est conserve, c'est un phenomene
que l'observateur voit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

from . import config, io, texte, tiff
from .calibrate import Calibration

# Motif RGGB. Le code OpenCV est nomme d'apres une autre convention que le
# mot-cle FITS : verifie sur les donnees, BayerBG2RGB rend bien un canal R
# identique au plan (0, 0) du brut.
BAYER_CODE = cv2.COLOR_BayerBG2RGB

CROP_R = 2.6  # cote de la sortie, en rayons solaires

# Niveau blanc du tirage a l'unite. Le 1,05 de la video ecrete jusqu'a 7 % des
# pixels : le disque deborde de sa mediane, qui sert de reference. Mesure sur
# 14 trames de classe A reparties sur la seance, le maximum atteint 1,58 en
# balance figee et 1,61 en couleur unifiee. 1,75 ne perd donc rien, et remplit
# encore 57 % de l'echelle, soit 37 000 niveaux sur 16 bits.
WHITE_SINGLE = 1.75

# Legende par defaut du tirage. Meme decoupage que dans `compose` : la premiere
# ligne va dans le coin superieur gauche, la suite dans le coin superieur droit.
# Texte visible et non commentaire, les accents y sont, la police les porte.
LEGENDE = "12 août 2026\nMontastruc (65)"

# Reglages par defaut de la ligne de commande. `render_frame` ne s'en sert
# qu'en dernier recours : il lit d'abord `render.json`, pour qu'une trame
# isolee sorte raccord avec le dernier timelapse rendu.
_CLI_DEFAULTS = {
    "mode": "disque",
    "size": 1080,
    "white": 1.05,
    "gamma": 1.0,
    "tint": 1.0,
    "couleur": "unifiee",
    "nettete": 0.0,
    "fps": 25,
    "workers": 6,
}

# Niveau minimal du disque, par canal, au-dessus duquel la normalisation
# chromatique par trame a un sens. En dessous on retombe sur la balance figee.
MIN_DISK_ADU = 400.0

# Nettete. Le plancher instrumental est la tache d'Airy d'un 66 mm, 2,1 arcsec,
# soit 1,4 px pleine resolution. La FWHM mesuree sur le limbe vaut 6,6 arcsec,
# donc 4,4 px : l'image est largement sur-echantillonnee par rapport a son
# propre contenu, et une deconvolution a de la marge avant de buter sur
# l'echantillonnage.
SHARPEN_FLOOR_ARCSEC = 3.0
SHARPEN_K_FLOOR = 2.0e-3
SHARPEN_PROTECT = 0.10  # sous ce niveau de disque, on n'accentue plus
SHARPEN_MAX_GAIN = 3.0


@dataclass
class RenderJob:
    path: Path
    cx_full: float
    cy_full: float
    norm: float  # diviseur d'intensite, 1.0 pour ne pas normaliser
    out: Path
    fwhm: float = float("nan")  # FWHM mesuree sur le limbe, arcsec
    snr: float = float("nan")


def white_balance(paths: list[Path], n: int = 6) -> tuple[float, float, float]:
    """Gains RVB fixes pour toute la session, mesures sur le disque non occulte.

    Fixes et non recalcules par trame : une balance automatique ferait respirer
    la couleur du timelapse au rythme des nuages et des changements de gain.

    Les trames doivent etre choisies claires. Prendre simplement les premieres
    non occultees ne suffit pas : dans cette session ce sont aussi les plus
    nuageuses, et un voile deforme la mesure.
    """
    med = []
    for path in paths[: n * 3]:
        with fits.open(path, memmap=False) as h:
            raw = h[0].data
        rgb = cv2.cvtColor(raw, BAYER_CODE).astype(np.float32)
        flat = rgb.reshape(-1, 3)
        sel = flat[:, 0] > np.percentile(flat[:, 0], 99.5)
        if sel.sum() < 1000:
            continue
        med.append(np.median(flat[sel], axis=0))
        if len(med) >= n:
            break
    if not med:
        return (1.0, 1.0, 1.0)
    m = np.median(np.stack(med), axis=0)
    return tuple(float(x) for x in (float(m[1]) / m))  # disque neutre, reference le vert


def apply_tint(wb: tuple[float, float, float], tint: float) -> tuple[float, float, float]:
    """Rechauffe la balance neutre. 0 = disque blanc, 1 = jaune orange marque.

    La couleur vraie d'une image derriere un filtre solaire n'a pas de sens :
    l'Astrosolar est un densite neutre large bande avec sa propre dominante. Le
    choix est esthetique, il est donc explicite.
    """
    warm = np.array([1.0, 0.87, 0.52])  # jaune solaire franc
    g = np.asarray(wb, float) * (1.0 - tint + tint * warm)
    return tuple(float(x) for x in g)


_CTX: dict = {}


def _init(wb, size, crop_px, white, gamma, couleur, target, r_full, fwhm_tgt):
    _CTX.update(
        wb=np.asarray(wb, np.float32),
        size=size,
        crop_px=crop_px,
        white=white,
        gamma=gamma,
        couleur=couleur,
        target=np.asarray(target, np.float32),
        r_full=r_full,
        fwhm_tgt=fwhm_tgt,
    )


def _disk_levels(rgb: np.ndarray, cx: float, cy: float, r_full: float) -> np.ndarray | None:
    """Niveau photospherique par canal, mesure sur la partie eclairee du disque.

    Sert a la normalisation chromatique par trame. Sous-echantillonne d'un
    facteur quatre : la mediane porte encore sur des dizaines de milliers de
    pixels, largement de quoi etre stable.
    """
    step = 4
    h, w, _ = rgb.shape
    y, x = np.mgrid[0:h:step, 0:w:step]
    inside = (x - cx) ** 2 + (y - cy) ** 2 < (0.97 * r_full) ** 2
    if inside.sum() < 4000:
        return None
    sub = rgb[::step, ::step]
    g = sub[..., 1][inside]
    lit = g > max(0.5 * np.percentile(g, 99.0), 1.0)
    # un croissant a 98 % d'obscuration ne couvre que 2 % du disque, soit
    # quelques centaines de points a ce pas d'echantillonnage
    if lit.sum() < 300:
        return None
    lev = np.median(sub[inside][lit], axis=0)
    # sous ce niveau les rapports entre canaux ne mesurent plus que le bruit :
    # les normaliser teinterait la trame au hasard, souvent en bleu
    if float(np.min(lev)) < MIN_DISK_ADU:
        return None
    return lev


def sharpen(
    v: np.ndarray,
    fwhm_cur_arcsec: float,
    fwhm_tgt_arcsec: float,
    snr: float,
    k_floor: float = SHARPEN_K_FLOOR,
    protect: float = SHARPEN_PROTECT,
    max_gain: float = SHARPEN_MAX_GAIN,
    weights: tuple[float, float, float] = (0.30, 0.59, 0.11),
) -> np.ndarray:
    """Deconvolution de Wiener calee sur la PSF mesuree de la trame.

    La largeur du limbe donne directement la FWHM de la PSF, trame par trame :
    le filtre n'est pas regle a l'oeil, il est deduit de la scene. L'anisotropie
    mesuree ne vaut que 6 % de la FWHM sur cette session, un noyau gaussien
    isotrope est donc justifie et non suppose.

    Filtre : W = Gc Gt / (Gc^2 + K), qui deconvolue de la gaussienne courante
    Gc et reconvolue vers la gaussienne cible Gt. K vient du rapport signal sur
    bruit mesure sur le disque, avec un plancher : c'est lui qui empeche
    l'amplification du bruit et limite les rebonds au limbe.

    L'accentuation est appliquee a la seule luminance, puis reportee sur les
    trois canaux comme un rapport. La teinte est donc rigoureusement conservee
    et aucun lisere colore n'apparait sur le bord.
    """
    if not np.isfinite(fwhm_cur_arcsec) or fwhm_cur_arcsec <= fwhm_tgt_arcsec:
        return v
    px = config.ARCSEC_PER_PX_FULL
    s_cur = fwhm_cur_arcsec / px / 2.3548
    s_tgt = max(fwhm_tgt_arcsec, SHARPEN_FLOOR_ARCSEC) / px / 2.3548
    lum = v @ np.asarray(weights, np.float32)

    h, w = lum.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f2 = fy**2 + fx**2
    gc = np.exp(-2.0 * np.pi**2 * s_cur**2 * f2)
    gt = np.exp(-2.0 * np.pi**2 * s_tgt**2 * f2)
    k = max(k_floor, (1.0 / snr) ** 2 if np.isfinite(snr) and snr > 0 else k_floor)
    wf = (gc * gt / (gc**2 + k)).astype(np.float32)

    sharp = np.fft.irfft2(np.fft.rfft2(lum) * wf, s=lum.shape).astype(np.float32)
    eps = 1e-4
    # le fond peut passer sous zero, le bruit etant centre sur le piedestal
    # soustrait : sans le plancher, un pixel a exactement -eps annule le
    # denominateur et sort un NaN que la ponderation propage
    gain = np.clip((sharp + eps) / (np.maximum(lum, 0.0) + eps), 0.0, max_gain)
    # le ciel reste intact : y accentuer ne ferait que remonter le bruit
    weight = np.clip(lum / protect, 0.0, 1.0)
    gain = 1.0 + (gain - 1.0) * weight
    return v * gain[..., None]


def render_array(
    path: Path,
    cx_full: float,
    cy_full: float,
    norm: float,
    wb,
    target,
    couleur: str,
    crop_px: int,
    size: int,
    r_full: float,
    white: float = 1.05,
    gamma: float = 1.0,
    fwhm_cur: float = float("nan"),
    fwhm_tgt: float = 0.0,
    snr: float = float("nan"),
    protect: float = SHARPEN_PROTECT,
    toile: tuple[int, int] | None = None,
    ancre: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Rend une trame et renvoie l'image RVB en flottant, entre 0 et 1.

    Isole du travailleur pour pouvoir servir aussi aux planches de comparaison,
    qui doivent rendre les memes trames sous plusieurs reglages.
    """
    with fits.open(path, memmap=False) as h:
        raw = h[0].data
    ped, _ = io.pedestal(raw[0::2, 0::2], center=(cx_full / 2, cy_full / 2))
    rgb = cv2.cvtColor(raw, BAYER_CODE).astype(np.float32) - ped

    if couleur == "unifiee":
        lev = _disk_levels(rgb, cx_full, cy_full, r_full)
        if lev is not None and np.all(lev > 0):
            # chaque canal ramene au niveau du disque, puis teinte cible
            # imposee : la couleur du timelapse ne depend plus ni de
            # l'extinction differentielle, qui rougit fortement le soleil sous
            # 5 degres, ni du voile
            rgb *= (np.asarray(target, np.float32) / lev).astype(np.float32)
            norm = 1.0
        else:
            rgb *= np.asarray(wb, np.float32)
    else:
        rgb *= np.asarray(wb, np.float32)

    # Translation pure et decoupe en une seule passe d'interpolation, sur
    # l'image d'origine non etiree. Lanczos 4.
    #
    # `toile` remplace le carre de 2,6 R par un cadre de format libre, le
    # disque pose a la fraction `ancre` de ses cotes. Aucune mise a l'echelle :
    # le disque garde son diametre natif et le cadre est complete en noir la ou
    # le capteur ne porte rien. Le fond derriere l'OD 3,8 est mesure a zero
    # au-dela de 2,6 R, le remplissage est donc la valeur vraie, pas un bouchon.
    w_out, h_out = toile or (crop_px, crop_px)
    m = np.array(
        [
            [1.0, 0.0, w_out * ancre[0] - cx_full],
            [0.0, 1.0, h_out * ancre[1] - cy_full],
        ],
        np.float32,
    )
    crop = cv2.warpAffine(
        rgb,
        m,
        (w_out, h_out),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    lin = crop / (norm * white)
    # accentuation avant le sous-echantillonnage, a l'echelle native ou la PSF
    # a ete mesuree
    if fwhm_tgt > 0:
        lin = sharpen(lin, fwhm_cur, fwhm_tgt, snr, protect=protect)
    if toile is None and size != crop_px:
        lin = cv2.resize(lin, (size,) * 2, interpolation=cv2.INTER_AREA)

    v = np.clip(lin, 0.0, 1.0)
    return v ** (1.0 / gamma) if gamma != 1.0 else v


def _work(job: RenderJob) -> tuple[str, bool]:
    v = render_array(
        job.path,
        job.cx_full,
        job.cy_full,
        job.norm,
        _CTX["wb"],
        _CTX["target"],
        _CTX["couleur"],
        _CTX["crop_px"],
        _CTX["size"],
        _CTX["r_full"],
        _CTX["white"],
        _CTX["gamma"],
        job.fwhm,
        _CTX["fwhm_tgt"],
        job.snr,
    )
    cv2.imwrite(str(job.out), cv2.cvtColor((v * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return job.out.name, True


def _norm_factors(rows: list[dict], mode: str) -> np.ndarray:
    """Diviseur d'intensite par trame.

    `disque` : niveau photospherique mesure. Timelapse stable, le nuage ne se
    lit plus que par la perte de nettete.

    `instrument` : pose et gain seuls. Garde l'extinction et les nuages
    lisibles comme tels, mais l'extinction couvre quatre ordres de grandeur
    entre 17 et 0,4 degre d'elevation : les dernieres trames sortent noires.

    `aucune` : brut. Inexploitable ici, la pose passe de 32 us a 700 ms au
    cours de la seance.
    """
    ref = np.array(
        [float(r["ref_level"]) if r["ref_level"] not in ("", "nan") else np.nan for r in rows]
    )
    inst = np.array([float(r["exptime"]) * 10 ** (float(r["gain"]) / 200.0) for r in rows])
    if mode == "disque":
        v = ref.copy()
        # trames sans mesure : reprise sur la voisine exploitable la plus proche
        bad = ~np.isfinite(v) | (v <= 0)
        if bad.all():
            return np.ones(len(rows))
        good = np.nonzero(~bad)[0]
        for i in np.nonzero(bad)[0]:
            v[i] = v[good[np.argmin(np.abs(good - i))]]
        return v
    if mode == "instrument":
        return inst / np.median(inst) * np.nanmedian(ref)
    return np.ones(len(rows))


def base_white_balance() -> tuple[float, float, float]:
    """Balance neutre de la session, mesuree une fois puis mise en cache.

    Les trames de reference sont les meilleures classes A non voilees et peu
    occultees : la balance doit sortir du disque, pas d'un croissant.
    """
    wb_path = config.ANALYSIS_DIR / "white_balance.json"
    if wb_path.exists():
        return tuple(json.loads(wb_path.read_text()))

    from . import diagnose, select

    d = diagnose.load()
    clean = (select.classify(d) == "A") & ~select.veiled(d) & (d["obsc_eph"] < 0.5)
    order = np.nonzero(clean)[0][np.argsort(d["limb_width"][clean])]
    by_name = {f.name: f for f in io.discover().frames}
    wb = white_balance([by_name[d["file"][i]].path for i in order])
    wb_path.parent.mkdir(parents=True, exist_ok=True)
    wb_path.write_text(json.dumps(list(wb)))
    return wb


def run(
    timelapse_csv: Path | None = None,
    out_dir: Path | None = None,
    mode: str = "disque",
    size: int = 1080,
    white: float = 1.05,
    gamma: float = 1.0,
    tint: float = 0.0,
    couleur: str = "unifiee",
    nettete: float = 4.5,
    fps: int = 25,
    workers: int = 6,
    nom: str = "timelapse",
) -> dict:
    config.exige_analyse("calibration.json", "metrics.csv", "timelapse.csv")
    timelapse_csv = timelapse_csv or (config.ANALYSIS_DIR / "timelapse.csv")
    out_dir = out_dir or (config.TIMELAPSE_DIR / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    # purge : ffmpeg lit la serie f%05d.png en entier, et une selection plus
    # courte que la precedente laisserait des trames perimees en fin de video
    for stale in out_dir.glob("f*.png"):
        stale.unlink()
    rows = list(csv.DictReader(timelapse_csv.open()))

    # les niveaux photospheriques viennent de la passe de mesure
    metrics = {r["file"]: r for r in csv.DictReader((config.ANALYSIS_DIR / "metrics.csv").open())}
    for r in rows:
        for k in ("ref_level", "exptime", "gain", "limb_width", "snr_disk"):
            r[k] = metrics[r["file"]][k]
    # une trame sans mesure de PSF herite de la mediane de la seance
    lws = [float(r["limb_width"]) for r in rows if r["limb_width"] not in ("", "nan")]
    lw_med = float(np.median(lws)) if lws else float("nan")

    session = io.discover()
    cal = Calibration.load()
    by_name = {f.name: f for f in session.frames}

    wb = apply_tint(base_white_balance(), tint)
    target = apply_tint((1.0, 1.0, 1.0), tint)

    crop_px = int(round(CROP_R * cal.r_sun_px * 2))  # rayon plan R -> pleine resolution
    norms = _norm_factors(rows, mode)
    jobs = [
        RenderJob(
            path=by_name[r["file"]].path,
            cx_full=float(r["cx_full"]),
            cy_full=float(r["cy_full"]),
            norm=float(n),
            out=out_dir / f"f{int(r['ordre']):05d}.png",
            fwhm=float(r["limb_width"]) if r["limb_width"] not in ("", "nan") else lw_med,
            snr=float(r["snr_disk"]) if r["snr_disk"] not in ("", "nan") else float("nan"),
        )
        for r, n in zip(rows, norms, strict=True)
    ]

    t0 = time.time()
    with Pool(
        workers,
        initializer=_init,
        initargs=(
            wb,
            size,
            crop_px,
            white,
            gamma,
            couleur,
            target,
            cal.r_sun_px * 2.0,
            nettete,
        ),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_work, jobs, chunksize=2)):
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    video = None
    if shutil.which("ffmpeg"):
        video = config.TIMELAPSE_DIR / f"{nom}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(out_dir / "f%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(video),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        print(
            "attention : ffmpeg introuvable dans le PATH, aucune video assemblee.\n"
            f"  les {len(jobs)} trames restent dans {out_dir}"
        )
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (config.ANALYSIS_DIR / "render.json").write_text(
        json.dumps(
            {
                "nom": nom,
                "mode": mode,
                "couleur": couleur,
                "tint": tint,
                "white": white,
                "gamma": gamma,
                "size": size,
                "crop_r": CROP_R,
                "fps": fps,
                "nettete": nettete,
                "fwhm_mediane": lw_med,
                "wb": list(wb),
                "target": list(target),
            },
            indent=2,
        )
    )
    return {
        "frames": len(jobs),
        "dir": out_dir,
        "video": video,
        "wb": wb,
        "target": target,
        "couleur": couleur,
        "crop_px": crop_px,
        "size": size,
        "seconds": time.time() - t0,
    }


def tirage_reglages(**over) -> dict:
    """Reglages d'un tirage, herites de la serie mais pas de ses compromis video.

    De la serie viennent la normalisation, la balance, la teinte cible et la
    nettete, etablies sur les 1350 trames. Ni le niveau blanc ni la taille de
    sortie n'en viennent : la video ecrete a 1,05 pour gagner du contraste, un
    tirage garde toute la dynamique. Les arguments non nuls surchargent.
    """
    garde = ("mode", "couleur", "tint", "nettete", "crop_r", "wb")
    cfg = {k: _CLI_DEFAULTS[k] for k in ("mode", "couleur", "tint", "nettete")}
    cfg.update(crop_r=CROP_R, white=WHITE_SINGLE)
    rj = config.ANALYSIS_DIR / "render.json"
    if rj.exists():
        cfg.update({k: v for k, v in json.loads(rj.read_text()).items() if k in garde})
    cfg.update({k: v for k, v in over.items() if v is not None and k in cfg})

    # une teinte imposee ici invalide la balance stockee, qui la porte deja
    if over.get("tint") is not None or "wb" not in cfg:
        cfg["wb"] = apply_tint(base_white_balance(), cfg["tint"])
    cfg["target"] = apply_tint((1.0, 1.0, 1.0), cfg["tint"])
    return cfg


def _find_frame(d: dict, at: str | None, fichier: str | None, tol_s: float) -> int:
    """Indice de la trame designee, par heure UTC ou par fragment de nom.

    L'heure est celle de `DATE-OBS`, pas celle du nom de fichier : celui-ci
    porte l'heure locale, deux heures en avance, et une trame de la seance a un
    horodatage repare qui ne colle pas a son nom.
    """
    if fichier:
        hit = np.nonzero(np.char.find(d["file"].astype(str), fichier) >= 0)[0]
        if hit.size == 0:
            raise SystemExit(f"aucune trame dont le nom contient {fichier!r}")
        return int(hit[0])
    if not at:
        raise SystemExit("preciser --at HHMMSS ou --fichier")

    digits = "".join(c for c in at if c.isdigit())
    if len(digits) == 4:
        digits += "00"
    if len(digits) != 6:
        raise SystemExit(f"heure illisible : {at!r}, attendu HHMMSS")
    cible = dt.datetime.combine(
        d["t"][0].date(), dt.time(int(digits[:2]), int(digits[2:4]), int(digits[4:]))
    )
    ecart = np.array([abs((x - cible).total_seconds()) for x in d["t"]])
    i = int(np.argmin(ecart))
    if ecart[i] > tol_s:
        raise SystemExit(
            f"trame la plus proche de {cible:%H:%M:%S} a {ecart[i]:.0f}s, au-dela de {tol_s:.0f}s"
        )
    return i


def render_frame(
    at: str | None = None,
    fichier: str | None = None,
    nom: str | None = None,
    centre: tuple[float, float] | None = None,
    toile: tuple[int, int] | None = None,
    ancre: tuple[float, float] = (0.5, 0.5),
    legende: str | None = None,
    etiquettes: bool = False,
    out_dir: Path | None = None,
    tol_s: float = 30.0,
    **over,
) -> dict:
    """Tirage d'une seule trame, aux reglages de la sequence, pour finition externe.

    Sort dans `out/single/` et non dans `out/frames/`, que `run` purge a chaque
    rendu et que ffmpeg lit en entier.

    Ce que la trame herite de la serie fait toute sa valeur : normalisation
    d'exposition mesuree sur le disque, balance et teinte cible, deconvolution
    calee sur la PSF de la trame. Ce sont des grandeurs etablies sur les 1350
    trames, pas des reglages a l'oeil.

    Ce qu'elle n'herite pas, ce sont les compromis de la video :

    - **echelle native, jamais de mise a l'echelle.** Le disque garde ses
      1265 px de diametre. Un cadre plus grand que le contenu est complete en
      noir, ce qui est la valeur mesuree du ciel derriere l'OD 3,8 au-dela de
      2,6 R, et non un bouchon ;
    - **aucun ecretage.** Le niveau blanc passe de 1,05 a `WHITE_SINGLE` ;
    - **16 bits avec courbe et profil sRGB embarques**, directement exploitable
      dans DxO ou Nik, la finition et l'export JPEG se faisant la.

    Le centre vient de `select.interpolate_centers`, pas de `timelapse.csv` :
    celui-ci ne contient que les trames elues, dans la seule plage du
    timelapse. La normalisation, elle, est calculee sur la table complete, sa
    reprise sur la voisine exploitable la plus proche ayant besoin de toute la
    serie.
    """
    from . import diagnose, select

    config.exige_analyse("calibration.json", "metrics.csv")
    out_dir = out_dir or (config.SINGLE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = diagnose.load()
    i = _find_frame(d, at, fichier, tol_s)

    cfg = tirage_reglages(**over)

    if centre is not None:
        cx_full, cy_full = float(centre[0]), float(centre[1])
        origine = "force"
    else:
        if select.is_black(d)[i]:
            raise SystemExit(f"{d['file'][i]} : trame noire, rien a rendre")
        ic = select.interpolate_centers(d)
        if not (ic["recoverable"][i] and np.isfinite(ic["cx"][i])):
            raise SystemExit(
                f"{d['file'][i]} : centre non recuperable, forcer avec --centre x,y "
                "en pixels pleine resolution"
            )
        # le pixel R (i, j) est le photosite (2i, 2j) de la trame complete
        cx_full, cy_full = float(ic["cx"][i]) * 2.0, float(ic["cy"][i]) * 2.0
        origine = "interpole" if ic["interpolated"][i] else "mesure"

    rows = list(csv.DictReader((config.ANALYSIS_DIR / "metrics.csv").open()))
    norm = float(_norm_factors(rows, cfg["mode"])[i])
    fwhm = float(d["limb_width"][i])
    if not np.isfinite(fwhm):
        fwhm = float(np.nanmedian(d["limb_width"]))

    cal = Calibration.load()
    crop_px = int(round(cfg["crop_r"] * cal.r_sun_px * 2))
    chemin = {f.name: f for f in io.discover().frames}[d["file"][i]].path

    v = render_array(
        chemin,
        cx_full,
        cy_full,
        norm,
        cfg["wb"],
        cfg["target"],
        cfg["couleur"],
        crop_px,
        crop_px,  # jamais de reechantillonnage : le tirage est a l'echelle native
        cal.r_sun_px * 2.0,
        cfg["white"],
        1.0,  # lineaire ici, la courbe sRGB est appliquee a l'ecriture
        fwhm,
        cfg["nettete"],
        float(d["snr_disk"][i]),
        # le seuil de protection du ciel est une fraction du niveau de disque,
        # lequel vaut 1 / white dans ces unites
        SHARPEN_PROTECT / cfg["white"],
        toile,
        ancre,
    )
    haut, large = v.shape[:2]
    plein = float(v.max())

    info = {
        "file": str(d["file"][i]),
        "t": d["t"][i].isoformat(),
        "obsc_eph": float(d["obsc_eph"][i]),
        "alt_true": float(d["alt_true"][i]),
        "centre_full": [cx_full, cy_full],
        "origine_centre": origine,
        "norm": norm,
        "fwhm_arcsec": fwhm,
        "snr_disk": float(d["snr_disk"][i]),
        "taille": [large, haut],
        "ancre": [float(ancre[0]), float(ancre[1])],
        "diametre_disque_px": 4.0 * cal.r_sun_px,
        "arcsec_par_px": config.ARCSEC_PER_PX_FULL,
        "remplissage_echelle": plein,
        "transfert": "sRGB IEC61966-2.1",
        "reglages": {k: (list(x) if isinstance(x, tuple) else x) for k, x in cfg.items()},
    }

    # concatenation et non with_suffix : un nom de trame contient des points
    stem = str(out_dir / (nom or f"{d['t'][i]:%H%M%S}"))
    enc = tiff.srgb_encode(v)

    # Le texte se compose apres la courbe de tonalite : c'est un graphisme, il
    # doit rester blanc pur et d'opacite constante.
    #
    # La legende est editoriale, les etiquettes sont les circonstances mesurees
    # de la trame. Toute la legende tient a gauche, un saut de ligne y passant a
    # la ligne, les etiquettes occupent le coin droit. Les deux blocs partagent
    # une ligne de base.
    obsc = 100 * float(d["obsc_eph"][i])
    etiq = [
        f"{d['t'][i]:%H:%M:%S} UTC",
        f"{obsc:.1f} %".replace(".", ",") + f"   {float(d['alt_true'][i]):.1f}°".replace(".", ","),
    ]
    if legende or etiquettes:
        petit = min(large, haut)
        # un \n tape en ligne de commande arrive en deux caracteres
        lignes = legende.replace("\\n", "\n").split("\n") if legende else None
        texte.coins(
            enc,
            lignes,
            etiq if etiquettes else None,
            texte.LEGENDE_FRAC * petit,
            texte.ETIQUETTE_FRAC * petit,
            texte.MARGE_FRAC * petit,
        )
        info["legende"] = legende
        info["etiquettes"] = etiq if etiquettes else None

    icc = tiff.srgb_icc()
    tiff.write_rgb16(
        Path(stem + ".tif"),
        (enc * 65535.0 + 0.5).astype(np.uint16),
        icc,
        description=json.dumps(info, separators=(",", ":")),
        software="eclipse render_frame",
    )
    # apercu 8 bits, meme encodage : sert au tri, pas au tirage
    apercu = cv2.cvtColor((enc * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(stem + ".png", apercu)
    Path(stem + ".json").write_text(json.dumps(info, indent=2))

    if icc is None:
        print("attention : aucun profil sRGB trouve, le TIFF sort non tagge")
    if plein > 0.999:
        print(
            f"attention : l'image touche le blanc, relever --white au-dela de "
            f"{cfg['white']:g} pour ne rien ecreter"
        )

    # sous cette elevation le disque refracte s'ecarte de plus de 0,5 px de la
    # meilleure ellipse : le modele geometrique ne tient plus, le rendu reste
    # lisible mais le centrage n'est plus garanti au sous-pixel
    if d["alt_true"][i] < config.MIN_ALT_DEG_TIMELAPSE:
        print(
            f"attention : elevation {d['alt_true'][i]:.2f}°, sous la coupure "
            f"{config.MIN_ALT_DEG_TIMELAPSE}° du modele geometrique"
        )
    return {"stem": Path(stem), "index": i, **info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    # Les valeurs par defaut sont a None : `render_frame` doit pouvoir
    # distinguer un reglage impose d'un reglage absent, qu'il reprend alors de
    # `render.json`. La serie retombe sur `_CLI_DEFAULTS`.
    ap.add_argument("--at", help="trame isolee, heure UTC HHMMSS ou HH:MM:SS")
    ap.add_argument("--fichier", help="trame isolee, fragment de nom de fichier")
    ap.add_argument(
        "--nom",
        help="nom de sortie. Pour un tirage, le fichier dans out/single, defaut HHMMSS. "
        "Pour la serie, la video dans out, defaut timelapse. Les trames intermediaires "
        "restent dans out/frames, que chaque rendu purge",
    )
    ap.add_argument(
        "--centre",
        type=lambda v: tuple(float(x) for x in v.replace(",", " ").split()),
        help="centre force en pixels pleine resolution, pour une trame sans fit",
    )
    ap.add_argument(
        "--toile",
        type=lambda v: tuple(int(x) for x in v.lower().replace("x", " ").split()),
        help="cadre du tirage, LxH en pixels, par exemple 3456x2234. Le disque n'est "
        "jamais mis a l'echelle, le cadre est complete en noir",
    )
    ap.add_argument(
        "--legende",
        nargs="?",
        const=LEGENDE,
        default=None,
        help=f"legende. La premiere ligne va dans le coin superieur gauche, la suite dans le "
        f"coin superieur droit, calee a droite. Sans valeur : {LEGENDE!r}",
    )
    ap.add_argument(
        "--etiquettes",
        action="store_true",
        help="circonstances de la trame dans le coin superieur droit : heure UTC, "
        "obscuration et elevation",
    )
    ap.add_argument(
        "--pos",
        type=lambda v: tuple(float(x) for x in v.replace(",", " ").split()),
        default=(0.5, 0.5),
        help="position du centre du disque dans le cadre, en fractions de la largeur et de la "
        "hauteur. Par defaut 0,5 0,5, soit le meme alignement que le timelapse, qui pose lui "
        "aussi le centre au milieu de la trame",
    )
    ap.add_argument("--mode", choices=("disque", "instrument", "aucune"), default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--white", type=float, default=None, help="niveau blanc, en unites de disque")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument(
        "--tint",
        type=float,
        default=None,
        help="rechauffement, 0 = disque blanc, 1 = jaune solaire",
    )
    ap.add_argument(
        "--couleur",
        choices=("fixe", "unifiee"),
        default=None,
        help="fixe : balance figee, le soleil rougit naturellement en descendant. "
        "unifiee : chaque canal ramene au niveau du disque, teinte constante",
    )
    ap.add_argument(
        "--nettete",
        type=float,
        default=None,
        help="FWHM cible en arcsec pour la deconvolution, 0 pour ne pas accentuer. "
        "La FWHM mesuree vaut environ 6,6 arcsec, le plancher instrumental 2,1",
    )
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    if a.at or a.fichier:
        if a.size is not None or a.gamma is not None:
            raise SystemExit(
                "--size et --gamma ne s'appliquent qu'a la video : le tirage sort a "
                "l'echelle native avec la courbe sRGB. Utiliser --toile pour le cadre"
            )
        r = render_frame(
            at=a.at,
            fichier=a.fichier,
            nom=a.nom,
            centre=a.centre,
            toile=a.toile,
            ancre=a.pos,
            legende=a.legende,
            etiquettes=a.etiquettes,
            mode=a.mode,
            white=a.white,
            tint=a.tint,
            couleur=a.couleur,
            nettete=a.nettete,
        )
        g = r["reglages"]
        print(
            f"{r['file']}\n{r['t'][11:19]} UTC, obscuration {100 * r['obsc_eph']:.2f} %, "
            f"elevation {r['alt_true']:.2f}°"
        )
        cxf, cyf = r["centre_full"]
        print(
            f"centre {cxf:.2f} / {cyf:.2f} ({r['origine_centre']}), "
            f"fwhm {r['fwhm_arcsec']:.2f} arcsec, couleur {g['couleur']}, teinte {g['tint']:g}"
        )
        print(
            f"{r['taille'][0]} x {r['taille'][1]} px, disque {r['diametre_disque_px']:.0f} px "
            f"centre a {r['ancre'][0]:.3f} / {r['ancre'][1]:.3f} du cadre, "
            f"echelle native {r['arcsec_par_px']:.4f} arcsec/px, "
            f"blanc a {g['white']:g}, echelle remplie a {100 * r['remplissage_echelle']:.1f} %"
        )
        print("->", str(r["stem"]) + ".tif")
        raise SystemExit(0)

    if a.legende is not None or a.etiquettes:
        raise SystemExit("--legende et --etiquettes ne s'appliquent qu'a un tirage, avec --at")

    opt = {
        k: (v if v is not None else _CLI_DEFAULTS[k])
        for k, v in vars(a).items()
        if k in _CLI_DEFAULTS
    }
    res = run(**opt, nom=a.nom or "timelapse")
    print(
        f"{res['frames']} trames rendues en {res['seconds']:.0f}s, "
        f"decoupe {res['crop_px']} px ramenee a {res['size']} px"
    )
    print(
        f"couleur {res['couleur']}, balance R/V/B = "
        f"{res['wb'][0]:.3f} / {res['wb'][1]:.3f} / {res['wb'][2]:.3f}, "
        f"cible {res['target'][0]:.3f} / {res['target'][1]:.3f} / {res['target'][2]:.3f}"
    )
    print("->", res["video"] or res["dir"])
