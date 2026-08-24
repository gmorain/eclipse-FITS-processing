"""Vignettes annotees et planches de controle."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from . import astro, config, io, limb
from .calibrate import Calibration

CROP_R = 2.6  # cote de la vignette, en rayons solaires


def vignette(
    ax,
    frame: io.Frame,
    cal: Calibration,
    k: float,
    obsc_eph: float,
    px: int = 400,
    seed: tuple[float, float] | None = None,
) -> dict:
    """Trace une vignette annotee dans `ax` et renvoie les mesures de la trame.

    Cadrage standardise a 2,6 R centre sur le centre ajuste : toutes les
    vignettes ont la meme echelle et le disque au meme endroit. Le
    reechantillonnage precede le trace, qui reste vectoriel : des traits fins
    restent nets.
    """
    img = io.load_r_plane(frame.path)
    ped, sigma = io.pedestal(img)
    eh, ev = astro.local_axes(cal.vert_angle_deg)
    # amorcee par le centre retenu par la passe de mesure : sans cela la
    # vignette re-mesure la trame isolement et n'illustre pas ce que le
    # pipeline a decide, notamment sur les croissants fins ou l'amorce fait
    # toute la difference
    m = limb.measure(img, ped, sigma, cal.r_sun_px, k, eh, ev, c0=seed)
    f = img.astype(np.float32) - ped

    half = CROP_R * cal.r_sun_px / 2
    cx, cy = m.center if m.ok else (img.shape[1] / 2, img.shape[0] / 2)
    x0, y0 = cx - half, cy - half
    xs = np.linspace(x0, x0 + 2 * half, px)
    ys = np.linspace(y0, y0 + 2 * half, px)
    from scipy.ndimage import map_coordinates

    gx, gy = np.meshgrid(xs, ys)
    crop = map_coordinates(f, [gy.ravel(), gx.ravel()], order=1, mode="constant", cval=0.0).reshape(
        px, px
    )

    scale = px / (2 * half)  # px vignette par px capteur
    ref = m.ref_level if (m.ok and np.isfinite(m.ref_level)) else max(np.percentile(crop, 99.9), 1)
    ax.imshow(
        np.clip(crop / ref, 0, 1.15),
        cmap="afmhot",
        vmin=0,
        vmax=1.15,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if m.ok:
        to_v = lambda p: ((p[..., 0] - x0) * scale, (p[..., 1] - y0) * scale)  # noqa: E731
        ang = np.degrees(np.arctan2(ev[1], ev[0])) + 90.0
        ax.add_patch(
            Ellipse(
                ((cx - x0) * scale, (cy - y0) * scale),
                2 * cal.r_sun_px * scale,
                2 * k * cal.r_sun_px * scale,
                angle=ang,
                fill=False,
                ec="#3af",
                lw=0.8,
            )
        )
        for r, st in ((0.85, (0, (2, 3))), (0.95, (0, (2, 3)))):
            ax.add_patch(
                Ellipse(
                    ((cx - x0) * scale, (cy - y0) * scale),
                    2 * r * cal.r_sun_px * scale,
                    2 * r * k * cal.r_sun_px * scale,
                    angle=ang,
                    fill=False,
                    ec="#3af",
                    lw=0.4,
                    ls=st,
                    alpha=0.45,
                )
            )
        px_in, py_in = to_v(m.points[m.inliers])
        px_out, py_out = to_v(m.points[~m.inliers])
        ax.plot(px_in, py_in, ".", ms=1.6, color="#00ff66", mew=0, alpha=0.9)
        ax.plot(px_out, py_out, ".", ms=3.0, color="#ff2090", mew=0)
        ax.plot([(cx - x0) * scale], [(cy - y0) * scale], "+", color="#3af", ms=7, mew=1.0)
        # verticale locale calibree
        ax.annotate(
            "",
            xy=(px * 0.07 + ev[0] * 34, px * 0.93 + ev[1] * 34),
            xytext=(px * 0.07, px * 0.93),
            arrowprops=dict(arrowstyle="->", color="w", lw=0.9),
        )
    ax.set_xlim(0, px)
    ax.set_ylim(px, 0)

    lab = (
        f"{frame.t:%H:%M:%S}  obsc {100 * obsc_eph:.1f}%\n"
        f'N={m.n_points} rms={m.rms:.2f}px  lw={m.limb_width:.1f}"\n'
        f"{frame.exptime * 1e3:g}ms g{frame.gain}  {m.reason}"
    )
    ax.set_title(lab, fontsize=6.2, family="monospace", pad=2)
    return {"m": m, "frame": frame}


def board(indices: list[int], out: Path, title: str, ncol: int = 4) -> Path:
    from . import diagnose

    session = io.discover()
    fr = session.frames
    cal = Calibration.load()
    eph = astro.ephemeris([f.t for f in fr])
    d = diagnose.load()
    seeds = {
        i: (float(d["cx"][i]), float(d["cy"][i]))
        for i in indices
        if np.isfinite(d["cx"][i]) and np.isfinite(d["cy"][i])
    }
    nrow = int(np.ceil(len(indices) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.35 * nrow), facecolor="0.12")
    for ax, i in zip(np.ravel(axes), indices, strict=False):
        vignette(
            ax,
            fr[i],
            cal,
            float(eph.flattening[i]),
            float(eph.obscuration[i]),
            seed=seeds.get(i),
        )
    for ax in np.ravel(axes)[len(indices) :]:
        ax.axis("off")
    for ax in np.ravel(axes):
        ax.title.set_color("0.9")
    fig.suptitle(title, color="0.95", y=0.998, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def selection_board(k: int = 1) -> Path:
    """La trame elue de chaque tranche de phase, dans l'ordre chronologique.

    C'est la planche qu'on regarde. Le classement vient de `rank`, donc du score
    composite normalise par rafale, et non du seul critere de nettete : une
    trame nette mais a demi voilee doit perdre des points.
    """
    from . import diagnose, rank

    d = diagnose.load()
    picks = rank.top_k(d, k=k)
    return board(
        [p.index for p in picks],
        config.REPORT_DIR / "planche_selection.png",
        "Planche de selection : trame elue de chaque tranche de phase",
    )


def control_board(n: int = 12) -> Path:
    """Les trames de plus mauvais rms parmi celles retenues, plus les saturees.

    C'est la planche qui revele les bugs : un paquet de points rejetes d'un
    seul cote signale un biais systematique, pas du bruit.
    """
    from . import diagnose

    d = diagnose.load()
    ok = d["ok"] & (d["alt_true"] > config.MIN_ALT_DEG_TIMELAPSE)
    idx = np.nonzero(ok)[0]
    worst = idx[np.argsort(-d["rms"][idx])][: n // 2]
    sat = idx[np.argsort(-d["sat_frac"][idx])][: n - len(worst)]
    sel = sorted(set(worst.tolist() + sat.tolist()))[:n]
    return board(
        sel,
        config.REPORT_DIR / "planche_controle.png",
        "Planche de controle : pires rms et trames ecretees",
    )


def maximum_board(n: int = 12) -> Path:
    """Trames retenues les plus proches du maximum, ordre chronologique.

    C'est la planche qui valide la detection sur croissant fin : ce sont les
    trames que la premiere version du detecteur rejetait toutes.
    """
    from . import diagnose, select

    d = diagnose.load()
    cls = select.classify(d)
    m = select.in_timelapse(d) & (cls != "C") & (d["obsc_eph"] > 0.85)
    idx = np.nonzero(m)[0]
    order = idx[np.argsort(-d["obsc_eph"][idx])][:n]
    return board(
        sorted(order.tolist()),
        config.REPORT_DIR / "planche_maximum.png",
        "Croissants fins : trames retenues au-dela de 85 % d'obscuration",
    )


def _colour_samples(n: int = 5) -> list[dict]:
    """Trames de reference pour les planches de couleur, etalees sur la sequence.

    Choisies claires et bien mesurees : c'est la teinte que l'on compare, un
    voile la fausserait.
    """
    import csv

    rows = list(csv.DictReader((config.ANALYSIS_DIR / "timelapse.csv").open()))
    good = [
        r for r in rows if r["classe"] == "A" and r["voile"] == "0" and float(r["qualite"]) > 0.45
    ]
    if len(good) < n:
        good = [r for r in rows if r["classe"] == "A"] or rows
    # etalees en obscuration et non en rang : c'est la progression de l'eclipse
    # que la planche doit montrer
    # borne haute : au-dela le croissant est trop mince pour juger d'une teinte
    good = [r for r in good if float(r["obsc"]) < 88.0] or good
    obsc = [float(r["obsc"]) for r in good]
    targets = np.linspace(min(obsc), max(obsc), n)
    picked, used = [], set()
    for t in targets:
        i = int(np.argmin([abs(o - t) + (1e6 if j in used else 0) for j, o in enumerate(obsc)]))
        used.add(i)
        picked.append(good[i])
    return picked


def _colour_grid(
    bands: list[tuple[str, list[np.ndarray]]], out: Path, title: str, subtitle: str
) -> Path:
    nrow, ncol = len(bands), len(bands[0][1])
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow + 0.55), facecolor="0.12", squeeze=False
    )
    for row, (label, imgs) in enumerate(bands):
        for col, im in enumerate(imgs):
            ax = axes[row][col]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if col == 0:
                ax.set_ylabel(label, color="0.92", fontsize=9)
    fig.suptitle(title, color="0.95", fontsize=11, y=0.985)
    fig.text(0.5, 0.938, subtitle, color="0.66", fontsize=8.5, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def colour_boards(size: int = 260) -> dict[str, Path]:
    """Deux planches : modes de couleur, et reglages de teinte.

    Les memes trames sont rendues sous chaque reglage, sans quoi la comparaison
    ne voudrait rien dire.
    """
    import json

    from . import render

    cal = Calibration.load()
    session = io.discover()
    by_name = {f.name: f for f in session.frames}
    crop_px = int(round(render.CROP_R * cal.r_sun_px * 2))
    r_full = cal.r_sun_px * 2.0
    wb0 = tuple(json.loads((config.ANALYSIS_DIR / "white_balance.json").read_text()))
    sam = _colour_samples()

    import csv as _csv

    metrics = {r["file"]: r for r in _csv.DictReader((config.ANALYSIS_DIR / "metrics.csv").open())}

    def rows_for(couleur: str, tint: float) -> list[np.ndarray]:
        wb = render.apply_tint(wb0, tint)
        target = render.apply_tint((1.0, 1.0, 1.0), tint)
        out = []
        for r in sam:
            ref = metrics[r["file"]]["ref_level"]
            norm = float(ref) if ref not in ("", "nan") else 1.0
            out.append(
                render.render_array(
                    by_name[r["file"]].path,
                    float(r["cx_full"]),
                    float(r["cy_full"]),
                    norm,
                    wb,
                    target,
                    couleur,
                    crop_px,
                    size,
                    r_full,
                )
            )
        return out

    tint = json.loads((config.ANALYSIS_DIR / "render.json").read_text())["tint"]
    modes = _colour_grid(
        [("balance figee", rows_for("fixe", tint)), ("couleur unifiee", rows_for("unifiee", tint))],
        config.REPORT_DIR / "planche_couleur_modes.png",
        "Deux modes de couleur, memes trames",
        f"teinte {tint:g}, obscuration croissante de gauche a droite",
    )
    tints = _colour_grid(
        [(f"teinte {t:g}", rows_for("unifiee", t)) for t in (0.4, 0.7, 1.0)],
        config.REPORT_DIR / "planche_couleur_teintes.png",
        "Reglage de la teinte, en couleur unifiee",
        "0 donne un disque neutre, 1 le jaune solaire",
    )
    return {"modes": modes, "teintes": tints}


def sharpen_board(targets=(0.0, 5.0, 4.2, 3.5), win: int = 300) -> Path:
    """Effet de l'accentuation, a l'echelle native, sur une tache et sur le limbe.

    Deux zones, parce qu'elles disent deux choses differentes : une tache
    solaire montre ce que la deconvolution rend en detail, le limbe montre ce
    qu'elle coute en rebonds. Juger sur la seule tache donnerait envie de
    pousser trop loin.
    """
    import csv
    import json

    from scipy.ndimage import uniform_filter

    from . import render

    cal = Calibration.load()
    session = io.discover()
    by_name = {f.name: f for f in session.frames}
    rows = list(csv.DictReader((config.ANALYSIS_DIR / "timelapse.csv").open()))
    metrics = {r["file"]: r for r in csv.DictReader((config.ANALYSIS_DIR / "metrics.csv").open())}
    wb0 = tuple(json.loads((config.ANALYSIS_DIR / "white_balance.json").read_text()))
    tint = json.loads((config.ANALYSIS_DIR / "render.json").read_text())["tint"]

    cand = [r for r in rows if r["classe"] == "A" and r["voile"] == "0" and float(r["obsc"]) < 45]
    cand.sort(key=lambda r: float(metrics[r["file"]]["limb_width"]))
    r = cand[0]
    m = metrics[r["file"]]
    crop_px = int(round(render.CROP_R * cal.r_sun_px * 2))
    R = cal.r_sun_px * 2.0
    args = dict(
        path=by_name[r["file"]].path,
        cx_full=float(r["cx_full"]),
        cy_full=float(r["cy_full"]),
        norm=float(m["ref_level"]),
        wb=render.apply_tint(wb0, tint),
        target=render.apply_tint((1.0, 1.0, 1.0), tint),
        couleur="unifiee",
        crop_px=crop_px,
        size=crop_px,
        r_full=R,
        fwhm_cur=float(m["limb_width"]),
        snr=float(m["snr_disk"]),
    )
    base = render.render_array(**args, fwhm_tgt=0.0)
    lum = base @ np.array([0.30, 0.59, 0.11])
    c = crop_px / 2

    # tache solaire : minimum local entoure de photosphere claire. Un simple
    # minimum trouverait le bord lunaire, qui est bien plus sombre.
    yy, xx = np.mgrid[0:crop_px, 0:crop_px]
    bg = uniform_filter(lum, 61)
    loc = uniform_filter(lum, 5)
    disk = ((xx - c) ** 2 + (yy - c) ** 2) < (0.80 * R) ** 2
    ok_bg = disk & (bg > 0.75 * np.median(lum[disk & (lum > 0.5)]))
    contrast = np.where(ok_bg, (bg - loc) / np.maximum(bg, 1e-3), -1.0)
    py, px = divmod(int(np.argmax(contrast)), crop_px)
    # limbe du cote eclaire
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    prof = [lum[int(c + 0.97 * R * np.sin(a)), int(c + 0.97 * R * np.cos(a))] for a in ang]
    a = ang[int(np.argmax(prof))]
    ly, lx = int(c + R * np.sin(a)), int(c + R * np.cos(a))

    def cut(img, x, y):
        h = win // 2
        x = int(np.clip(x, h, crop_px - h))
        y = int(np.clip(y, h, crop_px - h))
        return img[y - h : y + h, x - h : x + h]

    cols = []
    for t in targets:
        v = base if t == 0 else render.render_array(**args, fwhm_tgt=t)
        cols.append((("brut" if t == 0 else f'cible {t:g}"'), [cut(v, px, py), cut(v, lx, ly)]))

    ncol = len(cols)
    fig, axes = plt.subplots(2, ncol, figsize=(2.5 * ncol, 5.6), facecolor="0.12", squeeze=False)
    for col, (label, imgs) in enumerate(cols):
        for row, im in enumerate(imgs):
            ax = axes[row][col]
            ax.imshow(np.clip(im, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(label, color="0.92", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("tache solaire" if row == 0 else "limbe", color="0.92", fontsize=9)
    fig.suptitle(
        "Accentuation : deconvolution de Wiener calee sur la PSF mesuree",
        color="0.95",
        fontsize=11,
        y=0.985,
    )
    fig.text(
        0.5,
        0.943,
        f'{r["t"][11:19]}, FWHM mesuree {float(m["limb_width"]):.2f}", '
        f"snr_disk {float(m['snr_disk']):.0f}, echelle native "
        f"{config.ARCSEC_PER_PX_FULL:.3f} arcsec/px",
        color="0.66",
        fontsize=8.5,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = config.REPORT_DIR / "planche_nettete.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


ANNOTATION_MAX_WIDTH = 1500


def annotation_images() -> list[tuple[Path, str]]:
    """Copies reduites des captures d'annotation, pretes a embarquer dans le rapport.

    Les originaux font plusieurs megaoctets : embarques tels quels en base64 ils
    tripleraient la taille du rapport pour aucun gain de lisibilite.
    """
    import cv2

    src = config.ANNOTATIONS_DIR
    if src is None or not src.exists():
        return []
    out: list[tuple[Path, str]] = []
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.png")):
        if p.name.startswith("._"):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if w > ANNOTATION_MAX_WIDTH:
            s = ANNOTATION_MAX_WIDTH / w
            img = cv2.resize(
                img, (ANNOTATION_MAX_WIDTH, int(round(h * s))), interpolation=cv2.INTER_AREA
            )
        # JPEG et non PNG : ce sont des captures de cartes, le PNG y pese
        # quatre fois plus pour une difference invisible
        dst = config.REPORT_DIR / f"annotation_{len(out)}.jpg"
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append((dst, p.stem))
    return out
