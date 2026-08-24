"""Rapport HTML autonome : images en base64, aucune dependance externe."""

from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import subprocess
from pathlib import Path

import numpy as np

from . import config, diagnose, select
from .calibrate import Calibration
from .config import DEFAULT_LIMB

CSS = """
:root { color-scheme: dark; }
body { background:#14161a; color:#d8dbe0; font:15px/1.55 system-ui,-apple-system,sans-serif;
       margin:0; padding:0 0 6rem; }
main { max-width: 1180px; margin: 0 auto; padding: 0 1.5rem; }
h1 { font-size:1.7rem; margin:2.2rem 0 .3rem; }
h2 { font-size:1.2rem; margin:2.6rem 0 .8rem; padding-bottom:.35rem;
     border-bottom:1px solid #2b3038; color:#eef1f5; }
h3 { font-size:1rem; margin:1.6rem 0 .5rem; color:#aab3c0; font-weight:600; }
p, li { max-width: 78ch; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:.6rem 0 1.2rem; }
th, td { padding:.32rem .6rem; text-align:left; border-bottom:1px solid #23272e; }
th { color:#9aa4b2; font-weight:600; white-space:nowrap; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
img { max-width:100%; height:auto; display:block; margin:.6rem 0 1.4rem;
      border:1px solid #23272e; border-radius:3px; }
.kv { display:grid; grid-template-columns:auto 1fr; gap:.15rem 1.2rem; font-size:14px;
      max-width:70ch; }
.kv dt { color:#9aa4b2; } .kv dd { margin:0; font-variant-numeric:tabular-nums; }
code { background:#1c2027; padding:.1rem .3rem; border-radius:3px; font-size:.9em; }
code.fn { white-space:normal; word-break:break-all; font-size:11.5px; line-height:1.35;
          background:none; padding:0; color:#c3cad4; }
td.fn { min-width:31ch; }
pre.list { background:#1c2027; border:1px solid #23272e; border-radius:3px; padding:.7rem .9rem;
           font-size:12px; line-height:1.5; overflow-x:auto; max-height:26rem; }
.note { color:#9aa4b2; font-size:13.5px; }
.tag { display:inline-block; padding:0 .4rem; border-radius:3px; font-size:12px; }
.A { background:#1d4023; color:#8fe0a2; } .B { background:#463318; color:#e8c07d; }
.C { background:#2a2e36; color:#9aa4b2; }
.scroll { overflow-x:auto; }
"""


# Qualite JPEG du mode allege. 88 ne laisse aucun artefact visible sur les
# planches, qui sont des montages de disques sur fond noir.
JPEG_QUALITE = 88

# Mis par build(leger=True). Les images en sont le seul poids du rapport : 6,9 Mo
# sur 7,0 pour la session complete.
_LEGER = False


def _en_jpeg(path: Path) -> bytes | None:
    """Recode une image en JPEG. None si la lecture echoue."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITE])
    return buf.tobytes() if ok else None


def _img(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="note">image absente : {html.escape(path.name)}</p>'
    mime = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
    brut = path.read_bytes()
    # Le JPEG n'est retenu que s'il allege : il divise par trois les planches de
    # trames, mais double les courbes de controle, qui sont du trait sur aplat.
    if _LEGER and mime == "png":
        jpg = _en_jpeg(path)
        if jpg is not None and len(jpg) < len(brut):
            brut, mime = jpg, "jpeg"
    b64 = base64.b64encode(brut).decode("ascii")
    return f'<img src="data:image/{mime};base64,{b64}" alt="{html.escape(alt)}">'


def _git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5,
        )
        return r.stdout.strip() or "aucun commit"
    except Exception:
        return "indisponible"


def _kv(pairs: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{html.escape(k)}</dt><dd>{v}</dd>" for k, v in pairs)
    return f'<dl class="kv">{items}</dl>'


def _table(
    headers: list[str],
    rows: list[list[str]],
    num_cols: set[int] | None = None,
    fn_cols: set[int] | None = None,
) -> str:
    num, fn = num_cols or set(), fn_cols or set()

    def cls(i: int) -> str:
        return ' class="num"' if i in num else ' class="fn"' if i in fn else ""

    th = "".join(f"<th{cls(i)}>{html.escape(h)}</th>" for i, h in enumerate(headers))
    body = []
    for r in rows:
        body.append("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + th
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _render_line() -> str:
    """Parametres du dernier rendu, pour que le rapport soit reproductible."""
    p = config.ANALYSIS_DIR / "render.json"
    if not p.exists():
        return "aucun rendu produit"
    import json

    r = json.loads(p.read_text())
    cou = (
        "couleur unifiee, chaque canal ramene au niveau du disque"
        if r["couleur"] == "unifiee"
        else "balance figee, le soleil rougit naturellement en descendant"
    )
    return (
        f"decoupe {r['crop_r']} R ramenee a {r['size']} px, {r['fps']} images/s, "
        f"normalisation d'intensite « {r['mode']} », {cou}, teinte {r['tint']:g}, "
        f"cible R/V/B {r['target'][0]:.3f} / {r['target'][1]:.3f} / {r['target'][2]:.3f}. "
        + (
            f"Accentuation vers {r['nettete']:g} arcsec de FWHM, PSF mesuree par trame, "
            f"mediane {r.get('fwhm_mediane', float('nan')):.2f} arcsec"
            if r.get("nettete")
            else "Aucune accentuation"
        )
    )


def _annotations() -> list[tuple[Path, str]]:
    from . import report

    try:
        return report.annotation_images()
    except Exception:
        return []


def build(out: Path | None = None, leger: bool = False) -> Path:
    """Ecrit le rapport autonome.

    `leger` recode en JPEG les images que cela allege, pour une version
    publiable : environ 3 Mo au lieu de 7 pour la session complete.
    """
    global _LEGER
    _LEGER = leger
    out = out or (config.REPORT_DIR / "rapport.html")
    d = diagnose.load()
    cal = Calibration.load()
    cls = select.classify(d)
    tl = select.in_timelapse(d)
    veil = select.veiled(d)
    ok = d["ok"]
    o = config.ANALYSIS_DIR
    r = config.REPORT_DIR

    parts: list[str] = []
    parts.append(
        f"<h1>Eclipse partielle du 12 aout 2026, {config.SITE_NAME}</h1>"
        f'<p class="note">Rapport genere le {dt.datetime.now():%Y-%m-%d %H:%M} — '
        f"code <code>{_git_hash()}</code></p>"
    )

    # --- Resume
    i_max = int(np.argmax(d["obsc_eph"]))
    ab = ok & (cls != "C")
    deep = int(np.nonzero(tl & ab)[0][np.argmax(d["obsc_eph"][tl & ab])])
    tlrows = list((o / "timelapse.csv").open()) if (o / "timelapse.csv").exists() else []
    n_tl = max(len(tlrows) - 1, 0)
    parts.append("<h2>Resume</h2>")
    parts.append(
        _kv(
            [
                ("Trames traitees", f"{len(ok)}"),
                ("Plage horaire (UTC)", f"{d['t'][0]:%H:%M:%S} a {d['t'][-1]:%H:%M:%S}"),
                ("Elevation solaire", f"{d['alt_true'].max():.1f}° a {d['alt_true'].min():.1f}°"),
                (
                    "Obscuration maximale (ephemerides)",
                    f"{100 * d['obsc_eph'][i_max]:.2f} % a {d['t'][i_max]:%H:%M:%S}",
                ),
                (
                    "Trame exploitable la plus profonde",
                    f"{100 * d['obsc_eph'][deep]:.2f} % a {d['t'][deep]:%H:%M:%S} "
                    f"(classe {cls[deep]}, sigma {d['sigma_center'][deep]:.2f} px)",
                ),
                ("Centres mesures", f"{int(ab.sum())} ({100 * ab.mean():.0f} %)"),
                (
                    "Classes sur la plage timelapse",
                    "  ".join(
                        f'<span class="tag {c}">{c} {int((cls[tl] == c).sum())}</span>'
                        for c in "ABC"
                    ),
                ),
                ("Trames du timelapse", f"{n_tl}"),
                ("Duree du rendu", f"{n_tl / 25:.1f} s a 25 images/s"),
            ]
        )
    )

    # --- Site
    parts.append("<h2>Lieu et conditions</h2>")
    parts.append(
        "<p>Bord de champ a Montastruc, Hautes-Pyrenees, entre Castelbajac et Houeydets, au "
        f"nord de Lannemezan, a {config.SITE_ALT_M:.0f} m d'altitude.</p>"
    )
    try:
        from . import astro

        ev = astro.contacts()
    except Exception:
        ev = []
    if ev:
        parts.append(
            _table(
                ["Evenement", "Heure legale", "UTC", "Elevation", "Azimut", "Obscuration"],
                [
                    [
                        e["label"] + ("" if e["visible"] else " (soleil couche)"),
                        e["local"],
                        e["utc"],
                        f"{e['alt_app']:.1f}°",
                        f"{e['az']:.1f}°",
                        f"{100 * e['obsc']:.2f} %",
                    ]
                    for e in ev
                ],
                num_cols={3, 4, 5},
            )
        )
        parts.append(
            '<p class="note">Heure legale CEST, UTC+2. Une eclipse partielle n\'a que deux '
            "contacts : C2 et C3 marquent le debut et la fin de la phase totale, qui n'existe "
            "pas ici. Elevation apparente, refraction comprise ; le coucher est pris au sens "
            "usuel, bord superieur du disque a l'horizon. C4 tombe apres le coucher et n'a pas "
            "ete observable.</p>"
        )
    for path, name in _annotations():
        parts.append(_img(path, name))

    # --- Configuration
    parts.append("<h2>Configuration</h2>")
    parts.append(
        _kv(
            [
                ("Lunette", "SkyOptic 66/400, f/6"),
                ("Camera", "ZWO ASI 585 MC Air, IMX585, 3840 x 2160, 2,9 µm, RGGB"),
                ("Monture", "Celestron NexStar SLT, alt/az, sans rotateur"),
                ("Filtre", "Astrosolar OD 3,8 en avant d'ouverture, plus IR-cut"),
                (
                    "Site",
                    f"{config.SITE_NAME}, {config.SITE_LAT:.4f} N, {config.SITE_LON:.4f} E, "
                    f"{config.SITE_ALT_M:.0f} m",
                ),
                (
                    "Rayon solaire calibre",
                    f"{cal.r_sun_px:.2f} px plan R (dispersion {cal.r_scatter_px:.2f} px, "
                    f"{cal.n_radius} trames)",
                ),
                (
                    "Echelle mesuree",
                    f"{cal.arcsec_per_px_r:.4f} arcsec/px plan R, "
                    f"{cal.arcsec_per_px_r / 2:.4f} pleine resolution",
                ),
                ("Focale equivalente", f"{2.9e-3 * 206265 / (cal.arcsec_per_px_r / 2):.1f} mm"),
                (
                    "Verticale locale calibree",
                    f"{cal.vert_angle_deg:.2f}° (dispersion {cal.vert_scatter_deg:.2f}°, "
                    f"{cal.n_vertical} trames, mesuree sur l'angle de position de la lune)",
                ),
                ("Rayons de mesure", f"{DEFAULT_LIMB.n_rays}, seuil a 50 % d'une reference locale"),
                (
                    "Coupure du timelapse",
                    f"elevation > {config.MIN_ALT_DEG_TIMELAPSE}°, "
                    "au-dela le disque refracte s'ecarte de plus de 0,5 px de la meilleure ellipse",
                ),
                ("Rendu", _render_line()),
                (
                    "Classes",
                    f"A : sigma_centre &lt; {select.CLASS_A['sigma_center']} px et "
                    f"rms &lt; {select.CLASS_A['rms']} px. "
                    f"B : &lt; {select.CLASS_B['sigma_center']} px et "
                    f"&lt; {select.CLASS_B['rms']} px",
                ),
            ]
        )
    )

    # --- Courbes
    parts.append("<h2>Courbes de controle</h2>")
    parts.append(_img(r / "controle.png", "controles de la passe de mesure"))
    parts.append(_img(r / "controle_selection.png", "controles de la selection"))

    # --- Planches
    parts.append("<h2>Planches</h2>")
    parts.append("<h3>Selection : trame elue de chaque tranche de phase</h3>")
    parts.append(_img(r / "planche_selection.png", "planche de selection"))
    parts.append("<h3>Croissants fins, au-dela de 85 % d'obscuration</h3>")
    parts.append(_img(r / "planche_maximum.png", "planche des croissants fins"))
    parts.append("<h3>Controle : pires rms et trames ecretees</h3>")
    parts.append(
        '<p class="note">Points verts : limbe retenu. Points roses : rejets du sigma-clip. '
        "Un paquet de rose d'un seul cote signale un biais systematique, pas du bruit.</p>"
    )
    parts.append(_img(r / "planche_controle.png", "planche de controle"))

    # --- Nettete
    parts.append("<h2>Nettete</h2>")
    parts.append(
        "<p>L'accentuation est une deconvolution de Wiener dont la PSF vient de la mesure : "
        "la largeur du limbe donne la FWHM de chaque trame, et le rapport signal sur bruit du "
        "disque fixe la regularisation. Rien n'est regle a l'oeil.</p>"
        "<p>Trois faits mesures justifient le modele. L'anisotropie du limbe ne vaut que 6 % "
        "de la FWHM, un noyau gaussien isotrope suffit donc. La FWHM tient dans 6,46 a "
        "6,70 arcsec entre les premier et neuvieme deciles des trames retenues, la turbulence "
        "a donc peu derive et l'adaptation par trame change peu de chose ici. Enfin la FWHM "
        "vaut 4,4 px pleine resolution pour un plancher instrumental a 1,4 px : l'image est "
        "sur-echantillonnee par rapport a son propre contenu, la deconvolution a de la marge "
        "avant de buter sur l'echantillonnage.</p>"
        "<p>Le cout se lit en bruit, pas en rebonds. La sous-oscillation au limbe reste sous "
        "le seuil de visibilite, le ciel etant a zero, et la sur-oscillation ne bouge que de "
        "deux points. C'est l'amplification du bruit qui borne le reglage :</p>"
    )
    parts.append(
        _table(
            ["FWHM cible", "Bruit sur la photosphere", "Facteur", "Largeur de bord obtenue"],
            [
                ["brute", "0,862 %", "1,00x", '9,67"'],
                ['5,0"', "1,114 %", "1,29x", '7,97"'],
                ['4,5"', "1,346 %", "1,56x", '7,22"'],
                ['4,2"', "1,508 %", "1,75x", '6,80"'],
                ['3,5"', "1,957 %", "2,27x", '5,91"'],
            ],
            num_cols={1, 2, 3},
        )
    )
    parts.append(
        '<p class="note">La largeur obtenue est mesuree sur l\'image rendue, avec une '
        "reference prise plus haut que l'amplitude locale du bord : elle surestime la FWHM "
        "vraie et ne vaut que comme indicateur relatif. La cible n'est jamais atteinte "
        "exactement, la regularisation coupant les plus hautes frequences.</p>"
    )
    parts.append(_img(r / "planche_nettete.png", "effet de l'accentuation"))

    # --- Couleur
    parts.append("<h2>Couleur du rendu</h2>")
    parts.append(
        "<p>La balance des blancs est mesuree une fois pour toute la seance, sur les trames "
        "de classe A non voilees. Elle ne suffit pourtant pas a tenir la couleur : "
        "l'extinction differentielle rougit le soleil a mesure qu'il descend, la masse d'air "
        f"passant de {np.nanmin(d['airmass'][tl]):.1f} a {np.nanmax(d['airmass'][tl]):.1f} sur la "
        "plage retenue. Le rapport "
        "rouge sur vert du disque glisse alors de 1,00 a 1,55 sur la sequence retenue.</p>"
        "<p>En mode <code>unifiee</code>, chaque canal est ramene a son propre niveau "
        "photospherique mesure sur la trame, puis la teinte cible est imposee. La teinte "
        "devient constante au millieme. Sous 400 ADU par canal ou 300 pixels eclaires, la "
        "normalisation par trame est abandonnee au profit de la balance figee : en dessous, "
        "les rapports entre canaux ne mesurent plus que le bruit.</p>"
    )
    parts.append(_img(r / "planche_couleur_modes.png", "comparaison des deux modes de couleur"))
    parts.append(_img(r / "planche_couleur_teintes.png", "reglages de teinte"))
    parts.append(
        '<p class="note">Le mode <code>fixe</code> reste disponible si l\'on prefere garder '
        "le rougissement du coucher, qui est un phenomene reel.</p>"
    )

    # --- Tableau de selection
    parts.append("<h2>Selection par tranche de phase</h2>")
    sel_csv = o / "selection_phase.csv"
    if sel_csv.exists():
        rows = list(csv.DictReader(sel_csv.open()))
        parts.append(
            _table(
                [
                    "Heure",
                    "Obsc %",
                    "Elev °",
                    "Phase",
                    "Score",
                    'limb_width "',
                    "SNR",
                    "rms px",
                    "sigma px",
                    "Classe",
                    "Voile",
                    "Fichier",
                ],
                [
                    [
                        r["t"][11:19],
                        r["obsc"],
                        r["alt"],
                        f"{r['branche']} {r['bin']}",
                        r["score"],
                        r["limb_width"],
                        r["snr_disk"],
                        r["rms"],
                        r["sigma_center"],
                        f'<span class="tag {r["classe"]}">{r["classe"]}</span>',
                        "oui" if r["voile"] == "1" else "",
                        f'<code class="fn">{html.escape(r["file"])}</code>',
                    ]
                    for r in rows
                ],
                num_cols={1, 2, 4, 5, 6, 7, 8},
                fn_cols={11},
            )
        )
        parts.append(
            "<h3>Liste des trames elues, a copier</h3>"
            '<pre class="list">' + html.escape("\n".join(r["file"] for r in rows)) + "</pre>"
        )

    # --- Rafales
    parts.append("<h2>Couverture par rafale</h2>")
    rows = []
    for b in sorted(set(d["burst"].astype(int))):
        m = d["burst"] == b
        rows.append(
            [
                f"{b}",
                f"{d['t'][m][0]:%H:%M:%S}",
                f"{int(m.sum())}",
                f"{100 * d['obsc_eph'][m].mean():.1f}",
                f"{d['alt_true'][m].mean():.1f}",
                f"{float(d['exptime'][m][0]) * 1e3:g}",
                f"{int(d['gain'][m][0])}",
                f"{int((cls[m] == 'A').sum())}",
                f"{int((cls[m] == 'B').sum())}",
                f"{int((cls[m] == 'C').sum())}",
                f"{int((~veil & (cls != 'C'))[m].sum())}",
            ]
        )
    parts.append(
        _table(
            [
                "Rafale",
                "Debut",
                "Trames",
                "Obsc %",
                "Elev °",
                "Pose ms",
                "Gain",
                "A",
                "B",
                "C",
                "Claires",
            ],
            rows,
            num_cols={2, 3, 4, 5, 6, 7, 8, 9, 10},
        )
    )

    # --- Anomalies
    parts.append("<h2>Anomalies</h2>")
    an = []
    apath = o / "anomalies.txt"
    if apath.exists():
        an += [line.strip() for line in apath.read_text().splitlines() if line.strip()]
    sat = ok & (d["sat_frac"] > 1e-5)
    if sat.any():
        an.append(
            f"{int(sat.sum())} trames ecretees, jusqu'a "
            f"{100 * np.nanmax(d['sat_frac']):.1f} % des pixels ; "
            f"profil comprime, centrage et nettete biaises."
        )
    an.append(
        f"Les en-tetes FITS portent un site memorise a {config.HEADER_SITE_ERROR_KM} km au "
        "sud du lieu reel : l'ASIAIR a garde une position enregistree au lieu de relever la "
        "sienne. Le nom du dossier de session vient de la "
        "meme erreur. Corrige dans le calcul : l'ecart deplacait le maximum de 34 s et "
        "l'obscuration maximale de 0,36 point."
    )
    gains = sorted({int(g) for g in d["gain"]})
    an.append(
        "Gain modifie en cours de seance : "
        + ", ".join(str(g) for g in gains)
        + ". Le basculement HCG a 200 est traverse, tout le traitement reste en relatif."
    )
    exps = sorted({int(round(e * 1e6)) for e in d["exptime"]})
    an.append(
        f"{len(exps)} valeurs de pose employees, de {min(exps) / 1000:g} ms a "
        f"{max(exps) / 1000:g} ms, soit un facteur {max(exps) / min(exps):.0f}."
    )
    for b in sorted(set(d["burst"].astype(int))):
        m = d["burst"] == b
        if (cls[m] == "C").all():
            cause = (
                "sous la coupure d'elevation, le modele d'ellipse est hors domaine"
                if d["alt_true"][m].mean() < config.MIN_ALT_DEG_TIMELAPSE
                else "nuage"
            )
            an.append(
                f"Rafale {b} ({d['t'][m][0]:%H:%M:%S}, obscuration "
                f"{100 * d['obsc_eph'][m].mean():.1f} %, elevation "
                f"{d['alt_true'][m].mean():.1f}°) : {int(m.sum())} trames, aucune "
                f"exploitable, {cause}."
            )
    an.append(
        "Aucun recadrage manuel detecte : le centre balaie "
        f"{np.ptp(d['cx'][ok]):.0f} x {np.ptp(d['cy'][ok]):.0f} px sur la seance, "
        "avec une derive continue et des repositionnements entre rafales."
    )
    parts.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in an) + "</ul>")

    # --- Limites
    parts.append("<h2>Limites</h2>")
    lim = [
        "Pas de flats. Poussieres et vignettage ne sont pas corriges. Sans effet sur le "
        "centrage, qui normalise le seuil de limbe localement le long de chaque rayon.",
        "Pas de darks ni de bias. Le piedestal est la mediane de quatre fenetres de 128 px "
        "prises dans les coins, mesuree par trame, ce qui absorbe a la fois les changements "
        "de gain et la derive thermique. Le champ est trop etroit pour un anneau a 1,6 R des "
        "que le disque est decentre : les coins sont le seul fond disponible. Ils se tiennent "
        "a 949 px au minimum du centre ajuste, soit trois fois le rayon solaire, et ne "
        "contiennent donc jamais de lumiere du disque, verifie sur les trames mesurees.",
        "Le bruit de fond n'est pas pris sur l'ecart absolu median des coins, qui vaut souvent "
        f"exactement zero : le fond est si noir qu'il tient dans un pas de quantification de "
        f"{config.QUANT_ADU:.0f} ADU. Il est estime sur les differences de pixels voisins, avec "
        "plancher a ce pas. Sans ce plancher, tout seuil exprime en multiples de sigma "
        "s'ouvrait sur le bruit et fabriquait des faux bords.",
        "Refroidissement desactive. La temperature capteur passe de "
        f"{d['ccd_temp'].max():.1f} a {d['ccd_temp'].min():.1f} °C. Tracee, non corrigee.",
        "Aplatissement du disque par la refraction conserve au rendu : c'est un phenomene "
        "que l'observateur voit. Il n'est corrige que temporairement, pendant le fit "
        "geometrique, pour ne pas biaiser le centre.",
        f"Timelapse coupe a {config.MIN_ALT_DEG_TIMELAPSE}° d'elevation. Au-dela le disque "
        "n'est plus une ellipse et le modele geometrique ne tient plus. Ces trames restent "
        "classees et exportees pour traitement individuel.",
        "Trois reechantillonnages au rendu : dematricage, puis translation et decoupe "
        "fusionnees en une seule passe Lanczos, puis reduction a la taille de sortie. La "
        "specification en prevoyait un seul. Le supprimer demanderait un dematricage sur "
        "grille decalee et un rendu a la taille native, pour un gain marginal.",
        "L'accentuation est une deconvolution, donc une reconstruction. A la cible retenue "
        "elle multiplie par 1,56 le bruit de la photosphere, et la cible n'est jamais atteinte "
        "exactement, la regularisation coupant les plus hautes frequences. Desactivable.",
        "La couleur unifiee supprime un phenomene reel, le rougissement du soleil couchant par "
        "extinction differentielle. C'est un choix de rendu, pas une correction. Le mode a "
        "balance figee le conserve.",
        "sigma_centre est optimiste : les rayons voisins partagent la meme turbulence, leurs "
        "erreurs ne sont pas independantes. Grandeur de classement relative, pas barre "
        "d'erreur.",
        "La transparence brute mesure surtout l'extinction atmospherique, la masse d'air "
        f"passant de {np.nanmin(d['airmass']):.1f} a {np.nanmax(d['airmass']):.1f}. Le "
        "detecteur de voile utilise est l'ecart entre obscuration mesuree et ephemeride.",
        "Les coordonnees du site ne viennent pas d'un releve GPS mais de la carte "
        "d'annotation, validees en reproduisant les circonstances qu'elle affiche. "
        "L'incertitude sur le point exact est de l'ordre du kilometre, sans effet mesurable "
        "sur les grandeurs calculees.",
        "Le maximum reel de l'eclipse n'a aucune trame exploitable : les 50 trames de la "
        "rafale de 18:25:48 UTC plafonnent a un pas de quantification au-dessus du fond. "
        f"L'obscuration maximale atteinte sur une trame utilisable est de "
        f"{100 * d['obsc_eph'][deep]:.2f} % contre {100 * d['obsc_eph'][i_max]:.2f} % au "
        "maximum theorique.",
    ]
    parts.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in lim) + "</ul>")

    doc = (
        "<!doctype html><html lang=fr><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>Eclipse partielle 2026-08-12, {config.SITE_NAME}</title>"
        f"<style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rapport HTML autonome de la session.")
    ap.add_argument("--out", type=Path, help="fichier de sortie")
    ap.add_argument(
        "--leger",
        action="store_true",
        help="recode les planches en JPEG, pour une version publiable",
    )
    a = ap.parse_args()
    print("->", build(a.out, leger=a.leger))
