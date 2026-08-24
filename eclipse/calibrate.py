"""Auto-calibration de la geometrie : rayon solaire et verticale locale.

Deux grandeurs sont fixes pour toute la session et ne se deduisent pas des
specifications nominales :

- le rayon en pixels, qui depend de la focale reelle et non des 400 mm annonces ;
- la direction de la verticale locale dans le repere capteur, la monture etant
  alt/az sans rotateur.

Les deux sortent du meme fit d'ellipse libre, sur des trames choisies pour leur
disque complet (rayon) et pour leur basse elevation (aplatissement mesurable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from . import astro, config, io, limb
from .config import DEFAULT_LIMB, LimbParams


@dataclass
class Calibration:
    r_sun_px: float
    r_scatter_px: float
    vert_angle_deg: float
    vert_scatter_deg: float
    arcsec_per_px_r: float
    n_radius: int
    n_vertical: int

    def save(self, path=None) -> None:
        path = path or (config.ANALYSIS_DIR / "calibration.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=None) -> Calibration:
        path = path or (config.ANALYSIS_DIR / "calibration.json")
        return cls(**json.loads(path.read_text()))


def _mask_center(f: np.ndarray, frac: float = 0.5) -> tuple[float, float] | None:
    """Barycentre du disque, valable seulement sur un disque complet."""
    peak = float(np.percentile(f, 99.5))
    if peak < 500.0:
        return None
    ys, xs = np.nonzero(f > frac * peak)
    if xs.size < 5000:
        return None
    return float(xs.mean()), float(ys.mean())


def ellipse_of_frame(
    img: np.ndarray,
    pedestal: float,
    sigma_bg: float,
    r_guess: float,
    p: LimbParams = DEFAULT_LIMB,
    n_iter: int = 3,
) -> tuple[float, float, float, float, float] | None:
    """Ellipse libre ajustee sur les passages a 50 % du limbe."""
    f = img.astype(np.float32) - pedestal
    c = _mask_center(f)
    if c is None:
        return None
    eh, ev = astro.local_axes(0.0)
    res = None
    for _ in range(n_iter):
        prof, s, dirs = limb._sample_rays(f, c, r_guess, 1.0, eh, ev, p)
        level0 = float(np.percentile(f, 99.9))
        s_edge, _, _, _, valid = limb._crossings(prof, s, r_guess, sigma_bg, level0, p)
        if valid.sum() < 200:
            return None
        pts = np.asarray(c)[None, :] + s_edge[valid, None] * dirs[valid]
        res = limb.free_ellipse_fit(pts)
        if not np.isfinite(res[0]):
            return None
        c = (res[0], res[1])
    return res


def run(
    session: io.Session | None = None,
    n_radius: int = 25,
    n_vertical: int = 40,
    verbose: bool = True,
) -> Calibration:
    """Calibre le rayon sur les trames non occultees, la verticale sur les basses."""
    session = session or io.discover()
    fr = session.frames
    eph = astro.ephemeris([f.t for f in fr])

    # Rayon : disque complet exige, donc obscuration nulle.
    idx_r = [i for i in range(len(fr)) if eph.obscuration[i] < 0.005][:n_radius]
    radii, ratios, angles = [], [], []
    for i in idx_r:
        img = io.load_r_plane(fr[i].path)
        ped, mad = io.pedestal(img)
        e = ellipse_of_frame(img, ped, max(mad, 1.0), config.R_SUN_PX_R)
        if e is None:
            continue
        _, _, a, b, _ = e
        # le grand axe est l'horizontale locale, non affecte par la refraction
        radii.append(a)
    r_sun = float(np.median(radii))
    r_scatter = float(1.4826 * np.median(np.abs(np.array(radii) - r_sun)))

    # Verticale : la mesurer sur l'aplatissement demande une elevation basse,
    # or dans cette session les basses elevations sont aussi les croissants les
    # plus fins, ou le fit d'ellipse libre est mal conditionne. On la mesure
    # donc sur la lune, dont l'angle de position est connu par les ephemerides
    # et lisible dans l'image. Independant de la refraction, et disponible sur
    # toutes les trames partiellement occultees.
    cand = [
        i
        for i in range(len(fr))
        if 0.05 < eph.obscuration[i] < 0.95 and eph.alt_true_deg[i] > config.MIN_ALT_DEG_TIMELAPSE
    ]
    step = max(1, len(cand) // n_vertical)
    idx_v = cand[::step][:n_vertical]
    eh0, ev0 = astro.local_axes(0.0)
    for i in idx_v:
        img = io.load_r_plane(fr[i].path)
        ped, mad = io.pedestal(img)
        m = limb.measure(img, ped, max(mad, 1.0), r_sun, eph.flattening[i], eh0, ev0)
        if not m.ok or m.n_points < 150 or m.rms > 1.0 or not np.isfinite(m.crescent_pa):
            continue
        pa_img = m.crescent_pa + 180.0  # direction du centre lunaire
        d = (pa_img - eph.pa_moon_deg[i] + 180.0) % 360.0 - 180.0
        angles.append(d)
        ratios.append(m.rms)
    if angles:
        a = np.radians(np.array(angles))
        vert = float(np.degrees(np.arctan2(np.median(np.sin(a)), np.median(np.cos(a)))))
        dev = (np.array(angles) - vert + 180.0) % 360.0 - 180.0
        vert_scatter = float(1.4826 * np.median(np.abs(dev)))
    else:
        vert, vert_scatter = 0.0, float("nan")

    cal = Calibration(
        r_sun_px=r_sun,
        r_scatter_px=r_scatter,
        vert_angle_deg=vert,
        vert_scatter_deg=vert_scatter,
        arcsec_per_px_r=float(np.median(eph.r_sun_arcsec)) / r_sun,
        n_radius=len(radii),
        n_vertical=len(angles),
    )
    if verbose:
        print(
            f"rayon      {cal.r_sun_px:7.2f} px  (dispersion {cal.r_scatter_px:.2f} px, "
            f"{cal.n_radius} trames)"
        )
        print(
            f"echelle    {cal.arcsec_per_px_r:7.4f} arcsec/px plan R  "
            f"({cal.arcsec_per_px_r / 2:.4f} pleine resolution)"
        )
        print(f"focale     {2.9e-3 * 206265 / (cal.arcsec_per_px_r / 2):7.1f} mm equivalents")
        print(
            f"verticale  {cal.vert_angle_deg:7.2f} deg  (dispersion {cal.vert_scatter_deg:.2f}, "
            f"{cal.n_vertical} trames)"
        )
    return cal
