"""Profils radiaux, detection de limbe, fit a rayon fige, metriques par trame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from . import config
from .config import DEFAULT_LIMB, LimbParams


@dataclass
class LimbMeasurement:
    """Resultat complet de la mesure geometrique et photometrique d'une trame."""

    ok: bool
    center: tuple[float, float] = (np.nan, np.nan)  # (x, y) dans le plan R
    n_points: int = 0
    rms: float = np.nan
    sigma_center: float = np.nan  # incertitude 1 sigma sur le centre, px, pire direction
    arc_span: float = np.nan  # etendue azimutale des points retenus, deg
    limb_width: float = np.nan  # arcsec, distance 10-90 %
    limb_width_aniso: float = np.nan  # arcsec, ecart max-min selon l'azimut
    sharp: float = np.nan  # gradient max normalise, 1/px
    ref_level: float = np.nan  # niveau photospherique local, ADU
    disk_median: float = np.nan  # ADU
    snr_disk: float = np.nan
    sigma_bg: float = np.nan  # ADU
    pedestal: float = np.nan
    sat_frac: float = np.nan
    obsc_area: float = np.nan  # obscuration mesuree sur l'image
    crescent_pa: float = np.nan  # angle de position de la bissectrice, deg
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    inliers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    reason: str = ""


def _sample_rays(
    img: np.ndarray,
    center: tuple[float, float],
    r_px: float,
    k: float,
    e_h: np.ndarray,
    e_v: np.ndarray,
    p: LimbParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Echantillonne le long des rayons. Renvoie (profils, s, directions).

    Le rayon `s` est le rayon elliptique : la position echantillonnee est
    c + s (cos phi e_h + k sin phi e_v), donc s = R sur le limbe modelise.
    """
    phi = np.linspace(0.0, 2 * np.pi, p.n_rays, endpoint=False)
    # direction (non unitaire) de chaque rayon, en coordonnees image
    dirs = np.cos(phi)[:, None] * e_h[None, :] + k * np.sin(phi)[:, None] * e_v[None, :]
    s = np.arange(p.r_in * r_px, p.r_out * r_px, p.step_px)
    pts = np.asarray(center)[None, None, :] + s[None, :, None] * dirs[:, None, :]
    coords = np.stack([pts[..., 1].ravel(), pts[..., 0].ravel()])  # (row, col)
    prof = map_coordinates(img, coords, order=1, mode="constant", cval=np.nan)
    return prof.reshape(p.n_rays, s.size), s, dirs


def _crossings(
    prof: np.ndarray, s: np.ndarray, r_px: float, sigma_bg: float, level0: float, p: LimbParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Passage de bord le plus externe sur chaque rayon, en interpolation lineaire.

    Le limbe lunaire est toujours interne au limbe solaire et son bord contre
    le ciel n'existe pas : ne garder que le passage le plus externe suffit a
    ecarter la lune, sans aucune classification.

    Le seuil est a 50 % d'une reference **locale au bord**, prise juste en
    dedans du limbe et non dans un anneau fixe a 0,85-0,95 R : pres du maximum
    le croissant ne fait plus qu'une dizaine de pixels et un anneau fixe tombe
    dans l'ombre lunaire. Deux passes : seuil global grossier, puis reference
    locale une fois le bord connu.
    """
    n_rays, n_s = prof.shape
    ds = float(s[1] - s[0])
    if not np.isfinite(level0) or level0 < max(8.0 * sigma_bg, 1.0):
        z = np.full(n_rays, np.nan)
        return z, z.copy(), z.copy(), z.copy(), np.zeros(n_rays, bool)

    s_edge = np.full(n_rays, np.nan)
    width = np.full(n_rays, np.nan)
    grad = np.full(n_rays, np.nan)
    ref = np.full(n_rays, np.nan)
    valid = np.zeros(n_rays, dtype=bool)

    win_in = max(int(round(0.06 * r_px / ds)), 4)  # fenetre de reference locale
    win_w = max(int(round(0.03 * r_px / ds)), 6)  # demi-fenetre de mesure de largeur

    for i in range(n_rays):
        row = prof[i]
        if not np.isfinite(row).all():
            continue
        j = _outermost_above(row, 0.5 * level0)
        if j is None:
            continue
        lo = max(j - win_in, 0)
        r_loc = float(np.percentile(row[lo : j + 1], 85)) if j > lo else float(row[j])
        # contraste absolu exige : un rayon qui ne voit que du bruit doit sortir
        if r_loc < max(8.0 * sigma_bg, 0.10 * level0):
            continue
        j = _outermost_above(row, 0.5 * r_loc)
        if j is None or j >= n_s - 2:
            continue
        # fond au-dela du limbe : rejette les rayons ou le ciel n'est pas noir
        out = float(np.median(row[min(j + win_w, n_s - 1) :])) if j + win_w < n_s - 1 else 0.0
        if out > 0.3 * r_loc:
            continue
        f = (row[j] - 0.5 * r_loc) / (row[j] - row[j + 1])
        s_edge[i] = s[j] + f * ds
        ref[i] = r_loc
        grad[i] = np.max(np.abs(np.diff(row[max(j - 4, 0) : j + 5]))) / (r_loc * ds)
        width[i] = _edge_width(row, ds, j, win_w)
        valid[i] = np.isfinite(s_edge[i])
    return s_edge, width, grad, ref, valid


def _outermost_above(row: np.ndarray, level: float) -> int | None:
    """Dernier indice au-dessus de `level`, donc bord le plus externe."""
    above = np.nonzero(row > level)[0]
    if above.size == 0:
        return None
    j = int(above[-1])
    return j if j < row.size - 1 else None


def _edge_width(row: np.ndarray, ds: float, j: int, win: int) -> float:
    """Largeur du bord, en pixels, par le rapport amplitude sur pente maximale.

    Une mesure 10-90 % sur le profil brut traverse l'assombrissement
    centre-bord, qui s'etend sur des dizaines de pixels et n'a rien a voir avec
    la turbulence : elle donne 60 arcsec la ou la PSF en fait 8. Le second
    moment du gradient, lui, depend fortement de la fenetre a cause des ailes
    de lumiere diffusee.

    Pour un bord franc d'amplitude A convolue par une gaussienne d'ecart-type
    sigma, max|dI/ds| = A / (sigma racine(2 pi)). D'ou FWHM = 0,9394 A / max|dI/ds|,
    sans fenetre d'integration et sans sensibilite aux ailes.
    """
    n = row.size
    a_in = max(j - 8, 0)
    b_out0, b_out1 = min(j + 6, n), min(j + 14, n)
    if b_out1 - b_out0 < 3 or j - a_in < 3 or j + 5 > n:
        return np.nan
    amp = float(np.percentile(row[a_in : j + 1], 90) - np.median(row[b_out0:b_out1]))
    # lisser sur 1 px avant de deriver : l'echantillonnage est a 0,25 px et le
    # maximum brut de |diff| attrape un pic de bruit sur les trames faibles, ce
    # qui ecrase artificiellement la largeur mesuree jusqu'au plancher
    seg = row[max(j - 8, 0) : min(j + 9, n)]
    if seg.size < 9:
        return np.nan
    ker = np.ones(4) / 4.0
    gmax = float(np.max(np.abs(np.diff(np.convolve(seg, ker, mode="valid"))))) / ds
    if gmax <= 0 or amp <= 0:
        return np.nan
    w = 0.9394 * amp / gmax
    return w if w > 1.5 else np.nan  # sous 1,5 px la mesure n'est plus resolue


def fit_center(
    pts: np.ndarray,
    r_px: float,
    k: float,
    e_h: np.ndarray,
    e_v: np.ndarray,
    p: LimbParams,
    c0: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Ajuste le centre a rayon fige, avec rejet sigma-clip itere.

    Deux parametres au lieu de trois : le conditionnement reste bon sur un
    croissant fin, ou un rayon libre partirait a la derive.
    """
    basis = np.stack([e_h, e_v])  # (2, 2), lignes = axes locaux
    q = pts @ basis.T  # composantes locales (u, v)
    q[:, 1] /= k  # espace circularise
    c0_local = np.asarray(c0) @ basis.T
    c0_local[1] /= k

    keep = np.ones(len(q), dtype=bool)
    c = c0_local
    for _ in range(p.clip_iters):
        sub = q[keep]
        if sub.shape[0] < p.min_points:
            break
        res = least_squares(
            lambda cc, q=sub: np.hypot(q[:, 0] - cc[0], q[:, 1] - cc[1]) - r_px,
            c,
            method="lm",
        )
        c = res.x
        d = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) - r_px
        sigma = 1.4826 * np.median(np.abs(d[keep] - np.median(d[keep])))
        new = np.abs(d) < max(p.clip_sigma * sigma, 0.2)
        if new.sum() < p.min_points or np.array_equal(new, keep):
            keep = new if new.sum() >= p.min_points else keep
            break
        keep = new
    d = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) - r_px
    rms = float(np.sqrt(np.mean(d[keep] ** 2))) if keep.any() else np.nan
    sigma_c, span = _center_uncertainty(q[keep], c, rms)
    c_local = np.array([c[0], c[1] * k])
    center = c_local @ basis  # retour en coordonnees image
    return center, keep, rms, sigma_c, span


def _center_uncertainty(q: np.ndarray, c: np.ndarray, rms: float) -> tuple[float, float]:
    """Incertitude sur le centre et etendue azimutale des points retenus.

    Le nombre de points ne dit pas si le centre est contraint : quarante points
    etales sur cent degres d'arc le determinent mieux que mille serres sur
    vingt. Le residu vaut |p - c| - R, sa derivee par rapport au centre est le
    vecteur unitaire radial, d'ou une matrice normale somme(u u^T). Sa plus
    petite valeur propre porte la direction mal contrainte, celle de la normale
    a l'arc quand le croissant est court.

    Deux reserves. La valeur est calculee dans l'espace circularise, ou l'axe
    vertical est dilate de 1/k, soit moins de 4 % sur la plage retenue. Et elle
    est optimiste : des rayons voisins partagent la meme turbulence, leurs
    erreurs ne sont pas independantes. C'est une grandeur de classement
    relative, pas une barre d'erreur publiable.
    """
    if q.shape[0] < 3 or not np.isfinite(rms):
        return np.nan, np.nan
    v = q - c
    n = np.hypot(v[:, 0], v[:, 1])
    good = n > 1e-6
    if good.sum() < 3:
        return np.nan, np.nan
    u = v[good] / n[good, None]
    ev = np.linalg.eigvalsh(u.T @ u)
    if ev[0] <= 1e-9:
        return float("inf"), 0.0
    sigma = float(rms / np.sqrt(ev[0]))
    ang = np.sort(np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360.0)
    gaps = np.diff(np.concatenate([ang, ang[:1] + 360.0]))
    span = float(360.0 - gaps.max())
    return sigma, span


def free_ellipse_fit(pts: np.ndarray) -> tuple[float, float, float, float, float]:
    """Ajustement d'ellipse libre par la conique generale.

    Sert uniquement a l'auto-calibration de la verticale locale : la direction
    du petit axe donne l'angle d'aplatissement dans le repere capteur.
    Renvoie (cx, cy, a, b, theta_deg) avec theta l'angle du PETIT axe depuis -y.
    """
    x, y = pts[:, 0], pts[:, 1]
    x0, y0 = x.mean(), y.mean()
    sc = np.sqrt(((x - x0) ** 2 + (y - y0) ** 2).mean())
    xn, yn = (x - x0) / sc, (y - y0) / sc
    d = np.stack([xn**2, xn * yn, yn**2, xn, yn, np.ones_like(xn)], axis=1)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    a, b, c, dd, e, f = vt[-1]
    m = np.array([[a, b / 2], [b / 2, c]])
    if np.linalg.det(m) <= 0:
        return (np.nan,) * 5
    cen = np.linalg.solve(2 * m, [-dd, -e])
    val = a * cen[0] ** 2 + b * cen[0] * cen[1] + c * cen[1] ** 2 + dd * cen[0] + e * cen[1] + f
    evals, evecs = np.linalg.eigh(m / -val)
    axes = 1.0 / np.sqrt(evals)  # croissant : axes[0] est le grand
    i_min = int(np.argmin(axes))
    v = evecs[:, i_min]  # direction du petit axe
    theta = np.degrees(np.arctan2(v[0], -v[1]))
    theta = (theta + 90.0) % 180.0 - 90.0
    return (
        float(cen[0] * sc + x0),
        float(cen[1] * sc + y0),
        float(max(axes) * sc),
        float(min(axes) * sc),
        float(theta),
    )


def measure(
    img: np.ndarray,
    pedestal: float,
    sigma_bg: float,
    r_px: float,
    k: float,
    e_h: np.ndarray,
    e_v: np.ndarray,
    c0: tuple[float, float] | None = None,
    p: LimbParams = DEFAULT_LIMB,
    n_outer: int = 3,
    level0: float | None = None,
) -> LimbMeasurement:
    """Mesure complete d'une trame : centre, qualite geometrique, photometrie."""
    f = img.astype(np.float32) - pedestal
    if level0 is None:
        # Niveau photospherique de reference, pris sur l'image entiere et non
        # sur la fraction de rayons eclaires : pres du maximum le croissant ne
        # couvre plus qu'un rayon sur vingt, et un percentile sur les rayons
        # renvoie le fond. Le seuil de passe 1 tombait alors dans le bruit et
        # fabriquait un anneau complet de faux bords.
        level0 = float(np.percentile(f, 99.9))
    if c0 is None:
        c0 = coarse_center(f, sigma_bg, r_px, k, e_h, e_v)
        if c0 is None:
            return LimbMeasurement(
                ok=False, pedestal=pedestal, sigma_bg=sigma_bg, reason="aucun signal, trame noire"
            )
    center = np.asarray(c0, dtype=float)
    pts = np.empty((0, 2))
    keep = np.empty(0, dtype=bool)
    rms = np.nan
    width = grad = ref = np.empty(0)
    valid = np.empty(0, dtype=bool)

    for _ in range(n_outer):
        prof, s, dirs = _sample_rays(f, tuple(center), r_px, k, e_h, e_v, p)
        s_edge, width, grad, ref, valid = _crossings(prof, s, r_px, sigma_bg, level0, p)
        if valid.sum() < p.min_points:
            return LimbMeasurement(
                ok=False,
                pedestal=pedestal,
                sigma_bg=sigma_bg,
                n_points=int(valid.sum()),
                reason="limbe insuffisant",
            )
        pts = center[None, :] + s_edge[valid, None] * dirs[valid]
        center, keep, rms, sigma_c, span = fit_center(pts, r_px, k, e_h, e_v, p, tuple(center))

    if not np.isfinite(rms) or rms > 5.0:
        return LimbMeasurement(
            ok=False, pedestal=pedestal, sigma_bg=sigma_bg, rms=float(rms), reason="fit divergent"
        )

    m = LimbMeasurement(
        ok=True,
        center=(float(center[0]), float(center[1])),
        n_points=int(keep.sum()),
        rms=float(rms),
        sigma_center=float(sigma_c),
        arc_span=float(span),
        points=pts,
        inliers=keep,
        pedestal=pedestal,
        sigma_bg=sigma_bg,
    )
    _photometry(m, f, width, grad, ref, valid, keep, pts, r_px, k, e_h, e_v, img)
    return m


def _photometry(m, f, width, grad, ref, valid, keep, pts, r_px, k, e_h, e_v, raw) -> None:
    """Metriques de qualite. `keep` porte sur les rayons valides, dans l'ordre."""
    w = width[valid][keep]
    g = grad[valid][keep]
    r = ref[valid][keep]
    m.limb_width = float(np.nanmedian(w)) * config.ARCSEC_PER_PX_R
    m.sharp = float(np.nanmedian(g))
    m.ref_level = float(np.nanmedian(r))

    # anisotropie : ecart entre secteurs azimutaux, revele le file directionnel
    q = pts[keep] - np.asarray(m.center)
    ang = np.degrees(np.arctan2(q @ e_v, q @ e_h)) % 180.0
    bins = np.clip((ang // 30).astype(int), 0, 5)
    med = [np.nanmedian(w[bins == i]) for i in range(6) if (bins == i).sum() >= 5]
    m.limb_width_aniso = (
        float(np.nanmax(med) - np.nanmin(med)) * config.ARCSEC_PER_PX_R if len(med) >= 3 else np.nan
    )

    # masque du disque solaire ajuste
    h, wid = f.shape
    y, x = np.mgrid[0:h, 0:wid]
    du = (x - m.center[0]) * e_h[0] + (y - m.center[1]) * e_h[1]
    dv = (x - m.center[0]) * e_v[0] + (y - m.center[1]) * e_v[1]
    inside = (du / r_px) ** 2 + (dv / (k * r_px)) ** 2 <= 1.0
    disk = f[inside]
    lit = disk > 0.5 * m.ref_level
    m.obsc_area = float(1.0 - lit.mean()) if disk.size else np.nan
    m.disk_median = float(np.median(disk[lit])) if lit.any() else np.nan
    if lit.sum() > 500:
        # bruit estime sur les differences de pixels voisins : l'ecart-type brut
        # du disque mesurerait l'assombrissement centre-bord, pas le bruit
        lit_map = np.zeros_like(inside)
        lit_map[inside] = lit
        pair = lit_map[:, :-1] & lit_map[:, 1:]
        d = (f[:, 1:] - f[:, :-1])[pair]
        noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
        m.snr_disk = float(m.disk_median / noise) if noise > 0 else np.nan
    else:
        m.snr_disk = np.nan
    m.sat_frac = float((raw >= config.SATURATION_ADU).mean())

    if lit.any():
        cu = du[inside][lit].mean()
        cv = dv[inside][lit].mean()
        m.crescent_pa = float(np.degrees(np.arctan2(cu, cv)))


def coarse_center(
    f: np.ndarray,
    sigma_bg: float,
    r_px: float,
    k: float,
    e_h: np.ndarray,
    e_v: np.ndarray,
    down: int = 4,
) -> tuple[float, float] | None:
    """Centre approche par vote sur la direction du gradient.

    Le rayon etant connu, chaque pixel de bord vote pour le centre situe a R
    dans la direction du gradient. Sur le limbe solaire l'intensite croit vers
    l'interieur : les votes convergent. Sur le limbe lunaire le gradient pointe
    a l'oppose de la lune, les votes se dispersent. Aucune classification n'est
    necessaire, le pic de l'accumulateur est le centre solaire.
    """
    sub = f[::down, ::down]
    peak = float(np.percentile(sub, 99.9))
    if peak < max(20.0 * sigma_bg, 200.0):
        return None
    gy, gx = np.gradient(sub)
    g = np.hypot(gx, gy)
    thr = max(0.05 * peak / down, 4.0 * sigma_bg)
    ys, xs = np.nonzero(g > thr)
    if ys.size < 100:
        return None
    n = g[ys, xs]
    ux, uy = gx[ys, xs] / n, gy[ys, xs] / n
    # rayon local du modele elliptique dans la direction du vote
    du = ux * e_h[0] + uy * e_h[1]
    dv = ux * e_v[0] + uy * e_v[1]
    rho = r_px / np.sqrt(du**2 + (dv / k) ** 2) / down
    cx = xs + ux * rho
    cy = ys + uy * rho
    h, w = sub.shape
    m = (cx > -w) & (cx < 2 * w) & (cy > -h) & (cy < 2 * h)
    if m.sum() < 100:
        return None
    acc, ex, ey = np.histogram2d(
        cx[m], cy[m], bins=[np.arange(-w, 2 * w + 1, 2), np.arange(-h, 2 * h + 1, 2)]
    )
    from scipy.ndimage import uniform_filter

    sm = uniform_filter(acc, 3)
    i, j = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return float((ex[i] + 1) * down), float((ey[j] + 1) * down)
