"""Classification des trames et selection.

Pour l'instant : la classification A/B/C seule. La programmation dynamique de
selection temporelle et l'interpolation des centres viendront ici.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

# La specification classait sur N, le nombre de points de limbe. C'est le
# mauvais critere : N est borne par la geometrie, a 94 % d'obscuration le limbe
# solaire n'offre plus que 44 % de sa circonference et une trame parfaite n'en
# rend que quarante. Un seuil absolu declasserait tout le maximum.
#
# Le critere utile est l'incertitude sur le centre issue du fit, qui integre a
# la fois le nombre de points, leur etalement azimutal et la dispersion des
# residus. Elle repond directement a la question du timelapse : de combien de
# pixels le recalage peut-il se tromper.
CLASS_A = {"sigma_center": 0.10, "rms": 0.6}
CLASS_B = {"sigma_center": 0.60, "rms": 1.5}

# Seuil de voile, sur l'ecart entre l'obscuration mesuree et celle des
# ephemerides. Ne participe pas a la classe : une trame aux trois quarts voilee
# dont l'arc de limbe visible donne un centre a 0,08 px reste une excellente
# ancre de position. Le voile disqualifie l'image, pas la geometrie.
VEIL_GAP = 0.08


def classify(d: dict[str, np.ndarray]) -> np.ndarray:
    """Classe geometrique 'A', 'B' ou 'C', sur la fiabilite du centre seule.

    La specification classait sur N, le nombre de points de limbe. C'est le
    mauvais critere : N est borne par la geometrie, a 94 % d'obscuration le
    limbe solaire n'offre plus que 44 % de sa circonference et une trame
    parfaite n'en rend que quarante. Un seuil absolu declasserait tout le
    maximum.

    Le critere utile est l'incertitude sur le centre issue du fit, qui integre a
    la fois le nombre de points, leur etalement azimutal et la dispersion des
    residus. Elle repond directement a la question du timelapse : de combien de
    pixels le recalage peut-il se tromper.
    """
    ok = d["ok"] & np.isfinite(d["rms"]) & np.isfinite(d["sigma_center"])
    out = np.full(len(ok), "C", dtype="<U1")
    b = ok & (d["sigma_center"] < CLASS_B["sigma_center"]) & (d["rms"] < CLASS_B["rms"])
    a = ok & (d["sigma_center"] < CLASS_A["sigma_center"]) & (d["rms"] < CLASS_A["rms"])
    out[b] = "B"
    out[a] = "A"
    return out


def veiled(d: dict[str, np.ndarray]) -> np.ndarray:
    """Trames dont une part du disque est cachee par autre chose que la lune.

    Meilleur detecteur de nuage que la transparence, qui sur cette session est
    dominee par l'extinction atmospherique : la masse d'air passe de 3,4 a 16
    entre la premiere et la derniere rafale.
    """
    gap = np.abs(np.nan_to_num(d["obsc_gap"], nan=1.0))
    return gap >= VEIL_GAP


def in_timelapse(d: dict[str, np.ndarray]) -> np.ndarray:
    """Plage ou le modele d'ellipse refractee tient (residu sous 0,5 px)."""
    return d["alt_true"] > config.MIN_ALT_DEG_TIMELAPSE


def interpolate_centers(d: dict[str, np.ndarray], max_gap_s: float = 120.0) -> dict:
    """Complete la trajectoire du centre pour toutes les trames de chaque rafale.

    Segmentation par rafale et non par saut manuel : la session n'en contient
    aucun. Le centre balaie 120 x 89 px sur toute la duree, avec une derive
    mediane de 0,31 px/s, et les repositionnements ont lieu entre rafales. On
    n'interpole donc jamais d'une rafale a l'autre.

    Renvoie cx, cy completes, `interpolated` (vrai si la position vient de
    l'ajustement et non de la mesure) et `recoverable` (faux si la rafale
    n'offre aucune ancre ou si le trou temporel est trop grand).
    """
    cls = classify(d)
    anchor = (cls != "C") & np.isfinite(d["cx"])
    ts = np.array([(x - d["t"][0]).total_seconds() for x in d["t"]])
    cx = d["cx"].astype(float).copy()
    cy = d["cy"].astype(float).copy()
    interp = np.zeros(len(cx), dtype=bool)
    rec = np.zeros(len(cx), dtype=bool)

    for b in np.unique(d["burst"]).astype(int):
        m = d["burst"] == b
        a = m & anchor
        if not a.any():
            continue
        rec[m] = True
        idx = np.nonzero(m & ~anchor)[0]
        ta, xa, ya = ts[a], cx[a], cy[a]
        order = np.argsort(ta)
        ta, xa, ya = ta[order], xa[order], ya[order]
        for i in idx:
            # trou temporel jusqu'a l'ancre la plus proche
            if np.min(np.abs(ta - ts[i])) > max_gap_s:
                rec[i] = False
                continue
            cx[i] = np.interp(ts[i], ta, xa)
            cy[i] = np.interp(ts[i], ta, ya)
            interp[i] = True
        # les ancres restent telles quelles
    return {"cx": cx, "cy": cy, "interpolated": interp, "recoverable": rec, "anchor": anchor}


def is_black(d: dict[str, np.ndarray]) -> np.ndarray:
    """Trames sans aucun signal : nuage epais, rien a aligner ni a rendre.

    Conservees dans la selection avec une position interpolee, pour que le
    rendu decide seul entre garder le rythme reel de la seance et fondre d'une
    trame exploitable a la suivante.
    """
    return d["reason"] == "aucun signal, trame noire"


def _quality(d: dict[str, np.ndarray]) -> np.ndarray:
    """Qualite bornee dans [0, 1], derivee du score composite de `rank`."""
    from . import rank

    s = rank.score(d)
    q = 1.0 / (1.0 + np.exp(-s / 2.0))
    q[~np.isfinite(s)] = 0.0
    return q


@dataclass
class Timelapse:
    """Suite de trames retenues pour le rendu, et son diagnostic."""

    indices: list[int]
    delta_s: float
    objective: float
    skipped: list[tuple[int, int, int]]  # (index avant, index apres, intervalles sautes)


def select_timelapse(
    d: dict[str, np.ndarray],
    delta_s: float,
    lam: float = 1.0,
    mu: float = 0.5,
    k_cap: int = 3,
    candidates: np.ndarray | None = None,
) -> Timelapse:
    """Programmation dynamique : qualite totale sous contrainte de regularite.

    score(i) = max_j [ score(j) + q(i) - lam err^2 - mu min(k - 1, k_cap) ]
    avec k = round((t_i - t_j) / delta) et err = |t_i - t_j - k delta| / delta.

    Le terme en k autorise explicitement a sauter un intervalle quand un nuage
    a tout mange, plutot que de forcer une mauvaise trame ou de casser la
    serie. `mu` arbitre entre continuite et regularite stricte.

    Deux ecarts a la specification, imposes par la structure de la session.

    La penalite de saut est **plafonnee**. Les rafales sont espacees de 30 s a
    11 min : avec un pas de 4 s, franchir une attente de dix minutes couterait
    150 fois mu, davantage que tout ce que la suite de la seance rapporte. La
    version non plafonnee arretait la selection a 18:18, avant le maximum. Une
    attente entre rafales est un fait de la prise de vue, pas un defaut a punir
    proportionnellement a sa duree.

    Le chemin est **reconstruit depuis la derniere trame candidate** et non
    depuis le meilleur score cumule, qui est toujours atteint juste avant le
    premier grand trou. Le timelapse doit couvrir la seance.
    """
    q = _quality(d)
    ok = candidates if candidates is not None else (q > 0)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return Timelapse([], delta_s, -np.inf, [])
    ts = np.array([(d["t"][i] - d["t"][idx[0]]).total_seconds() for i in idx])

    best = q[idx].copy()
    prev = np.full(idx.size, -1)
    for i in range(1, idx.size):
        dt = ts[i] - ts[:i]
        k = np.maximum(np.round(dt / delta_s), 1.0)
        err = np.abs(dt - k * delta_s) / delta_s
        cand = best[:i] + q[idx[i]] - lam * err**2 - mu * np.minimum(k - 1.0, k_cap)
        j = int(np.argmax(cand))
        if cand[j] > best[i]:
            best[i] = cand[j]
            prev[i] = j

    end = idx.size - 1
    chain = []
    while end >= 0:
        chain.append(end)
        end = prev[end]
    chain.reverse()

    skipped = []
    for a, b in zip(chain[:-1], chain[1:], strict=False):
        k = int(round((ts[b] - ts[a]) / delta_s))
        if k > 1:
            skipped.append((int(idx[a]), int(idx[b]), k - 1))
    return Timelapse(
        indices=[int(idx[i]) for i in chain],
        delta_s=delta_s,
        objective=float(best[chain[-1]]),
        skipped=skipped,
    )


def sweep_delta(
    d: dict[str, np.ndarray], deltas: np.ndarray | None = None, **kw
) -> tuple[Timelapse, list[tuple[float, int, float]]]:
    """Balaye le pas temporel et garde le meilleur objectif.

    La specification proposait de balayer autour de la mediane des ecarts
    entre rafales. Cette session ne s'y prete pas : les rafales sont espacees
    de 30 s a 11 min sans regularite. On balaye donc une plage large et on
    rend la courbe, qui est le vrai element de decision.
    """
    if deltas is None:
        deltas = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60], dtype=float)
    curve, best = [], None
    for dt in deltas:
        tl = select_timelapse(d, float(dt), **kw)
        curve.append((float(dt), len(tl.indices), tl.objective))
        if best is None or tl.objective > best.objective:
            best = tl
    return best, curve


def export(
    d: dict[str, np.ndarray],
    delta_s: float = 8.0,
    out_dir=None,
    lam: float = 1.0,
    mu: float = 0.5,
) -> dict:
    """Ecrit la liste des trames du timelapse et leurs positions de recalage."""
    import csv

    out_dir = out_dir or config.ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ic = interpolate_centers(d)
    cls = classify(d)
    veil = veiled(d)
    q = _quality(d)
    black = is_black(d)
    cand = in_timelapse(d) & ic["recoverable"] & np.isfinite(ic["cx"])
    tl = select_timelapse(d, delta_s, lam=lam, mu=mu, candidates=cand)
    idx = np.array(tl.indices, dtype=int)

    # translation a appliquer, en pixels pleine resolution : le centre est
    # ramene a la mediane de la serie pour eviter un recadrage inutile. Le pixel
    # R (i, j) du plan sous-echantillonne est le photosite (2i, 2j) de la trame
    # complete, d'ou le facteur deux.
    ref_x = float(np.median(ic["cx"][idx])) * 2.0
    ref_y = float(np.median(ic["cy"][idx])) * 2.0

    path = out_dir / "timelapse.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "ordre",
                "t",
                "obsc",
                "alt",
                "classe",
                "interpole",
                "voile",
                "noir",
                "qualite",
                "cx_r",
                "cy_r",
                "cx_full",
                "cy_full",
                "dx_full",
                "dy_full",
                "file",
            ]
        )
        for n, i in enumerate(idx):
            xf, yf = ic["cx"][i] * 2.0, ic["cy"][i] * 2.0
            w.writerow(
                [
                    n,
                    d["t"][i].isoformat(),
                    f"{100 * d['obsc_eph'][i]:.2f}",
                    f"{d['alt_true'][i]:.2f}",
                    cls[i],
                    int(ic["interpolated"][i]),
                    int(veil[i]),
                    int(black[i]),
                    f"{q[i]:.3f}",
                    f"{ic['cx'][i]:.3f}",
                    f"{ic['cy'][i]:.3f}",
                    f"{xf:.3f}",
                    f"{yf:.3f}",
                    f"{ref_x - xf:.3f}",
                    f"{ref_y - yf:.3f}",
                    d["file"][i],
                ]
            )
    return {"path": path, "timelapse": tl, "centers": ic, "n": len(idx)}


if __name__ == "__main__":
    import argparse

    from . import diagnose

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=8.0, help="pas temporel vise, en secondes")
    args = ap.parse_args()
    dd = diagnose.load()
    res = export(dd, delta_s=args.delta)
    tlx = res["timelapse"]
    ii = np.array(tlx.indices)
    print(f"{res['n']} trames retenues, pas {args.delta:g}s, {len(tlx.skipped)} sauts")
    print(
        f"  {dd['t'][ii[0]]:%H:%M:%S} a {dd['t'][ii[-1]]:%H:%M:%S}, "
        f"obscuration {100 * dd['obsc_eph'][ii].min():.1f} -> "
        f"{100 * dd['obsc_eph'][ii].max():.1f} -> {100 * dd['obsc_eph'][ii[-1]]:.1f} %"
    )
    print("->", res["path"])
