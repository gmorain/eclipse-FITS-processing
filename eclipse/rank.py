"""Mesure de phase, metriques de qualite, classement par tranche de phase.

Objectif distinct du timelapse : proposer un jeu restreint de trames a traiter
individuellement, reparties sur les phases de l'eclipse, plus les listes de
rafales pretes a empiler.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, select

# Ponderations du score composite. limb_width domine : c'est la mesure de
# turbulence, et a 5 degres d'elevation la turbulence decide de tout.
WEIGHTS = {
    "limb_width": 1.0,
    "limb_width_aniso": 0.5,
    "snr_disk": 0.4,
    "transparency_norm": 0.6,
}
PENALTY_SAT = 6.0
PENALTY_RMS = 1.5
PENALTY_VEIL = 3.0


def robust_z(v: np.ndarray, groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cote centree reduite robuste, calculee a l'interieur de chaque rafale.

    Normaliser globalement reviendrait a garder toutes les trames de la
    meilleure demi-heure : la turbulence derive au cours de la session et
    l'obscuration change le contenu de l'image. On veut la meilleure trame de
    chaque moment.
    """
    out = np.full(v.shape, np.nan)
    for g in np.unique(groups):
        m = (groups == g) & mask & np.isfinite(v)
        if m.sum() < 3:
            if m.any():
                out[m] = 0.0
            continue
        med = np.median(v[m])
        mad = 1.4826 * np.median(np.abs(v[m] - med))
        out[m] = (v[m] - med) / mad if mad > 0 else 0.0
    return np.clip(out, -5.0, 5.0)


def score(d: dict[str, np.ndarray]) -> np.ndarray:
    """Score composite de qualite d'image, normalise par rafale."""
    cls = select.classify(d)
    usable = cls != "C"
    b = d["burst"]
    s = np.zeros(len(b))
    s += WEIGHTS["limb_width"] * robust_z(-d["limb_width"], b, usable)
    s += WEIGHTS["limb_width_aniso"] * robust_z(-d["limb_width_aniso"], b, usable)
    s += WEIGHTS["snr_disk"] * robust_z(d["snr_disk"], b, usable)
    s += WEIGHTS["transparency_norm"] * robust_z(np.log10(d["transparency_norm"]), b, usable)
    s -= PENALTY_SAT * np.nan_to_num(d["sat_frac"], nan=0.0) / 0.01
    s -= PENALTY_RMS * np.nan_to_num(d["rms"], nan=5.0)
    s -= PENALTY_VEIL * select.veiled(d)
    s[~usable] = -np.inf
    return s


def phase_bins(
    d: dict[str, np.ndarray], n_low: int = 8, n_high: int = 6, obsc_split: float = 0.90
) -> tuple[np.ndarray, np.ndarray]:
    """Decoupage en tranches de phase, branches montante et descendante separees.

    Uniforme en obscuration jusqu'a 90 %, puis uniforme en log10(1 - obsc), ce
    qui densifie l'approche du maximum ou le croissant s'affine vite. Les deux
    branches sont traitees a part : l'angle de position du croissant differe,
    l'aspect n'est pas symetrique.

    Renvoie (indice de tranche, branche) avec branche 0 montante, 1 descendante.
    """
    obsc = d["obsc_eph"]
    i_max = int(np.argmax(obsc))
    branch = (np.arange(len(obsc)) > i_max).astype(int)

    edges_low = np.linspace(0.0, obsc_split, n_low + 1)
    # au-dela du seuil on travaille sur -log10(1 - obsc), qui croit avec
    # l'obscuration : les bornes restent ordonnees et l'indice de tranche suit
    # la progression de l'eclipse
    u_lo = -np.log10(1.0 - obsc_split)
    u_hi = -np.log10(np.clip(1.0 - obsc.max(), 1e-4, None))
    edges_high = np.linspace(u_lo, max(u_hi, u_lo + 1e-6), n_high + 1)

    b = np.full(len(obsc), -1)
    b[obsc < obsc_split] = np.digitize(obsc, edges_low[1:-1])[obsc < obsc_split]
    m = obsc >= obsc_split
    if m.any():
        u = -np.log10(np.clip(1.0 - obsc[m], 1e-4, None))
        b[m] = n_low + np.clip(np.digitize(u, edges_high[1:-1]), 0, n_high - 1)
    return b, branch


@dataclass
class Pick:
    index: int
    bin: int
    branch: int
    rank: int
    score: float


def top_k(
    d: dict[str, np.ndarray], k: int = 1, min_dt_s: float = 30.0, in_timelapse: bool = False
) -> list[Pick]:
    """Les k meilleures trames de chaque tranche de phase et de chaque branche.

    Un ecart temporel minimal est impose entre trames retenues d'une meme
    tranche, sans quoi on renvoie k trames consecutives de la meme rafale, qui
    se ressemblent trait pour trait.
    """
    s = score(d)
    bins, branch = phase_bins(d)
    ok = np.isfinite(s)
    if in_timelapse:
        ok &= select.in_timelapse(d)
    ts = np.array([(x - d["t"][0]).total_seconds() for x in d["t"]])

    picks: list[Pick] = []
    for br in (0, 1):
        for b in np.unique(bins[bins >= 0]):
            m = ok & (bins == b) & (branch == br)
            if not m.any():
                continue
            cand = np.nonzero(m)[0][np.argsort(-s[m])]
            taken: list[int] = []
            for i in cand:
                if all(abs(ts[i] - ts[j]) >= min_dt_s for j in taken):
                    taken.append(int(i))
                if len(taken) >= k:
                    break
            for r, i in enumerate(taken):
                picks.append(Pick(index=i, bin=int(b), branch=br, rank=r, score=float(s[i])))
    picks.sort(key=lambda p: d["t"][p.index])
    return picks


def stack_lists(
    d: dict[str, np.ndarray], keep_frac: float = 0.15, min_frames: int = 8
) -> dict[int, list[int]]:
    """Meilleures trames de chaque rafale, pretes pour un empilement.

    Une rafale de 25 a 50 trames est un jeu de lucky imaging, pas une
    redondance : empiler les 15 % meilleures battra toujours la meilleure trame
    isolee, surtout a 5 degres d'elevation ou la turbulence domine.
    """
    s = score(d)
    out: dict[int, list[int]] = {}
    for b in np.unique(d["burst"]).astype(int):
        m = (d["burst"] == b) & np.isfinite(s)
        if m.sum() < 3:
            continue
        cand = np.nonzero(m)[0][np.argsort(-s[m])]
        n = max(min(min_frames, cand.size), int(round(keep_frac * m.sum())))
        out[b] = sorted(int(i) for i in cand[:n])
    return out


def export(d: dict[str, np.ndarray], out_dir: Path | None = None, k: int = 1) -> dict[str, Path]:
    """Ecrit la selection par phase et les listes d'empilement par rafale."""
    out_dir = out_dir or config.ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p_single = out_dir / "selection_phase.csv"
    with p_single.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rang_bin",
                "bin",
                "branche",
                "t",
                "obsc",
                "alt",
                "score",
                "limb_width",
                "snr_disk",
                "rms",
                "sigma_center",
                "classe",
                "voile",
                "file",
            ]
        )
        cls = select.classify(d)
        veil = select.veiled(d)
        for p in top_k(d, k=k):
            i = p.index
            w.writerow(
                [
                    p.rank,
                    p.bin,
                    "montante" if p.branch == 0 else "descendante",
                    d["t"][i].isoformat(),
                    f"{100 * d['obsc_eph'][i]:.2f}",
                    f"{d['alt_true'][i]:.2f}",
                    f"{p.score:.3f}",
                    f"{d['limb_width'][i]:.2f}",
                    f"{d['snr_disk'][i]:.0f}",
                    f"{d['rms'][i]:.3f}",
                    f"{d['sigma_center'][i]:.3f}",
                    cls[i],
                    int(veil[i]),
                    d["file"][i],
                ]
            )

    p_stack = out_dir / "empilement_par_rafale.txt"
    lists = stack_lists(d)
    with p_stack.open("w") as fh:
        for b, idx in sorted(lists.items()):
            fh.write(
                f"# rafale {b}  {d['t'][idx[0]]:%H:%M:%S}  "
                f"obsc {100 * d['obsc_eph'][idx[0]]:.1f}%  {len(idx)} trames\n"
            )
            for i in idx:
                fh.write(f"{d['file'][i]}\n")
            fh.write("\n")
    return {"single": p_single, "stack": p_stack}


if __name__ == "__main__":
    from . import diagnose

    dd = diagnose.load()
    paths = export(dd)
    for kk, v in paths.items():
        print(kk, "->", v)
