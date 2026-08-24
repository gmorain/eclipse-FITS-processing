"""Passe de mesure sur toute la session : une ligne de metriques par trame."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import astro, config, io
from . import limb as limb_mod
from .calibrate import Calibration
from .config import DEFAULT_LIMB

FIELDS = [
    "file",
    "t",
    "burst",
    "seq",
    "exptime",
    "gain",
    "egain",
    "ccd_temp",
    "alt_true",
    "alt_app",
    "az",
    "obsc_eph",
    "flat",
    "r_sun_px",
    "ok",
    "reason",
    "pedestal",
    "sigma_bg",
    "cx",
    "cy",
    "n_points",
    "rms",
    "limb_width",
    "limb_width_aniso",
    "sharp",
    "ref_level",
    "disk_median",
    "snr_disk",
    "sat_frac",
    "obsc_area",
    "crescent_pa",
    "transparency",
    "airmass",
    "transparency_norm",
    "obsc_gap",
    "arc_visible",
    "n_expected",
    "n_ratio",
    "sigma_center",
    "arc_span",
]

_CTX: dict = {}


@dataclass
class Job:
    """Une rafale entiere : c'est l'unite de travail, pas la trame.

    Le centre derive de 0,3 px/s environ et les trames d'une rafale sont
    espacees de 2 a 4 s : la trame precedente est de tres loin le meilleur
    point de depart possible. Sur un croissant fin le vote de Hough echoue,
    l'amorce temporelle non.
    """

    burst: int
    paths: list[Path]
    r_px: float
    ks: list[float]
    indices: list[int]
    ts: list[float] = field(default_factory=list)  # secondes depuis le debut de la rafale
    seeds: list[tuple[float, float] | None] = field(default_factory=list)


def _init(cal_dict: dict) -> None:
    cal = Calibration(**cal_dict)
    eh, ev = astro.local_axes(cal.vert_angle_deg)
    _CTX["e_h"], _CTX["e_v"] = eh, ev


def _measure_one(path: Path, r_px: float, k: float, seed=None) -> dict:
    img = io.load_r_plane(path)
    ped, sigma = io.pedestal(img)
    eh, ev = _CTX["e_h"], _CTX["e_v"]
    m = limb_mod.measure(img, ped, sigma, r_px, k, eh, ev, c0=seed)
    if seed is not None and not m.ok:
        m = limb_mod.measure(img, ped, sigma, r_px, k, eh, ev, c0=None)
    # le piedestal des coins peut englober le disque s'il est decentre : on le
    # reprend une fois le centre connu
    if m.ok:
        ped2, sig2 = io.pedestal(img, center=m.center, r=r_px)
        if abs(ped2 - ped) > 2.0:
            m = limb_mod.measure(img, ped2, sig2, r_px, k, eh, ev, c0=m.center)
            ped, sigma = ped2, sig2
    return {
        "ok": m.ok,
        "reason": m.reason,
        "pedestal": ped,
        "sigma_bg": sigma,
        "cx": m.center[0],
        "cy": m.center[1],
        "n_points": m.n_points,
        "rms": m.rms,
        "sigma_center": m.sigma_center,
        "arc_span": m.arc_span,
        "limb_width": m.limb_width,
        "limb_width_aniso": m.limb_width_aniso,
        "sharp": m.sharp,
        "ref_level": m.ref_level,
        "disk_median": m.disk_median,
        "snr_disk": m.snr_disk,
        "sat_frac": m.sat_frac,
        "obsc_area": m.obsc_area,
        "crescent_pa": m.crescent_pa,
    }


SEED_MAX_RMS = 1.0
# Le centre derive de 0,3 px/s et les trames d'une rafale sont espacees de
# 2 a 4 s. Un saut au-dela de cette borne n'est pas un mouvement de monture,
# c'est un fit qui s'est accroche ailleurs : on ne le propage pas.
SEED_MAX_DRIFT_PX_PER_S = 3.0
SEED_MAX_JUMP_PX = 40.0
TREND_MAX_DEV_PX = 25.0
# Un fit tres bien contraint est cru sur parole, meme s'il s'ecarte de la
# trajectoire de sa rafale : la monture peut encaisser une rafale de vent et
# revenir, ce qu'un ajustement affine local ne sait pas representer. Un fit qui
# s'est accroche a un bord de nuage, lui, n'atteint jamais ces valeurs.
TRUST_SIGMA_PX = 0.05
TRUST_RMS_PX = 0.6
TRUST_ARC_DEG = 120.0


def _work(job: Job) -> list[dict]:
    """Passe avant avec amorce, puis reprise des echecs depuis le voisin le plus proche."""
    n = len(job.paths)
    out: list[dict] = [None] * n  # type: ignore[list-item]
    seed, seed_t = None, 0.0
    for i in range(n):
        r = _measure_one(job.paths[i], job.r_px, job.ks[i], seed)
        out[i] = r
        if r["ok"] and r["rms"] < SEED_MAX_RMS:
            if seed is None:
                seed, seed_t = (r["cx"], r["cy"]), job.ts[i]
            else:
                budget = min(
                    SEED_MAX_JUMP_PX,
                    SEED_MAX_DRIFT_PX_PER_S * max(job.ts[i] - seed_t, 1.0),
                )
                if np.hypot(r["cx"] - seed[0], r["cy"] - seed[1]) <= budget:
                    seed, seed_t = (r["cx"], r["cy"]), job.ts[i]

    good = [i for i, r in enumerate(out) if r["ok"] and r["rms"] < SEED_MAX_RMS]
    if not good:
        return out
    for i, r in enumerate(out):
        if r["ok"] and r["rms"] < SEED_MAX_RMS:
            continue
        j = min(good, key=lambda g: abs(g - i))
        retry = _measure_one(job.paths[i], job.r_px, job.ks[i], (out[j]["cx"], out[j]["cy"]))
        if retry["ok"] and (not r["ok"] or retry["rms"] < r["rms"]):
            out[i] = retry
    return out


def _refit(args) -> tuple[int, dict]:
    idx, path, r_px, k, seed = args
    return idx, _measure_one(path, r_px, k, seed)


def burst_trend(
    ts: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    good: np.ndarray,
    half_window_s: float = 45.0,
):
    """Trajectoire robuste du centre dans une rafale, par ajustement affine local.

    Un fit a rayon fige sur un arc court peut se caler n'importe ou le long de
    la normale a l'arc et sortir un rms excellent. Le rms ne suffit donc pas :
    seule la coherence avec la trajectoire de la rafale separe une mesure d'un
    accrochage sur un bord de nuage.

    L'ajustement est local et non global : une rafale de 200 trames dure plus de
    500 s, la derive de monture y est courbe, et une droite unique s'ecarterait
    de plusieurs dizaines de pixels aux extremites.
    """
    keep = good.copy()
    tx = np.full(ts.size, np.nan)
    ty = np.full(ts.size, np.nan)
    for _ in range(3):
        if keep.sum() < 3:
            break
        for i in range(ts.size):
            m = keep & (np.abs(ts - ts[i]) <= half_window_s)
            if m.sum() < 4:
                m = keep
            if m.sum() >= 4 and np.ptp(ts[m]) > 1.0:
                tx[i] = np.polyval(np.polyfit(ts[m], cx[m], 1), ts[i])
                ty[i] = np.polyval(np.polyfit(ts[m], cy[m], 1), ts[i])
            else:
                tx[i] = np.median(cx[m])
                ty[i] = np.median(cy[m])
        dev = np.hypot(cx - tx, cy - ty)
        mad = 1.4826 * np.median(np.abs(dev[keep] - np.median(dev[keep])))
        new = good & (dev < max(6 * mad, TREND_MAX_DEV_PX))
        if new.sum() < 3 or np.array_equal(new, keep):
            keep = new if new.sum() >= 3 else keep
            break
        keep = new
    return tx, ty, keep


def run(limit: int | None = None, workers: int = 8, out: Path | None = None) -> Path:
    session = io.discover()
    frames = session.frames
    if limit:
        frames = frames[:: max(1, len(frames) // limit)][:limit]
    cal = Calibration.load()
    eph = astro.ephemeris([f.t for f in frames])
    out = out or (config.ANALYSIS_DIR / "metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort([f.burst for f in frames], kind="stable")
    jobs, job_idx = [], []
    for b in sorted({f.burst for f in frames}):
        idx = [i for i in order if frames[i].burst == b]
        jobs.append(
            Job(
                burst=b,
                paths=[frames[i].path for i in idx],
                r_px=cal.r_sun_px,
                ks=[float(eph.flattening[i]) for i in idx],
                indices=idx,
                ts=[(frames[i].t - frames[idx[0]].t).total_seconds() for i in idx],
            )
        )
        job_idx.append(idx)
    t0 = time.time()
    rows: list[dict] = [None] * len(frames)  # type: ignore[list-item]
    done = 0
    with Pool(workers, initializer=_init, initargs=(cal.__dict__,)) as pool:
        for job, res in zip(jobs, pool.imap(_work, jobs), strict=True):
            for i, r in zip(job.indices, res, strict=True):
                f = frames[i]
                row = {
                    "file": f.name,
                    "t": f.t.isoformat(),
                    "burst": f.burst,
                    "seq": f.seq,
                    "exptime": f.exptime,
                    "gain": f.gain,
                    "egain": f.egain,
                    "ccd_temp": f.ccd_temp,
                    "alt_true": eph.alt_true_deg[i],
                    "alt_app": eph.alt_app_deg[i],
                    "az": eph.az_deg[i],
                    "obsc_eph": eph.obscuration[i],
                    "flat": eph.flattening[i],
                    "r_sun_px": cal.r_sun_px,
                    "airmass": eph.airmass[i],
                    "arc_visible": eph.arc_visible[i],
                    "n_expected": DEFAULT_LIMB.n_rays * eph.arc_visible[i],
                }
                row.update(r)
                row["n_ratio"] = (
                    r["n_points"] / row["n_expected"] if row["n_expected"] > 1 else float("nan")
                )
                dm = r["disk_median"]
                row["transparency"] = dm * f.egain / f.exptime if np.isfinite(dm) else float("nan")
                # extinction retiree : 0,25 mag par masse d'air dans le rouge. Ce qui
                # reste est le voile, plus les erreurs du modele pres de l'horizon.
                row["transparency_norm"] = row["transparency"] * 10 ** (0.4 * 0.25 * eph.airmass[i])
                # ecart entre l'obscuration mesuree et celle des ephemerides : c'est
                # la part du disque cachee par autre chose que la lune, donc le nuage
                row["obsc_gap"] = (
                    r["obsc_area"] - eph.obscuration[i]
                    if np.isfinite(r["obsc_area"])
                    else float("nan")
                )
                rows[i] = row
            done += len(res)
            el = time.time() - t0
            print(
                f"  rafale {job.burst:2d}  {done}/{len(frames)}  {el:5.1f}s  "
                f"{done / el:5.1f} trames/s",
                flush=True,
            )

    rows = _enforce_trend(rows, frames, jobs, cal, workers)

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    ok = sum(1 for r in rows if r["ok"])
    print(
        f"{len(rows)} trames, {ok} mesurees, {len(rows) - ok} en echec, "
        f"{time.time() - t0:.0f}s -> {out}"
    )
    if session.anomalies:
        (config.ANALYSIS_DIR / "anomalies.txt").write_text("\n".join(session.anomalies) + "\n")
    return out


def _trusted(row: dict) -> bool:
    """Fit assez bien contraint pour primer sur la trajectoire de la rafale."""
    return bool(
        row["ok"]
        and np.isfinite(row["sigma_center"])
        and row["sigma_center"] < TRUST_SIGMA_PX
        and row["rms"] < TRUST_RMS_PX
        and np.isfinite(row["arc_span"])
        and row["arc_span"] > TRUST_ARC_DEG
    )


def _enforce_trend(rows, frames, jobs, cal, workers):
    """Rejette les centres incoherents avec leur rafale, puis tente une reprise.

    La reprise repart du centre attendu par la trajectoire : c'est la meilleure
    amorce disponible pour une trame que le vote de Hough et la trame voisine
    ont tous deux ratee.
    """
    todo, expected, prior = [], {}, {}
    for job in jobs:
        idx = job.indices
        ts = np.array(job.ts)
        cx = np.array([rows[i]["cx"] for i in idx], dtype=float)
        cy = np.array([rows[i]["cy"] for i in idx], dtype=float)
        good = np.array(
            [bool(rows[i]["ok"]) and rows[i]["rms"] < SEED_MAX_RMS for i in idx]
        ) & np.isfinite(cx)
        if good.sum() < 3:
            continue
        tx, ty, keep = burst_trend(ts, cx, cy, good)
        for pos, i in enumerate(idx):
            expected[i] = (float(tx[pos]), float(ty[pos]))
            if keep[pos] or _trusted(rows[i]):
                continue
            # une trame deja en echec garde son motif d'origine si la reprise
            # echoue aussi : ce n'est pas un probleme de coherence, c'est du nuage
            prior[i] = None if rows[i]["ok"] else rows[i]["reason"]
            todo.append((i, frames[i].path, cal.r_sun_px, job.ks[pos], expected[i]))

    if not todo:
        return rows
    n_before = sum(1 for r in rows if r["ok"])
    with Pool(workers, initializer=_init, initargs=(cal.__dict__,)) as pool:
        for i, r in pool.imap_unordered(_refit, todo, chunksize=4):
            ex = expected[i]
            dev = np.hypot(r["cx"] - ex[0], r["cy"] - ex[1]) if r["ok"] else np.inf
            if r["ok"] and dev < TREND_MAX_DEV_PX and r["rms"] < 1.5:
                rows[i].update(r)
                rows[i]["reason"] = ""
            else:
                rows[i]["ok"] = False
                rows[i]["reason"] = prior[i] or "centre incoherent avec la rafale"
    n_after = sum(1 for r in rows if r["ok"])
    n_inc = sum(1 for r in rows if r["reason"] == "centre incoherent avec la rafale")
    print(
        f"  coherence de trajectoire : {len(todo)} trames reprises, "
        f"{n_after - n_before:+d} mesurees, {n_inc} rejetees pour incoherence"
    )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="sous-echantillonne la session")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(limit=args.limit, workers=args.workers)
