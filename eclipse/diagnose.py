"""Courbes de controle. C'est ici que se reglent les seuils de classification."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from . import config


def load(path: Path | None = None) -> dict[str, np.ndarray]:
    path = path or (config.ANALYSIS_DIR / "metrics.csv")
    rows = list(csv.DictReader(path.open()))
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if k == "t":
            out[k] = np.array([dt.datetime.fromisoformat(v) for v in vals])
        elif k in ("file", "reason"):
            out[k] = np.array(vals)
        elif k == "ok":
            out[k] = np.array([v == "True" for v in vals])
        else:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in vals])
    return out


def rolling_norm(t: np.ndarray, v: np.ndarray, window_s: float = 240.0) -> np.ndarray:
    """Normalise une metrique par sa mediane glissante, en temps.

    La turbulence derive au cours de la session : on veut la meilleure trame de
    chaque instant, pas toutes les trames de la meilleure demi-heure.
    """
    ts = np.array([(x - t[0]).total_seconds() for x in t])
    out = np.full(v.shape, np.nan)
    for i in range(len(v)):
        m = (np.abs(ts - ts[i]) < window_s / 2) & np.isfinite(v)
        if m.sum() >= 5:
            med = np.median(v[m])
            if med > 0:
                out[i] = v[i] / med
    return out


def plot(d: dict[str, np.ndarray], out: Path | None = None) -> Path:
    out = out or (config.REPORT_DIR / "controle.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = d["ok"]
    t = d["t"]
    fig, ax = plt.subplots(4, 2, figsize=(16, 15), sharex=True)
    fmt = mdates.DateFormatter("%H:%M")

    def deco(a, ylab, log=False):
        a.set_ylabel(ylab)
        a.grid(alpha=0.25)
        a.xaxis.set_major_formatter(fmt)
        if log:
            a.set_yscale("log")

    a = ax[0, 0]
    a.plot(t[ok], d["n_points"][ok], ".", ms=2, color="C0", label="N mesure")
    a.plot(
        t,
        720 * (1 - d["obsc_eph"]) ** 0.5,
        "-",
        lw=1,
        color="C3",
        label=r"fond geometrique $720\sqrt{1-obsc}$",
    )
    a.legend(fontsize=8)
    deco(a, "N points de limbe")

    a = ax[0, 1]
    from . import select

    cls = select.classify(d)
    for c, col in (("A", "C2"), ("B", "C1"), ("C", "0.6")):
        m = ok & (cls == c)
        a.semilogy(
            t[m],
            np.maximum(d["sigma_center"][m], 1e-4),
            ".",
            ms=2,
            color=col,
            label=f"classe {c} ({m.sum()})",
        )
    for y, col in ((select.CLASS_A["sigma_center"], "C2"), (select.CLASS_B["sigma_center"], "C1")):
        a.axhline(y, color=col, lw=1, ls="--")
    a.legend(fontsize=8, markerscale=3)
    deco(a, "incertitude sur le centre (px)")

    a = ax[1, 0]
    a.plot(t[ok], d["limb_width"][ok], ".", ms=2, label="brut")
    a.plot(
        t[ok],
        6.3 * rolling_norm(t[ok], d["limb_width"][ok]),
        ".",
        ms=1.5,
        alpha=0.4,
        label="normalise x 6,3",
    )
    a.axhline(2.1, color="k", lw=1, ls=":", label="plancher instrumental")
    a.set_ylim(0, 20)
    a.legend(fontsize=8)
    deco(a, "limb_width (arcsec)")
    a2 = a.twinx()
    a2.plot(t, d["alt_true"], "-", color="C3", lw=1, alpha=0.6)
    a2.set_ylabel("elevation (deg)", color="C3")

    a = ax[1, 1]
    a.plot(t[ok], d["limb_width_aniso"][ok], ".", ms=2)
    a.set_ylim(0, 15)
    deco(a, "anisotropie du limbe (arcsec)")

    a = ax[2, 0]
    a.plot(t[ok], d["cx"][ok], ".", ms=2, label="x")
    a.plot(t[ok], d["cy"][ok], ".", ms=2, label="y")
    a.legend(fontsize=8)
    deco(a, "centre ajuste (px, plan R)")

    a = ax[2, 1]
    a.semilogy(t[ok], d["transparency"][ok], ".", ms=2)
    deco(a, "transparence (e-/s, relatif)")
    a2 = a.twinx()
    a2.plot(t, d["ccd_temp"], "-", color="C3", lw=1, alpha=0.6)
    a2.set_ylabel("CCD-TEMP (C)", color="C3")

    a = ax[3, 0]
    a.plot(t, 100 * d["obsc_eph"], "-", lw=1, color="C3", label="ephemerides")
    a.plot(t[ok], 100 * d["obsc_area"][ok], ".", ms=2, label="mesuree sur l'image")
    a.legend(fontsize=8)
    deco(a, "obscuration (%)")

    a = ax[3, 1]
    a.plot(t[ok], 100 * d["sat_frac"][ok], ".", ms=2)
    deco(a, "pixels ecretes (%)")
    for aa in ax[-1]:
        aa.set_xlabel("UTC")

    fig.suptitle(
        f"Eclipse partielle du 2026-08-12, {config.SITE_NAME}. Controles de la passe de mesure.",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def summary(d: dict[str, np.ndarray]) -> str:
    ok = d["ok"]
    n = len(ok)
    lines = [
        f"trames                  {n}",
        f"mesurees                {ok.sum()} ({100 * ok.mean():.1f} %)",
        f"en echec                {(~ok).sum()}",
    ]
    reasons: dict[str, int] = {}
    for r in d["reason"][~ok]:
        reasons[r] = reasons.get(r, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {r[:40]:<42} {c}")
    from . import select

    cls = select.classify(d)
    tl = select.in_timelapse(d)
    veil = select.veiled(d)
    lines.append(
        f"{'classes (plage timelapse)':<23} "
        + "  ".join(f"{c} {int((cls[tl] == c).sum())}" for c in "ABC")
        + f"   dont voilees {int((veil & tl & (cls != 'C')).sum())}"
    )
    for k, lab, f in (
        ("rms", "rms (px)", 3),
        ("sigma_center", "sigma_center (px)", 4),
        ("arc_span", "arc_span (deg)", 0),
        ("n_points", "N", 0),
        ("limb_width", "limb_width (arcsec)", 2),
        ("snr_disk", "snr_disk", 0),
    ):
        v = d[k][ok]
        v = v[np.isfinite(v)]
        lines.append(
            f"{lab:<23} median {np.median(v):.{f}f}   "
            f"p10 {np.percentile(v, 10):.{f}f}   p90 {np.percentile(v, 90):.{f}f}"
        )
    gap = np.abs(d["obsc_area"] - d["obsc_eph"])[ok]
    gap = gap[np.isfinite(gap)]
    lines.append(
        f"{'|obsc_img - obsc_eph|':<23} median {np.median(gap) * 100:.2f} %   "
        f"p90 {np.percentile(gap, 90) * 100:.2f} %"
    )
    sat = d["sat_frac"][ok]
    lines.append(f"{'trames ecretees':<23} {int(np.nansum(sat > 1e-5))}")
    return "\n".join(lines)


if __name__ == "__main__":
    d = load()
    print(summary(d))
    print("->", plot(d))


def plot_selection(d: dict[str, np.ndarray], delta_s: float = 8.0, out: Path | None = None) -> Path:
    """Controles de la selection : grille temporelle, qualite, couverture de phase."""
    from . import rank, select

    out = out or (config.REPORT_DIR / "controle_selection.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    ic = select.interpolate_centers(d)
    cand = select.in_timelapse(d) & ic["recoverable"] & np.isfinite(ic["cx"])
    tl = select.select_timelapse(d, delta_s, candidates=cand)
    idx = np.array(tl.indices, dtype=int)
    cls = select.classify(d)
    q = select._quality(d)
    ts = np.array([(x - d["t"][0]).total_seconds() for x in d["t"]])

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fmt = mdates.DateFormatter("%H:%M")

    a = ax[0, 0]
    for c, col in (("A", "C2"), ("B", "C1"), ("C", "0.6")):
        m = idx[cls[idx] == c]
        a.plot(d["t"][m], 100 * d["obsc_eph"][m], ".", ms=4, color=col, label=f"classe {c}")
    a.plot(d["t"], 100 * d["obsc_eph"], "-", lw=0.8, color="C3", alpha=0.5)
    a.set_ylabel("obscuration (%)")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)
    a.xaxis.set_major_formatter(fmt)
    a.set_title(f"trames retenues, pas vise {delta_s:g} s ({len(idx)} trames)", fontsize=10)

    a = ax[0, 1]
    dt = np.diff(ts[idx])
    a.semilogy(d["t"][idx][1:], dt, ".", ms=4)
    a.axhline(delta_s, color="C3", lw=1, ls="--", label=f"pas vise {delta_s:g} s")
    a.set_ylabel("ecart a la trame precedente (s)")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)
    a.xaxis.set_major_formatter(fmt)
    a.set_title(
        "regularite obtenue, les paliers hauts sont les attentes entre rafales", fontsize=10
    )

    a = ax[1, 0]
    a.plot(d["t"][cand], q[cand], ".", ms=2, color="0.75", label="candidates")
    a.plot(d["t"][idx], q[idx], ".", ms=4, color="C0", label="retenues")
    a.set_ylabel("qualite (score composite borne)")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)
    a.xaxis.set_major_formatter(fmt)
    a.set_xlabel("UTC")

    a = ax[1, 1]
    bins, branch = rank.phase_bins(d)
    picks = rank.top_k(d, k=1)
    width = 0.4
    for br, col, lab in ((0, "C0", "montante"), (1, "C1", "descendante")):
        m = cand & (branch == br) & (bins >= 0)
        cnt = np.bincount(bins[m], minlength=14)
        a.bar(np.arange(14) + (br - 0.5) * width, cnt, width, color=col, alpha=0.55, label=lab)
    for p in picks:
        a.plot(
            [p.bin + (p.branch - 0.5) * width],
            [1],
            "v",
            ms=6,
            color="C2" if p.branch == 0 else "C3",
        )
    a.set_yscale("symlog")
    a.set_xlabel("tranche de phase (0-7 uniforme en obscuration, 8-13 en log)")
    a.set_ylabel("trames disponibles")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)
    a.set_title("couverture des tranches, triangles = trame elue", fontsize=10)

    fig.suptitle("Selection temporelle et classement par phase", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
