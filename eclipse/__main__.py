"""Chaine complete : calibration, mesure, selection, classement, rendu, rapport."""

from __future__ import annotations

import argparse
import time

from . import calibrate, config, diagnose, html, rank, render, report, select


def main() -> None:
    ap = argparse.ArgumentParser(prog="eclipse", description=__doc__)
    ap.add_argument("--delta", type=float, default=8.0, help="pas temporel du timelapse, en s")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=1080, help="cote de la sortie video, en px")
    ap.add_argument("--tint", type=float, default=1.0)
    ap.add_argument("--couleur", choices=("fixe", "unifiee"), default="unifiee")
    ap.add_argument("--nettete", type=float, default=4.5, help="FWHM cible, arcsec, 0 = aucune")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    cal_path = config.ANALYSIS_DIR / "calibration.json"
    if a.recalibrate or not cal_path.exists():
        print("== calibration")
        calibrate.run().save()

    print("== mesure")
    from . import measure

    measure.run(workers=a.workers)

    print("== controles")
    d = diagnose.load()
    print(diagnose.summary(d))
    diagnose.plot(d)
    diagnose.plot_selection(d, a.delta)

    print("== selection et classement")
    res = select.export(d, delta_s=a.delta)
    print(f"  {res['n']} trames retenues -> {res['path'].name}")
    for k, v in rank.export(d).items():
        print(f"  {k} -> {v.name}")

    print("== planches")
    report.selection_board()
    report.maximum_board()
    report.control_board()
    if (config.ANALYSIS_DIR / "render.json").exists():
        report.colour_boards()
        report.sharpen_board()

    if not a.skip_render:
        print("== rendu")
        r = render.run(
            size=a.size,
            tint=a.tint,
            couleur=a.couleur,
            nettete=a.nettete,
            workers=min(a.workers, 6),
        )
        print(f"  {r['frames']} trames -> {r['video'] or r['dir']}")

    print("== rapport")
    print("  ->", html.build())
    print(f"termine en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
