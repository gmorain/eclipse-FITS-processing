"""Lecture des FITS, extraction du canal R, piedestal, inventaire de session."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import config

FNAME_RE = re.compile(
    r"^Light_SUN_(?P<exp>[\d.]+(?:ms|us|s))_Bin1_585MC_(?P<filt>\w+)_gain(?P<gain>\d+)_"
    r"(?P<stamp>\d{8}-\d{6})_(?P<temp>[\d.]+)C_(?P<seq>\d{4})\.fit$"
)


EPOCH = dt.datetime(2026, 8, 12)


@dataclass
class Session:
    """Inventaire de session : trames retenues et anomalies relevees."""

    frames: list[Frame]
    anomalies: list[str] = field(default_factory=list)

    @property
    def n_bursts(self) -> int:
        return self.frames[-1].burst + 1


@dataclass
class Frame:
    """Une trame et ses en-tetes, avant toute mesure pixel."""

    path: Path
    t: dt.datetime  # DATE-OBS, apres reparation eventuelle
    t_raw: dt.datetime  # DATE-OBS tel qu'ecrit
    seq: int  # numero dans la rafale, issu du nom de fichier
    exptime: float  # s
    gain: int
    egain: float  # e-/ADU
    offset: int
    ccd_temp: float
    filt: str
    focuspos: int
    burst: int = -1
    t_repaired: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def settings_key(self) -> tuple:
        return (round(self.exptime, 9), self.gain, self.filt)


def _parse_header(path: Path) -> dict:
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
    return {
        "t": dt.datetime.fromisoformat(h["DATE-OBS"]),
        "exptime": float(h["EXPTIME"]),
        "gain": int(h["GAIN"]),
        "egain": float(h["EGAIN"]),
        "offset": int(h["OFFSET"]),
        "ccd_temp": float(h["CCD-TEMP"]),
        "filt": str(h["FILTER"]).strip(),
        "focuspos": int(h.get("FOCUSPOS", -1)),
    }


def discover(session_dir: Path | None = None) -> Session:
    """Inventaire des trames retenues, triees par date, rafales numerotees.

    Ecarte les fichiers AppleDouble `._*`, les trames anterieures au debut de
    session (mise au point differente) et les filtres autres que IRCt.
    """
    session_dir = session_dir or config.session_dir()
    paths = sorted(p for p in session_dir.glob("*.fit") if not p.name.startswith("._"))
    if not paths:
        raise FileNotFoundError(f"aucun .fit dans {session_dir}")

    frames: list[Frame] = []
    for p in paths:
        m = FNAME_RE.match(p.name)
        if m is None:
            continue
        hdr = _parse_header(p)
        if hdr["filt"] not in config.KEEP_FILTERS:
            continue
        if hdr["t"] < config.SESSION_START_UTC:
            continue
        frames.append(
            Frame(
                path=p,
                t=hdr["t"],
                t_raw=hdr["t"],
                seq=int(m.group("seq")),
                exptime=hdr["exptime"],
                gain=hdr["gain"],
                egain=hdr["egain"],
                offset=hdr["offset"],
                ccd_temp=hdr["ccd_temp"],
                filt=hdr["filt"],
                focuspos=hdr["focuspos"],
            )
        )

    frames.sort(key=lambda f: f.t)
    anomalies = repair_timestamps(frames)
    assign_bursts(frames)
    return Session(frames=frames, anomalies=anomalies)


def _split_bursts(frames: list[Frame], gap_s: float) -> list[list[Frame]]:
    groups: list[list[Frame]] = [[frames[0]]]
    for prev, cur in zip(frames, frames[1:], strict=False):
        if (cur.t - prev.t).total_seconds() > gap_s:
            groups.append([])
        groups[-1].append(cur)
    return groups


def repair_timestamps(frames: list[Frame]) -> list[str]:
    """Repare les DATE-OBS aberrants.

    Une rafale ASIAIR numerote ses trames de 1 a n sans trou. Un DATE-OBS
    fautif isole la trame dans sa propre pseudo-rafale et laisse un trou dans
    la rafale d'origine. On rapatrie l'orpheline dans le trou correspondant et
    on recalcule sa date par la cadence locale. La session porte un cas connu :
    une trame decalee de +2 h exactement.
    """
    msgs: list[str] = []
    groups = _split_bursts(frames, config.BURST_GAP_S)
    complete = [g for g in groups if min(f.seq for f in g) == 1]
    orphans = [g for g in groups if min(f.seq for f in g) != 1]

    for g in orphans:
        for f in g:
            host = None
            for c in complete:
                seqs = {x.seq for x in c}
                if (
                    c[0].settings_key == f.settings_key
                    and f.seq not in seqs
                    and min(seqs) <= f.seq <= max(seqs) + 1
                ):
                    host = c
                    break
            if host is None:
                msgs.append(f"{f.name}: DATE-OBS isole, aucune rafale d'accueil trouvee")
                continue
            ts = np.array([(x.t - EPOCH).total_seconds() for x in host])
            seqs = np.array([x.seq for x in host], dtype=float)
            cadence = float(np.median(np.diff(ts) / np.diff(seqs)))
            anchor = float(np.median(ts - seqs * cadence))
            new_t = EPOCH + dt.timedelta(seconds=anchor + f.seq * cadence)
            delta = (f.t_raw - new_t).total_seconds()
            f.t = new_t
            f.t_repaired = True
            msgs.append(
                f"{f.name}: DATE-OBS {f.t_raw:%H:%M:%S} aberrant ({delta:+.0f} s), "
                f"recale a {new_t:%H:%M:%S} par la cadence de la rafale {host[0].t:%H:%M:%S}"
            )

    frames.sort(key=lambda f: f.t)
    return msgs


def assign_bursts(frames: list[Frame]) -> int:
    for i, g in enumerate(_split_bursts(frames, config.BURST_GAP_S)):
        for f in g:
            f.burst = i
    return frames[-1].burst + 1


def load_r_plane(path: Path) -> np.ndarray:
    """Canal R de la matrice de Bayer, en ADU 16 bits non signes.

    Le motif est RGGB sans ROWORDER : le pixel R occupe le coin (0, 0) de
    chaque cellule 2x2, verifie sur les donnees (canal le plus brillant).
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    dy, dx = config.R_PLANE_OFFSET
    return np.ascontiguousarray(data[dy::2, dx::2])


def corner_boxes(shape: tuple[int, int], size: int = 128) -> list[tuple[slice, slice]]:
    h, w = shape
    return [
        (slice(0, size), slice(0, size)),
        (slice(0, size), slice(w - size, w)),
        (slice(h - size, h), slice(0, size)),
        (slice(h - size, h), slice(w - size, w)),
    ]


def pedestal(
    img: np.ndarray,
    center: tuple[float, float] | None = None,
    r: float = config.R_SUN_PX_R,
    size: int = 128,
) -> tuple[float, float]:
    """Mediane et MAD du fond, mesurees dans les coins au-dela de 1,6 R.

    Le champ ne fait que 53,8' de haut pour un disque de 32' : un anneau a
    1,6 R ne tient pas dans l'image des que le disque est decentre. Les coins
    sont le seul fond disponible. Derriere un OD 3,8 ils sont noirs.
    """
    boxes = corner_boxes(img.shape, size)
    if center is not None:
        cx, cy = center
        keep = []
        for by, bx in boxes:
            yy = 0.5 * (by.start + by.stop)
            xx = 0.5 * (bx.start + bx.stop)
            if np.hypot(xx - cx, yy - cy) > 1.6 * r:
                keep.append((by, bx))
        if keep:
            boxes = keep
    vals = np.concatenate([img[by, bx].ravel() for by, bx in boxes]).astype(np.float32)
    med = float(np.median(vals))
    # Bruit estime sur les differences de pixels voisins, insensible a un
    # gradient de fond. Plancher au pas de quantification : sur un fond aussi
    # noir la MAD vaut souvent exactement zero, et un seuil a 8 sigma calcule
    # dessus laisserait passer n'importe quel pixel de bruit.
    diffs = np.concatenate([np.diff(img[by, bx], axis=1).ravel() for by, bx in boxes])
    sigma = 1.4826 * float(np.median(np.abs(diffs.astype(np.float32)))) / 2.0**0.5
    # flottants Python et non scalaires numpy : un np.float64 qui remonte
    # jusqu'a une operation sur un tableau float32 promeut tout en double
    return float(med), float(max(sigma, config.QUANT_ADU))
