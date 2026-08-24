"""Ephemerides, refraction atmospherique, geometrie de l'eclipse.

Toutes les fonctions sont vectorisees sur le temps. Le calcul astropy est le
poste le plus lourd de la chaine hors lecture disque, on le fait une fois pour
toute la session.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time

from . import config

R_SUN_KM = 696000.0
R_MOON_KM = 1737.4


def location() -> EarthLocation:
    return EarthLocation(
        lat=config.SITE_LAT * u.deg, lon=config.SITE_LON * u.deg, height=config.SITE_ALT_M * u.m
    )


def bennett_refraction(alt_true_deg: np.ndarray) -> np.ndarray:
    """Refraction en arcmin pour une altitude vraie en degres (Bennett 1982).

    Valide jusqu'a l'horizon, contrairement a la loi en cot h qui diverge sous
    5 degres. Corrigee de la pression et de la temperature du site.
    """
    h = np.asarray(alt_true_deg, dtype=float)
    r = 1.0 / np.tan(np.radians(h + 7.31 / (h + 4.4)))
    scale = (config.PRESSURE_HPA / 1010.0) * (283.0 / (273.0 + config.TEMP_C))
    return r * scale


def d_refraction_dh(alt_true_deg: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """dR/dh, sans dimension (arcmin par arcmin), par difference centree."""
    h = np.asarray(alt_true_deg, dtype=float)
    return (bennett_refraction(h + eps) - bennett_refraction(h - eps)) / (2 * eps * 60.0)


def flattening(alt_true_deg: np.ndarray) -> np.ndarray:
    """Rapport petit axe sur grand axe du disque refracte.

    Le disque est comprime selon la verticale locale d'un facteur
    b/a = 1 + dR/dh, negatif par construction puisque la refraction decroit
    avec l'altitude. Vaut 0,996 a 17 degres, 0,963 a 4 degres.
    """
    return 1.0 + d_refraction_dh(alt_true_deg)


@dataclass
class Ephemeris:
    """Grandeurs astronomiques par trame."""

    alt_true_deg: np.ndarray  # altitude geometrique, sans refraction
    alt_app_deg: np.ndarray  # altitude apparente
    az_deg: np.ndarray
    r_sun_arcsec: np.ndarray
    r_moon_arcsec: np.ndarray
    sep_arcsec: np.ndarray  # separation des centres soleil / lune
    pa_moon_deg: np.ndarray  # angle de position de la lune, 0 = zenith, sens trigo
    obscuration: np.ndarray  # fraction de surface du disque solaire couverte
    flattening: np.ndarray  # b/a
    arc_visible: np.ndarray  # fraction du limbe solaire non couverte par la lune
    airmass: np.ndarray  # masse d'air, Kasten-Young sur l'altitude apparente

    @property
    def r_sun_px(self) -> np.ndarray:
        return self.r_sun_arcsec / config.ARCSEC_PER_PX_R

    @property
    def r_moon_px(self) -> np.ndarray:
        return self.r_moon_arcsec / config.ARCSEC_PER_PX_R


def _overlap_fraction(d: np.ndarray, rs: np.ndarray, rm: np.ndarray) -> np.ndarray:
    """Fraction du disque de rayon rs couverte par un disque de rayon rm a distance d."""
    d = np.maximum(np.asarray(d, float), 1e-9)
    out = np.zeros_like(d)
    total = d <= np.abs(rs - rm)
    partial = (~total) & (d < rs + rm)
    out[total] = np.minimum(rs[total], rm[total]) ** 2 / rs[total] ** 2
    dp, rsp, rmp = d[partial], rs[partial], rm[partial]
    a1 = np.arccos(np.clip((dp**2 + rsp**2 - rmp**2) / (2 * dp * rsp), -1, 1))
    a2 = np.arccos(np.clip((dp**2 + rmp**2 - rsp**2) / (2 * dp * rmp), -1, 1))
    area = rsp**2 * (a1 - np.sin(2 * a1) / 2) + rmp**2 * (a2 - np.sin(2 * a2) / 2)
    out[partial] = area / (np.pi * rsp**2)
    return out


def ephemeris(times: list[dt.datetime]) -> Ephemeris:
    """Ephemerides pour une liste de dates UTC naives."""
    t = Time([x.isoformat() for x in times], scale="utc")
    loc = location()

    frame_geom = AltAz(obstime=t, location=loc)
    sun = get_sun(t)
    moon = get_body("moon", t, loc)
    sun_g = sun.transform_to(frame_geom)
    moon_g = moon.transform_to(frame_geom)

    alt_true = sun_g.alt.deg
    refr = bennett_refraction(alt_true) / 60.0
    alt_app = alt_true + refr

    r_sun = np.degrees(np.arcsin(R_SUN_KM / sun.distance.to_value(u.km))) * 3600.0
    r_moon = np.degrees(np.arcsin(R_MOON_KM / moon.distance.to_value(u.km))) * 3600.0
    sep = sun_g.separation(moon_g).to_value(u.arcsec)

    # Angle de position de la lune vue du soleil, dans le repere alt/az :
    # 0 vers le zenith, positif vers l'azimut croissant.
    d_alt = (moon_g.alt.deg - sun_g.alt.deg) * 3600.0
    d_az = (moon_g.az.deg - sun_g.az.deg) * 3600.0 * np.cos(np.radians(sun_g.alt.deg))
    pa_moon = np.degrees(np.arctan2(d_az, d_alt))

    return Ephemeris(
        alt_true_deg=alt_true,
        alt_app_deg=alt_app,
        az_deg=sun_g.az.deg,
        r_sun_arcsec=r_sun,
        r_moon_arcsec=r_moon,
        sep_arcsec=sep,
        pa_moon_deg=pa_moon,
        obscuration=_overlap_fraction(sep, r_sun, r_moon),
        flattening=flattening(alt_true),
        arc_visible=visible_arc(sep, r_sun, r_moon),
        airmass=airmass(alt_app),
    )


def visible_arc(d: np.ndarray, rs: np.ndarray, rm: np.ndarray) -> np.ndarray:
    """Fraction du limbe solaire encore visible, entre 0 et 1.

    Borne geometrique du nombre de points de limbe mesurables. Un seuil absolu
    sur N, comme le N >= 200 de la specification, confond une trame parfaite
    pres du maximum avec une trame nuageuse : a 94 % d'obscuration le limbe
    solaire n'offre plus que 20 % de sa circonference, soit 145 points sur 720,
    et une bonne trame en rend 45.
    """
    d = np.maximum(np.asarray(d, float), 1e-9)
    cos_a = np.clip((d**2 + rs**2 - rm**2) / (2 * d * rs), -1.0, 1.0)
    alpha = np.arccos(cos_a)  # demi-arc couvert, vu du centre solaire
    vis = 1.0 - alpha / np.pi
    vis[d >= rs + rm] = 1.0  # aucun recouvrement
    vis[d <= rm - rs] = 0.0  # soleil entierement couvert
    return vis


def airmass(alt_app_deg: np.ndarray) -> np.ndarray:
    """Masse d'air de Kasten-Young, valable jusqu'a l'horizon.

    Sur cette session elle passe de 3,4 a 16 h de la premiere a la derniere
    rafale : l'extinction domine la transparence mesuree de plusieurs ordres de
    grandeur et il faut la retirer avant d'y lire un voile nuageux.
    """
    h = np.maximum(np.asarray(alt_app_deg, float), 0.0)
    return 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)


def local_axes(vert_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Vecteurs unitaires (horizontale locale, verticale locale) en coordonnees image.

    Coordonnees tableau : x colonne vers la droite, y ligne vers le bas. Un
    angle nul place la verticale locale sur l'axe -y du capteur. La monture
    etant alt/az sans rotateur, cet angle est constant sur la session.
    """
    a = np.radians(vert_angle_deg)
    e_v = np.array([np.sin(a), -np.cos(a)])
    e_h = np.array([np.cos(a), np.sin(a)])
    return e_h, e_v


LOCAL_UTC_OFFSET_H = 2.0  # CEST le 12 aout 2026


def contacts(
    day: str = "2026-08-12", t0: str = "16:30:00", t1: str = "20:00:00", step_s: float = 1.0
) -> list[dict]:
    """Circonstances de l'eclipse pour le site, en heure legale.

    Une eclipse partielle n'a que deux contacts, C1 et C4 : C2 et C3 marquent le
    debut et la fin de la phase totale ou annulaire, qui n'existe pas ici.

    Le coucher est pris au sens usuel, bord superieur du disque a l'horizon avec
    la refraction standard, soit un centre a -0,833 degre.
    """
    t = (
        Time(f"{day}T{t0}")
        + np.arange(0, (Time(f"{day}T{t1}") - Time(f"{day}T{t0}")).sec, step_s) * u.s
    )
    loc = location()
    aa = AltAz(obstime=t, location=loc)
    sun, moon = get_sun(t), get_body("moon", t, loc)
    sa, ma = sun.transform_to(aa), moon.transform_to(aa)
    sep = sa.separation(ma).to_value(u.arcsec)
    r_sun = np.degrees(np.arcsin(R_SUN_KM / sun.distance.to_value(u.km))) * 3600.0
    r_moon = np.degrees(np.arcsin(R_MOON_KM / moon.distance.to_value(u.km))) * 3600.0
    obsc = _overlap_fraction(sep, r_sun, r_moon)
    alt_true = sa.alt.deg
    alt_app = alt_true + bennett_refraction(alt_true) / 60.0

    touching = sep < (r_sun + r_moon)
    idx = np.nonzero(touching)[0]
    events: list[tuple[str, int, str]] = []
    if idx.size:
        events.append(("C1, premier contact", int(idx[0]), "debut de l'eclipse partielle"))
    events.append(("Maximum", int(np.argmax(obsc)), "obscuration maximale"))
    below = np.nonzero(alt_app < -0.833)[0]
    if below.size:
        events.append(("Coucher du soleil", int(below[0]), "bord superieur a l'horizon"))
    if idx.size:
        events.append(("C4, dernier contact", int(idx[-1]), "fin de l'eclipse partielle"))

    out = []
    for label, i, note in sorted(events, key=lambda e: e[1]):
        local = t[i] + LOCAL_UTC_OFFSET_H * u.hour
        out.append(
            {
                "label": label,
                "note": note,
                "local": local.iso[11:19],
                "utc": t[i].iso[11:19],
                "alt_app": float(alt_app[i]),
                "az": float(sa.az.deg[i]),
                "obsc": float(obsc[i]),
                # le coucher est par definition a la limite : ne marquer que ce qui suit
                "visible": bool(alt_app[i] > -0.833) or label.startswith("Coucher"),
            }
        )
    return out
