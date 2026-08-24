"""Ecriture TIFF 16 bits par canal, avec profil ICC embarque.

OpenCV ecrit bien du 16 bits mais n'embarque aucun profil, et Pillow ne sait
pas ecrire de RGB 48 bits. Un TIFF non tagge est interprete en sRGB par defaut
par la plupart des applications, ce qui donne le bon resultat par accident :
DxO et Nik doivent lire le profil, pas le deviner.

Le format ecrit ici est le TIFF de base, non compresse, une seule bande, ordre
d'octets petit-boutiste. C'est le sous-ensemble le plus universellement lu.
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import numpy as np

# Emplacement du profil sRGB du systeme. Le generer avec littleCMS donne un
# profil equivalent, la version du systeme est preferee quand elle existe.
_SYS_SRGB = Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc")

SHORT, LONG, ASCII, RATIONAL, UNDEFINED = 3, 4, 2, 5, 7


def srgb_icc() -> bytes | None:
    """Profil sRGB a embarquer, ou None si aucune source n'est disponible."""
    if _SYS_SRGB.exists():
        return _SYS_SRGB.read_bytes()
    try:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def srgb_encode(v: np.ndarray) -> np.ndarray:
    """Fonction de transfert sRGB, par morceaux, pas une puissance 2,2.

    Elle doit correspondre exactement au profil embarque. Elle sert aussi la
    dynamique : a 16 bits, l'encodage non lineaire donne aux ombres une
    resolution que le lineaire depenserait dans les hautes lumieres.
    """
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def write_rgb16(
    path: Path,
    rgb: np.ndarray,
    icc: bytes | None = None,
    description: str = "",
    software: str = "eclipse",
    dpi: int = 300,
) -> Path:
    """Ecrit un tableau (H, W, 3) uint16 en TIFF de base non compresse."""
    a = np.ascontiguousarray(rgb, dtype="<u2")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"attendu (H, W, 3), recu {a.shape}")
    h, w = a.shape[:2]
    data = a.tobytes()

    aux = bytearray()  # zone de debordement, pour toute valeur au-dela de 4 octets

    def push(b: bytes) -> int:
        # les valeurs pointees doivent commencer sur un octet pair
        if len(aux) % 2:
            aux.append(0)
        off = len(aux)
        aux.extend(b)
        return off

    o_bits = push(struct.pack("<3H", 16, 16, 16))
    o_res = push(struct.pack("<2I", dpi, 1))
    o_soft = push(software.encode() + b"\0")
    o_date = push(f"{dt.datetime.now():%Y:%m:%d %H:%M:%S}".encode() + b"\0")
    o_desc = push(description.encode() + b"\0") if description else 0
    o_icc = push(icc) if icc else 0

    ifd = [
        (256, LONG, 1, w),  # ImageWidth
        (257, LONG, 1, h),  # ImageLength
        (258, SHORT, 3, o_bits),  # BitsPerSample
        (259, SHORT, 1, 1),  # Compression, aucune
        (262, SHORT, 1, 2),  # PhotometricInterpretation, RGB
        (273, LONG, 1, 8),  # StripOffsets, les donnees suivent l'en-tete
        (277, SHORT, 1, 3),  # SamplesPerPixel
        (278, LONG, 1, h),  # RowsPerStrip, une seule bande
        (279, LONG, 1, len(data)),  # StripByteCounts
        (282, RATIONAL, 1, o_res),  # XResolution
        (283, RATIONAL, 1, o_res),  # YResolution
        (296, SHORT, 1, 2),  # ResolutionUnit, pouce
        (305, ASCII, len(software) + 1, o_soft),  # Software
        (306, ASCII, 20, o_date),  # DateTime
    ]
    if o_desc or description:
        ifd.append((270, ASCII, len(description) + 1, o_desc))
    if icc:
        ifd.append((34675, UNDEFINED, len(icc), o_icc))
    ifd.sort()  # le TIFF impose des etiquettes croissantes

    ifd_off = 8 + len(data)
    aux_off = ifd_off + 2 + 12 * len(ifd) + 4
    inline = {SHORT: 2, LONG: 4, ASCII: 1, RATIONAL: 8, UNDEFINED: 1}

    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_off))
    out.extend(data)
    out.extend(struct.pack("<H", len(ifd)))
    for tag, typ, count, val in ifd:
        if inline[typ] * count <= 4:
            payload = struct.pack("<H2x", val) if typ == SHORT else struct.pack("<I", val)
        else:
            payload = struct.pack("<I", aux_off + val)
        out.extend(struct.pack("<HHI", tag, typ, count) + payload)
    out.extend(struct.pack("<I", 0))  # pas d'IFD suivant
    out.extend(aux)

    path = Path(path)
    path.write_bytes(bytes(out))
    return path
