"""Click-through for the transparent parts of a pet window.

Qt's setMask() cuts the window's visible shape with a 1-bit region. A pet's mask can only
be refreshed a few times a second, so it lags behind the animation and clips moving hair
and limbs. Each platform has a way to limit input without touching what is drawn:

- X11: a SHAPE extension "input" region. Qt has no API for it, so libxcb is called directly.
- Windows: Qt draws frameless translucent windows with UpdateLayeredWindow, and Windows
  already passes clicks through fully transparent pixels of those.
- Elsewhere: fall back to setMask() with a wide margin around the pet.
"""

from __future__ import annotations

import ctypes
import logging
import sys

import numpy as np
from PySide6.QtGui import QBitmap, QGuiApplication, QImage
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

_SO_SET = 0
_SK_INPUT = 2
_UNSORTED = 0
_NONE = 0
_FALLBACK_MARGIN = 12  # px kept visible around the pet when setMask() is the only option


def _load_xcb():
    """Open a private xcb connection. Qt's xcb plugin already needs both libraries."""
    try:
        xcb = ctypes.CDLL("libxcb.so.1")
        shape = ctypes.CDLL("libxcb-shape.so.0")
    except OSError as exc:
        log.warning("libxcb-shape unavailable: %s", exc)
        return None
    xcb.xcb_connect.restype = ctypes.c_void_p
    xcb.xcb_connect.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    xcb.xcb_connection_has_error.argtypes = [ctypes.c_void_p]
    xcb.xcb_flush.argtypes = [ctypes.c_void_p]
    shape.xcb_shape_rectangles.restype = ctypes.c_uint
    shape.xcb_shape_rectangles.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32,
        ctypes.c_int16, ctypes.c_int16, ctypes.c_uint32, ctypes.c_void_p,
    ]
    shape.xcb_shape_mask.restype = ctypes.c_uint
    shape.xcb_shape_mask.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32,
        ctypes.c_int16, ctypes.c_int16, ctypes.c_uint32,
    ]
    conn = xcb.xcb_connect(None, None)
    if not conn or xcb.xcb_connection_has_error(conn):
        log.warning("could not open an xcb connection for input shapes")
        return None
    return xcb, shape, conn


_xcb = None
_xcb_tried = False


def _x11():
    global _xcb, _xcb_tried
    if not _xcb_tried:
        _xcb_tried = True
        if QGuiApplication.platformName() == "xcb":
            _xcb = _load_xcb()
    return _xcb


def mask_to_rects(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean [h, w] mask to xcb_rectangle_t records, one per horizontal run."""
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    rows, cols = np.nonzero(np.diff(padded, axis=1))
    # Edges come in start/end pairs, ordered by row and then column.
    rects = np.zeros(len(rows) // 2, dtype=[("x", "i2"), ("y", "i2"), ("w", "u2"), ("h", "u2")])
    rects["x"] = cols[::2]
    rects["y"] = rows[::2]
    rects["w"] = cols[1::2] - cols[::2]
    rects["h"] = 1
    return rects


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by `radius` pixels (square kernel), without wrapping at the edges."""
    grown = mask.copy()
    step = 1
    while radius > 0:
        s = min(step, radius)
        for axis in (0, 1):
            before = np.zeros_like(grown)
            after = np.zeros_like(grown)
            if axis == 0:
                before[:-s, :] = grown[s:, :]
                after[s:, :] = grown[:-s, :]
            else:
                before[:, :-s] = grown[:, s:]
                after[:, s:] = grown[:, :-s]
            grown |= before | after
        radius -= s
        step *= 2
    return grown


def set_input_region(widget: QWidget, mask: np.ndarray) -> None:
    """Accept mouse input only where the boolean [h, w] `mask` is True.

    The mask is in device pixels, like the rendered frame. X11 regions use them too, while
    setMask() takes logical pixels.
    """
    if sys.platform == "win32":
        return
    x11 = _x11()
    if x11 is not None:
        xcb, shape, conn = x11
        rects = mask_to_rects(mask)
        shape.xcb_shape_rectangles(conn, _SO_SET, _SK_INPUT, _UNSORTED, int(widget.winId()), 0, 0,
                                   len(rects), rects.ctypes.data)
        xcb.xcb_flush(conn)
        return
    h, w = widget.height(), widget.width()
    ys = np.minimum((np.arange(h) * mask.shape[0] / max(h, 1)).astype(int), mask.shape[0] - 1)
    xs = np.minimum((np.arange(w) * mask.shape[1] / max(w, 1)).astype(int), mask.shape[1] - 1)
    grown = dilate(mask[np.ix_(ys, xs)], _FALLBACK_MARGIN)
    argb = np.where(grown, np.uint32(0xFF000000), np.uint32(0)).astype(np.uint32)
    img = QImage(argb.data, w, h, w * 4, QImage.Format.Format_ARGB32)
    widget.setMask(QBitmap.fromImage(img.createAlphaMask()))


def clear_input_region(widget: QWidget) -> None:
    """Accept mouse input over the whole window again."""
    if sys.platform == "win32":
        return
    x11 = _x11()
    if x11 is not None:
        xcb, shape, conn = x11
        shape.xcb_shape_mask(conn, _SO_SET, _SK_INPUT, int(widget.winId()), 0, 0, _NONE)
        xcb.xcb_flush(conn)
        return
    widget.clearMask()
