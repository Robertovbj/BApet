import numpy as np

from bapet.input_shape import dilate, mask_to_rects


def test_mask_to_rects_finds_runs_per_row():
    mask = np.array([
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=bool)
    rects = mask_to_rects(mask)
    assert [tuple(r) for r in rects.tolist()] == [(1, 0, 2, 1), (4, 0, 1, 1), (0, 2, 5, 1)]
    assert rects.dtype.itemsize == 8  # layout of xcb_rectangle_t


def test_mask_to_rects_empty():
    assert len(mask_to_rects(np.zeros((4, 4), dtype=bool))) == 0


def test_dilate_grows_a_square_without_wrapping():
    mask = np.zeros((31, 31), dtype=bool)
    mask[15, 15] = True
    mask[0, 0] = True
    grown = dilate(mask, 12)
    assert grown[3:28, 3:28].all()  # no holes from the growing step size
    assert not grown[2, 28]
    assert grown[12, 12] and not grown[30, 30]  # corner pixel does not wrap around
