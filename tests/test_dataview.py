import numpy as np
from SoundEditor.DataView import zoom_invariant_view, down_sample


def test_zoom_variant_view():
    data = np.ones((100, 1))
    xlim = (0, 10)

    xlim_zoom = (0,5)

    assert zoom_invariant_view(data, xlim, xlim, points_on_screen = 5) == 20
    assert zoom_invariant_view(data, xlim, xlim, points_on_screen = 10) == 10
    assert zoom_invariant_view(data, xlim, xlim, points_on_screen = 8) == 12

    assert zoom_invariant_view(data, xlim_zoom, xlim, points_on_screen = 5) == 10
    assert zoom_invariant_view(data, xlim_zoom, xlim, points_on_screen = 10) == 5
    assert zoom_invariant_view(data, xlim_zoom, xlim, points_on_screen = 8) == 6


def test_down_sample():
    array = np.array([89, 66, 29, 25, 36, 25, 30, 58, 64, 19, 25, 63, 76, 74, 44, 73, 94,
       88, 83, 88, 17, 91, 69, 65, 32, 73, 91, 20, 20, 14, 52, 65, 21, 58,
       14, 30, 26, 82, 61, 87, 24, 67, 83, 93, 57, 30, 81, 48, 84, 83, 59,
       19, 95, 55, 86, 57, 59, 77, 92, 44, 40, 29, 37, 42, 33, 89, 37, 57,
       18, 17, 85, 47, 19, 95, 96, 40, 13, 64, 18, 79, 95, 26, 31, 70, 35,
       65, 52, 93, 46, 63, 86, 77, 87, 48, 88, 62, 68, 82, 49, 86])

    down_sampled = down_sample(array, 7)

    assert down_sampled[0] == array[:7].mean()
    assert down_sampled[1] == array[7:14].mean()