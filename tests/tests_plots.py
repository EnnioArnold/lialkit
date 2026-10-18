#!/usr/bin/env python3
"""
Tests für lialkit.grafik.plots — laufen ohne Bildschirm (Agg-Backend).
Start: python3 tests/tests_plots.py
"""
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # Elternordner des Pakets lialkit

from matplotlib.figure import Figure  # noqa: E402

from lialkit import (Line, Matrix, Plane, Vector, has_plotly, plot_line, plot_plane,  # noqa: E402
                     plot_spat, plot_transformation, plot_vectors)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def _mpl(fig):
    assert isinstance(fig, Figure) and fig.axes
    fig.canvas.draw()                                  # wirklich rendern, nicht nur bauen
    return fig


def _plotly(fig):
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure) and len(fig.data) > 0
    return fig


def test_plot_vectors_2d():
    _mpl(plot_vectors([3, 4], [-1, 2]))
    _mpl(plot_vectors((1, 1), labels=["a"], title="Titel"))
    _mpl(plot_vectors(Vector([1, 2]), Vector([2, -1]), chain=True))
    _mpl(plot_vectors([1, 0], [0, 1], origins=[(1, 1), (2, 2)]))
    _mpl(plot_vectors([0, 0]))                         # Nullvektor darf nicht abstürzen


def test_plot_vectors_3d():
    _mpl(plot_vectors([1, 2, 3], [4, 5, 6], interactive=False))
    _mpl(plot_vectors([1, 2, 3], [-1, 0, 2], chain=True, labels=["u", "v"], interactive=False))
    _mpl(plot_vectors([-2, -2, -2], interactive=False))
    if has_plotly():
        _plotly(plot_vectors([1, 2, 3], [4, 5, 6], interactive=True))
    assert raises(Exception, plot_vectors)             # ohne Vektor gibt es nichts zu zeichnen


def test_plot_spat():
    _mpl(plot_spat([1, 2, 3], [4, 5, 6], [7, 8, 10], interactive=False))
    _mpl(plot_spat([1, 0, 0], [0, 1, 0], [0, 0, 1], title="Würfel", labels=["a", "b", "c"], interactive=False))
    _mpl(plot_spat([1, 2, 3], [2, 4, 6], [0, 0, 1], interactive=False))      # flach (Volumen 0)
    if has_plotly():
        _plotly(plot_spat([1, 2, 3], [4, 5, 6], [7, 8, 10], interactive=True))


def test_plot_line():
    _mpl(plot_line(Line((1, 1), (2, 1))))
    _mpl(plot_line(Line((0, 0), (1, 1)), span=(-1, 4), title="g"))
    _mpl(plot_line(Line((1, 1, 7), (2, 3, 4))))
    _mpl(plot_line(Line((0, 0, 0), (0, 0, 1))))


def test_plot_plane():
    E = Plane((1, 2, 4), u=(1, 0, 1), v=(0, 1, -2))
    _mpl(plot_plane(E, interactive=False))
    _mpl(plot_plane(E, span=(-1, 3), points=[(1, 2, 4), (0, 0, 0)], title="E", interactive=False))
    _mpl(plot_plane(Plane((0, 0, 1), normal=(0, 0, 2)), interactive=False))   # nur Normale gegeben
    if has_plotly():
        _plotly(plot_plane(E, points=[(1, 2, 4)], interactive=True))


def test_plot_transformation():
    _mpl(plot_transformation([[2, 0], [0, 1]]))
    _mpl(plot_transformation(Matrix([[1, 1], [0, 1]]), title="Scherung", grid=3))
    _mpl(plot_transformation([[0, 1], [1, 0]]))        # det < 0: Orientierung kippt
    _mpl(plot_transformation([[1, 2], [2, 4]]))        # singulär: Bild ist eine Gerade
    assert raises(ValueError, plot_transformation, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_figures_are_closed():
    import matplotlib.pyplot as plt
    plt.close("all")
    plot_vectors([1, 2])
    plot_transformation([[1, 0], [0, 1]])
    assert plt.get_fignums() == []                     # _finish schliesst, sonst doppelte Anzeige im Notebook


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print("FAIL", name, repr(e))
                traceback.print_exc()
    print("alle Tests bestanden" if not fails else f"{fails} Test(s) fehlgeschlagen")
    sys.exit(1 if fails else 0)
