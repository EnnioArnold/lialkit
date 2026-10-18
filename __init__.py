#!/usr/bin/env python3
"""
lialkit — lineare Algebra mit nachvollziehbarem Rechenweg.

Gebaut zum Modul I.BA_LIAL_MM.20 der HSLU. Alles rechnet exakt mit Brüchen
und kann jeden Schritt zeigen, so wie er in der Prüfung auf dem Papier
stünde.

Kurzübersicht
-------------

    from lialkit import *

    A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b = Vector([8, -11, -3])

    gauss(A, b)                 # Ergebnis
    gauss(A, b, weg=True)       # mit Rechenweg
    lu(A)                       # LR-Zerlegung
    inverse(A)                  # über Gauss-Jordan
    det(A)                      # Determinante

    u, v = Vector([1, 2, 3]), Vector([4, 5, 6])
    cross(u, v)
    spat(u, v, Vector([7, 8, 10]))
    plot_vectors(u, v)

Jedes Ergebnis zeigt sich im Notebook von selbst. `.as_decimal()` schaltet
auf vier Stellen um, `.latex()` liefert den Rechenweg zum Einfügen in eine
Abgabe.
"""

from __future__ import annotations

from .rechnen.core import (MINUS, Matrix, Num, Operation, Vector, fmt, fmt_factor,
                   is_exact, parse_number, tex_number, triple_product)
from .rechnen.determinant import (DetResult, cofactor, det, det_laplace, det_sarrus,
                          det_triangular, minor)
from .rechnen.factor import (LUResult, LUSolveResult, back_substitution,
                     elimination_matrix, forward_substitution,
                     permutation_matrix, lu, solve_lu, swap_matrix)
from .rechnen.inverse import (InverseResult, inverse, inverse_2x2, is_invertible,
                      solve_with_inverse)
from .rechnen.render import Result, Step, show_matrix, tex_matrix, tex_vector, text_matrix
from .rechnen.vectors import (AngleResult, CrossResult, DotResult, Line, Plane,
                      SpatResult, add, angle, cauchy_schwarz, cross, dot,
                      position, projection, scale, spat, triangle_inequality)
from .rechnen.solving import GaussResult, gauss, gauss_jordan, rank, solve
from .grafik.plots import (plot_line, plot_plane, plot_spat, plot_transformation,
                    plot_vectors, has_plotly)

# Die Module liegen in Unterordnern; die alten Namen (lialkit.core, lialkit.anim,
# lialkit.exercises, …) bleiben als Alias erhalten, damit Notebooks weiter laufen.
import importlib as _il, sys as _sys
for _alias, _real in (("core", "rechnen.core"), ("render", "rechnen.render"),
                      ("determinant", "rechnen.determinant"), ("factor", "rechnen.factor"),
                      ("inverse", "rechnen.inverse"), ("solving", "rechnen.solving"),
                      ("vectors", "rechnen.vectors"), ("plots", "grafik.plots"),
                      ("anim", "grafik.anim"), ("exercises", "aufgaben.exercises"),
                      ("uebung", "aufgaben.uebung")):
    _m = _il.import_module(f"{__name__}.{_real}")
    _sys.modules[f"{__name__}.{_alias}"] = _m
    globals().setdefault(_alias, _m)   # `inverse` bleibt die Funktion, nicht das Modul
del _il, _sys, _alias, _real, _m

__version__ = "0.1.0"

__all__ = [
    # Grundlagen
    "Matrix", "Vector", "Operation", "Num", "MINUS",
    "parse_number", "fmt", "fmt_factor", "tex_number", "is_exact",
    # Gauss
    "gauss", "gauss_jordan", "solve", "rank", "GaussResult",
    # Zerlegung
    "lu", "solve_lu", "elimination_matrix", "permutation_matrix",
    "swap_matrix", "forward_substitution", "back_substitution",
    "LUResult", "LUSolveResult",
    # Inverse
    "inverse", "inverse_2x2", "is_invertible", "solve_with_inverse",
    "InverseResult",
    # Determinante
    "det", "det_sarrus", "det_laplace", "det_triangular", "minor", "cofactor",
    "DetResult",
    # Vektoren
    "dot", "cross", "spat", "angle", "projection", "triple_product",
    "add", "position", "scale",
    "triangle_inequality", "cauchy_schwarz", "Line", "Plane",
    "DotResult", "CrossResult", "SpatResult", "AngleResult",
    # Zeichnen
    "plot_vectors", "plot_spat", "plot_line", "plot_plane",
    "plot_transformation", "has_plotly",
    # Darstellung
    "Result", "Step", "show_matrix", "tex_matrix", "tex_vector", "text_matrix",
]


def info() -> str:
    """Was dieses Paket kann und was es braucht."""
    lines = [
        f"lialkit {__version__} — lineare Algebra mit Rechenweg",
        "",
        "Verfahren:",
        "  gauss, gauss_jordan, solve, rank",
        "  lu (LR-Zerlegung, auch P·A = L·R), solve_lu",
        "  inverse (Gauss-Jordan), inverse_2x2, is_invertible",
        "  det (auto), det_sarrus, det_laplace, det_triangular",
        "  dot, cross, spat, angle, projection",
        "  triangle_inequality, cauchy_schwarz",
        "",
        "Zeichnen:",
        "  plot_vectors, plot_spat, plot_line, plot_plane, plot_transformation",
        f"  Plotly für Räumliches: {'vorhanden' if has_plotly() else 'fehlt'}",
        "",
        "Jedes Ergebnis kennt .as_decimal(), .as_exact(), .latex() und .show().",
    ]
    return "\n".join(lines)
