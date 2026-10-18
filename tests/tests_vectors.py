#!/usr/bin/env python3
"""Tests für lialkit.rechnen.vectors (Skalar-/Vektor-/Spatprodukt, Winkel, Geraden, Ebenen). Start: python3 tests/tests_vectors.py"""
import os
import random
import sys
from fractions import Fraction as F
from itertools import permutations

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # Elternordner des Pakets lialkit


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def ref_solve(A, b):
    """Unabhängiger Gauss mit Fraction (nur eindeutig lösbare Systeme)."""
    n = len(A)
    M = [[F(v) for v in row] + [F(bi)] for row, bi in zip(A, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c:
                M[r] = [a - M[r][c] * k for a, k in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def leibniz(A):
    """Determinante nach der Leibniz-Formel, unabhängig von lialkit."""
    n, total = len(A), F(0)
    for perm in permutations(range(n)):
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    sign = -sign
        prod = F(1)
        for i in range(n):
            prod *= A[i][perm[i]]
        total += sign * prod
    return total


def random_regular(n, rng):
    while True:
        A = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
        if leibniz(A) != 0:
            return A


EXA = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]     # Beispiel aus __init__.py
EXB = [8, -11, -3]                               # Lösung (2, 3, -1), det = -1

import math  # noqa: E402

from lialkit import (Line, Plane, Vector, add, angle, cauchy_schwarz, cross, dot,  # noqa: E402
                     position, projection, scale, spat, triangle_inequality)

U, V, W = Vector([1, 2, 3]), Vector([4, 5, 6]), Vector([7, 8, 10])


def _render(res):
    assert repr(res) and res.latex() and res._repr_html_()


def test_dot():
    r = dot(U, V, decimal=False)
    assert r.value == 32 and "Winkel" in r.summary
    assert dot([1, 0], [0, 1]).value == 0 and "senkrecht" in dot([1, 0], [0, 1]).summary
    assert dot([1, 0], [-1, 0]).value == -1 and "grösser" in dot([1, 0], [-1, 0]).summary
    assert raises(ValueError, dot, [1, 2], [1, 2, 3])
    _render(r)


def test_cross():
    r = cross(U, V)
    assert r.result == Vector([-3, 6, -3]) and r.area == Vector([-3, 6, -3]).norm()
    assert r.result.dot(U) == 0 and r.result.dot(V) == 0
    assert cross([1, 2, 3], [2, 4, 6]).result.is_zero() and "parallel" in cross([1, 2, 3], [2, 4, 6]).summary
    assert cross([2, -3, 6], [-1, 4, 8]).result == Vector([-48, -22, 5])       # SW1.5: -24-24, -6-16, 8-3
    assert raises(ValueError, cross, [1, 2], [3, 4])
    _render(r)


def test_spat():
    r = spat(U, V, W)
    assert r.value == -3 and r.volume == 3 and "Linkssystem" in r.summary
    assert spat(W, V, U).value == 3                                             # Vertauschen dreht das Vorzeichen
    assert spat(U, V, Vector([5, 7, 9])).value == 0 and "Ebene" in spat(U, V, Vector([5, 7, 9])).summary
    _render(r)


def test_norm_exact_and_angle():
    assert Vector([3, 4]).norm() == 5
    r = angle([1, 0], [1, 1])
    assert abs(r.degrees - 45) < 1e-9 and r.latex()
    o = angle([1, 0, 0], [0, 5, 0])
    assert abs(o.degrees - 90) < 1e-9 and o.warnings
    assert abs(angle([2, -3, 6], [-1, 4, 8]).degrees
               - math.degrees(math.acos((-2 - 12 + 48) / 63))) < 1e-9          # 34 / (7*9)
    assert raises(ValueError, angle, [0, 0], [1, 0])
    _render(r)


def test_add_position_scale():
    assert dict(add(U, V).highlights)["u + v"] == Vector([5, 7, 9])
    p = position([1, 1], [4, 5])
    assert dict(p.highlights)[r"\vec{AB}"] == Vector([3, 4]) and "5" in p.summary    # |(3,4)| = 5
    s = scale([1, -2], -2)
    assert dict(s.highlights)[r"k \cdot u"] == Vector([-2, 4]) and "kehrt" in s.summary
    assert "Nullvektor" in scale([1, 2], 0).summary
    for r in (add(U, V), p, s):
        _render(r)


def test_projection():
    r = projection([2, 3], [1, 0])
    assert dict(r.highlights)["p"] == Vector([2, 0])
    r2 = projection([3, 4], [1, 1])                  # lambda = 7/2
    assert dict(r2.highlights)["p"] == Vector([F(7, 2), F(7, 2)])
    assert raises(ValueError, projection, [1, 2], [0, 0])
    _render(r2)


def test_inequalities():
    for u, v in (([3, 4], [1, 2]), ([1, 0, 0], [0, 1, 0]), ([1, 2], [2, 4]), ([1, 2], [-2, -4])):
        t, c = triangle_inequality(u, v), cauchy_schwarz(u, v)
        assert t.summary == "Erfüllt." and not t.warnings, (u, v)
        assert not c.warnings and c.summary != "Verletzt."
        _render(t)
        _render(c)
    assert "Gleichheit" in cauchy_schwarz([1, 2], [2, 4]).summary


def test_line_basics_and_intersection():
    g, h = Line((1, 1), (2, 1)), Line((1, 4), (1, -1))     # SW1.3: lambda = 1, mu = 2
    assert g.at(1) == Vector([3, 2]) and g.contains((5, 3)) and not g.contains((5, 4))
    r = g.intersect(h)
    assert dict(r.highlights)["S"] == Vector([3, 2]) and h.at(2) == Vector([3, 2])
    _render(r)
    par = Line((0, 0), (1, 1)).intersect(Line((0, 1), (2, 2)))
    assert not par.highlights and "parallel" in par.summary
    same = Line((0, 0), (1, 1)).intersect(Line((1, 1), (3, 3)))
    assert "zusammen" in same.summary
    skew = Line((0, 0, 0), (1, 0, 0)).intersect(Line((0, 0, 1), (0, 1, 0)))
    assert not skew.highlights and ("windschief" in skew.summary or "Kein" in skew.summary)
    cut = Line((0, 0, 0), (1, 1, 0)).intersect(Line((2, 0, 0), (-1, 1, 0)))        # x = y, x = 2 - y -> (1,1,0)
    assert dict(cut.highlights)["S"] == Vector([1, 1, 0])
    assert raises(ValueError, Line, (0, 0), (0, 0))
    assert raises(ValueError, g.intersect, Line((0, 0, 0), (1, 1, 1)))


def test_plane_and_line_plane():
    E = Plane((1, 2, 4), u=(1, 0, 1), v=(0, 1, -2))        # SW1.4
    assert E.n == Vector([-1, 2, 1]) and E.c == 7 and E.contains((1, 2, 4))
    assert E.coordinate_form() == "−x + 2·y + z = 7" and "Plane" in repr(E)
    assert E.at(1, 1) == Vector([2, 3, 3]) and E.contains(E.at(2, -3))
    g = Line((1, 1, 7), (2, 3, 4))
    r = E.intersect_line(g)
    S = dict(r.highlights)["S"]
    assert S == Vector([F(3, 4), F(5, 8), F(13, 2)]) and E.contains(S) and g.contains(S)
    _render(r)
    assert "parallel" in E.intersect_line(Line((0, 0, 0), (1, 1, -1))).summary          # n.d = 0
    assert "ganz in der Ebene" in E.intersect_line(Line((1, 2, 4), (1, 0, 1))).summary
    N = Plane((0, 0, 1), normal=(0, 0, 2))                  # z = 1
    assert N.c == 2 and abs(N.distance((5, 5, 4)) - 3) < 1e-12
    assert raises(ValueError, N.at, 0, 0)
    assert raises(ValueError, Plane, (0, 0, 0), u=(1, 0, 0), v=(2, 0, 0))
    assert raises(ValueError, Plane, (0, 0, 0))


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
