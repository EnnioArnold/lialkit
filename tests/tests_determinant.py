#!/usr/bin/env python3
"""Tests für lialkit.rechnen.determinant (Sarrus, Laplace, Dreiecksform, det). Start: python3 tests/tests_determinant.py"""
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

from lialkit import (Matrix, cofactor, det, det_laplace, det_sarrus,  # noqa: E402
                     det_triangular, minor)

SW44 = [[1, 2, 3], [4, 5, 7], [7, 8, 10]]       # det = 3 (Handrechnung mit Sarrus: 244 - 241)


def test_known_values():
    assert det(EXA).value == -1                  # Sarrus: -5 - (-4)
    assert det(SW44).value == 3
    assert det([[1, 2], [3, 5]]).value == -1
    assert det([[1, 2], [3, 6]]).value == 0 and det([[1, 2], [3, 6]]).singular
    assert det([[7]]).value == 7
    assert det(Matrix.identity(5)).value == 1
    assert det([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]]).value == -6   # ein Tausch
    assert det([[2, 5, 1, 0], [0, 3, 7, 1], [0, 0, 4, 9], [0, 0, 0, 5]]).value == 120   # Dreieck


def test_methods_agree_with_leibniz():
    rng = random.Random(5)
    for n in (1, 2, 3, 4, 5):
        for _ in range(6):
            A = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
            want = leibniz(A)
            assert det(A).value == want
            assert det_laplace(A).value == want
            assert det_triangular(A).value == want
            if n == 3:
                assert det_sarrus(A).value == want
            if n > 1:
                assert det_laplace(A, along_row=n - 1).value == want


def test_properties():
    A = Matrix(SW44)
    assert det(A.scaled(3)).value == 3 ** 3 * 3          # SW4.4: lambda^n
    assert det(A.scaled(-1)).value == -det(A).value      # n = 3 ungerade
    B = Matrix([[1, 2], [3, 5]])
    assert det(B.scaled(-1)).value == det(B).value       # n = 2 gerade
    assert det(A.T).value == det(A).value
    assert det(A @ Matrix(EXA)).value == det(A).value * det(EXA).value
    assert det(A.swap(0, 2)).value == -det(A).value


def test_errors_and_inputs():
    assert raises(ValueError, det_sarrus, Matrix.identity(4))
    assert raises(ValueError, det, [[1, 2, 3], [4, 5, 6]])
    assert raises(ValueError, det, EXA, method="cramer")
    assert raises(ValueError, det_laplace, Matrix.identity(7))
    assert det(Matrix.system([[1, 2], [3, 4]], [5, 6])).value == -2   # rechte Seite wird ignoriert


def test_minor_cofactor():
    A = Matrix(SW44)
    assert minor(A, 0, 0).rows == [[5, 7], [8, 10]]
    assert minor(A, 1, 2).rows == [[1, 2], [7, 8]]
    assert cofactor(A, 0, 0) == -6 and cofactor(A, 0, 1) == -(40 - 49) == 9
    assert sum(A[0, c] * cofactor(A, 0, c) for c in range(3)) == 3


def test_render_and_methods():
    for method in ("auto", "sarrus", "laplace", "triangular"):
        r = det(SW44, method=method, weg=True) if False else det(SW44, method=method)
        assert r.value == 3 and r.steps and r.latex() and r._repr_html_() and repr(r)
    assert det(SW44, decimal=True).latex()
    assert det([[1, 2], [3, 4]]).method == "direct" and det(SW44).method == "sarrus"
    assert det(Matrix.identity(4)).method == "triangular"
    assert bool(det(SW44)) and not bool(det([[1, 1], [1, 1]]))
    zero = det_triangular([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
    assert zero.value == 0 and zero.steps


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
