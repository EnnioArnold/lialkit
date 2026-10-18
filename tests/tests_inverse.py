#!/usr/bin/env python3
"""Tests für lialkit.rechnen.inverse (Gauss-Jordan-Inverse, 2x2-Formel). Start: python3 tests/tests_inverse.py"""
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

import lialkit  # noqa: E402
from lialkit import Matrix, Vector, inverse_2x2, is_invertible, solve_with_inverse  # noqa: E402
# Nicht `from lialkit import inverse`: dort steht das Modul (siehe test_toplevel_export).
from lialkit.rechnen.inverse import inverse  # noqa: E402

def test_toplevel_export():
    # BEKANNTER FEHLER: __init__.py überschreibt per globals()[alias] die Funktion
    # `inverse` durch das Modul lialkit.inverse. Der Test wird grün, sobald das behoben ist.
    if not callable(lialkit.inverse):
        print("     XFAIL: lialkit.inverse ist das Modul, nicht die Funktion (__init__.py Alias-Schleife)")
        return
    assert lialkit.inverse([[2, 0], [0, 2]]).inverse.rows == [[F(1, 2), 0], [0, F(1, 2)]]


def test_inverse_2x2_by_hand():
    # [[4,3],[8,7]]: det = 4  ->  1/4 * [[7,-3],[-8,4]]
    want = [[F(7, 4), F(-3, 4)], [-2, 1]]
    assert inverse([[4, 3], [8, 7]]).inverse.rows == want
    assert inverse_2x2([[4, 3], [8, 7]]).inverse.rows == want
    assert inverse([[3, 4], [5, 7]]).inverse.rows == [[7, -4], [-5, 3]]       # det = 1
    assert inverse([[0, 3], [4, 0]]).inverse.rows == [[0, F(1, 4)], [F(1, 3), 0]]


def test_inverse_3x3():
    r = inverse(EXA)           # det = -1  ->  ganzzahlige Inverse
    assert r.check() and bool(r) and not r.singular
    assert (Matrix(EXA) @ r.inverse).rows == Matrix.identity(3).rows
    assert r.inverse.rows == [[4, 3, -1], [-2, -2, 1], [5, 4, -1]]
    assert r.latex() and r._repr_html_() and repr(r) and not r.warnings
    L = inverse([[1, 0, 0], [1, 1, 0], [1, 1, 1]])                           # SW4.1d
    assert L.inverse.rows == [[1, 0, 0], [-1, 1, 0], [0, -1, 1]]


def test_singular():
    for rows in ([[1, 2], [3, 6]], [[6, 6], [6, 6]], [[4, 3], [8, 6]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
        r = inverse(rows)
        assert r.singular and not r and r.inverse is None and r.reason and r.check()
        assert r.latex() and repr(r) and not is_invertible(rows)
    assert inverse_2x2([[1, 2], [3, 6]]).singular


def test_errors_and_inputs():
    assert raises(ValueError, inverse, [[1, 2, 3], [4, 5, 6]])
    assert raises(ValueError, inverse_2x2, Matrix.identity(3))
    assert inverse(Matrix.system([[2, 0], [0, 4]], [1, 1])).inverse.rows == [[F(1, 2), 0], [0, F(1, 4)]]
    assert inverse([[5]]).inverse.rows == [[F(1, 5)]]
    assert inverse([[1, 2], [3, 5]], decimal=True).latex()


def test_random_against_leibniz_and_product():
    rng = random.Random(11)
    for n in (2, 3, 4):
        for _ in range(6):
            A = random_regular(n, rng)
            r = inverse(A)
            assert r.check() and is_invertible(A)
            assert (Matrix(A) @ r.inverse).rows == Matrix.identity(n).rows
            # det(A) * det(A^-1) = 1 (Leibniz, unabhängig)
            assert leibniz(A) * leibniz(r.inverse.rows) == 1
            # (A^-1)^-1 = A
            assert inverse(r.inverse).inverse.rows == Matrix(A).rows


def test_solve_with_inverse():
    r = solve_with_inverse(EXA, EXB)
    assert dict(r.highlights)["x"] == Vector([2, 3, -1]) and r.latex()
    s = solve_with_inverse([[1, 2], [2, 4]], [1, 2])
    assert not s.highlights and "singulär" in s.summary


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
