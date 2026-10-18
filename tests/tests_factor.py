#!/usr/bin/env python3
"""Tests für lialkit.rechnen.factor (LR-Zerlegung, Dreieckssysteme, Hilfsmatrizen). Start: python3 tests/tests_factor.py"""
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

from lialkit import (Matrix, Vector, back_substitution, elimination_matrix,  # noqa: E402
                     forward_substitution, lu, permutation_matrix, solve_lu,
                     swap_matrix)

def test_lu_2x2_by_hand():
    # SW3.1a: l21 = -2/3, R = [[3,-6],[0,1]]
    r = lu([[3, -6], [-2, 5]])
    assert r.L.rows == [[1, 0], [F(-2, 3), 1]] and r.R.rows == [[3, -6], [0, 1]]
    assert not r.swapped and r.P == Matrix.identity(2) and r.check()
    assert r.solve([0, 1]).x == Vector([2, 1])


def test_lu_3x3_properties():
    A = [[2, 6, 2], [-3, -8, 0], [4, 9, 2]]
    r = lu(A)
    assert r.check() and r.L.is_triangular(lower=True) and r.R.is_triangular()
    assert all(r.L[i, i] == 1 for i in range(3))
    assert (r.L @ r.R).rows == (r.P @ Matrix(A)).rows
    assert not r.warnings and r.latex() and r._repr_html_() and repr(r)


def test_lu_pivoting():
    r = lu([[0, 2], [3, 1]])
    assert r.swapped and r.P.rows == [[0, 1], [1, 0]] and r.L == Matrix.identity(2)
    assert r.R.rows == [[3, 1], [0, 2]] and r.check()
    assert raises(ValueError, lu, [[0, 2], [3, 1]], pivot=False)
    assert lu([[2, 1], [4, 3]], pivot=False).R.rows == [[2, 1], [0, 1]]
    m = lu([[1, 3], [4, 1]], max_pivot=True)
    assert m.swapped and m.check()


def test_lu_errors_and_singular():
    assert raises(ValueError, lu, [[1, 2, 3], [4, 5, 6]])
    s = lu([[1, 2], [2, 4]])
    assert s.R.rows == [[1, 2], [0, 0]] and s.check()
    t = lu([[1, 2, 3], [2, 4, 7], [3, 6, 10]])
    assert t.check() and t.R.is_triangular()
    assert lu(Matrix.system([[1, 0], [0, 1]], [3, 4])).A == Matrix.identity(2)   # rechte Seite wird ignoriert


def test_lu_random_against_matmul():
    rng = random.Random(3)
    for n in (2, 3, 4, 5):
        for _ in range(6):
            A = random_regular(n, rng)
            r = lu(A)
            assert r.check() and r.R.is_triangular() and r.L.is_triangular(lower=True)
            b = [rng.randint(-9, 9) for _ in range(n)]
            assert r.solve(b).x == Vector(ref_solve(A, b))
            assert solve_lu(Matrix(A), b).x == Vector(ref_solve(A, b))


def test_example_from_init_via_lu():
    s = solve_lu(lu(EXA), EXB)
    assert s.x == Vector([2, 3, -1]) and not s.warnings and s.latex() and s.steps


def test_substitution():
    L = Matrix([[1, 0, 0], [2, 1, 0], [-1, 3, 1]])
    y = forward_substitution(L, Vector([1, 4, 2]))      # 1 ; 4-2 = 2 ; 2+1-6 = -3
    assert y == Vector([1, 2, -3])
    R = Matrix([[2, 1, 1], [0, 3, 2], [0, 0, 4]])
    x = back_substitution(R, Vector([6, 8, 4]))         # 1 ; (8-2)/3 = 2 ; (6-2-1)/2 = 3/2
    assert x == Vector([F(3, 2), 2, 1])
    assert raises(ValueError, back_substitution, Matrix([[1, 1], [0, 0]]), Vector([1, 1]))
    assert raises(ValueError, forward_substitution, Matrix([[0, 0], [1, 1]]), Vector([1, 1]))


def test_helper_matrices():
    E = elimination_matrix(3, 2, 0, -2)       # R3 <- R3 - 2 R1
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (E @ A).rows == A.add_multiple(2, -2, 0).rows
    assert raises(ValueError, elimination_matrix, 3, 1, 1, 5)
    P = permutation_matrix(3, [1, 0, 2])
    assert P.rows == [[0, 1, 0], [1, 0, 0], [0, 0, 1]] and (P @ A).rows == A.swap(0, 1).rows
    assert permutation_matrix(3, [2, 0, 1]).rows == [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert raises(ValueError, permutation_matrix, 3, [0, 0, 1])
    assert swap_matrix(3, 0, 2).rows == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


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
