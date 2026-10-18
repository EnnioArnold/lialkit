#!/usr/bin/env python3
"""Tests für lialkit.rechnen.solving (Gauss, Gauss-Jordan, solve, rank). Start: python3 tests/tests_gauss.py"""
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

from lialkit import Matrix, Vector, gauss, gauss_jordan, rank, solve  # noqa: E402

def test_example_from_init():
    r = gauss(EXA, EXB)
    assert r.kind == "unique" and r.solution == Vector([2, 3, -1])
    assert r.rank == 3 and r.final.form_state() == "ref" and not r.jordan
    j = gauss(EXA, EXB, jordan=True)
    assert j.final.form_state() == "rref" and j.final.rhs() == Vector([2, 3, -1])
    assert gauss_jordan(EXA, EXB).solution == Vector([2, 3, -1])


def test_weg_and_render():
    kurz, lang = gauss(EXA, EXB), gauss(EXA, EXB, weg=True)
    assert kurz.steps == [] and len(lang.steps) > 3
    for r in (kurz, lang, gauss(EXA, EXB, weg=True, decimal=True)):
        assert repr(r) and r.latex() and r._repr_html_()
    assert lang.as_decimal().latex() and lang.as_exact().latex()
    assert "Rangkriterium" in [s.title for s in lang.steps]


def test_augmented_matrix_input():
    M = Matrix.system(EXA, EXB)
    assert gauss(M).solution == Vector([2, 3, -1])
    assert gauss(M, EXB).solution == Vector([2, 3, -1])     # b ersetzt die rechte Seite


def test_no_solution_and_infinite():
    none = gauss([[1, 1, 1], [1, 1, 2]], weg=True)
    assert none.kind == "none" and none.solution is None and none.warnings
    inf = gauss([[1, 1, 2], [2, 2, 4]], weg=True)
    assert inf.kind == "infinite" and inf.rank == 1
    assert inf.final.solution_lines() == ["x1 = 2 − t1", "x2 = t1"]
    for r in (none, inf):
        assert r.latex() and r._repr_html_()


def test_without_rhs():
    r = gauss(Matrix([[1, 2], [3, 4]]), weg=True)
    assert r.kind is None and r.rank == 2 and r.final.is_triangular()
    # rohe Listen ohne b gelten als erweiterte Matrix (A | b)
    assert gauss([[1, 2], [3, 4]]).kind == "none"


def test_zero_pivot_needs_swap():
    r = gauss([[0, 2, 4], [3, 1, 7]], weg=True)    # 2y = 4, 3x + y = 7  ->  (5/3, 2)
    assert r.solution == Vector([F(5, 3), 2])
    assert any("↔" in s.title for s in r.steps)


def test_random_systems_against_reference():
    rng = random.Random(7)
    for n in (2, 3, 4, 5):
        for _ in range(8):
            A = random_regular(n, rng)
            b = [rng.randint(-9, 9) for _ in range(n)]
            want = Vector(ref_solve(A, b))
            assert gauss(A, b).solution == want
            assert gauss_jordan(A, b).solution == want
            assert Matrix(A) @ want == Vector(b)


def test_solve_methods_agree():
    for method in ("gauss", "lu", "inverse"):
        r = solve(EXA, EXB, method=method, weg=True)
        assert dict(r.highlights)["x"] == Vector([2, 3, -1]), method
        assert r.latex()
    assert raises(ValueError, solve, EXA, EXB, method="kramer")


def test_rank():
    r = rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]], weg=True)
    assert r.highlights[0][1] == 2 and r.steps and "Rang 2" in r.steps[-1].note
    assert rank(Matrix.identity(3)).highlights[0][1] == 3
    assert rank([[0, 0], [0, 0]]).highlights[0][1] == 0
    assert rank([[1, 2, 3], [2, 4, 6]]).highlights[0][1] == 1


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
