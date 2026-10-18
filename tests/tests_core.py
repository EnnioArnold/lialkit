#!/usr/bin/env python3
"""Tests für lialkit.rechnen.core (Zahlen, Matrix, Vector). Start: python3 tests/tests_core.py"""
import math
import os
import sys
from fractions import Fraction as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # Elternordner des Pakets lialkit

from lialkit.rechnen.core import (MINUS, Matrix, Operation, Vector, fmt,  # noqa: E402
                                  fmt_factor, is_exact, parse_number,
                                  tex_number, triple_product)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def test_parse_number():
    assert parse_number("3/4") == F(3, 4)
    assert parse_number("1,5") == F(3, 2)
    assert parse_number(f"{MINUS}2") == -2
    assert parse_number(" ( -7 ) ") == -7
    assert parse_number("") == 0
    assert parse_number(0.5) == F(1, 2)
    assert parse_number(F(1, 3)) == F(1, 3) and parse_number(4) == 4
    assert raises(ValueError, parse_number, "abc")
    assert raises(ValueError, parse_number, "1/0")


def test_fmt():
    assert fmt(F(-1, 2)) == f"{MINUS}1/2"
    assert fmt(F(1, 3), decimal=True) == "0.3333"
    assert fmt(F(2), decimal=True) == "2"
    assert fmt(F(-5, 4), decimal=True) == f"{MINUS}1.25"
    assert fmt_factor(F(-2)) == f"({MINUS}2)" and fmt_factor(F(3)) == "3"
    assert tex_number(F(-3, 4)) == r"-\frac{3}{4}" and tex_number(F(5)) == "5"
    assert is_exact(F(1)) and not is_exact(1.0)


def test_matrix_basics():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    assert A.shape == (2, 3) and A.m == 2 and A.n == 3 and A.width == 3
    aug = Matrix.system([[1, 2], [3, 4]], [5, 6])
    assert aug.augmented and aug.n == 2 and aug.width == 3
    assert aug.rhs() == Vector([5, 6]) and aug.coefficients() == Matrix([[1, 2], [3, 4]])
    assert A[1, 2] == 6 and A[0] == [1, 2, 3]
    assert Matrix.identity(2).rows == [[1, 0], [0, 1]]
    assert Matrix.zeros(2, 3).shape == (2, 3)
    assert Matrix.from_text("1 2\n3 4").rows == [[1, 2], [3, 4]]
    assert A.transpose().rows == [[1, 4], [2, 5], [3, 6]] and A.T == A.transpose()
    assert A.copy() == A and A.copy() is not A
    assert raises(ValueError, Matrix, [])
    assert raises(ValueError, Matrix, [[1, 2], [3]])
    assert raises(ValueError, Matrix.system, [[1, 2]], [1, 2])
    assert raises(ValueError, A.rhs)
    assert Matrix([[1, 0], [0, 1]]).augment(Matrix([[5], [6]])).rows == [[1, 0, 5], [0, 1, 6]]


def test_matrix_arithmetic():
    A, B = Matrix([[1, 2], [3, 4]]), Matrix([[5, 6], [7, 8]])
    assert (A @ B).rows == [[19, 22], [43, 50]]          # Handrechnung
    assert (A * B) == (A @ B)
    assert (A + B).rows == [[6, 8], [10, 12]] and (B - A).rows == [[4, 4], [4, 4]]
    assert A.scaled(F(1, 2)).rows == [[F(1, 2), 1], [F(3, 2), 2]] and (2 * A).rows == [[2, 4], [6, 8]]
    assert (A @ Vector([1, 1])) == Vector([3, 7])
    assert A.power(0) == Matrix.identity(2) and A.power(2).rows == [[7, 10], [15, 22]]
    assert A.trace() == 5
    assert Matrix([[1, 2], [2, 1]]).is_symmetric() and not A.is_symmetric()
    assert Matrix([[1, 2], [0, 3]]).is_triangular() and Matrix([[1, 0], [2, 3]]).is_triangular(lower=True)
    assert not A.is_triangular()
    assert raises(ValueError, A.matmul, Matrix([[1, 2, 3]]))
    assert raises(ValueError, A.matmul, Vector([1, 2, 3]))
    assert raises(ValueError, A.__add__, Matrix([[1, 2, 3]]))
    assert raises(ValueError, Matrix([[1, 2, 3]]).trace)
    assert raises(ValueError, A.power, -1)


def test_row_operations():
    A = Matrix([[1, 2], [3, 4]])
    assert A.swap(0, 1).rows == [[3, 4], [1, 2]]
    assert A.scale(0, F(1, 2)).rows == [[F(1, 2), 1], [3, 4]]
    assert A.add_multiple(1, -3, 0).rows == [[1, 2], [0, -2]]
    assert A.rows == [[1, 2], [3, 4]]                      # Original unverändert
    assert raises(ValueError, A.scale, 0, 0)
    assert raises(ValueError, A.add_multiple, 0, 1, 0)
    assert raises(ValueError, A.apply, Operation("edit", 0))
    assert str(Operation("swap", 0, j=1)) == "R1 ↔ R2"
    assert str(Operation("add", 1, j=0, k=F(-3))) == f"R2 ← R2 {MINUS} 3 · R1"
    assert "leftrightarrow" in Operation("swap", 0, j=1).latex()


def test_elimination_example():
    # Beispiel aus __init__.py
    M = Matrix.system([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
    assert M.rref().rhs() == Vector([2, 3, -1])
    assert M.ref().is_triangular() and M.ref().form_state() == "ref"
    assert M.rref().form_state() == "rref" and M.form_state() == "standard"
    info = M.analyze()
    assert info["type"] == "unique" and info["vector"] == Vector([2, 3, -1])
    assert M.solution_lines() == ["x1 = 2", "x2 = 3", "x3 = −1"]
    ops = M.steps_to(jordan=True)
    assert ops and ops[-1][1] == M.rref()
    assert M.next_step(jordan=False) is not None and M.rref().next_step(jordan=True) is None


def test_rank_and_solution_sets():
    assert Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).rank() == 2
    assert Matrix.identity(4).rank() == 4 and Matrix.zeros(2, 2).rank() == 0
    inf = Matrix([[1, 1, 2], [2, 2, 4]], augmented=True)
    a = inf.analyze()
    assert a["type"] == "infinite" and a["free"] == [1] and a["terms"][0] == (2, {1: -1})
    assert inf.solution_lines() == [f"x1 = 2 {MINUS} t1", "x2 = t1"]
    bad = Matrix([[1, 1, 1], [1, 1, 2]], augmented=True)
    assert bad.analyze()["type"] == "none" and bad.rref().inconsistent() and bad.solution_lines() == []
    assert raises(ValueError, Matrix([[1, 2]]).analyze)
    assert Matrix([[1, 2, 3], [0, 0, 0]], augmented=True).equation_lines() == ["x1 + 2·x2 = 3"]
    assert Matrix.zeros(1, 2).equation_lines() == ["0 = 0"]


def test_vector_basics():
    u, v = Vector([1, 2, 3]), Vector([4, 5, 6])
    assert len(u) == 3 and u.dim == 3 and u[1] == 2 and list(u) == [1, 2, 3]
    assert u + v == Vector([5, 7, 9]) and v - u == Vector([3, 3, 3]) and -u == Vector([-1, -2, -3])
    assert u.scaled(2) == Vector([2, 4, 6]) and 3 * u == Vector([3, 6, 9]) and u * 2 == Vector([2, 4, 6])
    assert u.dot(v) == 32 and u * v == 32
    assert u.as_matrix().shape == (3, 1) and u.as_row().shape == (1, 3)
    assert raises(ValueError, u.__add__, Vector([1, 2]))
    assert raises(ValueError, Vector, [])


def test_norm_and_angle():
    n = Vector([3, 4]).norm()
    assert n == 5 and isinstance(n, F)                       # |(3,4)| = 5 exakt
    assert Vector([2, -3, 6]).norm() == 7 and Vector([-1, 4, 8]).norm() == 9   # SW1.5
    assert Vector([F(3, 5), F(4, 5)]).norm() == 1
    assert abs(Vector([1, 1]).norm() - math.sqrt(2)) < 1e-12 and not isinstance(Vector([1, 1]).norm(), F)
    assert Vector([1, -2, 3]).norm(1) == 6 and Vector([1, -7, 3]).norm("inf") == 7
    assert Vector([3, 4]).norm_squared() == 25
    assert Vector([3, 4]).normalized() == Vector([F(3, 5), F(4, 5)])
    assert raises(ValueError, Vector([0, 0]).normalized)
    assert abs(Vector([1, 0]).angle(Vector([0, 1])) - 90) < 1e-9
    assert abs(Vector([1, 0]).angle(Vector([1, 1])) - 45) < 1e-9
    assert abs(Vector([1, 0]).angle(Vector([-1, 0])) - 180) < 1e-9
    assert abs(Vector([1, 0]).angle(Vector([0, 1]), degrees=False) - math.pi / 2) < 1e-12
    assert raises(ValueError, Vector([0, 0]).angle, Vector([1, 0]))


def test_cross_and_predicates():
    e1, e2, e3 = Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])
    assert e1.cross(e2) == e3 and e2.cross(e1) == -e3
    assert Vector([1, 2, 3]).cross(Vector([4, 5, 6])) == Vector([-3, 6, -3])
    assert raises(ValueError, Vector([1, 2]).cross, Vector([3, 4]))
    assert Vector([1, 2, 3]).parallel_to(Vector([2, 4, 6])) and not e1.parallel_to(e2)
    assert Vector([1, 2]).parallel_to(Vector([-2, -4])) and not Vector([1, 2]).parallel_to(Vector([2, 1]))
    assert e1.orthogonal_to(e2) and not e1.orthogonal_to(e1)
    assert Vector([0, 0]).is_zero()
    assert triple_product(Vector([1, 2, 3]), Vector([4, 5, 6]), Vector([7, 8, 10])) == -3
    assert triple_product(e1, e2, e3) == 1


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
