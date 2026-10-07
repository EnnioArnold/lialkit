#!/usr/bin/env python3
"""
Zerlegungen: LR (LU), Eliminations- und Permutationsmatrizen.

Die Vorlesung schreibt die Zerlegung als A = L·R, bei nötigem Zeilentausch
als P·A = L·R. L ist dabei normiert — auf ihrer Diagonale stehen Einsen,
darunter die Faktoren, mit denen eliminiert wurde.
"""

from __future__ import annotations

from fractions import Fraction

from .core import Matrix, Num, Operation, Vector, fmt, parse_number
from .render import Result, Step, tex_matrix, tex_number, tex_row_op


# ==========================================================================
# Hilfsmatrizen
# ==========================================================================

def elimination_matrix(size: int, target: int, source: int, factor) -> Matrix:
    """
    Die Eliminationsmatrix E, die R_target ← R_target + k · R_source bewirkt.

    Sie ist die Einheitsmatrix mit k an der Stelle (target, source). Von
    links an A multipliziert führt sie genau diese eine Umformung aus.
    """
    if target == source:
        raise ValueError("Ziel- und Quellzeile müssen verschieden sein")
    E = Matrix.identity(size)
    rows = [r[:] for r in E.rows]
    rows[target][source] = parse_number(factor)
    return Matrix(rows)


def permutation_matrix(size: int, order: list[int]) -> Matrix:
    """
    Die Permutationsmatrix zu einer Zeilenreihenfolge.

    order[i] = j heisst: Zeile j von A wird zur Zeile i von P·A.
    """
    if sorted(order) != list(range(size)):
        raise ValueError("Die Reihenfolge muss jede Zeile genau einmal nennen")
    return Matrix([[1 if c == order[r] else 0 for c in range(size)]
                   for r in range(size)])


def swap_matrix(size: int, i: int, j: int) -> Matrix:
    """Die Permutationsmatrix, die genau zwei Zeilen tauscht."""
    order = list(range(size))
    order[i], order[j] = order[j], order[i]
    return permutation_matrix(size, order)


# ==========================================================================
# LR-Zerlegung
# ==========================================================================

class LUResult(Result):
    """Das Ergebnis einer LR-Zerlegung samt Rechenweg."""

    title = "LR-Zerlegung"

    def __init__(self, A: Matrix, L: Matrix, R: Matrix, P: Matrix,
                 order: list[int], swapped: bool, decimal: bool = False):
        super().__init__(decimal)
        self.A, self.L, self.R, self.P = A, L, R, P
        self.order, self.swapped = order, swapped
        self.summary = ("P · A = L · R — es musste getauscht werden."
                        if swapped else "A = L · R ohne Zeilentausch.")
        self.highlights = [("L", L), ("R", R)] + ([("P", P)] if swapped else [])

    # -- Nachrechnen ------------------------------------------------------

    def check(self) -> bool:
        """Stimmt L·R wirklich mit P·A überein?"""
        return (self.L @ self.R).rows == (self.P @ self.A).rows

    def solve(self, b) -> "LUSolveResult":
        return solve_lu(self, b, decimal=self.decimal)


def lu(A, pivot: bool = True, decimal: bool = False,
       max_pivot: bool = False) -> LUResult:
    """
    Zerlegt A in L und R.

    pivot=False verbietet den Zeilentausch; stösst die Rechnung dann auf
    eine Null an der Pivotstelle, bricht sie mit einer Erklärung ab, statt
    stillschweigend etwas Falsches zu liefern.

    max_pivot=True wählt stets die betragsgrösste Zeile. Beim exakten
    Rechnen bringt das nichts — es ist nur für den Vergleich mit dem
    numerischen Vorgehen da, das du in scipy siehst.
    """
    A = A if isinstance(A, Matrix) else Matrix(A)
    if A.augmented:
        A = A.coefficients()
    if not A.is_square():
        raise ValueError("Die LR-Zerlegung ist hier nur für quadratische "
                         "Matrizen erklärt")

    size = A.m
    U = [r[:] for r in A.rows]
    L = [[Fraction(1) if i == j else Fraction(0) for j in range(size)]
         for i in range(size)]
    order = list(range(size))
    swapped = False
    steps: list[Step] = []

    steps.append(Step("Ausgangsmatrix",
                      items=[("A", A.copy())],
                      note="L beginnt als Einheitsmatrix, R als Kopie von A."))

    for c in range(size):
        # -- Pivot suchen
        candidates = [r for r in range(c, size) if U[r][c] != 0]
        if not candidates:
            steps.append(Step(
                f"Spalte {c + 1} hat kein Pivot",
                items=[("R", Matrix(U))],
                note="Unterhalb der Diagonale steht in dieser Spalte nur 0. "
                     "Die Matrix ist singulär; die Zerlegung bricht hier ab."))
            break
        choice = (max(candidates, key=lambda r: abs(U[r][c]))
                  if max_pivot else candidates[0])

        if choice != c:
            if not pivot:
                raise ValueError(
                    f"An der Pivotstelle ({c + 1}, {c + 1}) steht eine 0. "
                    f"Ohne Zeilentausch geht es hier nicht weiter — "
                    f"ruf lu(A, pivot=True) auf.")
            U[c], U[choice] = U[choice], U[c]
            # In L nur die schon berechneten Faktoren mitvertauschen.
            for k in range(c):
                L[c][k], L[choice][k] = L[choice][k], L[c][k]
            order[c], order[choice] = order[choice], order[c]
            swapped = True
            steps.append(Step(
                f"Zeilentausch R{c + 1} ↔ R{choice + 1}",
                detail=f"P_{{{c + 1}{choice + 1}}}",
                items=[("R", Matrix(U))],
                note="An der Pivotstelle stand eine 0. Die bereits berechneten "
                     "Faktoren in L wandern mit."))

        # -- Eliminieren
        for r in range(c + 1, size):
            if U[r][c] == 0:
                continue
            factor = U[r][c] / U[c][c]
            L[r][c] = factor
            U[r] = [a - factor * b for a, b in zip(U[r], U[c])]
            steps.append(Step(
                f"Zeile {r + 1} eliminieren",
                detail=tex_row_op(r, c, -factor, decimal),
                items=[("R", Matrix(U))],
                note=f"Der Faktor {fmt(factor, decimal)} wird an die Stelle "
                     f"({r + 1}, {c + 1}) von L geschrieben."))

    Lm, Rm = Matrix(L), Matrix(U)
    P = permutation_matrix(size, order)
    steps.append(Step("Ergebnis", items=[("L", Lm), ("R", Rm)],
                      note="Auf der Diagonale von L stehen Einsen, darunter "
                           "die Eliminationsfaktoren."))

    result = LUResult(A, Lm, Rm, P, order, swapped, decimal)
    result.steps = steps
    if not result.check():
        result.warnings.append("Achtung: L · R stimmt nicht mit P · A überein.")
    return result


# ==========================================================================
# Lösen über die Zerlegung
# ==========================================================================

class LUSolveResult(Result):
    title = "Lösen über die LR-Zerlegung"

    def __init__(self, decomposition: LUResult, b: Vector, y: Vector,
                 x: Vector, decimal: bool = False):
        super().__init__(decimal)
        self.lu, self.b, self.y, self.x = decomposition, b, y, x
        self.summary = ("A·x = b wird in zwei Dreieckssysteme zerlegt: "
                        "erst L·y = P·b, dann R·x = y.")
        self.highlights = [("x", x)]


def forward_substitution(L: Matrix, b: Vector) -> Vector:
    """L·y = b für eine untere Dreiecksmatrix — von oben nach unten."""
    size = L.m
    y: list[Num] = []
    for r in range(size):
        known = sum((L.rows[r][k] * y[k] for k in range(r)), Fraction(0))
        if L.rows[r][r] == 0:
            raise ValueError(f"Auf der Diagonale von L steht in Zeile {r + 1} "
                             f"eine 0 — das System ist so nicht lösbar.")
        y.append((b[r] - known) / L.rows[r][r])
    return Vector(y)


def back_substitution(R: Matrix, y: Vector) -> Vector:
    """R·x = y für eine obere Dreiecksmatrix — von unten nach oben."""
    size = R.m
    x: list[Num | None] = [None] * size
    for r in range(size - 1, -1, -1):
        known = sum((R.rows[r][k] * x[k] for k in range(r + 1, size)), Fraction(0))
        if R.rows[r][r] == 0:
            raise ValueError(f"Auf der Diagonale von R steht in Zeile {r + 1} "
                             f"eine 0 — die Matrix ist singulär.")
        x[r] = (y[r] - known) / R.rows[r][r]
    return Vector(x)


def solve_lu(decomposition, b, decimal: bool = False) -> LUSolveResult:
    """Löst A·x = b mit einer fertigen Zerlegung."""
    if isinstance(decomposition, Matrix):
        decomposition = lu(decomposition, decimal=decimal)
    b = b if isinstance(b, Vector) else Vector(b)
    Pb = decomposition.P @ b

    steps: list[Step] = []
    if decomposition.swapped:
        steps.append(Step("Rechte Seite mitvertauschen",
                          detail=r"P \cdot b",
                          items=[("P \\cdot b", Pb)],
                          note="Weil P·A = L·R gilt, muss auch b permutiert "
                               "werden."))
    y = forward_substitution(decomposition.L, Pb)
    steps.append(Step("Vorwärtseinsetzen", detail=r"L \cdot y = P \cdot b",
                      items=[("y", y)],
                      note="Von oben nach unten, weil L unter der Diagonale "
                           "voll und darüber leer ist."))
    x = back_substitution(decomposition.R, y)
    steps.append(Step("Rückwärtseinsetzen", detail=r"R \cdot x = y",
                      items=[("x", x)],
                      note="Von unten nach oben."))

    result = LUSolveResult(decomposition, b, y, x, decimal)
    result.steps = steps
    probe = decomposition.A @ x
    if probe.values != b.values:
        result.warnings.append("Achtung: A · x ergibt nicht b.")
    return result
