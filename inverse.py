#!/usr/bin/env python3
"""
Die inverse Matrix über das Gauss-Jordan-Verfahren.

Man schreibt A und die Einheitsmatrix nebeneinander und formt die linke
Hälfte zur Einheitsmatrix um. Was rechts entsteht, ist A⁻¹ — denn jede
Zeilenoperation ist eine Multiplikation von links, und das Produkt aller
dieser Matrizen ist gerade A⁻¹.
"""

from __future__ import annotations

from fractions import Fraction

from .core import Matrix, Num, Vector, fmt, parse_number
from .determinant import det
from .render import Result, Step, tex_matrix, tex_number, tex_row_op


class InverseResult(Result):
    title = "Inverse Matrix"

    def __init__(self, A: Matrix, inverse: Matrix | None, reason: str = "",
                 decimal: bool = False):
        super().__init__(decimal)
        self.A = A
        self.inverse = inverse
        self.singular = inverse is None
        self.reason = reason
        self.summary = (reason if self.singular
                        else "Links die Einheitsmatrix, rechts die Inverse.")
        self.highlights = ([] if self.singular else [(r"A^{-1}", inverse)])

    def __bool__(self) -> bool:
        return not self.singular

    def check(self) -> bool:
        """Ergibt A · A⁻¹ wirklich die Einheitsmatrix?"""
        if self.singular:
            return True
        size = self.A.m
        return ((self.A @ self.inverse).rows == Matrix.identity(size).rows
                and (self.inverse @ self.A).rows == Matrix.identity(size).rows)


def _split(wide: Matrix, size: int) -> tuple[Matrix, Matrix]:
    """Zerlegt [A | B] in seine beiden Hälften."""
    return (Matrix([r[:size] for r in wide.rows]),
            Matrix([r[size:] for r in wide.rows]))


def inverse(A, decimal: bool = False) -> InverseResult:
    """
    Berechnet A⁻¹ mit Gauss-Jordan und protokolliert jeden Schritt.

    Ist A singulär, kommt kein Ergebnis, sondern die Begründung: an welcher
    Stelle die Elimination stecken bleibt und was das bedeutet.
    """
    A = A if isinstance(A, Matrix) else Matrix(A)
    if A.augmented:
        A = A.coefficients()
    if not A.is_square():
        raise ValueError("Nur quadratische Matrizen können eine Inverse haben")

    size = A.m
    work = [list(row) + [Fraction(1) if i == j else Fraction(0)
                         for j in range(size)]
            for i, row in enumerate(A.rows)]
    steps = [Step("Einheitsmatrix anhängen",
                  items=[("[A \\mid E]", Matrix([r[:] for r in work]))],
                  note="Jede Umformung wirkt auf beide Hälften zugleich.")]

    for c in range(size):
        pivot = next((r for r in range(c, size) if work[r][c] != 0), None)
        if pivot is None:
            rank = A.rank()
            reason = (f"In Spalte {c + 1} lässt sich kein Pivot finden. "
                      f"Der Rang ist {rank} statt {size}, die Spalten sind also "
                      f"linear abhängig — A ist singulär und nicht invertierbar.")
            steps.append(Step(f"Spalte {c + 1} hat kein Pivot",
                              items=[("[A \\mid E]", Matrix([r[:] for r in work]))],
                              note=reason))
            result = InverseResult(A, None, reason, decimal)
            result.steps = steps
            return result

        if pivot != c:
            work[c], work[pivot] = work[pivot], work[c]
            steps.append(Step(
                f"Tausch R{c + 1} ↔ R{pivot + 1}",
                items=[("[A \\mid E]", Matrix([r[:] for r in work]))],
                note="An der Pivotstelle stand eine 0."))

        head = work[c][c]
        if head != 1:
            work[c] = [v / head for v in work[c]]
            steps.append(Step(
                f"Pivot in Zeile {c + 1} auf 1 bringen",
                detail=(f"R_{{{c + 1}}} \\leftarrow "
                        f"{tex_number(Fraction(1) / head, decimal)} \\cdot "
                        f"R_{{{c + 1}}}"),
                items=[("[A \\mid E]", Matrix([r[:] for r in work]))]))

        for r in range(size):
            if r == c or work[r][c] == 0:
                continue
            factor = work[r][c]
            work[r] = [a - factor * b for a, b in zip(work[r], work[c])]
            steps.append(Step(
                f"Spalte {c + 1} in Zeile {r + 1} räumen",
                detail=tex_row_op(r, c, -factor, decimal),
                items=[("[A \\mid E]", Matrix([r2[:] for r2 in work]))]))

    left, right = _split(Matrix(work), size)
    steps.append(Step("Links steht die Einheitsmatrix",
                      items=[("A^{-1}", right)],
                      note="Also ist die rechte Hälfte die Inverse."))

    result = InverseResult(A, right, decimal=decimal)
    result.steps = steps
    if not result.check():
        result.warnings.append("Achtung: A · A⁻¹ ergibt nicht die Einheitsmatrix.")
    return result


def is_invertible(A) -> bool:
    """Kurz gefragt: hat A eine Inverse?"""
    A = A if isinstance(A, Matrix) else Matrix(A)
    if A.augmented:
        A = A.coefficients()
    return A.is_square() and det(A).value != 0


def inverse_2x2(A, decimal: bool = False) -> InverseResult:
    """
    Die geschlossene Formel für 2×2: 1/det · [[d, −b], [−c, a]].

    Für die Prüfung schneller als Gauss-Jordan, aber nur hier gültig.
    """
    A = A if isinstance(A, Matrix) else Matrix(A)
    if A.m != 2 or A.n != 2:
        raise ValueError("Diese Formel gilt nur für 2×2-Matrizen")
    (a, b), (c, d) = A.rows[0], A.rows[1]
    value = a * d - b * c
    if value == 0:
        reason = ("Die Determinante ist 0 — die beiden Zeilen sind Vielfache "
                  "voneinander. A ist singulär.")
        result = InverseResult(A, None, reason, decimal)
        result.steps = [Step("Determinante", detail=r"ad - bc = 0",
                             items=[("A", A)])]
        return result
    inv = Matrix([[d / value, -b / value], [-c / value, a / value]])
    result = InverseResult(A, inv, decimal=decimal)
    result.steps = [
        Step("Determinante", detail=(f"ad - bc = {tex_number(value, decimal)}"),
             items=[("A", A)]),
        Step("Hauptdiagonale tauschen, Nebendiagonale negieren",
             items=[("\\operatorname{adj}(A)", Matrix([[d, -b], [-c, a]]))]),
        Step("Durch die Determinante teilen",
             detail=(r"\frac{1}{" + tex_number(value, decimal) + "}"),
             items=[("A^{-1}", inv)]),
    ]
    return result


def solve_with_inverse(A, b, decimal: bool = False) -> Result:
    """
    Löst A·x = b über x = A⁻¹·b.

    Für ein einzelnes System ist das der teurere Weg — die Elimination
    kommt schneller zum Ziel. Lohnend wird es erst, wenn viele rechte
    Seiten zur selben Matrix gehören.
    """
    A = A if isinstance(A, Matrix) else Matrix(A)
    b = b if isinstance(b, Vector) else Vector(b)
    inv = inverse(A, decimal)
    out = Result(decimal)
    out.title = "Lösen über die Inverse"
    if inv.singular:
        out.summary = inv.reason
        out.steps = inv.steps
        return out
    x = inv.inverse @ b
    out.summary = "x = A⁻¹ · b"
    out.steps = inv.steps + [
        Step("Inverse mit b multiplizieren", items=[("x", x)])]
    out.highlights = [("x", x)]
    return out
