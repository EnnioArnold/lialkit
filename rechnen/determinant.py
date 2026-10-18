#!/usr/bin/env python3
"""
Determinanten: Sarrus, Laplace-Entwicklung und über die Dreiecksform.

Welches Verfahren sinnvoll ist, hängt von der Grösse ab. Sarrus gilt nur
für 3×3 — die Regel lässt sich nicht auf 4×4 übertragen, auch wenn das
Schema verlockend aussieht. Laplace ist für Handrechnung bis etwa 4×4
brauchbar, darüber wächst der Aufwand mit n! ins Unhaltbare. Die
Dreiecksform braucht nur n³/3 Schritte und ist das Verfahren der Wahl,
sobald die Matrix grösser wird.
"""

from __future__ import annotations

from fractions import Fraction

from .core import Matrix, Num, fmt
from .render import Result, Step, tex_matrix, tex_number, tex_row_op


class DetResult(Result):
    title = "Determinante"

    def __init__(self, value: Num, method: str, decimal: bool = False):
        super().__init__(decimal)
        self.value = value
        self.method = method
        self.summary = {
            "sarrus": "Regel von Sarrus (nur für 3×3).",
            "laplace": "Laplace-Entwicklung nach der ersten Zeile.",
            "triangular": "Über die Dreiecksform: Produkt der Pivots.",
            "direct": "Direkt nach der Formel.",
        }[method]
        self.highlights = [(r"\det(A)", value)]

    def __bool__(self) -> bool:
        return self.value != 0

    @property
    def singular(self) -> bool:
        return self.value == 0


def _check_square(A: Matrix) -> Matrix:
    A = A if isinstance(A, Matrix) else Matrix(A)
    if A.augmented:
        A = A.coefficients()
    if not A.is_square():
        raise ValueError("Die Determinante ist nur für quadratische Matrizen "
                         "erklärt")
    return A


def _sum_tex(values, decimal: bool) -> str:
    """Summe als LaTeX, negative Summanden mit Minus statt '+ −'."""
    out = ""
    for v in values:
        piece = tex_number(abs(v), decimal)
        if not out:
            out = ("-" if v < 0 else "") + piece
        else:
            out += (" - " if v < 0 else " + ") + piece
    return out or "0"


def det_sarrus(A, decimal: bool = False) -> DetResult:
    """Regel von Sarrus — ausschliesslich für 3×3."""
    A = _check_square(A)
    if A.m != 3:
        raise ValueError(f"Sarrus gilt nur für 3×3, diese Matrix ist "
                         f"{A.m}×{A.m}. Für andere Grössen det_laplace oder "
                         f"det_triangular benutzen.")
    a = A.rows
    plus = [a[0][0] * a[1][1] * a[2][2],
            a[0][1] * a[1][2] * a[2][0],
            a[0][2] * a[1][0] * a[2][1]]
    minus = [a[0][2] * a[1][1] * a[2][0],
             a[0][0] * a[1][2] * a[2][1],
             a[0][1] * a[1][0] * a[2][2]]
    value = sum(plus, Fraction(0)) - sum(minus, Fraction(0))

    result = DetResult(value, "sarrus", decimal)
    result.steps = [
        Step("Die ersten beiden Spalten rechts anhängen",
             items=[("A", A)],
             note="Dann laufen drei Diagonalen nach rechts unten (positiv) "
                  "und drei nach rechts oben (negativ)."),
        Step("Diagonalen nach rechts unten addieren",
             detail=_sum_tex(plus, decimal),
             note=f"Summe: {fmt(sum(plus, Fraction(0)), decimal)}"),
        Step("Diagonalen nach rechts oben abziehen",
             detail=_sum_tex(minus, decimal),
             note=f"Summe: {fmt(sum(minus, Fraction(0)), decimal)}"),
    ]
    return result


def minor(A: Matrix, row: int, col: int) -> Matrix:
    """Die Matrix ohne Zeile `row` und Spalte `col`."""
    return Matrix([[v for c, v in enumerate(r) if c != col]
                   for i, r in enumerate(A.rows) if i != row])


def cofactor(A: Matrix, row: int, col: int) -> Num:
    """Der Kofaktor: Minor mal Vorzeichen (−1)^(i+j)."""
    sign = -1 if (row + col) % 2 else 1
    return sign * _det_raw(minor(A, row, col))


def _det_raw(A: Matrix) -> Num:
    """Determinante ohne Protokoll — rekursiv, für Minoren."""
    size = A.m
    if size == 1:
        return A.rows[0][0]
    if size == 2:
        return A.rows[0][0] * A.rows[1][1] - A.rows[0][1] * A.rows[1][0]
    return sum((A.rows[0][c] * cofactor(A, 0, c) for c in range(size)),
               Fraction(0))


def det_laplace(A, along_row: int = 0, decimal: bool = False) -> DetResult:
    """Laplace-Entwicklung nach einer Zeile."""
    A = _check_square(A)
    size = A.m
    if size > 6:
        raise ValueError(f"Laplace braucht {size}! Summanden. Bei {size}×{size} "
                         f"ist das nicht mehr sinnvoll — nimm det_triangular.")
    if size <= 2:
        value = _det_raw(A)
        result = DetResult(value, "direct", decimal)
        formula = (r"a_{11} a_{22} - a_{12} a_{21}" if size == 2 else "a_{11}")
        result.steps = [Step("Direkt ablesen", detail=formula, items=[("A", A)])]
        return result

    value = Fraction(0)
    steps = [Step(f"Nach Zeile {along_row + 1} entwickeln", items=[("A", A)],
                  note="Jeder Eintrag wird mit seinem Kofaktor multipliziert; "
                       "die Vorzeichen wechseln wie auf einem Schachbrett.")]
    for c in range(size):
        entry = A.rows[along_row][c]
        if entry == 0:
            steps.append(Step(f"Spalte {c + 1} überspringen",
                              note="Der Eintrag ist 0, der Summand fällt weg."))
            continue
        sub = minor(A, along_row, c)
        sign = -1 if (along_row + c) % 2 else 1
        sub_det = _det_raw(sub)
        contribution = sign * entry * sub_det
        value += contribution
        steps.append(Step(
            f"Summand zu a({along_row + 1},{c + 1})",
            detail=(f"{'+' if sign > 0 else '-'}\\, "
                    f"{tex_number(entry, decimal)} \\cdot "
                    f"{tex_number(sub_det, decimal)} = "
                    f"{tex_number(contribution, decimal)}"),
            items=[("M", sub)],
            note=f"Vorzeichen (−1)^({along_row + 1}+{c + 1}) = {sign}"))

    result = DetResult(value, "laplace", decimal)
    result.steps = steps
    return result


def det_triangular(A, decimal: bool = False) -> DetResult:
    """
    Über die Dreiecksform. Jeder Zeilentausch dreht das Vorzeichen,
    das Addieren von Vielfachen ändert nichts.
    """
    A = _check_square(A)
    size = A.m
    U = [r[:] for r in A.rows]
    sign = 1
    swaps = 0
    steps = [Step("Ausgangsmatrix", items=[("A", A)],
                  note="Addieren von Vielfachen lässt die Determinante "
                       "unverändert, ein Zeilentausch dreht ihr Vorzeichen.")]

    for c in range(size):
        pivot = next((r for r in range(c, size) if U[r][c] != 0), None)
        if pivot is None:
            steps.append(Step(f"Spalte {c + 1} ist unter der Diagonale leer",
                              items=[("U", Matrix(U))],
                              note="Damit entsteht eine Nullzeile: "
                                   "die Determinante ist 0."))
            result = DetResult(Fraction(0), "triangular", decimal)
            result.steps = steps
            return result
        if pivot != c:
            U[c], U[pivot] = U[pivot], U[c]
            sign = -sign
            swaps += 1
            steps.append(Step(f"Tausch R{c + 1} ↔ R{pivot + 1}",
                              items=[("U", Matrix(U))],
                              note=f"Vorzeichen jetzt {sign:+d}."))
        for r in range(c + 1, size):
            if U[r][c] == 0:
                continue
            factor = U[r][c] / U[c][c]
            U[r] = [a - factor * b for a, b in zip(U[r], U[c])]
            steps.append(Step(
                f"Zeile {r + 1} eliminieren",
                detail=tex_row_op(r, c, -factor, decimal),
                items=[("U", Matrix(U))]))

    diag = [U[i][i] for i in range(size)]
    value = sign
    for d in diag:
        value *= d
    steps.append(Step(
        "Produkt der Diagonale",
        detail=(("-" if sign < 0 else "") +
                r" \cdot ".join(tex_number(d, decimal) for d in diag)),
        items=[("U", Matrix(U))],
        note=f"{swaps} Zeilentausch{'e' if swaps != 1 else ''} → "
             f"Vorzeichen {sign:+d}."))

    result = DetResult(value, "triangular", decimal)
    result.steps = steps
    return result


def det(A, method: str = "auto", decimal: bool = False) -> DetResult:
    """
    Determinante, Verfahren nach Grösse gewählt.

    'auto' nimmt die direkte Formel bis 2×2, Sarrus bei 3×3 und sonst die
    Dreiecksform.
    """
    A = _check_square(A)
    if method == "auto":
        method = ("direct" if A.m <= 2 else
                  "sarrus" if A.m == 3 else "triangular")
    if method in ("direct", "laplace"):
        return det_laplace(A, decimal=decimal)
    if method == "sarrus":
        return det_sarrus(A, decimal=decimal)
    if method == "triangular":
        return det_triangular(A, decimal=decimal)
    raise ValueError(f"Unbekanntes Verfahren '{method}'")
