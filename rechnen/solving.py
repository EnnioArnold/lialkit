#!/usr/bin/env python3
"""
Gauss- und Gauss-Jordan-Elimination mit Rechenweg.

Der eigentliche Algorithmus steckt in core.Matrix — hier kommt dazu, was
den Weg erklärt: welcher Schritt warum, das Rangkriterium und die
Lösungsmenge in lesbarer Form.
"""

from __future__ import annotations

from fractions import Fraction

from .core import Matrix, Num, Operation, Vector, fmt
from .render import Result, Step, tex_matrix, tex_number


class GaussResult(Result):
    """Das Ergebnis einer Elimination samt Deutung der Lösungsmenge."""

    def __init__(self, start: Matrix, final: Matrix, jordan: bool,
                 decimal: bool = False):
        super().__init__(decimal)
        self.start, self.final, self.jordan = start, final, jordan
        self.title = ("Gauss-Jordan-Elimination" if jordan
                      else "Gauss-Elimination")
        self.info = final.analyze() if final.augmented else None
        self.kind = self.info["type"] if self.info else None
        self.solution = (self.info.get("vector")
                         if self.info and self.info["type"] == "unique" else None)

    @property
    def rank(self) -> int:
        return len(self.final.pivots())

    def __repr__(self) -> str:
        return super().__repr__()


def _criterion(A: Matrix, decimal: bool) -> Step:
    """Das Rangkriterium aus SW 2 in Worten."""
    rank_full = len(A.ref().pivots())
    coeff = A.coefficients() if A.augmented else A
    rank_coeff = coeff.rank()
    unknowns = A.n
    if rank_coeff < rank_full:
        verdict = ("Der Rang der erweiterten Matrix ist grösser als der der "
                   "Koeffizientenmatrix — das System ist unlösbar.")
    elif rank_coeff == unknowns:
        verdict = ("Rang gleich Anzahl Unbekannter — es gibt genau eine "
                   "Lösung.")
    else:
        free = unknowns - rank_coeff
        verdict = (f"Der Rang ist kleiner als die Zahl der Unbekannten: "
                   f"{free} Variable{'n' if free != 1 else ''} bleib"
                   f"{'en' if free != 1 else 't'} frei, es gibt unendlich "
                   f"viele Lösungen.")
    return Step(
        "Rangkriterium",
        detail=(f"\\operatorname{{rg}}(A) = {rank_coeff}, \\quad "
                f"\\operatorname{{rg}}(A \\mid b) = {rank_full}, \\quad "
                f"n = {unknowns}"),
        note=verdict)


def _solution_step(result: GaussResult, decimal: bool) -> Step | None:
    info = result.info
    if info is None:
        return None
    if info["type"] == "none":
        return Step("Lösungsmenge",
                    detail=(r"0 = " + tex_number(info["value"], decimal)),
                    note="Eine Zeile behauptet 0 = c mit c ungleich 0. "
                         "Das ist ein Widerspruch, das System ist unlösbar.")
    if info["type"] == "unique":
        return Step("Lösungsmenge", items=[("x", info["vector"])],
                    note="Jede Variable ist festgelegt.")
    names = ", ".join(f"t_{i + 1}" for i in range(len(info["free"])))
    lines = result.final.solution_lines(decimal)
    return Step("Lösungsmenge in Parameterform",
                detail=r" \\ ".join(
                    line.replace("−", "-").replace("·", r" \cdot ")
                    for line in lines),
                note=f"Frei wählbar: {names} ∈ ℝ")


def gauss(A, b=None, weg: bool = False, jordan: bool = False,
          decimal: bool = False) -> GaussResult:
    """
    Gauss-Elimination bis zur Zeilenstufenform.

    `weg=True` legt den vollständigen Rechenweg bei — jeden Schritt mit
    Begründung, so wie er auf dem Papier stünde. Ohne das kommt nur das
    Ergebnis, was zum Kontrollieren reicht.

    Mit `jordan=True` wird weiter bis zur reduzierten Zeilenstufenform
    gerechnet; dann steht die Lösung direkt in der letzten Spalte.
    """
    A = A if isinstance(A, Matrix) else Matrix(A, augmented=b is None)
    if b is not None:
        A = Matrix.system(A.coefficients() if A.augmented else A, b)

    steps: list[Step] = [Step("Ausgangsmatrix", items=[("(A \\mid b)", A)]
                              if A.augmented else [("A", A)])]
    current = A
    for _ in range(600):
        op = current.next_step(jordan)
        if op is None:
            break
        current = current.apply(op)
        steps.append(Step(str(op), detail=op.latex(),
                          items=[("", current)], note=op.reason))

    result = GaussResult(A, current, jordan, decimal)
    form = ("reduzierte Zeilenstufenform" if jordan else "Zeilenstufenform")
    result.summary = f"{len(steps) - 1} Schritte bis zur {form}."

    if A.augmented:
        steps.append(_criterion(A, decimal))
        tail = _solution_step(result, decimal)
        if tail:
            steps.append(tail)
        if result.kind == "unique":
            result.highlights = [("x", result.solution)]
        elif result.kind == "none":
            result.warnings.append("Das System hat keine Lösung.")
    else:
        result.highlights = [("", current)]

    result.steps = steps if weg else []
    if not weg and A.augmented and result.kind != "unique":
        result.highlights = [("", current)]
    return result


def gauss_jordan(A, b=None, weg: bool = False, decimal: bool = False):
    """Bis zur reduzierten Zeilenstufenform."""
    return gauss(A, b, weg=weg, jordan=True, decimal=decimal)


def solve(A, b, weg: bool = False, method: str = "gauss", decimal: bool = False):
    """
    Löst A·x = b.

    method='gauss' eliminiert, 'lu' zerlegt erst in L und R, 'inverse'
    multipliziert mit A⁻¹. Alle drei kommen zum selben x — sie unterscheiden
    sich im Aufwand und darin, was man dabei lernt.
    """
    if method == "gauss":
        return gauss_jordan(A, b, weg=weg, decimal=decimal)
    if method == "lu":
        from .factor import lu, solve_lu
        return solve_lu(lu(A, decimal=decimal), b, decimal=decimal)
    if method == "inverse":
        from .inverse import solve_with_inverse
        return solve_with_inverse(A, b, decimal=decimal)
    raise ValueError(f"Unbekanntes Verfahren '{method}' — "
                     f"erlaubt sind 'gauss', 'lu' und 'inverse'")


def rank(A, weg: bool = False, decimal: bool = False) -> Result:
    """Der Rang mit dem Weg zur Stufenform."""
    A = A if isinstance(A, Matrix) else Matrix(A)
    steps = [Step("Ausgangsmatrix", items=[("A", A)])]
    current = A
    for _ in range(600):
        op = current._next_forward()
        if op is None:
            break
        current = current.apply(op)
        steps.append(Step(str(op), detail=op.latex(), items=[("", current)]))
    pivots = current.pivots()
    steps.append(Step("Pivots zählen", items=[("", current)],
                      note=f"{len(pivots)} Zeile{'n' if len(pivots) != 1 else ''} "
                           f"ungleich null → Rang {len(pivots)}."))
    out = Result(decimal)
    out.title = "Rang"
    out.summary = (f"rg(A) = {len(pivots)} bei {A.m} Zeilen und {A.n} Spalten.")
    out.steps = steps if weg else []
    out.highlights = [(r"\operatorname{rg}(A)", len(pivots))]
    return out
