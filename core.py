#!/usr/bin/env python3
"""
Rechenkern von lialkit: Zahlen, Matrizen, Vektoren.

Dieses Modul kennt weder Qt noch Jupyter. Es rechnet exakt mit Brüchen,
damit Zwischenergebnisse nicht durch Rundung verfälscht werden — in der
linearen Algebra ist der Unterschied zwischen 0 und 1e-17 der Unterschied
zwischen singulär und regulär.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

Num = Fraction
MINUS = "−"          # typografisches Minus, nur für die Anzeige


# ==========================================================================
# Zahlen
# ==========================================================================

def parse_number(text) -> Num:
    """Akzeptiert 3, '3', '-2', '0.5', '1,5', '3/4', Fraction, float."""
    if isinstance(text, Fraction):
        return text
    if isinstance(text, int):
        return Fraction(text)
    if isinstance(text, float):
        return Fraction(text).limit_denominator(10 ** 9)
    s = str(text).strip().replace(MINUS, "-").replace(",", ".").replace(" ", "")
    s = s.strip("()")
    if s == "":
        return Fraction(0)
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            num, den = Fraction(a), Fraction(b)
            if den == 0:
                raise ZeroDivisionError
            return num / den
        return Fraction(s)
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        raise ValueError(f"'{text}' ist keine gültige Zahl") from exc


def fmt(x, decimal: bool = False, digits: int = 4) -> str:
    """
    Zahl als Text. Brüche bleiben Brüche, ausser decimal=True.

    Das Übungsblatt verlangt vier Stellen; darum ist das die Vorgabe,
    sobald dezimal gerechnet wird.
    """
    if isinstance(x, float):
        return f"{x:.{digits}f}".rstrip("0").rstrip(".").replace("-", MINUS)
    if decimal:
        v = float(x)
        out = (f"{v:g}" if v == int(v)
               else f"{v:.{digits}f}".rstrip("0").rstrip("."))
        return out.replace("-", MINUS)
    return str(x).replace("-", MINUS)


def fmt_factor(k: Num) -> str:
    """Faktor in einer Operationsbeschreibung, negative Werte geklammert."""
    s = str(k).replace("-", MINUS)
    return s if k >= 0 else f"({s})"


def is_exact(x) -> bool:
    return isinstance(x, Fraction)


# ==========================================================================
# Zeilenoperationen
# ==========================================================================

@dataclass(frozen=True)
class Operation:
    """
    Eine elementare Zeilenumformung, Zeilenindizes 0-basiert.

    'edit' ist keine Zeilenoperation, sondern das Überschreiben einer Zelle
    von Hand. Es trägt seinen Text in `text` und wird nie auf eine Matrix
    angewendet, sondern nur im Verlauf vermerkt.
    """
    kind: str                     # 'swap' | 'scale' | 'add' | 'edit'
    i: int
    j: int | None = None
    k: Num | None = None
    reason: str = ""
    text: str = ""

    def __str__(self) -> str:
        if self.kind == "edit":
            return self.text or f"R{self.i + 1} von Hand geändert"
        ri, rj = f"R{self.i + 1}", f"R{(self.j or 0) + 1}"
        if self.kind == "swap":
            return f"{ri} ↔ {rj}"
        if self.kind == "scale":
            return f"{ri} ← {fmt_factor(self.k)} · {ri}"
        sign = MINUS if self.k < 0 else "+"
        mag = abs(self.k)
        factor = "" if mag == 1 else f"{mag} · "
        return f"{ri} ← {ri} {sign} {factor}{rj}"

    def latex(self) -> str:
        if self.kind == "edit":
            return r"\text{" + (self.text or "von Hand geändert") + "}"
        ri, rj = f"R_{{{self.i + 1}}}", f"R_{{{(self.j or 0) + 1}}}"
        if self.kind == "swap":
            return f"{ri} \\leftrightarrow {rj}"
        if self.kind == "scale":
            return f"{ri} \\leftarrow {tex_number(self.k)} \\cdot {ri}"
        sign = "-" if self.k < 0 else "+"
        mag = abs(self.k)
        factor = "" if mag == 1 else tex_number(mag) + r" \cdot "
        return f"{ri} \\leftarrow {ri} {sign} {factor}{rj}"


def tex_number(x, decimal: bool = False, digits: int = 4) -> str:
    """Zahl als LaTeX; echte Brüche als \\frac."""
    if isinstance(x, float):
        return f"{x:.{digits}f}".rstrip("0").rstrip(".")
    if decimal or x.denominator == 1:
        return fmt(x, decimal, digits).replace(MINUS, "-")
    sign = "-" if x < 0 else ""
    return f"{sign}\\frac{{{abs(x.numerator)}}}{{{x.denominator}}}"


# ==========================================================================
# Matrix
# ==========================================================================

class Matrix:
    """
    Eine Matrix mit exakten Einträgen.

    Sie kann als reine Koeffizientenmatrix auftreten oder als erweiterte
    Matrix (A | b). `augmented` sagt, ob die letzte Spalte zur rechten Seite
    gehört; davon hängt ab, was `n` zählt und wo die Trennlinie liegt.
    """

    def __init__(self, rows: Sequence[Sequence], augmented: bool = False):
        if not rows:
            raise ValueError("Die Matrix braucht mindestens eine Zeile")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("Alle Zeilen müssen gleich lang sein")
        self.rows: list[list[Num]] = [[parse_number(v) for v in r] for r in rows]
        self.augmented = augmented

    # -- Grundlegendes ----------------------------------------------------

    @property
    def m(self) -> int:
        """Anzahl Zeilen."""
        return len(self.rows)

    @property
    def n(self) -> int:
        """Anzahl Spalten der Koeffizientenmatrix (ohne rechte Seite)."""
        return len(self.rows[0]) - (1 if self.augmented else 0)

    @property
    def width(self) -> int:
        """Anzahl aller Spalten, rechte Seite eingeschlossen."""
        return len(self.rows[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.m, self.n

    def copy(self) -> "Matrix":
        return type(self)(self.rows, self.augmented)

    def __eq__(self, other) -> bool:
        return (isinstance(other, Matrix) and self.rows == other.rows
                and self.augmented == other.augmented)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.rows[key[0]][key[1]]
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def __repr__(self) -> str:
        inner = "; ".join(" ".join(fmt(v) for v in row) for row in self.rows)
        kind = "A|b" if self.augmented else "A"
        return f"Matrix[{self.m}x{self.n} {kind}]({inner})"

    # -- Konstruktoren ----------------------------------------------------

    @classmethod
    def identity(cls, size: int) -> "Matrix":
        return cls([[1 if i == j else 0 for j in range(size)]
                    for i in range(size)])

    @classmethod
    def zeros(cls, m: int, n: int) -> "Matrix":
        return cls([[0] * n for _ in range(m)])

    @classmethod
    def from_text(cls, text: str, augmented: bool = False) -> "Matrix":
        """Eine Zeile pro Matrixzeile, Werte durch Leerzeichen getrennt."""
        rows = [line.split() for line in text.strip().splitlines() if line.strip()]
        return cls(rows, augmented)

    @classmethod
    def system(cls, A, b) -> "Matrix":
        """Erweiterte Matrix (A | b) aus Koeffizienten und rechter Seite."""
        A = A if isinstance(A, Matrix) else Matrix(A)
        values = b.values if isinstance(b, Vector) else [parse_number(v) for v in b]
        if len(values) != A.m:
            raise ValueError("Die rechte Seite hat die falsche Länge")
        return cls([row + [val] for row, val in zip(A.rows, values)],
                   augmented=True)

    # -- Teile ------------------------------------------------------------

    def coefficients(self) -> "Matrix":
        """Nur A, ohne die rechte Seite."""
        return Matrix([row[:self.n] for row in self.rows])

    def rhs(self) -> "Vector":
        """Die rechte Seite b."""
        if not self.augmented:
            raise ValueError("Diese Matrix hat keine rechte Seite")
        return Vector([row[-1] for row in self.rows])

    def column(self, c: int) -> "Vector":
        return Vector([row[c] for row in self.rows])

    def row(self, r: int) -> "Vector":
        return Vector(self.rows[r])

    def with_rhs(self, b) -> "Matrix":
        return Matrix.system(self.coefficients() if self.augmented else self, b)

    def augment(self, other: "Matrix") -> "Matrix":
        """[A | B] — für die Inverse über [A | E]."""
        if other.m != self.m:
            raise ValueError("Beide Matrizen brauchen gleich viele Zeilen")
        return Matrix([a + b for a, b in zip(self.rows, other.rows)])

    # -- Arithmetik -------------------------------------------------------

    def transpose(self) -> "Matrix":
        return Matrix([[self.rows[r][c] for r in range(self.m)]
                       for c in range(self.width)])

    @property
    def T(self) -> "Matrix":
        return self.transpose()

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.shape != other.shape:
            raise ValueError("Für die Addition müssen die Formate gleich sein")
        return Matrix([[a + b for a, b in zip(ra, rb)]
                       for ra, rb in zip(self.rows, other.rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        return self + other.scaled(-1)

    def scaled(self, k) -> "Matrix":
        k = parse_number(k)
        return Matrix([[v * k for v in row] for row in self.rows], self.augmented)

    def __mul__(self, other):
        if isinstance(other, (Matrix, Vector)):
            return self.matmul(other)
        return self.scaled(other)

    __rmul__ = scaled

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other):
        """
        Matrixprodukt. Zeile mal Spalte — die innere Dimension muss passen,
        sonst ist das Produkt nicht erklärt.
        """
        if isinstance(other, Vector):
            if self.width != len(other):
                raise ValueError(
                    f"{self.m}×{self.width} mal Vektor der Länge {len(other)} "
                    f"geht nicht — nötig wären {self.width} Einträge")
            return Vector([sum((a * b for a, b in zip(row, other.values)),
                               Fraction(0)) for row in self.rows])
        if self.width != other.m:
            raise ValueError(
                f"{self.m}×{self.width} mal {other.m}×{other.width} geht nicht — "
                f"die inneren Masse {self.width} und {other.m} müssen gleich sein")
        return Matrix([[sum((self.rows[i][k] * other.rows[k][j]
                             for k in range(self.width)), Fraction(0))
                        for j in range(other.width)] for i in range(self.m)])

    def power(self, exponent: int) -> "Matrix":
        if self.m != self.width:
            raise ValueError("Nur quadratische Matrizen lassen sich potenzieren")
        if exponent < 0:
            raise ValueError("Negative Potenzen über .inverse() rechnen")
        result = Matrix.identity(self.m)
        for _ in range(exponent):
            result = result @ self
        return result

    def is_square(self) -> bool:
        return self.m == self.n

    def is_symmetric(self) -> bool:
        return self.is_square() and self.rows == self.transpose().rows

    def is_triangular(self, lower: bool = False) -> bool:
        for r in range(self.m):
            for c in range(self.width):
                if (c < r if not lower else c > r) and self.rows[r][c] != 0:
                    return False
        return True

    def trace(self) -> Num:
        if not self.is_square():
            raise ValueError("Die Spur ist nur für quadratische Matrizen erklärt")
        return sum((self.rows[i][i] for i in range(self.m)), Fraction(0))

    # -- Zeilenoperationen ------------------------------------------------

    def apply(self, op: Operation) -> "Matrix":
        """Gibt eine neue Matrix zurück, das Original bleibt unverändert."""
        rows = [r[:] for r in self.rows]
        if op.kind == "swap":
            rows[op.i], rows[op.j] = rows[op.j], rows[op.i]
        elif op.kind == "scale":
            if op.k == 0:
                raise ValueError("Multiplizieren mit 0 ist keine erlaubte Umformung")
            rows[op.i] = [v * op.k for v in rows[op.i]]
        elif op.kind == "add":
            if op.i == op.j:
                raise ValueError("Ziel- und Quellzeile müssen verschieden sein")
            rows[op.i] = [v + op.k * s for v, s in zip(rows[op.i], rows[op.j])]
        else:
            raise ValueError(f"'{op.kind}' lässt sich nicht anwenden")
        return type(self)(rows, self.augmented)

    def swap(self, i: int, j: int) -> "Matrix":
        return self.apply(Operation("swap", i, j=j))

    def scale(self, i: int, k) -> "Matrix":
        return self.apply(Operation("scale", i, k=parse_number(k)))

    def add_multiple(self, i: int, k, j: int) -> "Matrix":
        """R_i ← R_i + k · R_j"""
        return self.apply(Operation("add", i, j=j, k=parse_number(k)))

    # -- Struktur ---------------------------------------------------------

    def _leading_col(self, r: int) -> int:
        """Spalte der ersten Zahl ungleich 0 in Zeile r, sonst -1."""
        for c in range(self.n):
            if self.rows[r][c] != 0:
                return c
        return -1

    def pivots(self) -> list[tuple[int, int]]:
        out = []
        for r in range(self.m):
            c = self._leading_col(r)
            if c >= 0:
                out.append((r, c))
        return out

    # -- Elimination ------------------------------------------------------

    def _next_forward(self) -> Operation | None:
        r = 0
        for c in range(self.n):
            if r >= self.m:
                break
            pivot = next((i for i in range(r, self.m) if self.rows[i][c] != 0), None)
            if pivot is None:
                continue
            if pivot != r:
                return Operation("swap", r, j=pivot, reason=(
                    f"An der Pivotstelle von Spalte {c + 1} steht eine 0. "
                    f"Tausche mit einer Zeile darunter."))
            for i in range(r + 1, self.m):
                if self.rows[i][c] != 0:
                    return Operation("add", i, j=r,
                                     k=-self.rows[i][c] / self.rows[r][c],
                                     reason=f"Eintrag unter dem Pivot in Spalte "
                                            f"{c + 1} zu 0 machen.")
            r += 1
        return None

    def _next_backward(self) -> Operation | None:
        for r, c in reversed(self.pivots()):
            if self.rows[r][c] != 1:
                return Operation("scale", r, k=1 / self.rows[r][c],
                                 reason=f"Pivot in Zeile {r + 1} auf 1 normieren.")
            for i in range(r - 1, -1, -1):
                if self.rows[i][c] != 0:
                    return Operation("add", i, j=r, k=-self.rows[i][c],
                                     reason=f"Eintrag über dem Pivot in Spalte "
                                            f"{c + 1} zu 0 machen.")
        return None

    def next_step(self, jordan: bool = False) -> Operation | None:
        """Die nächste nötige Umformung, oder None wenn die Form erreicht ist."""
        step = self._next_forward()
        if step is not None:
            return step
        return self._next_backward() if jordan else None

    def ref(self) -> "Matrix":
        """Zeilenstufenform."""
        A = self
        for _ in range(600):
            op = A._next_forward()
            if op is None:
                return A
            A = A.apply(op)
        return A

    def rref(self) -> "Matrix":
        """Reduzierte Zeilenstufenform."""
        A = self
        for _ in range(600):
            op = A._next_forward() or A._next_backward()
            if op is None:
                return A
            A = A.apply(op)
        return A

    def rank(self) -> int:
        """Rang — die Anzahl der Pivots in der Stufenform."""
        return len(self.ref().pivots())

    def form_state(self) -> str:
        """'standard', 'ref' oder 'rref'."""
        last_lead = -1
        seen_zero_row = False
        is_ref = True
        is_rref = True
        for r in range(self.m):
            c = self._leading_col(r)
            if c < 0:
                seen_zero_row = True
                continue
            if seen_zero_row or c <= last_lead:
                is_ref = False
            last_lead = c
            if self.rows[r][c] != 1:
                is_rref = False
            for k in range(self.m):
                if k != r and self.rows[k][c] != 0:
                    is_rref = False
        if not is_ref:
            return "standard"
        return "rref" if is_rref else "ref"

    def steps_to(self, jordan: bool = True) -> list[tuple[Operation, "Matrix"]]:
        """Der ganze Weg bis zur (reduzierten) Zeilenstufenform."""
        out: list[tuple[Operation, Matrix]] = []
        A = self
        for _ in range(600):
            op = A.next_step(jordan)
            if op is None:
                break
            A = A.apply(op)
            out.append((op, A))
        return out

    def inconsistent(self) -> bool:
        """Enthält eine Zeile 0 … 0 | b mit b ungleich 0?"""
        if not self.augmented:
            return False
        return any(all(v == 0 for v in row[:self.n]) and row[self.n] != 0
                   for row in self.rows)

    # -- Lösungsmenge -----------------------------------------------------

    def analyze(self) -> dict:
        """
        Klassifiziert die Lösungsmenge der erweiterten Matrix.

        {'type': 'none' | 'unique' | 'infinite', …}
        """
        if not self.augmented:
            raise ValueError("Dafür braucht es eine erweiterte Matrix (A | b)")
        A = self.rref()
        pivot_of_col: dict[int, int] = {}
        for r in range(A.m):
            c = A._leading_col(r)
            if c < 0:
                if A.rows[r][A.n] != 0:
                    return {"type": "none", "value": A.rows[r][A.n], "row": r}
            else:
                pivot_of_col[c] = r
        free = [c for c in range(A.n) if c not in pivot_of_col]
        if not free:
            x = [Fraction(0)] * A.n
            for c, r in pivot_of_col.items():
                x[c] = A.rows[r][A.n]
            return {"type": "unique", "x": x, "vector": Vector(x)}
        terms = {}
        for c in range(A.n):
            if c in free:
                continue
            r = pivot_of_col[c]
            terms[c] = (A.rows[r][A.n],
                        {f: -A.rows[r][f] for f in free if A.rows[r][f] != 0})
        return {"type": "infinite", "free": free, "terms": terms}

    def solution_lines(self, decimal: bool = False) -> list[str]:
        info = self.analyze()
        if info["type"] == "none":
            return []
        if info["type"] == "unique":
            return [f"x{c + 1} = {fmt(v, decimal)}" for c, v in enumerate(info["x"])]
        names = {c: f"t{i + 1}" for i, c in enumerate(info["free"])}
        lines = []
        for c in range(self.n):
            if c in info["free"]:
                lines.append(f"x{c + 1} = {names[c]}")
                continue
            const, coeffs = info["terms"][c]
            expr = fmt(const, decimal) if (const != 0 or not coeffs) else ""
            for f, k in coeffs.items():
                mag = abs(k)
                piece = names[f] if mag == 1 else f"{fmt(mag, decimal)}·{names[f]}"
                if expr == "":
                    expr = piece if k > 0 else MINUS + piece
                else:
                    expr += " " + ("+" if k > 0 else MINUS) + " " + piece
            lines.append(f"x{c + 1} = {expr}")
        return lines

    def equation_lines(self, decimal: bool = False) -> list[str]:
        out = []
        for row in self.rows:
            if all(v == 0 for v in row):
                continue
            left = ""
            for c in range(self.n):
                k = row[c]
                if k == 0:
                    continue
                mag = abs(k)
                piece = f"x{c + 1}" if mag == 1 else f"{fmt(mag, decimal)}·x{c + 1}"
                if left == "":
                    left = piece if k > 0 else MINUS + piece
                else:
                    left += " " + ("+" if k > 0 else MINUS) + " " + piece
            right = fmt(row[self.n], decimal) if self.augmented else "0"
            out.append(f"{left or '0'} = {right}")
        return out or ["0 = 0"]

    def same_as(self, other: "Matrix") -> bool:
        return self.rows == other.rows


# ==========================================================================
# Vektor
# ==========================================================================

class Vector:
    """Ein Spaltenvektor mit exakten Einträgen."""

    def __init__(self, values: Iterable):
        self.values: list[Num] = [parse_number(v) for v in values]
        if not self.values:
            raise ValueError("Ein Vektor braucht mindestens einen Eintrag")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other) -> bool:
        return isinstance(other, Vector) and self.values == other.values

    def __repr__(self) -> str:
        return "Vector(" + ", ".join(fmt(v) for v in self.values) + ")"

    def copy(self) -> "Vector":
        return Vector(self.values)

    @property
    def dim(self) -> int:
        return len(self.values)

    def as_matrix(self) -> Matrix:
        """Als Spaltenmatrix n×1."""
        return Matrix([[v] for v in self.values])

    def as_row(self) -> Matrix:
        return Matrix([list(self.values)])

    # -- Rechnen ----------------------------------------------------------

    def _check(self, other: "Vector"):
        if len(self) != len(other):
            raise ValueError(f"Vektoren verschiedener Länge: "
                             f"{len(self)} und {len(other)}")

    def __add__(self, other: "Vector") -> "Vector":
        self._check(other)
        return Vector([a + b for a, b in zip(self.values, other.values)])

    def __sub__(self, other: "Vector") -> "Vector":
        self._check(other)
        return Vector([a - b for a, b in zip(self.values, other.values)])

    def __neg__(self) -> "Vector":
        return Vector([-v for v in self.values])

    def scaled(self, k) -> "Vector":
        k = parse_number(k)
        return Vector([v * k for v in self.values])

    def __mul__(self, k):
        if isinstance(k, Vector):
            return self.dot(k)
        return self.scaled(k)

    __rmul__ = scaled

    def dot(self, other: "Vector") -> Num:
        """Skalarprodukt."""
        self._check(other)
        return sum((a * b for a, b in zip(self.values, other.values)), Fraction(0))

    def norm_squared(self) -> Num:
        """Quadrat der Länge — exakt, im Gegensatz zur Länge selbst."""
        return self.dot(self)

    def norm(self, p=2) -> float | Num:
        """
        Norm. p=2 ist die euklidische Länge und im Allgemeinen irrational,
        darum Gleitkomma; p=1 und p='inf' bleiben exakt.
        """
        if p == 1:
            return sum((abs(v) for v in self.values), Fraction(0))
        if p in ("inf", math.inf):
            return max(abs(v) for v in self.values)
        if p == 2:
            q = self.norm_squared()
            root = math.isqrt(q.numerator), math.isqrt(q.denominator)
            if root[0] ** 2 == q.numerator and root[1] ** 2 == q.denominator:
                return Fraction(root[0], root[1])      # geht exakt auf
            return math.sqrt(float(q))
        return float(sum(abs(float(v)) ** p for v in self.values)) ** (1 / p)

    def is_zero(self) -> bool:
        return all(v == 0 for v in self.values)

    def normalized(self) -> "Vector":
        length = self.norm()
        if self.is_zero():
            raise ValueError("Der Nullvektor lässt sich nicht normieren")
        if isinstance(length, Fraction):
            return self.scaled(Fraction(1) / length)
        return Vector([Fraction(float(v) / length).limit_denominator(10 ** 9)
                       for v in self.values])

    def angle(self, other: "Vector", degrees: bool = True) -> float:
        """Winkel zwischen zwei Vektoren."""
        self._check(other)
        if self.is_zero() or other.is_zero():
            raise ValueError("Zum Nullvektor gibt es keinen Winkel")
        cos = float(self.dot(other)) / (float(self.norm()) * float(other.norm()))
        cos = max(-1.0, min(1.0, cos))         # Rundung abfangen
        rad = math.acos(cos)
        return math.degrees(rad) if degrees else rad

    def cross(self, other: "Vector") -> "Vector":
        """Vektorprodukt — nur im Raum erklärt."""
        if len(self) != 3 or len(other) != 3:
            raise ValueError("Das Vektorprodukt gibt es nur für Vektoren mit "
                             "drei Einträgen")
        a, b = self.values, other.values
        return Vector([a[1] * b[2] - a[2] * b[1],
                       a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]])

    def parallel_to(self, other: "Vector") -> bool:
        if len(self) == 3 and len(other) == 3:
            return self.cross(other).is_zero()
        return Matrix([list(self.values), list(other.values)]).rank() < 2

    def orthogonal_to(self, other: "Vector") -> bool:
        return self.dot(other) == 0


def triple_product(u: Vector, v: Vector, w: Vector) -> Num:
    """Spatprodukt (u × v) · w — das Volumen des aufgespannten Spats."""
    return u.cross(v).dot(w)
