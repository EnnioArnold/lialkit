#!/usr/bin/env python3
"""
Vektorrechnung mit Rechenweg: Skalarprodukt, Vektorprodukt, Spatprodukt,
Winkel, Projektionen und die beiden Ungleichungen aus SW 1.

Der Vector selbst steht in core.py. Hier kommt dazu, was einen Rechenweg
hat oder erklärt werden will.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .core import (MINUS, Matrix, Num, Vector, fmt, is_exact, parse_number,
                   triple_product)
from .render import Result, Step, tex_any, tex_number, tex_vector


def _vec(v) -> Vector:
    return v if isinstance(v, Vector) else Vector(v)


# ==========================================================================
# Skalarprodukt
# ==========================================================================

class DotResult(Result):
    title = "Skalarprodukt"

    def __init__(self, u: Vector, v: Vector, value: Num, decimal: bool = False):
        super().__init__(decimal)
        self.u, self.v, self.value = u, v, value
        if value == 0:
            self.summary = ("Das Skalarprodukt ist 0 — die Vektoren stehen "
                            "senkrecht aufeinander.")
        elif value > 0:
            self.summary = "Positiv — der Winkel ist kleiner als 90°."
        else:
            self.summary = "Negativ — der Winkel ist grösser als 90°."
        self.highlights = [(r"u \cdot v", value)]


def dot(u, v, decimal: bool = False) -> DotResult:
    """Skalarprodukt mit ausgeschriebener Summe."""
    u, v = _vec(u), _vec(v)
    u._check(v)
    products = [a * b for a, b in zip(u.values, v.values)]
    value = sum(products, Fraction(0))
    terms = " + ".join(
        f"{tex_number(a, decimal)} \\cdot {tex_number(b, decimal)}"
        for a, b in zip(u.values, v.values))
    result = DotResult(u, v, value, decimal)
    result.steps = [
        Step("Komponentenweise multiplizieren", detail=terms,
             items=[("u", u), ("v", v)]),
        Step("Produkte addieren",
             detail=" + ".join(tex_number(p, decimal) for p in products)
                    + " = " + tex_number(value, decimal)),
    ]
    return result


# ==========================================================================
# Vektorprodukt und Spatprodukt
# ==========================================================================

class CrossResult(Result):
    title = "Vektorprodukt"

    def __init__(self, u: Vector, v: Vector, w: Vector, decimal: bool = False):
        super().__init__(decimal)
        self.u, self.v, self.result = u, v, w
        area = w.norm()
        self.area = area
        if w.is_zero():
            self.summary = ("Das Ergebnis ist der Nullvektor — die beiden "
                            "Vektoren sind parallel und spannen keine Fläche auf.")
        else:
            self.summary = (f"Steht senkrecht auf beiden. Seine Länge "
                            f"{fmt(area, decimal)} ist der Flächeninhalt des "
                            f"aufgespannten Parallelogramms.")
        self.highlights = [(r"u \times v", w)]


def cross(u, v, decimal: bool = False) -> CrossResult:
    """Vektorprodukt mit der Merkregel über die Determinante."""
    u, v = _vec(u), _vec(v)
    w = u.cross(v)
    a, b = u.values, v.values
    rows = [
        (f"(u_2 v_3 - u_3 v_2)", a[1] * b[2], a[2] * b[1]),
        (f"(u_3 v_1 - u_1 v_3)", a[2] * b[0], a[0] * b[2]),
        (f"(u_1 v_2 - u_2 v_1)", a[0] * b[1], a[1] * b[0]),
    ]
    result = CrossResult(u, v, w, decimal)
    result.steps = [
        Step("Nach der ersten Zeile entwickeln",
             detail=r"u \times v = \begin{vmatrix} e_1 & e_2 & e_3 \\ "
                    + " & ".join(tex_number(x, decimal) for x in a) + r" \\ "
                    + " & ".join(tex_number(x, decimal) for x in b)
                    + r"\end{vmatrix}",
             items=[("u", u), ("v", v)]),
    ] + [
        Step(f"Komponente {i + 1}",
             detail=(f"{formula} = {tex_number(p, decimal)} - "
                     f"{tex_number(q, decimal)} = "
                     f"{tex_number(p - q, decimal)}"))
        for i, (formula, p, q) in enumerate(rows)
    ]
    if not w.is_zero():
        result.steps.append(Step(
            "Probe: senkrecht auf beiden",
            detail=(f"(u \\times v) \\cdot u = "
                    f"{tex_number(w.dot(u), decimal)}, \\quad "
                    f"(u \\times v) \\cdot v = {tex_number(w.dot(v), decimal)}"),
            note="Beide Skalarprodukte müssen 0 sein."))
    return result


class SpatResult(Result):
    title = "Spatprodukt"

    def __init__(self, u, v, w, value, decimal: bool = False):
        super().__init__(decimal)
        self.u, self.v, self.w, self.value = u, v, w, value
        self.volume = abs(value)
        if value == 0:
            self.summary = ("Das Spatprodukt ist 0 — die drei Vektoren liegen "
                            "in einer Ebene und sind linear abhängig.")
        else:
            orientation = "Rechtssystem" if value > 0 else "Linkssystem"
            self.summary = (f"Das Volumen des Spats ist {fmt(abs(value), decimal)}. "
                            f"Das Vorzeichen zeigt die Orientierung: {orientation}.")
        self.highlights = [(r"(u \times v) \cdot w", value),
                           (r"V", abs(value))]


def spat(u, v, w, decimal: bool = False) -> SpatResult:
    """Spatprodukt (u × v) · w — Volumen des aufgespannten Parallelepipeds."""
    u, v, w = _vec(u), _vec(v), _vec(w)
    crossed = u.cross(v)
    value = crossed.dot(w)
    result = SpatResult(u, v, w, value, decimal)
    result.steps = [
        Step("Erst das Vektorprodukt", items=[("u \\times v", crossed)],
             note="Seine Länge ist die Grundfläche des Spats."),
        Step("Dann das Skalarprodukt mit w",
             detail=(" + ".join(f"{tex_number(a, decimal)} \\cdot "
                                f"{tex_number(b, decimal)}"
                                for a, b in zip(crossed.values, w.values))
                     + " = " + tex_number(value, decimal)),
             note="Das entspricht Grundfläche mal Höhe."),
        Step("Als Determinante gelesen",
             detail=r"\det\begin{pmatrix}"
                    + r" \\ ".join(" & ".join(tex_number(x, decimal) for x in vec)
                                   for vec in (u.values, v.values, w.values))
                    + r"\end{pmatrix} = " + tex_number(value, decimal),
             note="Das Spatprodukt ist die Determinante der drei Vektoren."),
    ]
    return result


# ==========================================================================
# Winkel, Projektion, Ungleichungen
# ==========================================================================

class AngleResult(Result):
    title = "Winkel zwischen zwei Vektoren"

    def __init__(self, u, v, degrees, decimal: bool = True):
        super().__init__(decimal)
        self.u, self.v, self.degrees = u, v, degrees
        self.summary = (f"{degrees:.4f}° ≈ "
                        f"{math.radians(degrees):.4f} rad")
        self.highlights = [(r"\varphi", f"{degrees:.4f}^\\circ")]


def angle(u, v, decimal: bool = True) -> AngleResult:
    """Winkel über cos φ = (u·v) / (|u| · |v|)."""
    u, v = _vec(u), _vec(v)
    value = u.dot(v)
    nu, nv = u.norm(), v.norm()
    deg = u.angle(v)
    result = AngleResult(u, v, deg, decimal)
    result.steps = [
        Step("Skalarprodukt", detail=f"u \\cdot v = {tex_number(value, decimal)}",
             items=[("u", u), ("v", v)]),
        Step("Längen", detail=(f"|u| = {tex_number(nu, True)}, \\quad "
                               f"|v| = {tex_number(nv, True)}"),
             note=("Beide Längen sind exakt." if is_exact(nu) and is_exact(nv)
                   else "Mindestens eine Länge ist irrational, hier auf vier "
                        "Stellen gerundet.")),
        Step("Kosinus bilden",
             detail=(r"\cos\varphi = \frac{u \cdot v}{|u|\,|v|} = "
                     f"{float(value) / (float(nu) * float(nv)):.4f}")),
        Step("Umkehrfunktion", detail=f"\\varphi = {deg:.4f}^\\circ"),
    ]
    if value == 0:
        result.warnings.append("Die Vektoren stehen senkrecht aufeinander.")
    return result


def projection(u, onto, decimal: bool = False) -> Result:
    """Die orthogonale Projektion von u auf einen anderen Vektor."""
    u, a = _vec(u), _vec(onto)
    if a.is_zero():
        raise ValueError("Auf den Nullvektor lässt sich nicht projizieren")
    factor = u.dot(a) / a.norm_squared()
    p = a.scaled(factor)
    rest = u - p
    out = Result(decimal)
    out.title = "Orthogonale Projektion"
    out.summary = "u zerfällt in einen Anteil längs a und einen senkrecht dazu."
    out.steps = [
        Step("Faktor bestimmen",
             detail=(r"\lambda = \frac{u \cdot a}{a \cdot a} = "
                     f"{tex_number(factor, decimal)}"),
             items=[("u", u), ("a", a)]),
        Step("Projektion", detail=r"p = \lambda\, a", items=[("p", p)]),
        Step("Senkrechter Rest", detail=r"u - p", items=[("u - p", rest),
             ],
             note=f"Probe: (u − p) · a = {fmt(rest.dot(a), decimal)}"),
    ]
    out.highlights = [("p", p)]
    return out


def triangle_inequality(u, v, decimal: bool = True) -> Result:
    """Die Dreiecksungleichung |u + v| ≤ |u| + |v| am Beispiel."""
    u, v = _vec(u), _vec(v)
    left = (u + v).norm()
    right = u.norm() + v.norm()
    out = Result(decimal)
    out.title = "Dreiecksungleichung"
    holds = float(left) <= float(right) + 1e-12
    out.summary = ("Erfüllt." if holds else "Verletzt — das darf nicht sein.")
    out.steps = [
        Step("Summe bilden", items=[("u + v", u + v)]),
        Step("Linke Seite", detail=f"|u + v| = {tex_number(left, True)}"),
        Step("Rechte Seite",
             detail=(f"|u| + |v| = {tex_number(u.norm(), True)} + "
                     f"{tex_number(v.norm(), True)} = {tex_number(right, True)}")),
        Step("Vergleich",
             detail=f"{float(left):.4f} \\le {float(right):.4f}",
             note="Gleichheit gilt genau dann, wenn die Vektoren gleich "
                  "gerichtet sind."),
    ]
    if not holds:
        out.warnings.append("Die Ungleichung ist verletzt — das deutet auf "
                            "einen Rechenfehler hin.")
    return out


def cauchy_schwarz(u, v, decimal: bool = True) -> Result:
    """Die Schwarzsche Ungleichung |u · v| ≤ |u| · |v| am Beispiel."""
    u, v = _vec(u), _vec(v)
    left = abs(u.dot(v))
    right = float(u.norm()) * float(v.norm())
    out = Result(decimal)
    out.title = "Schwarzsche Ungleichung"
    holds = float(left) <= right + 1e-12
    equal = abs(float(left) - right) < 1e-12
    out.summary = ("Gleichheit — die Vektoren sind parallel." if equal
                   else "Erfüllt." if holds else "Verletzt.")
    out.steps = [
        Step("Linke Seite", detail=f"|u \\cdot v| = {tex_number(left, decimal)}",
             items=[("u", u), ("v", v)]),
        Step("Rechte Seite", detail=f"|u| \\cdot |v| = {right:.4f}"),
        Step("Vergleich", detail=f"{float(left):.4f} \\le {right:.4f}",
             note="Gleichheit genau dann, wenn die Vektoren linear abhängig sind."),
    ]
    if not holds:
        out.warnings.append("Die Ungleichung ist verletzt — das deutet auf "
                            "einen Rechenfehler hin.")
    return out


# ==========================================================================
# Geraden und Ebenen
# ==========================================================================

class Line:
    """Eine Gerade in Parameterform: x = p + t · d."""

    def __init__(self, point, direction):
        self.p = _vec(point)
        self.d = _vec(direction)
        if self.d.is_zero():
            raise ValueError("Der Richtungsvektor darf nicht der Nullvektor sein")

    def at(self, t) -> Vector:
        return self.p + self.d.scaled(t)

    def contains(self, point) -> bool:
        q = _vec(point) - self.p
        return q.parallel_to(self.d) if not q.is_zero() else True

    def intersect(self, other: "Line", decimal: bool = False) -> Result:
        """
        Schnittpunkt zweier Geraden.

        Gesucht sind λ und μ mit p + λd = q + μe. Das ist ein lineares
        Gleichungssystem mit zwei Unbekannten und so vielen Gleichungen wie
        der Raum Dimensionen hat — in der Ebene also quadratisch, im Raum
        überbestimmt. Darum kann es im Raum auch windschiefe Geraden geben.
        """
        from .solving import gauss

        out = Result(decimal)
        out.title = "Schnittpunkt zweier Geraden"
        if len(self.p) != len(other.p):
            raise ValueError("Beide Geraden müssen im selben Raum liegen")

        rhs = other.p - self.p
        system = Matrix([[self.d[i], -other.d[i], rhs[i]]
                         for i in range(len(self.p))], augmented=True)
        out.steps = [
            Step("Gleichsetzen",
                 detail=r"p + \lambda d = q + \mu e \;\Rightarrow\; "
                        r"\lambda d - \mu e = q - p",
                 items=[("(A \\mid b)", system)],
                 note="Die Unbekannten sind λ und μ."),
        ]
        info = system.analyze()
        if info["type"] == "none":
            out.summary = ("Kein Schnittpunkt. In der Ebene heisst das "
                           "parallel, im Raum windschief."
                           if len(self.p) == 3 else
                           "Kein Schnittpunkt — die Geraden sind parallel.")
            out.steps.append(Step("Das System ist unlösbar", note=out.summary))
            return out
        if info["type"] == "infinite":
            out.summary = "Die Geraden fallen zusammen."
            out.steps.append(Step("Unendlich viele Lösungen", note=out.summary))
            return out
        lam, mu = info["vector"][0], info["vector"][1]
        point = self.at(lam)
        probe = other.at(mu)
        out.summary = f"Schnittpunkt bei λ = {fmt(lam, decimal)}."
        out.steps.append(Step(
            "Parameter einsetzen",
            detail=(f"\\lambda = {tex_number(lam, decimal)}, \\quad "
                    f"\\mu = {tex_number(mu, decimal)}"),
            items=[("S", point)],
            note=("Probe über die zweite Gerade: "
                  + ("stimmt überein." if probe == point
                     else "stimmt NICHT überein."))))
        out.highlights = [("S", point)]
        if probe != point:
            out.warnings.append("Die Probe schlägt fehl — im Raum sind die "
                                "Geraden dann windschief.")
            out.highlights = []
            out.summary = "Kein echter Schnittpunkt: die Geraden sind windschief."
        return out

    def __repr__(self) -> str:
        return (f"Line(p=({', '.join(fmt(v) for v in self.p)}), "
                f"d=({', '.join(fmt(v) for v in self.d)}))")


class Plane:
    """
    Eine Ebene, entweder über zwei Spannvektoren oder über die Normale.

    Die Koordinatenform n · x = c ergibt sich aus der Normalen und einem
    Punkt; umgekehrt liefert die Normale sofort, ob zwei Ebenen parallel
    sind.
    """

    def __init__(self, point, u=None, v=None, normal=None):
        self.p = _vec(point)
        if normal is not None:
            self.n = _vec(normal)
            self.u = self.v = None
        else:
            if u is None or v is None:
                raise ValueError("Entweder zwei Spannvektoren oder eine Normale")
            self.u, self.v = _vec(u), _vec(v)
            self.n = self.u.cross(self.v)
            if self.n.is_zero():
                raise ValueError("Die Spannvektoren sind parallel — sie "
                                 "spannen keine Ebene auf")
        self.c = self.n.dot(self.p)

    def at(self, s, t) -> Vector:
        if self.u is None:
            raise ValueError("Diese Ebene ist nur über ihre Normale gegeben")
        return self.p + self.u.scaled(s) + self.v.scaled(t)

    def contains(self, point) -> bool:
        return self.n.dot(_vec(point)) == self.c

    def distance(self, point) -> float:
        q = _vec(point)
        return abs(float(self.n.dot(q) - self.c)) / float(self.n.norm())

    def intersect_line(self, line: "Line", decimal: bool = False) -> Result:
        """
        Schnittpunkt einer Geraden mit dieser Ebene.

        Man setzt die Geradengleichung in die Koordinatenform ein; das gibt
        eine einzige Gleichung für den Parameter. Steht die Gerade senkrecht
        zur Normalen, läuft sie parallel zur Ebene — dann gibt es entweder
        keinen oder unendlich viele Schnittpunkte.
        """
        out = Result(decimal)
        out.title = "Schnittpunkt Gerade und Ebene"
        denominator = self.n.dot(line.d)
        offset = self.c - self.n.dot(line.p)
        out.steps = [
            Step("Ebene in Koordinatenform",
                 detail=r"n \cdot x = c",
                 items=[("n", self.n), ("c", self.c)],
                 note=self.coordinate_form(decimal)),
            Step("Gerade einsetzen",
                 detail=(r"n \cdot (p + \lambda d) = c \;\Rightarrow\; "
                         r"\lambda\, (n \cdot d) = c - n \cdot p"),
                 note=f"n·d = {fmt(denominator, decimal)}, "
                      f"c − n·p = {fmt(offset, decimal)}"),
        ]
        if denominator == 0:
            if offset == 0:
                out.summary = ("Die Gerade liegt ganz in der Ebene — jeder "
                               "ihrer Punkte ist ein Schnittpunkt.")
            else:
                out.summary = ("Die Gerade läuft parallel zur Ebene und trifft "
                               "sie nie.")
            out.steps.append(Step("Kein eindeutiger Schnittpunkt",
                                  note=out.summary))
            return out
        lam = offset / denominator
        point = line.at(lam)
        out.summary = f"Ein Schnittpunkt bei λ = {fmt(lam, decimal)}."
        out.steps.append(Step("Parameter einsetzen",
                              detail=f"\\lambda = {tex_number(lam, decimal)}",
                              items=[("S", point)],
                              note=f"Probe: n·S = {fmt(self.n.dot(point), decimal)}, "
                                   f"c = {fmt(self.c, decimal)}"))
        out.highlights = [("S", point)]
        return out

    def coordinate_form(self, decimal: bool = False) -> str:
        names = ["x", "y", "z"]
        parts = []
        for i, k in enumerate(self.n.values):
            if k == 0:
                continue
            mag = abs(k)
            piece = names[i] if mag == 1 else f"{fmt(mag, decimal)}·{names[i]}"
            if not parts:
                parts.append(piece if k > 0 else MINUS + piece)
            else:
                parts.append(("+ " if k > 0 else MINUS + " ") + piece)
        return " ".join(parts) + " = " + fmt(self.c, decimal)

    def __repr__(self) -> str:
        return f"Plane({self.coordinate_form()})"
