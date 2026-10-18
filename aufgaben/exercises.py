#!/usr/bin/env python3
"""
Die Übungsaufgaben aus den Semesterwochen 1 bis 4.

Die Aufgaben stehen hier als Daten, nicht als fertige Antworten. Du holst
dir eine Aufgabe, rechnest selbst und vergleichst — das ist der Zweck.
`loese(...)` rechnet sie durch, wenn du kontrollieren willst.

    from lialkit.exercises import aufgabe, liste, loese

    liste("SW3")              # was es gibt
    a = aufgabe("SW3.1a")     # die Daten
    loese("SW3.1a")           # mit Rechenweg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..rechnen.core import Matrix, Vector


@dataclass
class Exercise:
    """Eine Aufgabe mit ihren Daten und dem passenden Verfahren."""
    key: str
    week: str
    title: str
    task: str
    data: dict = field(default_factory=dict)
    method: str = ""
    note: str = ""

    def __repr__(self) -> str:
        lines = [f"{self.key} — {self.title}", f"  {self.task}"]
        for name, value in self.data.items():
            if isinstance(value, Matrix):
                lines.append(f"  {name} = {value!r}")
            elif isinstance(value, Vector):
                lines.append(f"  {name} = {value!r}")
            else:
                lines.append(f"  {name} = {value}")
        if self.note:
            lines.append(f"  Hinweis: {self.note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        rows = []
        for name, value in self.data.items():
            if isinstance(value, (Matrix, Vector)):
                from ..rechnen.render import tex_any
                rows.append(f"<div>$$ {name} = {tex_any(value)} $$</div>")
            else:
                rows.append(f"<div>{html.escape(name)} = "
                            f"{html.escape(str(value))}</div>")
        note = (f"<p style='color:#86868b;font-size:12px'>"
                f"{html.escape(self.note)}</p>" if self.note else "")
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,"
            "'Segoe UI',Helvetica,sans-serif\">"
            f"<p style='font-weight:600;margin:0'>{html.escape(self.key)} — "
            f"{html.escape(self.title)}</p>"
            f"<p style='color:#6e6e73;font-size:13px;margin:2px 0 10px'>"
            f"{html.escape(self.task)}</p>" + "".join(rows) + note + "</div>")


# ==========================================================================
# Die Sammlung
# ==========================================================================

def _m(rows, augmented=False) -> Matrix:
    return Matrix(rows, augmented)


EXERCISES: dict[str, Exercise] = {}


def _add(ex: Exercise):
    EXERCISES[ex.key] = ex


# -- Semesterwoche 1: Vektoren, Geraden, Ebenen ----------------------------

_add(Exercise(
    "SW1.2", "SW1", "Gleichungssystem in Stufenform",
    "Erweiterte Koeffizientenmatrix angeben und das System lösen.",
    {"A|b": _m([[2, 5, -2, -3], [0, 3, 2, 7], [0, 0, 5, 10]], True)},
    "gauss",
    "Die Matrix steht schon in Stufenform — reines Rückwärtseinsetzen."))

_add(Exercise(
    "SW1.3", "SW1", "Schnittpunkt zweier Geraden",
    "Schnittpunkt von g und h bestimmen.",
    {"g": ((1, 1), (2, 1)), "h": ((1, 4), (1, -1))},
    "lines",
    "Gleichsetzen führt auf zwei Gleichungen für λ und μ."))

_add(Exercise(
    "SW1.4", "SW1", "Gerade schneidet Ebene",
    "Schnittpunkt der Geraden g mit der Ebene E bestimmen.",
    {"g": ((1, 1, 7), (2, 3, 4)),
     "E": ((1, 2, 4), (1, 0, 1), (0, 1, -2))},
    "line_plane",
    "Erst die Ebene in Koordinatenform, dann die Gerade einsetzen."))

_add(Exercise(
    "SW1.5", "SW1", "Skalar- und Vektorprodukt",
    "Längen, Skalarprodukt, Winkel und Vektorprodukt von u und v.",
    {"u": Vector([2, -3, 6]), "v": Vector([-1, 4, 8])},
    "vectors",
    "Beide Längen gehen hier exakt auf — 49 und 81 sind Quadratzahlen."))

# -- Semesterwoche 2: Gauss-Elimination ------------------------------------

_add(Exercise(
    "SW2.3", "SW2", "System mit eindeutiger Lösung",
    "Erweiterte Matrix angeben, auf Zeilenstufenform bringen, lösen.",
    {"A|b": _m([[6, -1, -1, 4], [1, 1, 10, -6], [2, -1, 1, -2]], True)},
    "gauss"))

_SW2_4 = [
    ([[1, 1, 2, 8], [-1, -2, 3, 1], [3, -7, 4, 10]], "eindeutig"),
    ([[2, 2, 2, 0], [-2, 5, 2, 1], [8, 1, 4, -1]], "eindeutig"),
    ([[1, -1, 2, -1, -1], [2, 1, -2, -2, -2], [-1, 2, -4, 1, 1],
      [3, 0, 0, -3, -3]], "unendlich viele"),
    ([[0, -2, 3, 1], [3, 6, -3, -2], [6, 6, 3, 5]], "eindeutig"),
    ([[2, -3, -2], [2, 1, 1], [3, 2, 1]], "überbestimmt"),
    ([[3, 2, -1, -15], [5, 3, 2, 0], [3, 1, 3, 11], [-6, -4, 2, 30]],
     "überbestimmt"),
    ([[4, -8, 12], [3, -6, 9], [-2, 4, -6]], "unendlich viele"),
    ([[0, 10, -4, 1, 1], [1, 4, -1, 1, 2], [3, 2, 1, 2, 5],
      [-2, -8, 2, -2, -4], [1, -6, 3, 0, 1]], "unendlich viele"),
]
for _i, (_rows, _kind) in enumerate(_SW2_4, 1):
    _add(Exercise(
        f"SW2.4.{_i}", "SW2", f"Lösungsmenge bestimmen ({_i})",
        "Die Lösungsmenge des Gleichungssystems bestimmen.",
        {"A|b": _m(_rows, True)}, "gauss",
        f"Erwartet: {_kind}."))

# -- Semesterwoche 3: LR-Zerlegung -----------------------------------------

_add(Exercise(
    "SW3.1a", "SW3", "LR-Zerlegung, 2×2",
    "Das System mittels LR-Zerlegung lösen.",
    {"A": _m([[3, -6], [-2, 5]]), "b": Vector([0, 1])}, "lu"))

_add(Exercise(
    "SW3.1b", "SW3", "LR-Zerlegung, 3×3",
    "Das System mittels LR-Zerlegung lösen.",
    {"A": _m([[2, 6, 2], [-3, -8, 0], [4, 9, 2]]), "b": Vector([2, 2, 3])},
    "lu"))

_add(Exercise(
    "SW3.1c", "SW3", "LR-Zerlegung, 4×4",
    "Das System mittels LR-Zerlegung lösen.",
    {"A": _m([[2, -1, 3, 4], [1, 0, -2, 7], [3, -3, 1, 5], [2, 1, 4, 4]]),
     "b": Vector([9, 11, 8, 10])}, "lu"))

_add(Exercise(
    "SW3.3", "SW3", "Eigener LR-Algorithmus",
    "Einen Algorithmus zur LR-Zerlegung programmieren, Schleifen ausschreiben.",
    {}, "program",
    "Diese Aufgabe sollst du selbst programmieren. In lialkit.uebung "
    "liegt ein Gerüst mit Tests, an denen du deine Fassung prüfen kannst."))

# -- Semesterwoche 4: Inverse und Determinanten ----------------------------

for _key, _rows in [("a", [[4, 3], [8, 6]]), ("b", [[4, 3], [8, 7]]),
                    ("c", [[6, 6], [6, 6]]),
                    ("d", [[1, 0, 0], [1, 1, 0], [1, 1, 1]])]:
    _add(Exercise(
        f"SW4.1{_key}", "SW4", f"Regulär oder singulär? ({_key})",
        "Die Inverse berechnen, falls sie existiert — sonst begründen, "
        "warum nicht.",
        {"A": _m(_rows)}, "inverse"))

for _key, _rows in [("a", [[0, 3], [4, 0]]), ("b", [[2, 0], [4, 2]]),
                    ("c", [[3, 4], [5, 7]])]:
    _add(Exercise(
        f"SW4.2{_key}", "SW4", f"Inverse direkt und mit Gauss-Jordan ({_key})",
        "Die Inverse einmal direkt mit den Einheitsvektoren als rechten "
        "Seiten und einmal mit Gauss-Jordan berechnen.",
        {"A": _m(_rows)}, "inverse"))

_add(Exercise(
    "SW4.3", "SW4", "Nicht invertierbar",
    "Zeigen, dass die Matrix nicht invertierbar ist.",
    {"A": _m([[1, 2], [3, 6]])}, "inverse",
    "Die zweite Zeile ist das Dreifache der ersten."))

_add(Exercise(
    "SW4.4", "SW4", "Determinante eines Vielfachen",
    "Wie hängt det(λ·A) von det(A) ab? Entspricht das der Anschauung?",
    {"A": _m([[1, 2, 3], [4, 5, 7], [7, 8, 10]]), "lambda": 3},
    "det_scale",
    "Jede der n Zeilen wird mit λ multipliziert — darum λ^n."))

_add(Exercise(
    "SW4.5", "SW4", "Gilt det(−A) = −det(A)?",
    "Die Aussage prüfen und begründen.",
    {"A2": _m([[1, 2], [3, 5]]), "A3": _m([[1, 2, 3], [4, 5, 7], [7, 8, 10]])},
    "det_negate",
    "Ein Sonderfall von Aufgabe 4 mit λ = −1: es gilt (−1)^n."))

_add(Exercise(
    "SW4.6", "SW4", "Determinante der Inversen",
    "Wie berechnet man det(A⁻¹), wenn det(A) bekannt ist?",
    {"A": _m([[4, 3], [8, 7]])}, "det_inverse",
    "Aus A·A⁻¹ = E folgt det(A)·det(A⁻¹) = 1."))


# ==========================================================================
# Zugriff
# ==========================================================================

def liste(week: str | None = None) -> list[str]:
    """Alle Aufgabenschlüssel, auf Wunsch nur einer Woche."""
    keys = sorted(EXERCISES)
    if week:
        keys = [k for k in keys if EXERCISES[k].week == week.upper()]
    return keys


def aufgabe(key: str) -> Exercise:
    """Eine Aufgabe mit ihren Daten."""
    if key not in EXERCISES:
        near = [k for k in EXERCISES if k.lower().startswith(key.lower()[:5])]
        hint = f" Gemeint war vielleicht: {', '.join(near[:5])}" if near else ""
        raise KeyError(f"Aufgabe '{key}' gibt es nicht.{hint}")
    return EXERCISES[key]


def loese(key: str, weg: bool = True, decimal: bool = False):
    """
    Rechnet eine Aufgabe durch.

    Gedacht zum Kontrollieren, nachdem du selbst gerechnet hast — der
    Rechenweg ist dieselbe Abfolge, die auf dem Papier stünde.
    """
    from ..rechnen.factor import lu, solve_lu
    from ..rechnen.inverse import inverse
    from ..rechnen.determinant import det
    from ..rechnen.solving import gauss_jordan
    from ..rechnen.vectors import Line, Plane, angle, cross, dot

    ex = aufgabe(key)
    data = ex.data

    if ex.method == "gauss":
        return gauss_jordan(data["A|b"], weg=weg, decimal=decimal)

    if ex.method == "lu":
        decomposition = lu(data["A"], decimal=decimal)
        if "b" in data:
            return solve_lu(decomposition, data["b"], decimal=decimal)
        return decomposition

    if ex.method == "inverse":
        return inverse(data["A"], decimal=decimal)

    if ex.method == "lines":
        g = Line(*data["g"])
        h = Line(*data["h"])
        return g.intersect(h, decimal=decimal)

    if ex.method == "line_plane":
        g = Line(*data["g"])
        point, u, v = data["E"]
        return Plane(point, u=u, v=v).intersect_line(g, decimal=decimal)

    if ex.method == "vectors":
        from ..rechnen.render import Result, Step
        u, v = data["u"], data["v"]
        out = Result(decimal)
        out.title = ex.title
        out.summary = ex.task
        out.steps = [
            Step("Längen",
                 detail=(f"|u| = {u.norm()}, \\quad |v| = {v.norm()}"),
                 items=[("u", u), ("v", v)]),
            Step("Skalarprodukt", detail=f"u \\cdot v = {u.dot(v)}"),
            Step("Winkel", detail=f"\\varphi = {u.angle(v):.4f}^\\circ"),
            Step("Vektorprodukt", items=[("u \\times v", u.cross(v))]),
        ]
        out.highlights = [(r"u \cdot v", u.dot(v)),
                          (r"u \times v", u.cross(v))]
        return out

    if ex.method in ("det_scale", "det_negate", "det_inverse"):
        return _det_property(ex, decimal)

    if ex.method == "program":
        raise ValueError(
            "Diese Aufgabe sollst du selbst programmieren. Das Gerüst mit "
            "Tests steht in lialkit.uebung — ruf dort pruefe_lr(deine_funktion) "
            "auf.")

    raise ValueError(f"Für '{ex.method}' gibt es noch keinen Rechenweg")


def _det_property(ex: Exercise, decimal: bool):
    """Die drei Determinanten-Eigenschaften aus SW4 am Beispiel."""
    from ..rechnen.determinant import det
    from ..rechnen.inverse import inverse
    from ..rechnen.render import Result, Step

    out = Result(decimal)
    out.title = ex.title
    out.summary = ex.task

    if ex.method == "det_scale":
        A, lam = ex.data["A"], ex.data["lambda"]
        base = det(A).value
        scaled = det(A.scaled(lam)).value
        size = A.n
        out.steps = [
            Step("Determinante von A", detail=f"\\det(A) = {base}", items=[("A", A)]),
            Step(f"Jede Zeile mit {lam} multiplizieren",
                 items=[(f"{lam}A", A.scaled(lam))],
                 note=f"Das sind {size} Zeilen, also {size} Faktoren {lam}."),
            Step("Ergebnis",
                 detail=(f"\\det({lam}A) = {lam}^{{{size}}} \\cdot \\det(A) = "
                         f"{lam ** size} \\cdot {base} = {scaled}")),
            Step("Anschauung",
                 note=f"Die Determinante misst ein Volumen. Streckt man alle "
                      f"{size} Richtungen um {lam}, wächst das Volumen um "
                      f"{lam}^{size} = {lam ** size}."),
        ]
        out.highlights = [(f"\\det({lam}A)", scaled)]
        return out

    if ex.method == "det_negate":
        A2, A3 = ex.data["A2"], ex.data["A3"]
        d2, d2n = det(A2).value, det(A2.scaled(-1)).value
        d3, d3n = det(A3).value, det(A3.scaled(-1)).value
        out.steps = [
            Step("2×2 geprüft",
                 detail=(f"\\det(A) = {d2}, \\quad \\det(-A) = {d2n}, \\quad "
                         f"-\\det(A) = {-d2}"),
                 items=[("A", A2)],
                 note=("Hier gilt det(−A) = +det(A) — die Behauptung stimmt "
                       "also nicht.")),
            Step("3×3 geprüft",
                 detail=(f"\\det(A) = {d3}, \\quad \\det(-A) = {d3n}, \\quad "
                         f"-\\det(A) = {-d3}"),
                 items=[("A", A3)],
                 note="Hier stimmt sie."),
            Step("Allgemein",
                 detail=r"\det(-A) = (-1)^n \det(A)",
                 note="Die Aussage gilt genau dann, wenn n ungerade ist. "
                      "Bei geradem n ist det(−A) = +det(A)."),
        ]
        out.summary = ("Nein — es gilt det(−A) = (−1)ⁿ·det(A), die Behauptung "
                       "stimmt nur für ungerades n.")
        return out

    A = ex.data["A"]
    base = det(A).value
    inv = inverse(A)
    inv_det = det(inv.inverse).value
    out.steps = [
        Step("Determinante von A", detail=f"\\det(A) = {base}", items=[("A", A)]),
        Step("Produktregel anwenden",
             detail=r"A \cdot A^{-1} = E \;\Rightarrow\; "
                    r"\det(A) \cdot \det(A^{-1}) = 1"),
        Step("Umstellen",
             detail=(r"\det(A^{-1}) = \frac{1}{\det(A)} = "
                     r"\frac{1}{" + str(base) + "} = " + str(inv_det)),
             items=[("A^{-1}", inv.inverse)],
             note="Nachgerechnet stimmt es mit der direkten Determinante "
                  "der Inversen überein."),
    ]
    out.highlights = [(r"\det(A^{-1})", inv_det)]
    return out
