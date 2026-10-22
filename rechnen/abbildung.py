#!/usr/bin/env python3
"""
Lineare Abbildungen (SW 5): Matrizen der Standardabbildungen, Bildpunkte,
Zusammensetzung, Linearitätsprüfung und die beiden Aufgaben des Blatts.

Die Spalten einer Matrix sind die Bilder der Einheitsvektoren; kennt man
sie, kennt man die ganze Abbildung (Folie 8). Punkte stehen spaltenweise in
einer Matrix P, ihre Bilder sind Q = A·P.

Einträge wie cos 30° sind irrational. Solche Matrizen halten hier eine
Näherung (auf rund neun Stellen); die Ergebnisse zeigen dann vier Stellen
und sagen es dazu. Bei Vielfachen von 90° bleibt alles exakt.

Das Vektorprodukt (ein Lernziel von SW 5) steht schon in vectors.py:
`cross(u, v)` und `Vector.cross`. Hier wird es nicht noch einmal gebaut.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Sequence

from .core import Matrix, Num, Vector, parse_number
from .render import Result, Step, tex_matrix, tex_number, tex_vector

# ==========================================================================
# Beispieldaten aus Folien, Blatt und Musterlösung
# ==========================================================================

#: Buchstabe F der Folien 4–7: zehn Eckpunkte, spaltenweise.
BUCHSTABE_F_FOLIEN = Matrix([[1, 2, 2, 3, 3, 2, 2, 4, 4, 1],
                             [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])

#: Buchstabe F aus Aufgabe 2 (Musterlösung): elf Punkte, geschlossen.
BUCHSTABE_F_AUFGABE = Matrix([[0, 1, 1, 2, 2, 1, 1, 3, 3, 0, 0],
                              [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]])

#: Einheitsquadrat als geschlossener Linienzug (Aufgabe 1).
EINHEITSQUADRAT = Matrix([[0, 1, 1, 0, 0],
                          [0, 0, 1, 1, 0]])

#: Die neuen Basisvektoren f1, f2 von Folie 6.
BASIS_FOLIE_6 = ((3, -1), (-1, 2))

#: Winkel der Drehreihe in Aufgabe 2: 0°, 30°, …, 330°.
PHI_REIHE = tuple(range(0, 360, 30))

_NAEHERUNG_AB = 10 ** 6     # shortcut: Nenner ab hier gelten als Näherung


# ==========================================================================
# Hilfen
# ==========================================================================

def _vec(v) -> Vector:
    return v if isinstance(v, Vector) else Vector(v)


def _mat(M) -> Matrix:
    M = M if isinstance(M, Matrix) else Matrix(M)
    return M.coefficients() if M.augmented else M


def _spalten(vektoren: Sequence[Vector]) -> Matrix:
    """Vektoren als Spalten einer Matrix."""
    dims = {len(v) for v in vektoren}
    if len(dims) != 1:
        raise ValueError("Alle Punkte brauchen gleich viele Koordinaten")
    return Matrix([[v[i] for v in vektoren] for i in range(dims.pop())])


def punkte(liste) -> Matrix:
    """
    Punkte als Spalten einer Matrix.

    Eine Matrix bleibt, wie sie ist; ein Vektor wird zur Spalte; eine Liste
    wie [(0, 0), (1, 0), (1, 1)] ergibt die Matrix mit diesen Punkten als
    Spalten.
    """
    if isinstance(liste, Matrix):
        return liste
    if isinstance(liste, Vector):
        return liste.as_matrix()
    return _spalten([_vec(p) for p in liste])


def _naeherung(*objekte) -> bool:
    for o in objekte:
        zeilen = o.rows if isinstance(o, Matrix) else [o.values]
        if any(v.denominator > _NAEHERUNG_AB for r in zeilen for v in r):
            return True
    return False


def _glatt(M: Matrix) -> Matrix:
    """Rundungsstaub entfernen: 1e-18 statt 0 oder 0.9999999999 statt 1."""
    def snap(v: Fraction) -> Fraction:
        s = v.limit_denominator(100)
        return s if abs(v - s) < Fraction(1, 10 ** 12) else v
    return Matrix([[snap(v) for v in r] for r in M.rows])


def _nah(a: Matrix, b: Matrix, tol: float = 1e-9) -> bool:
    return (a.shape == b.shape
            and all(abs(float(x) - float(y)) < tol
                    for ra, rb in zip(a.rows, b.rows) for x, y in zip(ra, rb)))


_NAEHERUNGS_HINWEIS = ("Die Einträge sind irrational (Wurzeln aus cos/sin) und "
                       "hier auf vier Stellen gerundet angezeigt.")


# ==========================================================================
# Matrizen der Standardabbildungen (Folien 10–16)
# ==========================================================================

def skalierung(sx, sy=None) -> Matrix:
    """Skalierung diag(sx, sy); mit einem Argument gleichmässig (Folie 10)."""
    sx = parse_number(sx)
    sy = sx if sy is None else parse_number(sy)
    return Matrix([[sx, 0], [0, sy]])


def drehung(winkel_grad) -> Matrix:
    """
    Drehung um den Ursprung, Winkel im Gradmass, gegen den Uhrzeigersinn:
    [[cos φ, −sin φ], [sin φ, cos φ]] (Folie 12).

    Bei Vielfachen von 90° exakt. Sonst sind cos und sin Näherungen
    (rund neun Stellen), ausser dort, wo sie rational sind (cos 60° = 1/2).
    """
    g = parse_number(winkel_grad)
    k = g % 360
    if k % 90 == 0:
        c, s = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[int(k)]
    else:
        t = math.radians(float(g))
        c = Fraction(math.cos(t)).limit_denominator(10 ** 9)
        s = Fraction(math.sin(t)).limit_denominator(10 ** 9)
    return Matrix([[c, -s], [s, c]])


#: Aufgabe 2: Rotationsmatrix zu einem Winkel im Gradmass.
rotationsmatrix = drehung


def drehstreckung(a, b) -> Matrix:
    """
    Drehstreckung [[a, −b], [b, a]] (Folie 11): Drehung um arctan(b/a),
    gestreckt um √(a² + b²). Folie 11 hat a = b = 1: 45° und Faktor √2.
    """
    a, b = parse_number(a), parse_number(b)
    return Matrix([[a, -b], [b, a]])


def scherung(k, richtung: str = "x") -> Matrix:
    """Scherung (Folie 16): "x" gibt [[1, k], [0, 1]], "y" gibt [[1, 0], [k, 1]]."""
    k = parse_number(k)
    if richtung == "x":
        return Matrix([[1, k], [0, 1]])
    if richtung == "y":
        return Matrix([[1, 0], [k, 1]])
    raise ValueError("richtung ist 'x' oder 'y'")


def spiegelung_x() -> Matrix:
    """Spiegelung an der x-Achse (Folie 13)."""
    return Matrix([[1, 0], [0, -1]])


def spiegelung_y() -> Matrix:
    """Spiegelung an der y-Achse (Folie 14)."""
    return Matrix([[-1, 0], [0, 1]])


def spiegelung_achse(phi_grad) -> Matrix:
    """
    Spiegelung an der Geraden durch den Ursprung, die mit der x-Achse den
    Winkel φ einschliesst: [[cos 2φ, sin 2φ], [sin 2φ, −cos 2φ]] (Folie 15, 20).
    """
    d = drehung(2 * parse_number(phi_grad))
    c, s = d[0, 0], d[1, 0]
    return Matrix([[c, s], [s, -c]])


def homogen(A=None, t=(0, 0)) -> Matrix:
    """
    3×3-Matrix in homogenen Koordinaten: [[A, t], [0 0 1]].

    A ist 2×2 (Standard: Einheitsmatrix), t die Verschiebung. Damit lässt
    sich auch eine Verschiebung als Matrix schreiben, was ohne die dritte
    Koordinate nicht geht: jede lineare Abbildung lässt den Nullpunkt fest.
    """
    A = Matrix.identity(2) if A is None else _mat(A)
    if A.shape != (2, 2):
        raise ValueError("A muss eine 2×2-Matrix sein")
    t = _vec(t)
    if len(t) != 2:
        raise ValueError("t braucht zwei Einträge")
    return Matrix([[A[0, 0], A[0, 1], t[0]],
                   [A[1, 0], A[1, 1], t[1]],
                   [0, 0, 1]])


def verschiebung(tx, ty) -> Matrix:
    """Verschiebung um (tx, ty) als 3×3-Matrix in homogenen Koordinaten."""
    return homogen(None, (tx, ty))


def homogen_punkte(P) -> Matrix:
    """Hängt eine Zeile mit Einsen an: aus 2×n wird 3×n."""
    P = punkte(P)
    return Matrix(P.rows + [[1] * P.width])


# ==========================================================================
# Punkte abbilden, Basiswechsel (Folien 5–8)
# ==========================================================================

class AbbildungResult(Result):
    title = "Bildpunkte einer linearen Abbildung"

    def __init__(self, A: Matrix, P: Matrix, Q: Matrix, homogen_: bool,
                 decimal: bool = False):
        super().__init__(decimal or _naeherung(A, P))
        self.A, self.P, self.Q = A, P, Q
        self.homogen = homogen_
        self.summary = "Q = A·P: die Spalten von Q sind die Bilder der Spalten von P."
        self.highlights = [("Q", Q)]

    @property
    def bild(self) -> Matrix:
        """Die Bildpunkte zum Zeichnen: bei homogenen Koordinaten ohne die 1-Zeile."""
        return Matrix(self.Q.rows[:2]) if self.homogen else self.Q


def abbilden(A, P, beispiel: int = 0, decimal: bool = False) -> AbbildungResult:
    """
    Bildpunkte Q = A·P, Punkte spaltenweise, mit Rechenweg.

    `beispiel` ist die Spalte von P, an der die Linearkombination
    p₁·A(e₁) + p₂·A(e₂) ausgeschrieben wird. Hat A drei Zeilen und P zwei,
    kommt automatisch die Zeile mit Einsen dazu (homogene Koordinaten).
    """
    A, P = _mat(A), punkte(P)
    steps = []
    homogen_ = A.shape == (3, 3) and A.rows[2] == [0, 0, 1] and (
        P.m == 2 or P.rows[2] == [1] * P.width)
    if A.shape == (3, 3) and P.m == 2:
        P = homogen_punkte(P)
        steps.append(Step(
            "Homogene Koordinaten: Zeile mit Einsen anhängen",
            items=[("P", P)],
            note="Die 3×3-Matrix kann verschieben, eine 2×2-Matrix nie — "
                 "jede lineare Abbildung lässt den Nullpunkt fest."))
    if not 0 <= beispiel < P.width:
        raise ValueError(f"beispiel muss zwischen 0 und {P.width - 1} liegen")

    Q = A @ P
    if _naeherung(A, P):
        Q = _glatt(Q)
    p = P.column(beispiel)
    spalten = [A.column(k) for k in range(A.n)]
    steps.insert(0, Step(
        "Die Spalten von A sind die Bilder der Einheitsvektoren",
        items=[(f"A\\,e_{k + 1}", c) for k, c in enumerate(spalten)],
        note="Kennt man diese Bilder, kennt man die ganze Abbildung."))
    steps.append(Step(
        f"Ein Punkt als Linearkombination (Spalte {beispiel + 1} von P)",
        detail=" + ".join(f"{tex_number(p[k])} \\cdot {tex_vector(spalten[k])}"
                          for k in range(A.n)),
        items=[("A p", A @ p)],
        note="Die Koordinaten von p sind die Faktoren vor den Bildern der "
             "Einheitsvektoren."))
    steps.append(Step("Alle Punkte auf einmal", detail=r"Q = A \cdot P",
                      items=[("A", A), ("P", P), ("Q", Q)]))
    out = AbbildungResult(A, P, Q, homogen_, decimal)
    if _naeherung(A, P):
        steps[-1].note = _NAEHERUNGS_HINWEIS
    out.steps = steps
    return out


def basiswechsel(f1, f2, P, beispiel: int = 0, decimal: bool = False) -> AbbildungResult:
    """
    Dieselben Punkte bezüglich der Basis (f₁, f₂) statt (e₁, e₂) (Folien 6–7).

    Die Matrix mit f₁, f₂ als Spalten leistet es: Q = A·P. Folie 6:
    w = 3·f₁ + 2·f₂ = (7, 1).
    """
    A = _spalten([_vec(f1), _vec(f2)])
    out = abbilden(A, P, beispiel, decimal)
    out.title = "Vektoren bezüglich einer anderen Basis"
    out.steps[0].title = "Die neuen Basisvektoren f₁, f₂ als Spalten"
    return out


# ==========================================================================
# Zusammensetzung (Folien 18–20)
# ==========================================================================

class ZusammensetzungResult(Result):
    title = "Zusammensetzung linearer Abbildungen"

    def __init__(self, matrizen, C: Matrix, decimal: bool = False):
        super().__init__(decimal or _naeherung(C, *matrizen))
        self.matrizen, self.matrix = list(matrizen), C
        self.summary = ("Die zuerst angewendete Abbildung steht im Produkt "
                        "rechts: C = B·A heisst erst A, dann B.")
        self.highlights = []


def zusammensetzen(*matrizen, namen: Sequence[str] | None = None, P=None,
                   decimal: bool = False) -> ZusammensetzungResult:
    """
    Setzt Abbildungen hintereinander zusammen. Die Matrizen stehen in der
    Reihenfolge, in der die Abbildungen wirken: zusammensetzen(A, B) = B·A.

    Mit P werden zur Probe auch die Punkte abgebildet.
    """
    Ms = [_mat(m) for m in matrizen]
    n = len(Ms)
    if n < 2:
        raise ValueError("Zum Zusammensetzen braucht es mindestens zwei Matrizen")
    if namen is None:
        namen = ["A", "B"] if n == 2 else [f"A_{{{i + 1}}}" for i in range(n)]
    elif len(namen) != n:
        raise ValueError("Zu jeder Matrix gehört ein Name")
    name_c = "C" if n == 2 else "M"

    kette = "x" + "".join(f" \\xrightarrow{{{namen[i]}}} x_{{{i + 1}}}"
                          for i in range(n))
    steps = [Step("Reihenfolge der Abbildungen", detail=kette,
                  items=list(zip(namen, Ms)),
                  note="Die zuerst angewendete Abbildung steht im Produkt "
                       "rechts, die zuletzt angewendete links.")]
    acc, label = Ms[0], namen[0]
    for k in range(1, n):
        acc = Ms[k] @ acc          # wirft ValueError, wenn die Formate nicht passen
        if _naeherung(*Ms):
            acc = _glatt(acc)
        label = f"{namen[k]} \\cdot {label}"
        steps.append(Step(f"Mit {namen[k]} von links multiplizieren",
                          items=[(label, acc)]))
    out = ZusammensetzungResult(Ms, acc, decimal)
    steps.append(Step("Ergebnis", detail=f"{name_c} = {label}",
                      items=[(name_c, acc)]))
    out.highlights = [(name_c, acc)]
    if _naeherung(*Ms):
        steps[-1].note = _NAEHERUNGS_HINWEIS
    if n == 2 and Ms[0].shape == Ms[1].shape and Ms[0].is_square():
        umgekehrt = Ms[0] @ Ms[1]
        steps.append(Step(
            "Reihenfolge vertauscht?", items=[(f"{namen[0]} \\cdot {namen[1]}", umgekehrt)],
            note=("Eine andere Abbildung — die Reihenfolge zählt."
                  if not _nah(umgekehrt, acc) else
                  "Hier kommt dasselbe heraus; im Allgemeinen ist AB ≠ BA.")))
    if P is not None:
        P = punkte(P)
        schritt = P
        for M in Ms:
            schritt = M @ schritt
        direkt = acc @ P
        steps.append(Step(
            "Probe an den Punkten", detail=r"M_n \cdots M_1 P \overset{?}{=} C P",
            items=[("C\\,P", direkt)],
            note="Stimmt mit dem schrittweisen Abbilden überein."
                 if _nah(schritt, direkt) else "ACHTUNG: weicht ab."))
    out.steps = steps
    return out


def spiegelung_achse_weg(phi_grad, decimal: bool = False) -> ZusammensetzungResult:
    """
    Spiegelung an einer Achse durch den Ursprung als Zusammensetzung (Folie 20):
    Drehung um −φ, Spiegelung an der x-Achse, Drehung um φ.
    """
    phi = parse_number(phi_grad)
    out = zusammensetzen(drehung(-phi), spiegelung_x(), drehung(phi),
                         namen=(r"D_{-\varphi}", "S_x", r"D_{\varphi}"),
                         decimal=decimal)
    closed = spiegelung_achse(phi)
    out.title = "Spiegelung an einer Achse durch den Ursprung"
    out.steps.append(Step(
        "Geschlossene Form",
        detail=(r"\begin{pmatrix}\cos 2\varphi & \sin 2\varphi \\ "
                r"\sin 2\varphi & -\cos 2\varphi\end{pmatrix}"),
        items=[(r"S_\varphi", closed)],
        note=("Stimmt mit dem Produkt überein." if _nah(closed, out.matrix)
              else "ACHTUNG: weicht vom Produkt ab.")))
    out.highlights = [(r"S_\varphi", closed)]
    return out


# ==========================================================================
# Die transponierte Abbildung (Notiz SW 6: »transponiert anschauen«)
# ==========================================================================

class TransponierteResult(Result):
    title = "Die transponierte Abbildung"

    def __init__(self, A: Matrix, At: Matrix, Q: Matrix, Qt: Matrix,
                 eigenschaft: str, decimal: bool = False):
        super().__init__(decimal or _naeherung(A, Q))
        self.A, self.At, self.Q, self.Qt = A, At, Q, Qt
        self.eigenschaft = eigenschaft
        self.summary = eigenschaft
        self.highlights = [("A^T", At), ("A^T P", Qt)]

    @property
    def bild(self) -> Matrix:
        """Die Bildpunkte von A^T zum Zeichnen (bei homogenen Koordinaten ohne die 1-Zeile)."""
        return Matrix(self.Qt.rows[:2]) if self.Qt.m == 3 else self.Qt


def transponierte_abbildung(A, P=None, decimal: bool = False) -> TransponierteResult:
    """
    Was A^T mit denselben Punkten macht wie A (quadratische A, Standard: Einheitsquadrat).

    Die Zeilen von A werden zu den Spalten von A^T, also zu den Bildern von
    e₁, e₂. Gleich bleibt |det|: die Fläche ändert sich um denselben Faktor.
    Sonst ist A^T im Allgemeinen eine andere Abbildung; bei symmetrischem A
    dieselbe, bei Drehungen und Spiegelungen (A^T·A = E) die Umkehrung.
    """
    A = _mat(A)
    if not A.is_square():
        raise ValueError("Die transponierte Abbildung gibt es hier nur für quadratische Matrizen")
    P = EINHEITSQUADRAT if P is None else punkte(P)
    At = A.T
    Q, Qt = abbilden(A, P).Q, abbilden(At, P).Q
    if A.rows == At.rows:
        eig = "A ist symmetrisch: A^T = A, beide Abbildungen sind dieselbe."
    elif _nah(At @ A, Matrix.identity(A.n)):
        eig = ("A^T·A = E: A^T ist die Umkehrung von A (A^T = A⁻¹); "
               "sie macht A wieder rückgängig.")
    else:
        eig = ("A^T ist eine andere Abbildung als A: die Bilder von e₁, e₂ "
               "sind die Zeilen von A.")
    steps = [
        Step("Zeilen und Spalten vertauschen",
             items=[("A", A), ("A^T", At)],
             note="Die Spalten von A^T sind die Zeilen von A, also die Bilder "
                  "von e₁ und e₂ unter A^T."),
        Step("Dieselben Punkte, zwei Abbildungen",
             detail=r"Q = A \cdot P \qquad Q' = A^T \cdot P",
             items=[("P", P), ("A\\,P", Q), ("A^T P", Qt)]),
    ]
    if A.n == 2:
        dA, dAt = A.rows[0][0] * A.rows[1][1] - A.rows[0][1] * A.rows[1][0], \
            At.rows[0][0] * At.rows[1][1] - At.rows[0][1] * At.rows[1][0]
        steps.append(Step("Fläche", detail=f"\\det A = {tex_number(dA)} = \\det A^T = {tex_number(dAt)}",
                          note="Die Determinante ändert sich beim Transponieren nicht."))
    steps.append(Step("Einordnung", note=eig))
    out = TransponierteResult(A, At, Q, Qt, eig, decimal)
    out.steps = steps
    return out


# ==========================================================================
# Ist eine Abbildung linear? (Folie 8)
# ==========================================================================

class LinearResult(Result):
    title = "Ist die Abbildung linear?"

    def __init__(self, linear: bool, summary: str, decimal: bool = False):
        super().__init__(decimal)
        self.linear, self.summary = linear, summary

    def __bool__(self) -> bool:
        return self.linear


def pruefe_linear(f, dim: int = 2, vektoren=None, skalare=(2, -3),
                  decimal: bool = False) -> LinearResult:
    """
    Prüft f(u+v) = f(u)+f(v) und f(s·u) = s·f(u) an Beispielen.

    f ist eine Matrix oder eine Funktion Vektor → Vektor (dim Einträge).
    Ein verletztes Beispiel beweist, dass f nicht linear ist; erfüllte
    Beispiele beweisen die Linearität nicht. Notwendig ist f(0) = 0, denn
    f(0) = f(0·u) = 0·f(u) = 0.
    """
    if isinstance(f, Matrix):
        f = _mat(f)
        dim, F = f.n, (lambda v: f @ v)
    else:
        F = lambda v: _vec(f(v))                              # noqa: E731
    if vektoren is None:
        u = Vector(range(1, dim + 1))
        v = Vector([(-1) ** i * (dim + 1 - i) for i in range(dim)])
        w = Vector([(-1) ** (i + 1) * (i + 2) for i in range(dim)])
        vektoren = [u, v, w]
    vs = [_vec(x) for x in vektoren]
    if len(vs) < 2 or any(len(x) != dim for x in vs):
        raise ValueError(f"Mindestens zwei Beispielvektoren mit je {dim} Einträgen")

    fehler: list[str] = []
    steps = []
    null = F(Vector([0] * dim))
    if not null.is_zero():
        fehler.append("f(0) ist nicht der Nullvektor")
    steps.append(Step(
        "Der Nullvektor muss auf sich selbst abgebildet werden",
        detail=r"f(0) = f(0 \cdot u) = 0 \cdot f(u) = 0",
        items=[("f(0)", null)],
        note="Erfüllt." if null.is_zero() else
             "VERLETZT — damit ist f nicht linear. (Verschiebungen sind nie "
             "linear; mit homogenen Koordinaten lassen sie sich trotzdem als "
             "Matrix schreiben.)"))
    for u, v in [(vs[0], vs[1])] + ([(vs[0], vs[2])] if len(vs) > 2 else []):
        lhs, rhs = F(u + v), F(u) + F(v)
        if lhs != rhs:
            fehler.append("f(u+v) ≠ f(u)+f(v)")
        steps.append(Step(
            "Additivität", detail=r"f(u+v) \overset{?}{=} f(u) + f(v)",
            items=[("u", u), ("v", v), ("f(u+v)", lhs), ("f(u)+f(v)", rhs)],
            note="Erfüllt." if lhs == rhs else "VERLETZT."))
    for s, u in zip(skalare, vs):
        s = parse_number(s)
        lhs, rhs = F(u.scaled(s)), F(u).scaled(s)
        if lhs != rhs:
            fehler.append("f(s·u) ≠ s·f(u)")
        steps.append(Step(
            f"Homogenität mit s = {tex_number(s)}",
            detail=r"f(s \cdot u) \overset{?}{=} s \cdot f(u)",
            items=[("u", u), ("f(s u)", lhs), ("s f(u)", rhs)],
            note="Erfüllt." if lhs == rhs else "VERLETZT."))

    if fehler:
        summary = f"Nicht linear: {fehler[0]}."
    elif isinstance(f, Matrix):
        summary = ("Linear — die Multiplikation einer Matrix mit einem Vektor "
                   "erfüllt beide Bedingungen immer.")
    else:
        summary = ("Auf diesen Beispielen erfüllt. Ein Beweis ist das nicht, "
                   "ein einziges Gegenbeispiel würde aber genügen.")
    out = LinearResult(not fehler, summary, decimal)
    out.steps = steps
    return out


def ist_linear(f, **kwargs) -> bool:
    """Kurzform von pruefe_linear: True, wenn kein Beispiel die Linearität verletzt."""
    return pruefe_linear(f, **kwargs).linear


# ==========================================================================
# Aufgabe 1: Gibt es eine lineare Abbildung zu diesem Bild?
# ==========================================================================

class FindeResult(Result):
    title = "Gibt es eine lineare Abbildung?"

    def __init__(self, Q: Matrix, decimal: bool = False):
        super().__init__(decimal)
        self.Q = Q
        self.linear = False
        self.A: Matrix | None = None      # 2×2, falls linear
        self.H: Matrix | None = None      # 3×3, falls nur mit homogenen Koordinaten

    @property
    def affin(self) -> bool:
        """Lässt sich das Bild wenigstens mit homogenen Koordinaten beschreiben?"""
        return self.linear or self.H is not None


def finde_abbildung(bild_punkte, decimal: bool = False) -> FindeResult:
    """
    Aufgabe 1: Das Einheitsquadrat (0,0), (1,0), (1,1), (0,1), (0,0) wird auf
    die gegebenen Punkte abgebildet. Gibt es eine lineare Abbildung?

    Ja: A hat die Bilder von e₁ und e₂ als Spalten, und A·(1,1) muss das
    Bild der dritten Ecke sein. Nein: wird 0 nicht auf 0 abgebildet (oder
    stimmt (1,1) nicht), lässt sich das nicht durch eine 2×2-Matrix
    beschreiben; ist das Bild ein Parallelogramm, liefert `.H` die
    3×3-Matrix in homogenen Koordinaten.
    """
    Q = punkte(bild_punkte)
    if Q.m != 2 or Q.width not in (4, 5):
        raise ValueError("Gebraucht werden die Bilder der Ecken (0,0), (1,0), "
                         "(1,1), (0,1) und zum Schluss wieder (0,0)")
    q0, q1, q2, q3 = (Q.column(i) for i in range(4))
    if Q.width == 5 and Q.column(4) != q0:
        raise ValueError("Der Linienzug muss geschlossen sein: der fünfte "
                         "Punkt ist das Bild von (0,0) wie der erste")
    P = Matrix(EINHEITSQUADRAT.rows) if Q.width == 5 else \
        Matrix([r[:4] for r in EINHEITSQUADRAT.rows])
    out = FindeResult(Q, decimal)
    steps = [Step("Einheitsquadrat und gegebene Figur", items=[("P", P), ("Q", Q)],
                  note="Eine lineare Abbildung ist durch die Bilder von e₁ = (1,0) "
                       "und e₂ = (0,1) festgelegt (sie bilden die Spalten von A).")]
    out.steps = steps

    def probe_ecke(t: Vector, a: Vector, b: Vector) -> bool:
        return t + a + b == q2

    if q0.is_zero():
        A = Matrix([[q1[0], q3[0]], [q1[1], q3[1]]])
        steps.append(Step("Der Nullpunkt bleibt fest: f(0) = 0", items=[("f(0,0)", q0)]))
        steps.append(Step("Matrix aus den Bildern von e₁ und e₂", items=[("A", A)]))
        ecke = A @ Vector([1, 1])
        steps.append(Step(
            "Probe an der Ecke (1,1)", detail=r"A\,(1,1)^T = A e_1 + A e_2",
            items=[("A (1,1)", ecke), ("Bild laut Figur", q2)],
            note="Stimmt überein." if ecke == q2 else
                 "Stimmt NICHT überein: wegen (1,1) = e₁ + e₂ müsste f(1,1) = "
                 "f(e₁) + f(e₂) gelten."))
        if ecke == q2:
            out.linear, out.A = True, A
            out.summary = "Ja, es gibt eine lineare Abbildung."
            out.highlights = [("A", A)]
            return out
        out.summary = ("Nein: Additivität verletzt, es gibt keine lineare "
                       "Abbildung (auch mit homogenen Koordinaten nicht: das "
                       "Bild ist kein Parallelogramm).")
        return out

    steps.append(Step(
        "Der Nullpunkt wird nicht auf sich selbst abgebildet",
        detail=r"f(0) = 0 \text{ wäre nötig}", items=[("f(0,0)", q0)],
        note="Wegen f(0) = f(0·u) = 0·f(u) = 0 kann es keine lineare Abbildung "
             "geben."))
    a, b = q1 - q0, q3 - q0
    steps.append(Step(
        "Mit homogenen Koordinaten geht es trotzdem?",
        detail=r"\begin{pmatrix}x'\\y'\\1\end{pmatrix} = "
               r"\begin{pmatrix}A & t\\0\;0 & 1\end{pmatrix}"
               r"\begin{pmatrix}x\\y\\1\end{pmatrix}",
        items=[("t = f(0,0)", q0), ("A e_1", a), ("A e_2", b)],
        note="Verschiebung t = Bild des Nullpunkts; die Spalten von A sind die "
             "Bilder von e₁ und e₂ minus t."))
    if not probe_ecke(q0, a, b):
        out.summary = ("Nein, auch mit homogenen Koordinaten nicht: das Bild ist "
                       "kein Parallelogramm.")
        return out
    H = homogen(Matrix([[a[0], b[0]], [a[1], b[1]]]), q0)
    Ph, Qh = homogen_punkte(P), homogen_punkte(Q)
    steps.append(Step(
        "Probe: H · P_hom = Q_hom", items=[("H", H), ("H P_hom", H @ Ph)],
        note="Stimmt mit den gegebenen Punkten überein." if H @ Ph == Qh
             else "ACHTUNG: weicht ab."))
    if a[0] * b[1] - a[1] * b[0] == 0:
        out.warnings.append("Die Figur ist entartet (Fläche 0).")
    out.H = H
    out.summary = ("Keine lineare Abbildung (der Nullvektor wird nicht auf sich "
                   "selbst abgebildet), aber mit homogenen Koordinaten.")
    out.highlights = [("H", H)]
    return out


# ==========================================================================
# Aufgabe 2: Drehreihe des Buchstabens F
# ==========================================================================

class RotationsreiheResult(Result):
    title = "Drehreihe des Buchstabens F"

    def __init__(self, variante: str, reihe, decimal: bool = True):
        super().__init__(decimal)
        self.variante, self.reihe = variante, reihe

    def folgen(self) -> list[tuple[Num, list[float], list[float]]]:
        """(φ, x-Werte, y-Werte) je Winkel — die Punktfolgen zum Zeichnen."""
        return [(phi, [float(v) for v in M.rows[0]], [float(v) for v in M.rows[1]])
                for phi, M in self.reihe]


def _plus(F: Matrix, T) -> Matrix:
    """F + T, die Spalte T zu jedem Punkt addiert (wie numpy-Broadcasting)."""
    T = _vec(T)
    return Matrix([[v + T[i] for v in row] for i, row in enumerate(F.rows)])


def rotationsreihe(variante: str = "a", F=BUCHSTABE_F_AUFGABE, T=(15, 0),
                   winkel=PHI_REIHE, decimal: bool = True) -> RotationsreiheResult:
    """
    Aufgabe 2. Variante "a": Rot(φ)·(F + T), erst verschieben, dann drehen.
    Variante "b": F + Rot(φ)·T, nur der Verschiebungsvektor wird gedreht.
    """
    variante = str(variante).lower()
    if variante not in ("a", "b"):
        raise ValueError("variante ist 'a' oder 'b'")
    F, T = punkte(F), _vec(T)
    reihe = []
    for phi in winkel:
        R = rotationsmatrix(phi)
        M = R @ _plus(F, T) if variante == "a" else _plus(F, R @ T)
        reihe.append((phi, _glatt(M)))
    out = RotationsreiheResult(variante, reihe, decimal)
    beispiel = next((p for p, _ in reihe if p == 30), reihe[-1][0])
    R = rotationsmatrix(beispiel)
    vorschrift = (r"F_\varphi = \mathrm{Rot}(\varphi)\,(F + T)" if variante == "a"
                  else r"F_\varphi = F + \mathrm{Rot}(\varphi)\,T")
    zwischen = (_plus(F, T) if variante == "a" else R @ T)
    out.steps = [
        Step("Rotationsmatrix",
             detail=r"\mathrm{Rot}(\varphi) = \begin{pmatrix}\cos\varphi & "
                    r"-\sin\varphi\\ \sin\varphi & \cos\varphi\end{pmatrix}",
             items=[(f"Rot({tex_number(beispiel)}°)", R)],
             note=_NAEHERUNGS_HINWEIS),
        Step("Vorschrift", detail=vorschrift,
             items=[("F", F), ("T", T)],
             note=f"Für φ = {', '.join(str(p) for p in winkel)} Grad."),
        Step(f"Beispiel φ = {tex_number(beispiel)}°",
             items=[("F + T" if variante == "a" else "Rot(φ) T", zwischen),
                    (r"F_\varphi", dict(reihe)[beispiel])]),
    ]
    out.summary = (
        "Erst verschieben, dann alles drehen: der Buchstabe dreht sich mit und "
        "wandert auf einem Kreis mit Radius |T| = 15 um den Ursprung."
        if variante == "a" else
        "Nur der Verschiebungsvektor wird gedreht: der Buchstabe behält seine "
        "Ausrichtung und wandert auf einem Kreis mit Radius |T| = 15.")
    return out
