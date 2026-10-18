#!/usr/bin/env python3
"""Tests für lialkit.abbildung (SW 5), gegen Folien und Musterlösung. Aufruf: python3 tests/tests_abbildung.py"""
import math
import os
import sys
import tempfile
from fractions import Fraction as Fr

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from lialkit.aufgaben.exercises import EXERCISES, aufgabe, liste, loese
from lialkit.rechnen.abbildung import (BASIS_FOLIE_6, BUCHSTABE_F_AUFGABE,
                                       BUCHSTABE_F_FOLIEN, EINHEITSQUADRAT, PHI_REIHE,
                                       abbilden, basiswechsel, drehstreckung, drehung,
                                       finde_abbildung, homogen, homogen_punkte,
                                       ist_linear, pruefe_linear, punkte,
                                       rotationsmatrix, rotationsreihe, scherung,
                                       skalierung, spiegelung_achse,
                                       spiegelung_achse_weg, spiegelung_x, spiegelung_y,
                                       verschiebung, zusammensetzen)
from lialkit.rechnen.core import Matrix, Vector

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def M(rows):
    return Matrix(rows)


def werte(A):
    return [[float(v) for v in r] for r in A.rows]


def nah(A, B, tol=1e-9):
    a, b = werte(A), B if isinstance(B, list) else werte(B)
    return len(a) == len(b) and all(abs(x - y) < tol for ra, rb in zip(a, b)
                                    for x, y in zip(ra, rb))


def faellt(f, *args, **kw):
    try:
        f(*args, **kw)
    except ValueError:
        return True
    return False


def alles_anzeigen(res, name):
    """Jedes Ergebnis hat Schritte und lässt sich als Text, HTML und LaTeX ausgeben."""
    check(getattr(res, "steps", None), f"{name}: keine Schritte")
    check(repr(res) and res._repr_html_() and res.latex(), f"{name}: Anzeige")
    check(res.as_decimal().latex() and res.as_exact() is res, f"{name}: as_decimal/as_exact")


try:
    import numpy as np
except ImportError:
    np = None

# --------------------------------------------------------------------------
print("Standardabbildungen (Folien 10-16)")
check(skalierung(2).rows == M([[2, 0], [0, 2]]).rows, "Skalierung 2")
check(skalierung(2, 3).rows == M([[2, 0], [0, 3]]).rows, "Skalierung 2,3")
check(drehstreckung(1, 1).rows == M([[1, -1], [1, 1]]).rows, "Drehstreckung (Folie 11)")
check(scherung(1).rows == M([[1, 1], [0, 1]]).rows, "Scherung (Folie 16)")
check(scherung(2, "y").rows == M([[1, 0], [2, 1]]).rows, "Scherung in y")
check(faellt(scherung, 1, "z"), "Scherung: falsche Richtung")
check(spiegelung_x().rows == M([[1, 0], [0, -1]]).rows, "Spiegelung x (Folie 13)")
check(spiegelung_y().rows == M([[-1, 0], [0, 1]]).rows, "Spiegelung y (Folie 14)")
check(drehung(0).rows == Matrix.identity(2).rows, "Drehung 0 exakt")
check(drehung(90).rows == M([[0, -1], [1, 0]]).rows, "Drehung 90 exakt")
check(drehung(180).rows == M([[-1, 0], [0, -1]]).rows, "Drehung 180 exakt")
check(drehung(-90).rows == M([[0, 1], [-1, 0]]).rows, "Drehung -90 exakt")
check(drehung(450).rows == drehung(90).rows, "Drehung 450 = 90")
check(drehung(60)[0, 0] == Fr(1, 2), "cos 60° = 1/2 exakt")
check(drehung(30)[1, 0] == Fr(1, 2), "sin 30° = 1/2 exakt")
for g in (30, 45, 60, 100, 330, -45):
    t = math.radians(g)
    check(nah(drehung(g), [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]),
          f"Drehung {g}° gegen math")
check(rotationsmatrix(30).rows == drehung(30).rows, "rotationsmatrix = drehung")
for phi in (0, 30, 45, 60, 90, 120):                       # Folie 15 und 20
    t = math.radians(2 * phi)
    check(nah(spiegelung_achse(phi), [[math.cos(t), math.sin(t)], [math.sin(t), -math.cos(t)]]),
          f"spiegelung_achse({phi})")
check(spiegelung_achse(0).rows == spiegelung_x().rows, "Achse 0° = x-Achse")
check(spiegelung_achse(90).rows == spiegelung_y().rows, "Achse 90° = y-Achse")
check(spiegelung_achse(45).rows == M([[0, 1], [1, 0]]).rows, "Achse 45° vertauscht x, y")
check(homogen(M([[1, -1], [0, 2]]), (1, 0)).rows == M([[1, -1, 1], [0, 2, 0], [0, 0, 1]]).rows,
      "homogen (Lösung 1d)")
check(homogen().rows == Matrix.identity(3).rows, "homogen() = Einheit")
check(verschiebung(3, -2).rows == M([[1, 0, 3], [0, 1, -2], [0, 0, 1]]).rows, "verschiebung")
check(faellt(homogen, M([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), "homogen: nur 2×2")

# --------------------------------------------------------------------------
print("Punkte, Bildpunkte, Basiswechsel (Folien 4-7)")
check(BUCHSTABE_F_FOLIEN.shape == (2, 10) and BUCHSTABE_F_AUFGABE.shape == (2, 11),
      "Buchstaben: Formate")
check(BUCHSTABE_F_FOLIEN.column(3) == Vector([3, 2]), "Folie 5: 4. Punkt (3,2)")
check(punkte([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]).rows == EINHEITSQUADRAT.rows,
      "punkte(Liste)")
check(punkte(Vector([1, 2])).rows == [[1], [2]], "punkte(Vektor)")
check(faellt(punkte, [(1, 2), (1, 2, 3)]), "punkte: ungleiche Längen")

A6 = M([[3, -1], [-1, 2]])
res = abbilden(A6, BUCHSTABE_F_FOLIEN, beispiel=3)
Q_FOLIE = [[2, 5, 4, 7, 6, 3, 2, 8, 7, -2], [1, 0, 2, 1, 3, 4, 6, 4, 6, 9]]
check(res.Q.rows == M(Q_FOLIE).rows, "Folie 7: Q = A·P")
check(res.bild.rows == res.Q.rows and not res.homogen, "bild = Q ohne Homogenes")
check(res.steps[1].items[0][1] == Vector([7, 1]), "Folie 6/7: A·(3,2) = (7,1)")
check("3" in res.steps[1].detail and "2" in res.steps[1].detail, "Linearkombination ausgeschrieben")
alles_anzeigen(res, "abbilden")
bw = basiswechsel(*BASIS_FOLIE_6, BUCHSTABE_F_FOLIEN, beispiel=3)
check(bw.Q.rows == M(Q_FOLIE).rows and bw.A.rows == A6.rows, "basiswechsel = Folie 7")
check(bw.title != res.title, "basiswechsel: eigener Titel")
w = Vector([3, 2])
check(w[0] * Vector([3, -1]) + w[1] * Vector([-1, 2]) == Vector([7, 1]), "Folie 6: w = 3f1 + 2f2")
check(faellt(abbilden, A6, BUCHSTABE_F_FOLIEN, beispiel=99), "beispiel ausserhalb")
check(faellt(abbilden, M([[1, 2, 3], [4, 5, 6]]), BUCHSTABE_F_FOLIEN), "abbilden: Formate")
check(abbilden(skalierung(2), EINHEITSQUADRAT).Q.rows == M([[0, 2, 2, 0, 0], [0, 0, 2, 2, 0]]).rows,
      "Lösung 1a: Bild des Quadrats")
# Homogene Koordinaten: 3×3-Matrix auf 2×n-Punkte hängt die Einsen selbst an
hres = abbilden(M([[1, -1, 1], [0, 2, 0], [0, 0, 1]]), EINHEITSQUADRAT)
check(hres.homogen and hres.Q.rows[2] == [1] * 5, "homogen: Einsenzeile")
check(hres.bild.rows == M([[1, 2, 1, 0, 1], [0, 0, 2, 2, 0]]).rows, "Lösung 1d: Bild")
check(homogen_punkte(EINHEITSQUADRAT).rows ==
      [[0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 1]], "Lösung [7]: Q mit Einsen")
alles_anzeigen(hres, "abbilden homogen")
# Näherungen: Anzeige dezimal und mit Hinweis
rres = abbilden(drehung(30), EINHEITSQUADRAT)
check(rres.decimal and "irrational" in rres.steps[-1].note, "Näherung: dezimal + Hinweis")
check(abbilden(drehung(90), EINHEITSQUADRAT).decimal is False, "exakte Drehung bleibt exakt")
check(abbilden(drehung(90), EINHEITSQUADRAT).Q.rows == M([[0, 0, -1, -1, 0], [0, 1, 1, 0, 0]]).rows,
      "Drehung 90° des Quadrats")
check(nah(rres.Q, [[0, 0.866025, 0.366025, -0.5, 0], [0, 0.5, 1.366025, 0.866025, 0]], 1e-6),
      "Drehung 30° des Quadrats")

# --------------------------------------------------------------------------
print("Zusammensetzung (Folien 18-20)")
A, B = scherung(1), skalierung(2, 3)
z = zusammensetzen(A, B)
check(z.matrix.rows == (B @ A).rows, "zusammensetzen(A, B) = B·A")
check(z.matrix.rows != (A @ B).rows, "nicht A·B")
check("rechts" in z.steps[0].note, "Hinweis: zuerst angewendete rechts")
check(any(s.title == "Reihenfolge vertauscht?" for s in z.steps), "Vergleich A·B")
alles_anzeigen(z, "zusammensetzen")
v = Vector([1, 1])
check(z.matrix @ v == B @ (A @ v), "C·x = B(A·x)")
z3 = zusammensetzen(A, B, spiegelung_x())
check(z3.matrix.rows == (spiegelung_x() @ B @ A).rows, "drei Abbildungen: C·B·A")
check(len(z3.matrizen) == 3 and z3.highlights[0][0] == "M", "drei Abbildungen: Aufbau")
zp = zusammensetzen(A, B, P=EINHEITSQUADRAT)
check(zp.steps[-1].title == "Probe an den Punkten" and "Stimmt" in zp.steps[-1].note, "Probe")
check(faellt(zusammensetzen, A), "nur eine Matrix")
check(faellt(zusammensetzen, M([[1, 2, 3]]), M([[1, 2, 3]])), "Formate passen nicht")
check(faellt(zusammensetzen, A, B, namen=["A"]), "falsche Namenszahl")
# Spiegelung an beliebiger Achse = Rot(φ)·S_x·Rot(−φ)
for phi in (0, 30, 45, 60, 90, 120, 15):
    sw = spiegelung_achse_weg(phi)
    check(nah(sw.matrix, spiegelung_achse(phi)), f"Folie 20: Produkt = Formel bei {phi}°")
    check(nah(sw.matrix, werte(drehung(phi) @ spiegelung_x() @ drehung(-phi))),
          f"Folie 20: D(φ)·S·D(−φ) bei {phi}°")
    check("Stimmt" in sw.steps[-1].note, f"Folie 20: Schritt 'Geschlossene Form' bei {phi}°")
alles_anzeigen(spiegelung_achse_weg(30), "spiegelung_achse_weg")
sw45 = spiegelung_achse_weg(45)
check(sw45.matrix.rows == M([[0, 1], [1, 0]]).rows, "45°: Rundungsstaub entfernt, exakt (0 1; 1 0)")
check(sw45.decimal, "45°: Näherung wird dezimal gezeigt")
# Spiegelung ist eine Involution, Drehungen addieren sich
check(spiegelung_achse(30) @ spiegelung_achse(30) == Matrix.identity(2) or
      nah(spiegelung_achse(30) @ spiegelung_achse(30), [[1, 0], [0, 1]]), "S·S = E")
check(nah(zusammensetzen(drehung(30), drehung(60)).matrix, werte(drehung(90))), "Dreh(30)+Dreh(60)=Dreh(90)")

# --------------------------------------------------------------------------
print("Linearität (Folie 8)")
check(pruefe_linear(M([[1, 2], [3, 4]])).linear and ist_linear(skalierung(2)), "Matrix ist linear")
check(pruefe_linear(M([[1, 2, 0], [3, 4, 5]])).linear, "Matrix 2×3 ist linear")
check(ist_linear(lambda v: (2 * v[0], 3 * v[1])), "diag linear")
check(ist_linear(lambda v: (v[0] + v[1], v[0] - v[1])), "x+y, x-y linear")
check(ist_linear(lambda v: Vector([v[0], 0, v[2]]), dim=3), "Projektion im Raum")
verschiebt = pruefe_linear(lambda v: (v[0] + 1, v[1]))
check(not verschiebt and "f(0)" in verschiebt.summary, "Verschiebung: f(0) ≠ 0")
check(verschiebt.steps[0].items[0][1] == Vector([1, 0]), "Verschiebung: f(0) = (1,0)")
check("VERLETZT" in verschiebt.steps[0].note, "Verschiebung: Hinweis")
check(not ist_linear(lambda v: (abs(v[0]), v[1])), "Betrag nicht linear")
check(not ist_linear(lambda v: (v[0] ** 2, v[1])), "Quadrat nicht linear")
check(not ist_linear(lambda v: (v[0] * v[1], v[1])), "Produkt nicht linear")
check(not ist_linear(lambda v: (v[0], v[1] ** 2)), "y² nicht linear")
nl = pruefe_linear(lambda v: (v[0] ** 2, v[1]))
check("Additivität" in nl.summary or "f(u+v)" in nl.summary, "Begründung nennt Additivität")
check(not bool(nl) and bool(pruefe_linear(skalierung(3))), "__bool__")
check("Beweis ist das nicht" in pruefe_linear(lambda v: v).summary, "Callable: ehrlicher Hinweis")
check(faellt(pruefe_linear, lambda v: v, vektoren=[(1, 2)]), "zu wenige Beispiele")
alles_anzeigen(nl, "pruefe_linear")

# --------------------------------------------------------------------------
print("Aufgabe 1 (Musterlösung)")
Q1 = {
    "a": [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
    "b": [(0, 0), (1, 0), (1, 2), (0, 2), (0, 0)],
    "c": [(0, 0), (1, 0), (2, 2), (1, 2), (0, 0)],
    "d": [(1, 0), (2, 0), (1, 2), (0, 2), (1, 0)],
    "e": [(1, 0), (2, 2), (1, 3), (0, 1), (1, 0)],
}
LOES_A = {"a": [[2, 0], [0, 2]], "b": [[1, 0], [0, 2]], "c": [[1, 1], [0, 2]]}
LOES_H = {"d": [[1, -1, 1], [0, 2, 0], [0, 0, 1]], "e": [[1, -1, 1], [2, 1, 0], [0, 0, 1]]}
for k in "abc":
    f = finde_abbildung(Q1[k])
    check(f.linear and f.A.rows == M(LOES_A[k]).rows and f.H is None, f"1{k}: Matrix")
    check(abbilden(f.A, EINHEITSQUADRAT).Q.rows == punkte(Q1[k]).rows, f"1{k}: Probe")
    check(f.summary.startswith("Ja") and f.highlights[0][0] == "A", f"1{k}: Aussage")
    alles_anzeigen(f, f"finde 1{k}")
for k in "de":
    f = finde_abbildung(Q1[k])
    check(not f.linear and f.A is None, f"1{k}: keine lineare Abbildung")
    check(f.H is not None and f.H.rows == M(LOES_H[k]).rows, f"1{k}: homogene Matrix")
    check(f.affin, f"1{k}: affin")
    check("Nullvektor" in f.summary and "homogen" in f.summary, f"1{k}: Begründung")
    check(abbilden(f.H, EINHEITSQUADRAT).bild.rows == punkte(Q1[k]).rows, f"1{k}: H bildet das Quadrat ab")
    alles_anzeigen(f, f"finde 1{k}")
    check(f.steps[1].title.startswith("Der Nullpunkt wird nicht"), f"1{k}: Schritt Nullpunkt")
# Nullpunkt fest, aber (1,1) falsch: weder linear noch affin
f = finde_abbildung([(0, 0), (1, 0), (3, 3), (0, 1), (0, 0)])
check(not f.linear and not f.affin and f.H is None, "Ecke (1,1) passt nicht: keine Abbildung")
check("Additivität" in f.summary, "Additivität verletzt genannt")
# Nullpunkt verschoben UND kein Parallelogramm
f = finde_abbildung([(1, 0), (2, 0), (3, 3), (0, 1), (1, 0)])
check(not f.affin and "kein Parallelogramm" in f.summary, "verschoben und verzerrt")
# Entartetes Bild
f = finde_abbildung([(0, 0), (1, 0), (2, 0), (1, 0), (0, 0)])
check(f.linear and f.A.rows == M([[1, 1], [0, 0]]).rows, "Strecke: lineare Abbildung (singulär)")
# vier Punkte genügen, falsche Eingaben
check(finde_abbildung([(0, 0), (2, 0), (2, 2), (0, 2)]).A.rows == M(LOES_A["a"]).rows, "4 Punkte")
check(faellt(finde_abbildung, [(0, 0), (1, 0)]), "zu wenige Punkte")
check(faellt(finde_abbildung, [(0, 0), (1, 0), (1, 1), (0, 1), (5, 5)]), "nicht geschlossen")
check(faellt(finde_abbildung, [(0, 0, 0)] * 5), "3D abgelehnt")

# --------------------------------------------------------------------------
print("Aufgabe 2 (Musterlösung)")
check(len(PHI_REIHE) == 12 and PHI_REIHE[0] == 0 and PHI_REIHE[-1] == 330, "Phi = 0, 30, …, 330")
F = BUCHSTABE_F_AUFGABE
check(F.rows == M([[0, 1, 1, 2, 2, 1, 1, 3, 3, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]]).rows,
      "F wie in der Lösung")
ra, rb = rotationsreihe("a"), rotationsreihe("b")
check(len(ra.reihe) == 12 and len(rb.reihe) == 12, "zwölf Bilder")
check([p for p, _ in ra.reihe] == list(PHI_REIHE), "Winkel in der Reihe")
T = Vector([15, 0])
check(ra.reihe[0][1].rows == M([[x + 15 for x in F.rows[0]], F.rows[1]]).rows, "a) φ=0: F+T")
check(rb.reihe[0][1].rows == ra.reihe[0][1].rows, "b) φ=0 = a) φ=0")
check(ra.reihe[3][1].rows == M([[-y for y in F.rows[1]], [x + 15 for x in F.rows[0]]]).rows,
      "a) φ=90: (−y, x+15) exakt")
check(rb.reihe[3][1].rows == M([list(F.rows[0]), [y + 15 for y in F.rows[1]]]).rows,
      "b) φ=90: F+(0,15) exakt")
check(rb.reihe[6][1].rows == M([[x - 15 for x in F.rows[0]], list(F.rows[1])]).rows,
      "b) φ=180: F−T exakt")
if np is not None:
    def Rot(phi):
        t = phi * np.pi / 180
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])
    Fn = np.array([[0, 1, 1, 2, 2, 1, 1, 3, 3, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]])
    Tn = np.array([[15], [0]])
    for i, phi in enumerate(np.arange(12) * 30):
        check(nah(ra.reihe[i][1], (Rot(phi) @ (Fn + Tn)).tolist(), 1e-9), f"a) numpy bei {phi}°")
        check(nah(rb.reihe[i][1], (Fn + Rot(phi) @ Tn).tolist(), 1e-9), f"b) numpy bei {phi}°")
    check(nah(drehung(30), Rot(30).tolist()), "Rot(30) = numpy")
    check(nah(zusammensetzen(drehung(-30), spiegelung_x(), drehung(30)).matrix,
              (Rot(30) @ np.array([[1, 0], [0, -1]]) @ Rot(-30)).tolist()),
          "Folie 20 gegen numpy")
    # Zahl für Zahl auf 4 Stellen: gerundete Ausgabe der Lösungsplots
    for i, phi in enumerate(np.arange(12) * 30):
        ref = np.round(Rot(phi) @ (Fn + Tn), 4)
        got = [[round(float(v), 4) for v in r] for r in ra.reihe[i][1].rows]
        check(np.allclose(ref, got, atol=1.5e-4), f"a) 4 Stellen bei {phi}°")
else:
    print("  (numpy fehlt: Vergleich mit der Musterlösung übersprungen)")
check(ra.decimal and "Rot(30°)" in ra.steps[0].items[0][0], "Reihe: dezimal und Beispielmatrix")
check(ra.steps[0].items[0][1].rows[0][0] != Fr(1, 1), "Rotationsmatrix im Schritt")
alles_anzeigen(ra, "rotationsreihe a")
alles_anzeigen(rb, "rotationsreihe b")
check("dreht sich mit" in ra.summary and "Ausrichtung" in rb.summary, "Zusammenfassungen unterscheiden sich")
fa = ra.folgen()
check(len(fa) == 12 and fa[1][0] == 30 and len(fa[1][1]) == 11 and isinstance(fa[1][1][0], float),
      "folgen(): Punktfolgen als float")
check(abs(fa[1][1][0] - 15 * math.cos(math.radians(30))) < 1e-9
      and abs(fa[1][2][0] - 15 * math.sin(math.radians(30))) < 1e-9, "folgen(): Startpunkt auf Kreis")
check(rotationsreihe("A").variante == "a", "variante gross geschrieben")
check(faellt(rotationsreihe, "c"), "variante falsch")
check(rotationsreihe("a", winkel=[0, 90]).reihe[1][1].rows == ra.reihe[3][1].rows, "eigene Winkel")
check(abs(sum(float(v) ** 2 for v in T.values) ** 0.5 - 15) < 1e-12, "|T| = 15")

# --------------------------------------------------------------------------
print("Aufgaben im Katalog")
check(liste("SW5") == ["SW5.1a", "SW5.1b", "SW5.1c", "SW5.1d", "SW5.1e", "SW5.2a", "SW5.2b"],
      f"liste('SW5') = {liste('SW5')}")
check(liste("sw5") == liste("SW5"), "liste klein geschrieben")
check(all(EXERCISES[k].week == "SW5" for k in liste("SW5")), "Woche SW5")
for k in liste("SW5"):
    ex = aufgabe(k)
    check(ex.title and ex.task and repr(ex) and ex._repr_html_(), f"{k}: Anzeige")
    alles_anzeigen(loese(k), f"loese({k})")
check(aufgabe("SW5.1a").data["Q"].rows == punkte(Q1["a"]).rows, "SW5.1a: Daten")
for k in "abc":
    check(loese(f"SW5.1{k}").A.rows == M(LOES_A[k]).rows, f"loese SW5.1{k}")
for k in "de":
    r = loese(f"SW5.1{k}")
    check(not r.linear and r.H.rows == M(LOES_H[k]).rows, f"loese SW5.1{k}")
check(loese("SW5.2a").variante == "a" and loese("SW5.2b").variante == "b", "loese SW5.2")
check(loese("SW5.2a").reihe[1][1].rows == ra.reihe[1][1].rows, "loese SW5.2a = rotationsreihe")
check(len(aufgabe("SW5.2b").data["phi"]) == 12, "SW5.2b: Winkel")
check(len([k for k in EXERCISES if k.startswith("SW4.")]) >= 10, "SW4 unberührt")
check("SW5.1f" not in EXERCISES, "keine Aufgabe zu viel")
try:
    aufgabe("SW5.9")
    check(False, "aufgabe('SW5.9') hätte scheitern müssen")
except KeyError:
    pass

# --------------------------------------------------------------------------
print("Paket")
import lialkit
check(lialkit.abbildung.drehung is drehung and lialkit.drehung is drehung, "Export und Alias abbildung")
check(lialkit.plots_abbildung.plot_abbildung is lialkit.plot_abbildung, "Alias plots_abbildung")
check(all(hasattr(lialkit, n) for n in lialkit.__all__), "__all__ vollständig")
check("abbilden" in lialkit.info() and "plot_rotationsreihe" in lialkit.info(), "info()")
check(lialkit.cross is not None, "Vektorprodukt bleibt in vectors.py")

# --------------------------------------------------------------------------
print("Zeichnungen")
from lialkit.grafik.plots_abbildung import (plot_abbildung, plot_basiswechsel,
                                            plot_rotationsreihe)
with tempfile.TemporaryDirectory() as tmp:
    figs = {
        "quadrat": plot_abbildung(skalierung(2), titel="Skalierung"),
        "homogen": plot_abbildung(M([[1, -1, 1], [0, 2, 0], [0, 0, 1]])),
        "buchstabe": plot_abbildung(spiegelung_achse(30), BUCHSTABE_F_FOLIEN),
        "liste": plot_abbildung(drehung(45), [(0, 0), (1, 0), (1, 1)]),
        "basis": plot_basiswechsel(),
        "reihe": plot_rotationsreihe(),
        "reihe_a": plot_rotationsreihe("a"),
        "reihe_b": plot_rotationsreihe("b"),
    }
    for name, fig in figs.items():
        path = os.path.join(tmp, name + ".png")
        fig.savefig(path, dpi=60)
        check(os.path.getsize(path) > 2000, f"{name}: PNG leer")
    check(len(figs["reihe"].axes) == 2 and len(figs["reihe_a"].axes) == 1, "Reihe: Anzahl Achsen")
    check(len(figs["basis"].axes) == 2, "Basiswechsel: zwei Achsen")
    ax = figs["quadrat"].axes[0]
    lines = [l for l in ax.get_lines() if len(l.get_xdata()) == 5]
    check(len(lines) >= 2, "Quadrat: Original und Bild")
    farben = {l.get_label(): l.get_color() for l in ax.get_lines() if l.get_label() in ("Original", "Bild")}
    check(farben == {"Original": "#e60000", "Bild": "#000000"}, f"Swiss: Rot für das Original, Schwarz fürs Bild: {farben}")
    check(ax.get_aspect() in (1.0, "equal"), "gleiche Achsenmassstäbe")
    bild = [l for l in ax.get_lines() if l.get_label() == "Bild"][0]
    check(list(bild.get_xdata()) == [0, 2, 2, 0, 0] and list(bild.get_ydata()) == [0, 0, 2, 2, 0],
          "gezeichnetes Bild = Lösung 1a")
check(faellt(plot_abbildung, M([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [(1, 2, 3)]), "3D wird abgelehnt")

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
