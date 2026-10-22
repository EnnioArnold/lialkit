#!/usr/bin/env python3
"""Tests für SW 6 (transponierte Abbildung, eigene Punkte, Aufgaben SW6.*). Aufruf: python3 tests/tests_sw6.py"""
import os
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from lialkit.aufgaben.exercises import aufgabe, liste, loese
from lialkit.grafik.plots_abbildung import plot_abbildung
from lialkit.rechnen.abbildung import (BASIS_FOLIE_6, BUCHSTABE_F_FOLIEN, EINHEITSQUADRAT, abbilden, basiswechsel,
                                       drehung, scherung, skalierung, transponierte_abbildung)
from lialkit.rechnen.core import Matrix

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


print("Transponierte Abbildung")
r = transponierte_abbildung(scherung(1))               # A = (1 1; 0 1), Einheitsquadrat
check(r.At.rows == [[1, 0], [1, 1]], "A^T")
check(r.Q.rows == [[0, 1, 2, 1, 0], [0, 0, 1, 1, 0]], "A·P")
check(r.Qt.rows == [[0, 1, 1, 0, 0], [0, 1, 2, 1, 0]], "A^T·P")
check("andere Abbildung" in r.eigenschaft, "allgemeiner Fall")
check(len(r.steps) == 4 and r.latex() and repr(r), "Rechenweg und Anzeige")
check("symmetrisch" in transponierte_abbildung([[2, 1], [1, 3]]).eigenschaft, "symmetrisch")
d = transponierte_abbildung(drehung(90))
check("Umkehrung" in d.eigenschaft and d.At.rows == [[0, 1], [-1, 0]], "Drehung: A^T = A^-1")
check(transponierte_abbildung(skalierung(2)).Qt.rows == abbilden(skalierung(2), EINHEITSQUADRAT).Q.rows,
      "Skalierung: A^T = A")
# Folie 7: A = (f1 f2) = (3 -1; -1 2) ist symmetrisch, Q = A·P wie auf der Folie
Af = basiswechsel(*BASIS_FOLIE_6, BUCHSTABE_F_FOLIEN)
check(Af.Q.rows[0] == [2, 5, 4, 7, 6, 3, 2, 8, 7, -2], "Folie 7: Q, erste Zeile")
check("symmetrisch" in transponierte_abbildung(Af.A, BUCHSTABE_F_FOLIEN).eigenschaft, "Folie 7: A symmetrisch")
# Eigene Punkte (Notiz SW6): Liste statt Viereck oder F
e = transponierte_abbildung([[1, 2], [0, 1]], [(0, 0), (2, 1), (3, 3)])
check(e.Qt.rows == [[0, 2, 3], [0, 5, 9]], f"eigene Punkte: {e.Qt.rows}")
try:
    transponierte_abbildung([[1, 2, 3], [4, 5, 6]])
    check(False, "nicht quadratisch hätte scheitern müssen")
except ValueError:
    pass

print("Zeichnung")
with tempfile.TemporaryDirectory() as tmp:
    fig = plot_abbildung(scherung(1), [(0, 0), (2, 0), (2, 1)], transponiert=True)
    path = os.path.join(tmp, "t.png")
    fig.savefig(path, dpi=60)
    check(os.path.getsize(path) > 2000, "PNG leer")
    labels = [l.get_label() for l in fig.axes[0].get_lines()]
    check("Bild unter $A^T$" in labels, f"gestricheltes Bild: {labels}")
    check("Bild unter $A^T$" not in [l.get_label() for l in plot_abbildung(scherung(1)).axes[0].get_lines()],
          "ohne Flag keine Aᵀ-Linie")

print("Aufgaben SW6 (gleiches Blatt wie SW5)")
check(liste("SW6") == [k.replace("SW5", "SW6") for k in liste("SW5")] and len(liste("SW6")) == 7, f"liste: {liste('SW6')}")
for k in liste("SW6"):
    a5, a6 = aufgabe(k.replace("SW6", "SW5")), aufgabe(k)
    check(a6.week == "SW6" and a6.method == a5.method and a6.task == a5.task, f"{k}: wie SW5")
    check(repr(a6) and a6._repr_html_() and loese(k) is not None, f"{k}: Anzeige/Lösung")
check(loese("SW6.1a").A.rows == [[2, 0], [0, 2]], "SW6.1a: A = 2E (Musterlösung 1a)")
check(loese("SW6.1c").A.rows == [[1, 1], [0, 2]], "SW6.1c: A (Musterlösung 1c)")
check(loese("SW6.1d").H.rows == [[1, -1, 1], [0, 2, 0], [0, 0, 1]], "SW6.1d: D (Musterlösung)")
check(len(liste("SW5")) == 7, "SW5 unberührt")

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
