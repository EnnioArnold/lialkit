#!/usr/bin/env python3
"""
Gerüst für SW3 Aufgabe 3 — den LR-Algorithmus selbst programmieren.

Das Übungsblatt verlangt ausdrücklich, dass du den Algorithmus selbst
schreibst und die Schleifen ausschreibst. Darum steht hier keine Lösung,
sondern ein Prüfstand: du schreibst deine Funktion, rufst `pruefe_lr` damit
auf und siehst sofort, ob sie stimmt — an denselben Beispielen, die auf dem
Übungsblatt stehen, und an ein paar, die gern übersehen werden.

    from lialkit.uebung import pruefe_lr, vorlage

    print(vorlage())          # Gerüst zum Abschreiben

    def meine_lr(A):
        ...
        return L, R

    pruefe_lr(meine_lr)
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

from .core import Matrix, Vector, fmt
from .factor import lu


VORLAGE = '''
def meine_lr(A):
    """
    Zerlegt die quadratische Matrix A in L und R mit A = L @ R.

    A ist eine Liste von Listen. Gib zwei solche Listen zurück:
    L mit Einsen auf der Diagonale und den Faktoren darunter,
    R als obere Dreiecksmatrix.
    """
    n = len(A)
    # Mit Kopien arbeiten, damit A unverändert bleibt.
    R = [row[:] for row in A]
    L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for c in range(n):          # über die Spalten
        for r in range(c + 1, n):   # Zeilen darunter
            # TODO: Faktor bestimmen, in L eintragen,
            #       Zeile r um faktor * Zeile c verringern
            pass

    return L, R


def meine_loesung(L, R, b):
    """Löst L·y = b vorwärts, dann R·x = y rückwärts."""
    n = len(b)
    y = [0] * n
    for i in range(n):
        # TODO: y[i] aus b[i] und den schon bekannten y[k] bestimmen
        pass
    x = [0] * n
    for i in range(n - 1, -1, -1):
        # TODO: x[i] aus y[i] und den schon bekannten x[k] bestimmen
        pass
    return x
'''.strip()


#: Prüffälle. Die ersten drei sind die Aufgaben vom Übungsblatt, die
#: weiteren decken ab, was dabei leicht untergeht.
FAELLE: list[tuple[str, list[list], list | None, str]] = [
    ("Blatt 1a", [[3, -6], [-2, 5]], [0, 1], ""),
    ("Blatt 1b", [[2, 6, 2], [-3, -8, 0], [4, 9, 2]], [2, 2, 3], ""),
    ("Blatt 1c", [[2, -1, 3, 4], [1, 0, -2, 7], [3, -3, 1, 5], [2, 1, 4, 4]],
     [9, 11, 8, 10], ""),
    ("Einheitsmatrix", [[1, 0], [0, 1]], [5, 7],
     "L und R müssen beide die Einheitsmatrix sein."),
    ("schon Dreieck", [[2, 3], [0, 4]], [7, 8],
     "Hier ist nichts zu eliminieren — L bleibt die Einheitsmatrix."),
    ("Brüche", [[2, 1], [1, 3]], [1, 1],
     "Der Faktor ist 1/2 — mit float kommt 0.5, das ist hier noch harmlos."),
    ("kleines Pivot", [[1, 1000], [1, 1001]], [1, 2],
     "Grosse Zahlenunterschiede; exakt gerechnet kein Problem."),
]

#: Fälle, an denen ein einfacher Algorithmus scheitern DARF — aber nicht
#: still falsch rechnen soll.
STOLPERSTEINE: list[tuple[str, list[list], str]] = [
    ("Null an der Pivotstelle", [[0, 2], [3, 1]],
     "Ohne Zeilentausch geht es nicht weiter. Deine Funktion sollte das "
     "melden, statt durch Null zu teilen."),
    ("singulär", [[1, 2], [2, 4]],
     "Die zweite Zeile ist das Doppelte der ersten. Es gibt keine "
     "eindeutige Zerlegung."),
]


def vorlage() -> str:
    """Das Gerüst zum Abschreiben."""
    return VORLAGE


def _to_fractions(rows) -> list[list[Fraction]]:
    return [[Fraction(v) for v in row] for row in rows]


def _matmul(A, B) -> list[list[Fraction]]:
    n, m, k = len(A), len(B[0]), len(B)
    return [[sum(Fraction(A[i][t]) * Fraction(B[t][j]) for t in range(k))
             for j in range(m)] for i in range(n)]


def _same(A, B, tol: float = 1e-9) -> bool:
    """Vergleicht zwei Matrizen — exakt wenn möglich, sonst mit Toleranz."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    for ra, rb in zip(A, B):
        for a, b in zip(ra, rb):
            try:
                if Fraction(a) != Fraction(b):
                    return False
            except (TypeError, ValueError):
                if abs(float(a) - float(b)) > tol:
                    return False
    return True


def pruefe_lr(funktion: Callable, loeser: Callable | None = None,
              ausfuehrlich: bool = True) -> bool:
    """
    Prüft deine LR-Zerlegung an den Aufgaben vom Blatt und an Sonderfällen.

    `funktion` bekommt eine Liste von Listen und soll (L, R) zurückgeben.
    Mit `loeser` kannst du zusätzlich deine Vorwärts-/Rückwärtslösung
    prüfen lassen; sie bekommt (L, R, b) und soll x liefern.

    Gibt True zurück, wenn alles stimmt.
    """
    bestanden = 0
    gesamt = 0
    meldungen: list[str] = []

    for name, rows, b, hinweis in FAELLE:
        gesamt += 1
        try:
            L, R = funktion([row[:] for row in rows])
        except Exception as exc:
            meldungen.append(f"  ✕ {name}: Fehler beim Aufruf — {exc}")
            continue

        try:
            produkt = _matmul(L, R)
        except Exception as exc:
            meldungen.append(f"  ✕ {name}: L und R lassen sich nicht "
                             f"multiplizieren — {exc}")
            continue

        if not _same(produkt, _to_fractions(rows)):
            meldungen.append(f"  ✕ {name}: L · R ergibt nicht A.")
            if ausfuehrlich:
                meldungen.append(f"      L·R = {[[str(v) for v in r] for r in produkt]}")
                meldungen.append(f"      A   = {rows}")
            continue

        # L muss unten dreieckig sein mit Einsen auf der Diagonale.
        n = len(rows)
        fehler = []
        for i in range(n):
            if Fraction(L[i][i]) != 1:
                fehler.append(f"L[{i}][{i}] = {L[i][i]}, erwartet 1")
            for j in range(i + 1, n):
                if Fraction(L[i][j]) != 0:
                    fehler.append(f"L[{i}][{j}] = {L[i][j]}, erwartet 0")
        for i in range(n):
            for j in range(i):
                if Fraction(R[i][j]) != 0:
                    fehler.append(f"R[{i}][{j}] = {R[i][j]}, erwartet 0")
        if fehler:
            meldungen.append(f"  ✕ {name}: Form stimmt nicht — "
                             f"{fehler[0]}" + (f" (und {len(fehler) - 1} weitere)"
                                               if len(fehler) > 1 else ""))
            continue

        if loeser is not None and b is not None:
            try:
                x = loeser(L, R, b[:])
                probe = [sum(Fraction(rows[i][j]) * Fraction(x[j])
                             for j in range(n)) for i in range(n)]
                if not _same([probe], [[Fraction(v) for v in b]]):
                    meldungen.append(f"  ✕ {name}: A · x ergibt nicht b "
                                     f"(x = {[str(v) for v in x]}).")
                    continue
            except Exception as exc:
                meldungen.append(f"  ✕ {name}: Löser wirft einen Fehler — {exc}")
                continue

        bestanden += 1
        meldungen.append(f"  ✓ {name}" + (f" — {hinweis}" if hinweis and
                                          ausfuehrlich else ""))

    print(f"LR-Zerlegung: {bestanden} von {gesamt} Fällen bestanden")
    for zeile in meldungen:
        print(zeile)

    if ausfuehrlich:
        print()
        print("Stolpersteine — hier darf deine Funktion abbrechen, aber sie")
        print("soll nicht still etwas Falsches liefern:")
        for name, rows, hinweis in STOLPERSTEINE:
            try:
                L, R = funktion([row[:] for row in rows])
                produkt = _matmul(L, R)
                if _same(produkt, _to_fractions(rows)):
                    print(f"  ✓ {name}: sauber gelöst.")
                else:
                    print(f"  ! {name}: liefert ein Ergebnis, aber L · R ≠ A. "
                          f"{hinweis}")
            except ZeroDivisionError:
                print(f"  ! {name}: Division durch Null. {hinweis}")
            except Exception as exc:
                print(f"  ✓ {name}: bricht mit einer Meldung ab ({type(exc).__name__}). "
                      f"Das ist in Ordnung.")

    return bestanden == gesamt


def vergleiche_mit_lialkit(rows) -> None:
    """Zeigt die Zerlegung von lialkit zum Nachschauen — nach dem Üben."""
    result = lu(Matrix(rows))
    print(result)
