#!/usr/bin/env python3
"""
Zeichnungen zu SW 5 (lineare Abbildungen): Original und Bild, Basiswechsel
und die Drehreihen aus Aufgabe 2.

Gestaltung wie in plots.py (Swiss Style): Schwarz/Weiss, das Original in
Rot (#e60000), das Bild in Schwarz, keine Rundungen. Matplotlib mit
beliebigem Backend, also auch Agg ohne Fenster.
"""

from __future__ import annotations

import math

from ..rechnen.abbildung import (BASIS_FOLIE_6, BUCHSTABE_F_AUFGABE,
                                 BUCHSTABE_F_FOLIEN, EINHEITSQUADRAT, PHI_REIHE,
                                 abbilden, punkte, rotationsreihe)
from ..rechnen.core import Matrix, Vector
from .plots import INK, RED, _finish, _style_2d, _swiss


def _schritt(spanne: float) -> float:
    """Gitterabstand 1, 2, 5, 10, … so, dass höchstens rund zehn Linien entstehen."""
    for s in (1, 2, 5, 10, 20, 50, 100):
        if spanne / s <= 10:
            return float(s)
    return 100.0


def _grenzen(ax, xs, ys, schritt: float | None = None):
    """Achsengrenzen um die Daten (und den Ursprung), Gitter im ganzen Schritt."""
    import matplotlib.ticker as tk
    lo_x, hi_x = min(0, min(xs)), max(0, max(xs))
    lo_y, hi_y = min(0, min(ys)), max(0, max(ys))
    s = schritt or _schritt(max(hi_x - lo_x, hi_y - lo_y, 1))
    ax.set_xlim(math.floor(lo_x / s) * s - 0.25 * s, math.ceil(hi_x / s) * s + 0.25 * s)
    ax.set_ylim(math.floor(lo_y / s) * s - 0.25 * s, math.ceil(hi_y / s) * s + 0.25 * s)
    ax.xaxis.set_major_locator(tk.MultipleLocator(s))
    ax.yaxis.set_major_locator(tk.MultipleLocator(s))


def _figur(plt, spalten: int, xs, ys, groesse: float):
    """Figur, deren Seitenverhältnis zu den Daten passt (gleiche Massstäbe)."""
    breite = max(max(xs) - min(0, min(xs)), 1)
    hoehe = max(max(ys) - min(0, min(ys)), 1)
    verhaeltnis = min(max(hoehe / breite, 0.6), 1.6)
    return plt.subplots(1, spalten, figsize=(groesse * spalten, groesse * verhaeltnis))


def _zeilen(M: Matrix):
    return [float(v) for v in M.rows[0]], [float(v) for v in M.rows[1]]


def _pfeil(ax, ziel, farbe, name: str, start=(0, 0)):
    ax.annotate("", xy=ziel, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=farbe, lw=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=11))
    ax.annotate(name, xy=ziel, xytext=(5, 5), textcoords="offset points",
                color=farbe, fontsize=11, fontweight="bold")


@_swiss
def plot_abbildung(A, P=None, titel: str = "", groesse: float = 5.5):
    """
    Original (rot) und Bild (schwarz) einer Abbildung A, wie in der Musterlösung.

    P sind die Punkte spaltenweise (Standard: das Einheitsquadrat); ein
    Buchstabe wie BUCHSTABE_F_AUFGABE geht genauso. Eine 3×3-Matrix wirkt auf
    die homogenen Koordinaten; die Zeile mit den Einsen wird nicht gezeichnet.
    """
    import matplotlib.pyplot as plt

    P = EINHEITSQUADRAT if P is None else punkte(P)
    res = abbilden(A, P)
    if res.bild.m != 2:
        raise ValueError("Gezeichnet werden nur Abbildungen der Ebene")
    px, py = _zeilen(Matrix(P.rows[:2]))
    qx, qy = _zeilen(res.bild)

    fig, ax = _figur(plt, 1, px + qx, py + qy, groesse)
    _style_2d(ax, titel)
    ax.plot(px, py, color=RED, lw=3, label="Original", zorder=3)
    ax.plot(qx, qy, color=INK, lw=1.6, label="Bild", zorder=4)
    _grenzen(ax, px + qx, py + qy)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    return _finish(fig)


@_swiss
def plot_basiswechsel(f1=BASIS_FOLIE_6[0], f2=BASIS_FOLIE_6[1],
                      P=BUCHSTABE_F_FOLIEN, punkt: int = 3, groesse: float = 5.0):
    """
    Folie 6: derselbe Buchstabe im Gitter von e₁, e₂ (links) und im
    verformten Gitter von f₁, f₂ (rechts).

    `punkt` ist die Spalte von P, die als Linearkombination markiert wird
    (Standard: der vierte Punkt, v = 3·e₁ + 2·e₂ = (3, 2)).
    """
    import matplotlib.pyplot as plt

    f1, f2 = Vector(f1), Vector(f2)
    P = punkte(P)
    res = abbilden(Matrix([[f1[0], f2[0]], [f1[1], f2[1]]]), P, punkt)
    px, py = _zeilen(P)
    qx, qy = _zeilen(res.Q)
    fig, (links, rechts) = plt.subplots(1, 2, figsize=(2 * groesse, groesse))

    n = 14
    for ax, (ax1, ax2), xs, ys in ((links, ((1, 0), (0, 1)), px, py),
                                   (rechts, (tuple(map(float, f1)), tuple(map(float, f2))),
                                    qx, qy)):
        _style_2d(ax)
        a, b = ax1, ax2
        for i in range(-n, n + 1):          # Gitterlinien parallel zu b und a
            ax.plot([i * a[0] - n * b[0], i * a[0] + n * b[0]],
                    [i * a[1] - n * b[1], i * a[1] + n * b[1]],
                    color="#e0e0e0", lw=0.6, zorder=1)
            ax.plot([i * b[0] - n * a[0], i * b[0] + n * a[0]],
                    [i * b[1] - n * a[1], i * b[1] + n * a[1]],
                    color="#e0e0e0", lw=0.6, zorder=1)
        ax.plot(xs + xs[:1], ys + ys[:1], color=RED if ax is links else INK,
                lw=2, zorder=3)
        ax.plot([xs[punkt]], [ys[punkt]], marker="s", ms=6, color=INK, zorder=5)
        _grenzen(ax, xs + [a[0], b[0]], ys + [a[1], b[1]], schritt=1 if ax is links else 2)
        ax.grid(False)
    _pfeil(links, (1, 0), RED, "e₁")
    _pfeil(links, (0, 1), RED, "e₂")
    _pfeil(rechts, tuple(map(float, f1)), INK, "f₁")
    _pfeil(rechts, tuple(map(float, f2)), INK, "f₂")
    links.set_title("Basis e₁, e₂", fontsize=12, loc="left", pad=12)
    rechts.set_title("Basis f₁, f₂", fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    return _finish(fig)


@_swiss
def plot_rotationsreihe(variante: str = "beide", F=BUCHSTABE_F_AUFGABE, T=(15, 0),
                        winkel=PHI_REIHE, groesse: float = 5.0):
    """
    Aufgabe 2: der Buchstabe F für φ = 0°, 30°, …, 330°.

    variante "a": Rot(φ)·(F + T), "b": F + Rot(φ)·T, "beide": nebeneinander.
    φ = 0° ist rot hervorgehoben.
    """
    import matplotlib.pyplot as plt

    varianten = ["a", "b"] if variante == "beide" else [variante]
    reihen = [rotationsreihe(v, F, T, winkel) for v in varianten]
    alle_x = [x for r in reihen for _, xs, _ in r.folgen() for x in xs]
    alle_y = [y for r in reihen for _, _, ys in r.folgen() for y in ys]
    fig, axs = _figur(plt, len(varianten), alle_x, alle_y, groesse)
    axs = [axs] if len(varianten) == 1 else list(axs)
    titel = {"a": r"a)  Rot(φ) · (F + T)", "b": r"b)  F + Rot(φ) · T"}
    for ax, v, r in zip(axs, varianten, reihen):
        _style_2d(ax, titel[v])
        for phi, xs, ys in r.folgen():
            erste = phi == 0
            ax.plot(xs, ys, color=RED if erste else INK, lw=2.4 if erste else 1.1,
                    zorder=4 if erste else 3)
        _grenzen(ax, alle_x, alle_y, schritt=5)
    fig.tight_layout()
    return _finish(fig)
