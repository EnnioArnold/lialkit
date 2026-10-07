#!/usr/bin/env python3
"""
Zeichnen: Vektoren, Geraden, Ebenen und Spate.

Flaches mit Matplotlib — das passt zu euren Notebooks und lässt sich
drucken. Räumliches zusätzlich mit Plotly, weil ein Spat erst verständlich
wird, wenn man ihn drehen kann. Fehlt Plotly, fällt alles auf Matplotlib
zurück, statt abzubrechen.
"""

from __future__ import annotations

import math
from typing import Sequence

from .core import Matrix, Vector, fmt

# Dieselbe Farbfolge wie in der Qt-App, damit Fenster und Notebook
# zusammengehören.
COLORS = ["#0071e3", "#ff9500", "#248a3d", "#d70015", "#8944ab", "#00a2a8"]


def _vec(v) -> Vector:
    return v if isinstance(v, Vector) else Vector(v)


def _floats(v: Vector) -> list[float]:
    return [float(x) for x in v.values]


def has_plotly() -> bool:
    try:
        import plotly.graph_objects  # noqa: F401
        return True
    except ImportError:
        return False


def _finish(fig):
    """
    Gibt die Figur zurück, ohne dass sie im Notebook doppelt erscheint.

    Mit dem Inline-Backend zeigt Jupyter jede offene Figur am Zellenende
    von selbst an — und die zurückgegebene gleich noch einmal. plt.close
    nimmt sie aus der Verwaltung; angezeigt wird dann nur noch das
    Rückgabeobjekt.
    """
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig


def _style_2d(ax, title: str = ""):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d2d2d7", linewidth=0.6, alpha=0.7)
    ax.axhline(0, color="#86868b", linewidth=1)
    ax.axvline(0, color="#86868b", linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d2d2d7")
    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=12)


# ==========================================================================
# Vektoren
# ==========================================================================

def plot_vectors(*vectors, labels: Sequence[str] | None = None,
                 origins=None, title: str = "", chain: bool = False,
                 interactive: bool | None = None, size: float = 5.5):
    """
    Zeichnet Vektoren als Pfeile.

    chain=True hängt sie aneinander — so sieht man die Summe als Streckenzug
    und versteht, warum die Dreiecksungleichung gilt.

    Zwei Einträge je Vektor ergeben eine flache Zeichnung, drei eine
    räumliche. interactive=True erzwingt Plotly, False erzwingt Matplotlib.
    """
    vecs = [_vec(v) for v in vectors]
    if not vecs:
        raise ValueError("Mindestens ein Vektor wird gebraucht")
    dims = {len(v) for v in vecs}
    if len(dims) > 1:
        raise ValueError("Alle Vektoren brauchen gleich viele Einträge")
    dim = dims.pop()
    if dim not in (2, 3):
        raise ValueError("Zeichnen lässt sich nur in der Ebene oder im Raum")

    labels = list(labels) if labels else [f"v{i + 1}" for i in range(len(vecs))]
    if origins is None:
        if chain:
            starts, running = [], Vector([0] * dim)
            for v in vecs:
                starts.append(running)
                running = running + v
            origins = starts
        else:
            origins = [Vector([0] * dim)] * len(vecs)
    else:
        origins = [_vec(o) for o in origins]

    if interactive is None:
        interactive = dim == 3 and has_plotly()
    if dim == 3 and interactive and has_plotly():
        return _plot_vectors_plotly(vecs, origins, labels, title, chain)
    return _plot_vectors_mpl(vecs, origins, labels, title, dim, chain, size)


def _plot_vectors_mpl(vecs, origins, labels, title, dim, chain, size):
    import matplotlib.pyplot as plt

    if dim == 2:
        fig, ax = plt.subplots(figsize=(size, size))
        for i, (v, o) in enumerate(zip(vecs, origins)):
            x0, y0 = _floats(o)
            dx, dy = _floats(v)
            color = COLORS[i % len(COLORS)]
            ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                        shrinkA=0, shrinkB=0))
            ax.text(x0 + dx / 2, y0 + dy / 2, f"  {labels[i]}", color=color,
                    fontsize=11, fontweight="demibold")
        if chain:
            total = _floats(origins[-1] + vecs[-1])
            ax.annotate("", xy=tuple(total), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#86868b",
                                        lw=2, ls="--", shrinkA=0, shrinkB=0))
            ax.text(total[0] / 2, total[1] / 2, "  Summe", color="#86868b",
                    fontsize=10)
        points = [[0.0, 0.0]] + [_floats(o + v) for o, v in zip(origins, vecs)] \
                 + [_floats(o) for o in origins]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        pad = max(1.0, 0.18 * max(max(map(abs, xs + ys)), 1))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        _style_2d(ax, title)
        fig.tight_layout()
        return _finish(fig)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(size + 1, size))
    ax = fig.add_subplot(111, projection="3d")
    for i, (v, o) in enumerate(zip(vecs, origins)):
        x0, y0, z0 = _floats(o)
        dx, dy, dz = _floats(v)
        color = COLORS[i % len(COLORS)]
        ax.quiver(x0, y0, z0, dx, dy, dz, color=color, arrow_length_ratio=0.12,
                  linewidth=2.2)
        ax.text(x0 + dx, y0 + dy, z0 + dz, f"  {labels[i]}", color=color,
                fontsize=10)
    all_pts = [[0.0, 0.0, 0.0]] + [_floats(o + v) for o, v in zip(origins, vecs)]
    bound = max(1.0, max(abs(c) for p in all_pts for c in p) * 1.2)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()
    return _finish(fig)


def _plot_vectors_plotly(vecs, origins, labels, title, chain):
    import plotly.graph_objects as go

    data = []
    for i, (v, o) in enumerate(zip(vecs, origins)):
        x0, y0, z0 = _floats(o)
        dx, dy, dz = _floats(v)
        color = COLORS[i % len(COLORS)]
        data.append(go.Scatter3d(
            x=[x0, x0 + dx], y=[y0, y0 + dy], z=[z0, z0 + dz],
            mode="lines", line=dict(color=color, width=6), name=labels[i]))
        data.append(go.Cone(
            x=[x0 + dx], y=[y0 + dy], z=[z0 + dz], u=[dx], v=[dy], w=[dz],
            sizemode="absolute", sizeref=0.28, anchor="tip",
            showscale=False, colorscale=[[0, color], [1, color]]))
    figure = go.Figure(data=data)
    figure.update_layout(
        title=title or None, template="plotly_white", height=560,
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y",
                   zaxis_title="z"))
    return figure


# ==========================================================================
# Spat
# ==========================================================================

def plot_spat(u, v, w, title: str = "", interactive: bool | None = None,
              labels: Sequence[str] | None = None):
    """
    Zeichnet das von drei Vektoren aufgespannte Parallelepiped.

    Sein Volumen ist der Betrag des Spatprodukts; liegen die Vektoren in
    einer Ebene, fällt der Körper flach zusammen — das sieht man dann auch.
    """
    u, v, w = _vec(u), _vec(v), _vec(w)
    names = list(labels) if labels else ["u", "v", "w"]
    for vec in (u, v, w):
        if len(vec) != 3:
            raise ValueError("Ein Spat braucht drei Vektoren mit je drei "
                             "Einträgen")
    corners = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                corners.append(_floats(u.scaled(a) + v.scaled(b) + w.scaled(c)))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    if interactive is None:
        interactive = has_plotly()
    if interactive and has_plotly():
        import plotly.graph_objects as go
        data = []
        for a, b in edges:
            data.append(go.Scatter3d(
                x=[corners[a][0], corners[b][0]],
                y=[corners[a][1], corners[b][1]],
                z=[corners[a][2], corners[b][2]],
                mode="lines", line=dict(color="#86868b", width=3),
                showlegend=False, hoverinfo="skip"))
        for i, (vec, name) in enumerate(zip((u, v, w), names)):
            dx, dy, dz = _floats(vec)
            color = COLORS[i]
            data.append(go.Scatter3d(x=[0, dx], y=[0, dy], z=[0, dz],
                                     mode="lines",
                                     line=dict(color=color, width=7),
                                     name=name))
        data.append(go.Mesh3d(
            x=[c[0] for c in corners], y=[c[1] for c in corners],
            z=[c[2] for c in corners], alphahull=0, opacity=0.18,
            color="#0071e3", hoverinfo="skip", showlegend=False))
        figure = go.Figure(data=data)
        figure.update_layout(
            title=title or None, template="plotly_white", height=580,
            margin=dict(l=0, r=0, t=40 if title else 10, b=0),
            scene=dict(aspectmode="data"))
        return figure

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color="#86868b", linewidth=1.2)
    for i, (vec, name) in enumerate(zip((u, v, w), names)):
        dx, dy, dz = _floats(vec)
        ax.quiver(0, 0, 0, dx, dy, dz, color=COLORS[i],
                  arrow_length_ratio=0.12, linewidth=2.4)
        ax.text(dx, dy, dz, f"  {name}", color=COLORS[i], fontsize=11)
    # Die Grenzen aus den Ecken ableiten und gleich gross halten, sonst
    # erscheint der Spat verzerrt.
    lows = [min(p[i] for p in corners) for i in range(3)]
    highs = [max(p[i] for p in corners) for i in range(3)]
    span = max(1.0, max(h - l for l, h in zip(lows, highs))) * 1.12
    for axis, lo, hi in zip("xyz", lows, highs):
        mid = (lo + hi) / 2
        getattr(ax, f"set_{axis}lim")(mid - span / 2, mid + span / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()
    return _finish(fig)


# ==========================================================================
# Geraden und Ebenen
# ==========================================================================

def plot_line(line, span=(-3, 3), title: str = "", ax=None):
    """Zeichnet eine Gerade in Parameterform."""
    import matplotlib.pyplot as plt
    dim = len(line.p)
    ts = [span[0] + (span[1] - span[0]) * i / 60 for i in range(61)]
    points = [_floats(line.at(t)) for t in ts]
    if dim == 2:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
        else:
            fig = ax.figure
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=COLORS[0], linewidth=2)
        px, py = _floats(line.p)
        dx, dy = _floats(line.d)
        ax.plot([px], [py], "o", color=COLORS[3], markersize=7)
        ax.annotate("", xy=(px + dx, py + dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS[3], lw=2))
        _style_2d(ax, title)
        fig.tight_layout()
        return _finish(fig)
    raise ValueError("Für räumliche Geraden plot_vectors oder plot_plane nutzen")


def plot_plane(plane, span=(-2, 2), points=None, title: str = "",
               interactive: bool | None = None):
    """
    Zeichnet eine Ebene mitsamt ihrer Normalen.

    Die Normale macht sichtbar, warum die Koordinatenform n · x = c alle
    Punkte der Ebene beschreibt: für sie ist das Skalarprodukt konstant.
    """
    if plane.u is None:
        raise ValueError("Zum Zeichnen braucht die Ebene zwei Spannvektoren")
    lo, hi = span
    grid = [lo + (hi - lo) * i / 12 for i in range(13)]
    surface = [[_floats(plane.at(s, t)) for t in grid] for s in grid]

    if interactive is None:
        interactive = has_plotly()
    if interactive and has_plotly():
        import plotly.graph_objects as go
        data = [go.Surface(
            x=[[p[0] for p in row] for row in surface],
            y=[[p[1] for p in row] for row in surface],
            z=[[p[2] for p in row] for row in surface],
            opacity=0.55, showscale=False,
            colorscale=[[0, "#cfe3fb"], [1, "#0071e3"]], name="Ebene")]
        px, py, pz = _floats(plane.p)
        nx, ny, nz = _floats(plane.n.normalized()
                             if not plane.n.is_zero() else plane.n)
        data.append(go.Scatter3d(x=[px, px + nx], y=[py, py + ny],
                                 z=[pz, pz + nz], mode="lines",
                                 line=dict(color=COLORS[3], width=7),
                                 name="Normale n"))
        for i, pt in enumerate(points or []):
            q = _floats(_vec(pt))
            data.append(go.Scatter3d(x=[q[0]], y=[q[1]], z=[q[2]],
                                     mode="markers",
                                     marker=dict(size=5, color=COLORS[2]),
                                     name=f"P{i + 1}"))
        figure = go.Figure(data=data)
        figure.update_layout(
            title=title or None, template="plotly_white", height=580,
            margin=dict(l=0, r=0, t=40 if title else 10, b=0),
            scene=dict(aspectmode="data"))
        return figure

    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    # plot_surface erwartet Arrays, keine verschachtelten Listen.
    ax.plot_surface(
        np.array([[p[0] for p in row] for row in surface], dtype=float),
        np.array([[p[1] for p in row] for row in surface], dtype=float),
        np.array([[p[2] for p in row] for row in surface], dtype=float),
        alpha=0.45, color=COLORS[0], edgecolor="none")
    px, py, pz = _floats(plane.p)
    nx, ny, nz = _floats(plane.n)
    ax.quiver(px, py, pz, nx, ny, nz, color=COLORS[3], linewidth=2.2,
              arrow_length_ratio=0.15)
    for pt in points or []:
        q = _floats(_vec(pt))
        ax.scatter(*q, color=COLORS[2], s=40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()
    return _finish(fig)


def plot_transformation(A, title: str = "", grid: int = 5):
    """
    Zeigt, was eine 2×2-Matrix mit dem Einheitsquadrat und dem Gitter macht.

    Die Fläche des Bildes ist |det A| — damit wird die Determinante
    anschaulich: sie misst, wie stark die Abbildung Flächen streckt, und ihr
    Vorzeichen sagt, ob die Orientierung kippt.
    """
    import matplotlib.pyplot as plt
    from .determinant import det

    A = A if isinstance(A, Matrix) else Matrix(A)
    if A.m != 2 or A.n != 2:
        raise ValueError("Dafür braucht es eine 2×2-Matrix")
    a, b = float(A.rows[0][0]), float(A.rows[0][1])
    c, d = float(A.rows[1][0]), float(A.rows[1][1])

    def apply(x, y):
        return a * x + b * y, c * x + d * y

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, transform, name in ((axes[0], False, "vorher"),
                                (axes[1], True, "nachher")):
        for i in range(-grid, grid + 1):
            line_a = [(i, t) for t in range(-grid, grid + 1)]
            line_b = [(t, i) for t in range(-grid, grid + 1)]
            for line in (line_a, line_b):
                pts = [apply(*p) if transform else p for p in line]
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color="#d2d2d7", linewidth=0.7, zorder=1)
        square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        pts = [apply(*p) if transform else p for p in square]
        ax.fill([p[0] for p in pts], [p[1] for p in pts], color=COLORS[0],
                alpha=0.25, zorder=2)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[0],
                linewidth=2, zorder=3)
        for i, base in enumerate(((1, 0), (0, 1))):
            tip = apply(*base) if transform else base
            ax.annotate("", xy=tip, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=COLORS[1 + i], lw=2.4), zorder=4)
        limit = grid if not transform else max(
            grid, abs(a) + abs(b), abs(c) + abs(d)) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        _style_2d(ax, name)
    value = det(A).value
    fig.suptitle(title or f"Flächenfaktor |det A| = {fmt(abs(value))}",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return _finish(fig)
