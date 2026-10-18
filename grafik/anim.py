#!/usr/bin/env python3
"""
Manim-Animationen zur linearen Algebra: Gauss-Elimination, Pivots,
Vektorprodukte.

Swiss Style wie die App: Schwarz/Weiss mit einem Rot (#e60000), keine
Rundungen, schlichte Linien, Helvetica-artige Schrift.

Engine: manimgx wird bevorzugt (schnell, bringt die Videoausgabe selbst mit,
kein ffmpeg und kein Cairo/Pango nötig). Ist es nicht installiert, fällt
alles auf Manim CE zurück. LIALKIT_ENGINE=manim erzwingt den Rückfall.

Bewusst ohne LaTeX: Die Matrix wird aus Text-Objekten und selbst gezeichneten
Klammern aufgebaut. Dadurch ist eine TeX-Installation nicht nötig.

Installation:
    pip install manimgx
    (Rückfall: brew install cairo pango pkg-config ffmpeg; pip install manim)

Verwendung aus der App:
    from lialkit.grafik.anim import AnimationJob, render
    job = AnimationJob(frames=[...], captions=[...], ...)
    pfad = render(job, progress=print)

Direkt von der Kommandozeile (rendert ein Beispiel):
    python3 -m lialkit.grafik.anim [full|step|pivots]
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence

# --------------------------------------------------------------------------
# Auftragsbeschreibung — kennt weder Manim noch Qt
# --------------------------------------------------------------------------

#: Grösse, Bildrate und Manim-Kürzel je Qualitätsstufe.
QUALITIES = {
    "low": ("480p15", "-ql"),
    "medium": ("720p30", "-qm"),
    "high": ("1080p60", "-qh"),
    "ultra": ("1440p60", "-qp"),
}


@dataclass
class Theme:
    """Farben der Animation: Swiss Style, Schwarz/Weiss mit einem Rot."""
    background: str = "#ffffff"
    text: str = "#000000"
    muted: str = "#666666"
    accent: str = "#e60000"
    highlight: str = "#f2f2f2"     # Zeilenband bei einer Operation
    pivot: str = "#e60000"         # Pivotfeld (wird halbtransparent gefüllt)
    good: str = "#000000"          # Lösungszeilen

    @staticmethod
    def dark() -> "Theme":
        return Theme(background="#000000", text="#ffffff", muted="#999999",
                     accent="#ff3333", highlight="#1a1a1a", pivot="#ff3333",
                     good="#ffffff")


@dataclass
class AnimationJob:
    """
    Ein Rendering-Auftrag.

    frames    — die Matrizen als Text, Zustand vor und nach jedem Schritt
    captions  — Beschriftung je Zustand, gleich lang wie frames
    reasons   — Begründung je Zustand, gleich lang wie frames
    changed   — je Übergang die Menge der geänderten Zellen als (Zeile, Spalte)
    kinds     — je Übergang die Art: 'swap' | 'scale' | 'add'
    moved     — je Übergang bei 'swap' das Zeilenpaar, sonst None
    targets   — je Übergang die Zielzeile
    pivots    — je Zustand die Pivotposition oder None
    scene     — 'step' | 'full' | 'pivots' oder eine der Vektorszenen
    divider_at— vor welcher Spalte die gestrichelte Linie steht; -1 heisst
                vor der letzten (die erweiterte Matrix), None heisst keine
    vectors   — für die Vektorszenen: die Vektoren als Zahlen
    vector_labels, vector_note — deren Namen und das Ergebnis als Text
    matrix    — für die Szene 'abbild': zwei Zeilen [a, b] oder [a, b, t]; ein
                dritter Eintrag ist die Verschiebung (homogene Koordinaten).
                Der Körper steht dann in `vectors` als Punkte (x, y).
    """
    frames: list[list[list[str]]] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    changed: list[list[tuple[int, int]]] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    moved: list[tuple[int, int] | None] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    pivots: list[tuple[int, int] | None] = field(default_factory=list)
    solution: list[str] = field(default_factory=list)
    divider_at: int | None = -1
    vectors: list[list[float]] = field(default_factory=list)
    vector_labels: list[str] = field(default_factory=list)
    vector_note: str = ""
    matrix: list[list[float]] = field(default_factory=list)
    title: str = ""
    scene: str = "full"
    theme: Theme = field(default_factory=Theme)
    quality: str = "low"
    fmt: str = "gif"
    gif_width: int = 760
    gif_fps: int = 15

    def to_json(self) -> str:
        data = asdict(self)
        data["theme"] = asdict(self.theme)
        return json.dumps(data)

    @staticmethod
    def from_json(text: str) -> "AnimationJob":
        data = json.loads(text)
        theme = Theme(**data.pop("theme"))
        job = AnimationJob(theme=theme, **data)
        # JSON kennt keine Tupel — Listen zurückverwandeln.
        job.changed = [[tuple(c) for c in step] for step in job.changed]
        job.moved = [tuple(m) if m else None for m in job.moved]
        job.pivots = [tuple(p) if p else None for p in job.pivots]
        return job


def job_from_history(history, *, scene: str = "full", theme: Theme | None = None,
                     quality: str = "low", fmt: str = "gif",
                     decimal: bool = False, title: str = "",
                     solution: Sequence[str] = ()) -> AnimationJob:
    """
    Baut einen Auftrag aus der history der App.

    history ist eine Folge von (Matrix, Operation | None, auto). Die Matrix
    muss .rows (Brüche), .m, .n und _leading_col(r) anbieten — das trifft auf
    die Matrix-Klasse der App zu. So bleibt der Rechenweg die einzige Quelle.
    """
    from ..rechnen.core import fmt as fmt_value  # dieselbe Zahlendarstellung

    frames, captions, reasons, pivots = [], [], [], []
    changed, kinds, moved, targets = [], [], [], []
    previous = None

    for index, (matrix, op, _auto) in enumerate(history):
        rows = [[fmt_value(v, decimal) for v in row] for row in matrix.rows]
        frames.append(rows)
        captions.append("Ausgangsmatrix" if op is None else str(op))
        reasons.append("" if op is None else op.reason)

        # Das Pivot gehört zur Operation, die als Nächstes folgt — dieselbe
        # Zuordnung wie die blaue Markierung im Fenster.
        following = history[index + 1][1] if index + 1 < len(history) else None
        pivot = None
        if following is not None:
            # 'edit' ist eine Handänderung und hat kein Pivot.
            if following.kind == "swap":
                pivot = (following.i, matrix._leading_col(following.j))
            elif following.kind == "scale":
                pivot = (following.i, matrix._leading_col(following.i))
            elif following.kind == "add":
                pivot = (following.j, matrix._leading_col(following.j))
            if pivot is not None and pivot[1] < 0:
                pivot = None
        pivots.append(pivot)

        if previous is not None and op is not None:
            cells = [(r, c) for r in range(len(rows))
                     for c in range(len(rows[0])) if rows[r][c] != previous[r][c]]
            changed.append(cells)
            kinds.append(op.kind)
            moved.append((op.i, op.j) if op.kind == "swap" else None)
            targets.append(op.i)
        previous = rows

    return AnimationJob(
        frames=frames, captions=captions, reasons=reasons, changed=changed,
        kinds=kinds, moved=moved, targets=targets, pivots=pivots,
        solution=list(solution), title=title, scene=scene,
        theme=theme or Theme(), quality=quality, fmt=fmt)


def job_from_steps(steps, *, scene: str = "full", theme: Theme | None = None,
                   quality: str = "low", fmt: str = "mp4",
                   decimal: bool = False, title: str = "",
                   label: str | None = None,
                   solution: Sequence[str] = ()) -> AnimationJob:
    """
    Baut einen Auftrag aus dem Rechenweg eines lialkit-Ergebnisses.

    Ein Schritt zählt, wenn er eine Matrix mitführt — bei der LR-Zerlegung
    die wachsende rechte obere Dreiecksmatrix, bei der Inversen die Tafel
    [A | E]. Welche Zellen sich geändert haben, ergibt der Vergleich mit dem
    vorigen Bild; das ist verlässlicher, als die Zeilenoperation aus dem
    Begründungstext zurückzulesen.

    `label` wählt aus, welcher Eintrag eines Schrittes die Matrix ist — nötig
    beim letzten Schritt der LR-Zerlegung, der L und R nebeneinander zeigt.
    """
    from ..rechnen.core import Matrix as _M
    from ..rechnen.core import fmt as fmt_value
    from ..rechnen.render import tex_to_text

    frames, captions, reasons = [], [], []
    changed, kinds, moved, targets, pivots = [], [], [], [], []
    previous = None

    for step in steps:
        matrix = None
        for name, obj in step.items:
            if isinstance(obj, _M) and (label is None or name == label
                                        or matrix is None):
                matrix = obj
                if label is not None and name == label:
                    break
        if matrix is None:
            continue
        rows = [[fmt_value(v, decimal) for v in row] for row in matrix.rows]
        if previous is not None and len(rows) != len(previous):
            continue                      # Formwechsel lässt sich nicht zeigen
        frames.append(rows)
        captions.append(tex_to_text(step.title))
        # Die Begründung steht im Rechenweg als LaTeX; in der Animation gibt
        # es keinen Formelsatz, also hier in lesbaren Text wandeln.
        reasons.append(tex_to_text(step.detail))
        pivots.append(None)
        if previous is not None:
            cells = [(r, c) for r in range(len(rows))
                     for c in range(len(rows[0]))
                     if rows[r][c] != previous[r][c]]
            changed.append(cells)
            kinds.append("add")
            moved.append(None)
            # Zielzeile ist die erste geänderte Zeile.
            targets.append(cells[0][0] if cells else 0)
        previous = rows

    if len(frames) < 2:
        raise ValueError("Dieser Rechenweg hat keine Zwischenschritte zum "
                         "Zeigen.")
    if scene == "step":
        frames, captions, reasons = frames[-2:], captions[-2:], reasons[-2:]
        changed, kinds = changed[-1:], kinds[-1:]
        moved, targets, pivots = moved[-1:], targets[-1:], pivots[-2:]

    # Bei der Inversen steht [A | E] auf der Tafel: die Trennlinie gehört in
    # die Mitte. Sonst ist die Matrix nicht erweitert und braucht keine.
    width = len(frames[0][0])
    height = len(frames[0])
    divider = width // 2 if width == 2 * height else None

    return AnimationJob(
        frames=frames, captions=captions, reasons=reasons, changed=changed,
        kinds=kinds, moved=moved, targets=targets, pivots=pivots,
        solution=list(solution), title=title, divider_at=divider,
        scene="full" if scene == "step" else scene,
        theme=theme or Theme(), quality=quality, fmt=fmt)


def job_from_vectors(scene: str, vectors, labels, note: str = "", *,
                     theme: Theme | None = None, quality: str = "low",
                     fmt: str = "mp4", title: str = "") -> AnimationJob:
    """
    Baut einen Auftrag für eine der räumlichen Szenen.

    `vectors` sind die beteiligten Vektoren in Zeichenreihenfolge — bei
    Vektorprodukt und Projektion gehört das Ergebnis als dritter dazu.
    Gerechnet wird in der App exakt; fürs Zeichnen genügt Gleitkomma.
    """
    rows = [[float(c) for c in v] for v in vectors]
    if any(len(r) not in (2, 3) for r in rows):
        raise ValueError("Zeichnen lässt sich nur in der Ebene und im Raum.")
    return AnimationJob(
        frames=[], captions=[], vectors=rows, vector_labels=list(labels),
        vector_note=note, scene=scene, title=title,
        theme=theme or Theme(), quality=quality, fmt=fmt)


def job_from_abbildung(matrix, punkte, note: str = "", *,
                       theme: Theme | None = None, quality: str = "low",
                       fmt: str = "mp4", title: str = "") -> AnimationJob:
    """
    Auftrag für die Szene 'abbild': Gitter und Körper gehen stetig von der
    Identität zur Abbildung über.

    `matrix` ist 2×2 oder 3×3 (homogen, letzte Zeile 0 0 1); `punkte` sind die
    Eckpunkte des Körpers als Paare (x, y), in Zeichenreihenfolge.
    """
    rows = [[float(c) for c in r] for r in matrix]
    if len(rows) == 3 and rows[2] == [0.0, 0.0, 1.0]:
        rows = [r for r in rows[:2]]          # 2×3: A | t
    elif len(rows) == 2 and len(rows[0]) == 2:
        pass
    else:
        raise ValueError("Animiert werden nur Abbildungen der Ebene (2×2 oder "
                         "homogen 3×3).")
    pts = [[float(c) for c in p] for p in punkte]
    if not pts or any(len(p) != 2 for p in pts):
        raise ValueError("Die Punkte brauchen je zwei Koordinaten.")
    return AnimationJob(
        matrix=rows, vectors=pts, vector_note=note, scene="abbild",
        title=title, theme=theme or Theme(), quality=quality, fmt=fmt)


# --------------------------------------------------------------------------
# Verfügbarkeit
# --------------------------------------------------------------------------

def engine() -> str:
    """'manimgx' (schnell, bringt die Videoausgabe mit) wenn installiert,
    sonst 'manim'. LIALKIT_ENGINE=manim erzwingt den Rückfall."""
    forced = os.environ.get("LIALKIT_ENGINE")
    if forced in ("manim", "manimgx"):
        return forced
    import importlib.util
    return "manimgx" if importlib.util.find_spec("manimgx") else "manim"


def manim_available() -> tuple[bool, str]:
    """(verfügbar, Begründung) — prüft die Engine (bei Manim auch ffmpeg) ohne Import-Fehler."""
    name = engine()
    try:
        importlib.import_module(name)
    except ImportError:
        return False, "manimgx (oder Manim CE) ist nicht installiert."
    except Exception as exc:                       # defekte Installation
        return False, f"{name} lässt sich nicht laden: {exc}"
    if name == "manim" and shutil.which("ffmpeg") is None:
        return False, "ffmpeg fehlt — nötig für die Videoausgabe."
    return True, ""


INSTALL_HINT = ("pip install manimgx   # Rendering und Videoausgabe in einem Paket\n"
                "(Rückfall: Manim CE mit brew install cairo pango pkg-config ffmpeg; pip install manim)")


def _pick_font() -> str | None:
    """Eine Helvetica-artige Schrift, die auf dem Rechner da ist (sonst der Standard der Engine)."""
    if engine() == "manimgx":
        return "Noto Sans"                         # bringt manimgx selbst mit
    try:
        import manimpango
        have = set(manimpango.list_fonts())
    except Exception:
        return None
    for name in ("Helvetica Neue", "Helvetica", "Arial", "Inter", "DejaVu Sans"):
        if name in have:
            return name
    return None


# --------------------------------------------------------------------------
# Die Szenen
# --------------------------------------------------------------------------

def _swiss_text(Text):
    """Text-Klasse, die die Helvetica-artige Schrift als Standard setzt."""
    font = _pick_font()

    def text(*args, **kw):
        if font:
            kw.setdefault("font", font)
        return Text(*args, **kw)
    return text


def _build_scenes():
    """
    Erzeugt die Szenenklassen erst beim Aufruf, damit das Modul auch ohne
    installiertes Manim importierbar bleibt.
    """
    mod = importlib.import_module(engine())
    Scene, VGroup, VMobject, Rectangle, FadeIn, FadeOut, FadeTransform, \
        Transform, Write, DOWN, UP, ORIGIN = (
            getattr(mod, n) for n in (
                "Scene VGroup VMobject Rectangle FadeIn FadeOut FadeTransform "
                "Transform Write DOWN UP ORIGIN").split())
    Text = _swiss_text(mod.Text)

    class MatrixBoard(VGroup):
        """Die erweiterte Matrix: Zellen, Klammern, Trennlinie."""

        def __init__(self, rows: Sequence[Sequence[str]], theme: Theme,
                     cell_w: float, cell_h: float, font_size: int,
                     divider_at: int | None = -1, **kw):
            super().__init__(**kw)
            self.theme = theme
            self.cell_w, self.cell_h = cell_w, cell_h
            self.font_size = font_size
            self.n_rows = len(rows)
            self.n_cols = len(rows[0])
            # Vor welcher Spalte die gestrichelte Linie steht. -1 meint die
            # letzte Spalte, also die rechte Seite einer erweiterten Matrix.
            self.div_col = (None if divider_at is None
                            else (self.n_cols - 1 if divider_at < 0
                                  else divider_at))
            if self.div_col is not None and not 0 < self.div_col < self.n_cols:
                self.div_col = None

            self.cells = VGroup()
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    t = Text(value, font_size=font_size, color=theme.text)
                    t.move_to(self.cell_center(r, c))
                    self.cells.add(t)

            width = self.cell_w * self.n_cols + 0.7
            height = self.cell_h * self.n_rows + 0.5
            arm = 0.22
            self.brackets = VGroup(
                VMobject(color=theme.text, stroke_width=3.5).set_points_as_corners(
                    [[-width / 2 + arm, height / 2, 0], [-width / 2, height / 2, 0],
                     [-width / 2, -height / 2, 0], [-width / 2 + arm, -height / 2, 0]]),
                VMobject(color=theme.text, stroke_width=3.5).set_points_as_corners(
                    [[width / 2 - arm, height / 2, 0], [width / 2, height / 2, 0],
                     [width / 2, -height / 2, 0], [width / 2 - arm, -height / 2, 0]]))

            # Gestrichelte Linie vor der abgetrennten Spalte
            self.divider = VGroup()
            if self.div_col is not None:
                div_x = (self.cell_center(0, self.div_col - 1)[0]
                         + self.cell_center(0, self.div_col)[0]) / 2
                y = height / 2 - 0.12
                while y > -height / 2:
                    seg = VMobject(color=theme.muted, stroke_width=2)
                    seg.set_points_as_corners([[div_x, y, 0],
                                               [div_x, y - 0.1, 0]])
                    self.divider.add(seg)
                    y -= 0.22

            self.add(self.brackets, self.divider, self.cells)

        def cell_center(self, r: int, c: int):
            x = (c - (self.n_cols - 1) / 2) * self.cell_w
            # Lücke hinter der Trennlinie
            if self.div_col is not None and c >= self.div_col:
                x += 0.18
            y = ((self.n_rows - 1) / 2 - r) * self.cell_h
            return [x, y, 0]

        def cell(self, r: int, c: int) -> Text:
            return self.cells[r * self.n_cols + c]

        def row_cells(self, r: int) -> VGroup:
            return VGroup(*[self.cell(r, c) for c in range(self.n_cols)])

        def row_band(self, r: int, color: str, opacity: float = 0.5) -> Rectangle:
            band = Rectangle(
                width=self.cell_w * self.n_cols + 0.45,
                height=self.cell_h * 0.82,
                fill_color=color, fill_opacity=opacity, stroke_width=0)
            band.move_to([self.get_center()[0],
                          self.cell_center(r, 0)[1] + self.get_center()[1], 0])
            band.set_z_index(-1)
            return band

        def cell_box(self, r: int, c: int, color: str,
                     opacity: float = 0.95) -> Rectangle:
            box = Rectangle(width=self.cell_w * 0.82, height=self.cell_h * 0.7,
                            fill_color=color, fill_opacity=opacity, stroke_width=0)
            centre = self.cell_center(r, c)
            box.move_to([centre[0] + self.get_center()[0],
                         centre[1] + self.get_center()[1], 0])
            box.set_z_index(-1)
            return box

    class GaussScene(Scene):
        """Gemeinsame Grundlage: Auftrag laden, Grössen wählen, Beschriftung."""

        def setup(self):
            self.job = AnimationJob.from_json(os.environ["LIALKIT_JOB"])
            self.theme = self.job.theme
            self.camera.background_color = self.theme.background
            cols = len(self.job.frames[0][0])
            rows = len(self.job.frames[0])
            # Mit mehr Spalten wird alles kleiner, damit nichts hinausläuft.
            self.cell_w = min(1.6, 10.5 / max(cols, 1))
            self.cell_h = min(1.0, 5.0 / max(rows, 1))
            self.font_size = int(min(34, self.cell_w * 22))
            self.caption = None
            self.reason = None

        def board(self, index: int) -> "MatrixBoard":
            return MatrixBoard(self.job.frames[index], self.theme,
                               self.cell_w, self.cell_h, self.font_size,
                               divider_at=self.job.divider_at)

        def show_caption(self, board, text: str, reason: str = "",
                         animate: bool = True):
            new_cap = Text(text, font_size=int(self.font_size * 0.82),
                           color=self.theme.text)
            new_cap.next_to(board, DOWN, buff=0.55)
            new_reason = None
            if reason:
                new_reason = Text(reason, font_size=int(self.font_size * 0.6),
                                  color=self.theme.muted)
                new_reason.next_to(new_cap, DOWN, buff=0.22)

            if self.caption is None or not animate:
                self.add(new_cap)
                if new_reason:
                    self.add(new_reason)
            else:
                self.play(FadeTransform(self.caption, new_cap), run_time=0.3)
                if self.reason is not None:
                    self.remove(self.reason)
                if new_reason:
                    self.play(FadeIn(new_reason, shift=UP * 0.1), run_time=0.2)
            self.caption = new_cap
            self.reason = new_reason

        def transition(self, board, index: int, *, run_time: float = 0.95):
            """Übergang von Zustand index zu index+1 auf demselben Board."""
            target = self.board(index + 1).move_to(board.get_center())
            kind = self.job.kinds[index]
            cells = self.job.changed[index]

            if kind == "swap" and self.job.moved[index]:
                a, b = self.job.moved[index]
                band_a = board.row_band(a, self.theme.highlight)
                band_b = board.row_band(b, self.theme.highlight)
                self.play(FadeIn(band_a), FadeIn(band_b), run_time=0.3)
                row_a = board.row_cells(a)
                row_b = board.row_cells(b)
                y_a = row_a.get_center()[1]
                y_b = row_b.get_center()[1]
                self.play(row_a.animate.shift([0, y_b - y_a, 0]),
                          row_b.animate.shift([0, y_a - y_b, 0]),
                          run_time=run_time)
                # Reihenfolge im Datenmodell nachziehen
                for c in range(board.n_cols):
                    i_a, i_b = a * board.n_cols + c, b * board.n_cols + c
                    board.cells.submobjects[i_a], board.cells.submobjects[i_b] = (
                        board.cells.submobjects[i_b], board.cells.submobjects[i_a])
                self.play(FadeOut(band_a), FadeOut(band_b), run_time=0.25)
                return

            row = self.job.targets[index]
            band = board.row_band(row, self.theme.highlight)
            self.play(FadeIn(band), run_time=0.3)
            anims = [FadeTransform(board.cell(r, c), target.cell(r, c))
                     for r, c in cells]
            if anims:
                self.play(*anims, run_time=run_time)
                for r, c in cells:
                    board.cells.submobjects[r * board.n_cols + c] = target.cell(r, c)
            else:
                self.wait(run_time * 0.5)
            self.play(FadeOut(band), run_time=0.25)

    class StepScene(GaussScene):
        """Ein einzelner Umformungsschritt."""

        def construct(self):
            board = self.board(0)
            board.move_to(ORIGIN + UP * 0.35)
            self.add(board)
            self.show_caption(board, self.job.captions[1] if len(self.job.captions) > 1
                              else self.job.captions[0],
                              self.job.reasons[1] if len(self.job.reasons) > 1 else "",
                              animate=False)
            self.wait(0.6)
            if self.job.changed:
                self.transition(board, 0, run_time=1.15)
            self.wait(0.9)

    class FullScene(GaussScene):
        """Der ganze Weg von der Ausgangsmatrix bis zur Lösung."""

        def construct(self):
            board = self.board(0)
            board.move_to(ORIGIN + UP * 0.5)
            counter = Text(f"Schritt 0 von {len(self.job.frames) - 1}",
                           font_size=int(self.font_size * 0.6),
                           color=self.theme.muted)
            counter.next_to(board, UP, buff=0.5)
            self.add(board, counter)
            self.show_caption(board, self.job.captions[0], animate=False)
            self.wait(0.7)

            for index in range(len(self.job.frames) - 1):
                new_counter = Text(
                    f"Schritt {index + 1} von {len(self.job.frames) - 1}",
                    font_size=int(self.font_size * 0.6), color=self.theme.muted)
                new_counter.move_to(counter)
                self.play(FadeTransform(counter, new_counter), run_time=0.25)
                counter = new_counter
                self.show_caption(board, self.job.captions[index + 1],
                                  self.job.reasons[index + 1])
                self.transition(board, index)
                self.wait(0.15)

            if self.job.solution:
                self.wait(0.3)
                self.play(FadeOut(counter), run_time=0.3)
                lines = VGroup(*[
                    Text(line, font_size=int(self.font_size * 0.8),
                         color=self.theme.good) for line in self.job.solution])
                lines.arrange(DOWN, buff=0.18)
                lines.next_to(board, UP, buff=0.45)
                self.play(Write(lines), run_time=0.9)
            self.wait(1.1)

    class PivotScene(GaussScene):
        """Zeigt, wie das Pivot wandert und darunter Nullen entstehen."""

        def construct(self):
            board = self.board(0)
            board.move_to(ORIGIN + UP * 0.5)
            self.add(board)
            title = Text("Die Pivots wandern über die Diagonale",
                         font_size=int(self.font_size * 0.62),
                         color=self.theme.muted)
            title.next_to(board, UP, buff=0.5)
            self.add(title)
            self.show_caption(board, self.job.captions[0], animate=False)
            self.wait(0.6)

            marker = None
            trail = VGroup()
            for index in range(len(self.job.frames) - 1):
                pivot = self.job.pivots[index]
                if pivot is not None:
                    box = board.cell_box(pivot[0], pivot[1], self.theme.pivot,
                                         opacity=0.2)
                    if marker is None:
                        marker = box
                        self.play(FadeIn(marker), run_time=0.3)
                    else:
                        self.play(Transform(marker, box), run_time=0.45)
                    ghost = box.copy().set_opacity(0.3)
                    trail.add(ghost)
                    self.add(ghost)
                self.show_caption(board, self.job.captions[index + 1],
                                  self.job.reasons[index + 1])
                self.transition(board, index, run_time=0.8)

            if marker is not None:
                self.play(FadeOut(marker), run_time=0.3)
            self.wait(0.9)

    return {"step": StepScene, "full": FullScene, "pivots": PivotScene}


def _build_vector_scenes():
    """
    Die räumlichen Szenen — Vektoren, Produkte, Winkel, Projektion.

    Auch hier ohne LaTeX: Beschriftungen sind Text-Objekte. Die Kamera dreht
    sich langsam, weil erst die Bewegung zeigt, dass etwa u × v wirklich
    senkrecht auf beiden steht und nicht nur so gezeichnet ist.
    """
    mod = importlib.import_module(engine())
    Arrow3D, Create, DEGREES, DashedLine, FadeIn, Polygon, ThreeDAxes, \
        ThreeDScene, VGroup, VMobject, DOWN, UP, LEFT, ORIGIN = (
            getattr(mod, n) for n in (
                "Arrow3D Create DEGREES DashedLine FadeIn Polygon ThreeDAxes "
                "ThreeDScene VGroup VMobject DOWN UP LEFT ORIGIN").split())
    Text = _swiss_text(mod.Text)
    import numpy as np

    #: Farben der einzelnen Vektoren, in dieser Reihenfolge vergeben.
    PALETTE = ["#000000", "#666666", "#e60000", "#999999"]

    class VectorScene(ThreeDScene):
        """Grundlage: Achsen, Vektoren, Beschriftung, Kamerafahrt."""

        def setup(self):
            self.job = AnimationJob.from_json(os.environ["LIALKIT_JOB"])
            self.theme = self.job.theme
            self.camera.background_color = self.theme.background
            self.vecs = [np.array(v + [0.0] * (3 - len(v)), dtype=float)
                         for v in self.job.vectors]
            self.names = list(self.job.vector_labels)
            self.flat = all(abs(v[2]) < 1e-9 for v in self.vecs)
            span = max(2.0, max((abs(c) for v in self.vecs for c in v),
                                default=2.0))
            self.span = span * 1.25
            # Achsen starten bei 0; nur wo ein Vektor ins Negative reicht,
            # gehen sie darunter. Eine Einheit ist auf allen Achsen gleich lang.
            self.unit = 7 / (2 * self.span)      # Zeichenlänge je Einheit
            lows = [min(0.0, *(v[i] for v in self.vecs)) for i in range(3)]
            lows = [lo * 1.25 for lo in lows]
            step = max(1, round(self.span / 4))
            self.axes = ThreeDAxes(
                x_range=[lows[0], self.span, step],
                y_range=[lows[1], self.span, step],
                z_range=[lows[2], self.span, step],
                x_length=self.unit * (self.span - lows[0]),
                y_length=self.unit * (self.span - lows[1]),
                z_length=self.unit * (self.span - lows[2]))
            self.axes.set_color(self.theme.muted)
            self.axes.set_stroke(width=2)

        def pin(self, tag):
            """Beschriftung, die der Kamera zugewandt bleibt. manimgx kennt
            das nicht; dort steht sie fest im Raum."""
            if hasattr(self, "add_fixed_orientation_mobjects"):
                self.add_fixed_orientation_mobjects(tag)
            else:
                self.add(tag)

        def point(self, v) -> np.ndarray:
            """Rechnet einen Vektor in Zeichenkoordinaten um."""
            return self.axes.c2p(*v)

        def arrow(self, v, color: str, start=None) -> Arrow3D:
            base = ORIGIN if start is None else start
            return Arrow3D(start=self.point(base), end=self.point(v + base),
                           color=color, thickness=0.018,
                           base_radius=0.055, height=0.28)

        def tag(self, text: str, v, color: str, start=None):
            base = np.zeros(3) if start is None else start
            label = Text(text, font_size=26, color=color)
            label.move_to(self.point((v + base) * 1.12))
            return label

        def caption(self, text: str, sub: str = ""):
            """Beschriftung, die bei der Kamerafahrt stehen bleibt."""
            group = VGroup(Text(text, font_size=30, color=self.theme.text))
            if sub:
                group.add(Text(sub, font_size=22, color=self.theme.muted))
            group.arrange(DOWN, buff=0.18)
            group.to_corner(UP + LEFT, buff=0.5)
            self.add_fixed_in_frame_mobjects(group)
            return group

        def open_view(self, extra=()):
            """
            Blickwinkel setzen und das Bild mittig rücken.

            Alle Vektoren gehen vom Ursprung aus; liegen sie im selben
            Oktanten, hinge die Zeichnung sonst in einer Ecke. Darum die
            Achsen so verschieben, dass das Gezeigte in der Mitte steht.
            """
            if self.flat:
                self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
            else:
                self.set_camera_orientation(phi=66 * DEGREES,
                                            theta=-52 * DEGREES)
            points = ([np.zeros(3)] + list(self.vecs)
                      + [np.array(p, dtype=float) for p in extra])
            low = np.min(points, axis=0)
            high = np.max(points, axis=0)
            middle = (low + high) / 2
            self.axes.shift(self.point(np.zeros(3)) - self.point(middle))
            self.add(self.axes)

        def turn(self, seconds: float = 4.0):
            if self.flat:
                self.wait(seconds * 0.5)
                return
            self.begin_ambient_camera_rotation(rate=0.22)
            self.wait(seconds)
            self.stop_ambient_camera_rotation()

        def show_vectors(self, count: int = 2):
            drawn = []
            for i in range(count):
                arrow = self.arrow(self.vecs[i], PALETTE[i])
                tag = self.tag(self.names[i], self.vecs[i], PALETTE[i])
                self.pin(tag)
                self.play(Create(arrow), FadeIn(tag), run_time=0.7)
                drawn.append((arrow, tag))
            return drawn

        def arc(self, u, v, radius: float, color: str) -> VMobject:
            """Ein Bogen zwischen zwei Richtungen, Punkt für Punkt gesetzt."""
            a = u / max(np.linalg.norm(u), 1e-12)
            b = v / max(np.linalg.norm(v), 1e-12)
            cos = float(np.clip(np.dot(a, b), -1, 1))
            theta = np.arccos(cos)
            perp = b - cos * a
            norm = np.linalg.norm(perp)
            if norm < 1e-9:                       # parallel: kein Bogen
                return VMobject()
            perp = perp / norm
            points = [self.point(radius * (np.cos(t) * a + np.sin(t) * perp))
                      for t in np.linspace(0, theta, 40)]
            curve = VMobject(color=color, stroke_width=4)
            curve.set_points_as_corners(points)
            return curve

        def parallelogram(self, a, b, color: str, opacity=0.25, start=None):
            o = np.zeros(3) if start is None else start
            return Polygon(self.point(o), self.point(o + a),
                           self.point(o + a + b), self.point(o + b),
                           color=color, fill_opacity=opacity,
                           stroke_width=2)

    class DotScene(VectorScene):
        """u und v mit dem eingeschlossenen Winkel — darum geht es beim
        Skalarprodukt."""

        def construct(self):
            self.open_view()
            self.caption("Skalarprodukt", self.job.vector_note)
            self.show_vectors(2)
            bogen = self.arc(self.vecs[0], self.vecs[1],
                             0.45 * self.span / 1.25, self.theme.accent)
            self.play(Create(bogen), run_time=0.8)
            self.turn(4.5)

    class CrossScene(VectorScene):
        """u × v fährt senkrecht aus der von u und v aufgespannten Fläche."""

        def construct(self):
            self.open_view(extra=[self.vecs[0] + self.vecs[1], self.vecs[2]])
            self.caption("Vektorprodukt", self.job.vector_note)
            self.show_vectors(2)
            flaeche = self.parallelogram(self.vecs[0], self.vecs[1],
                                         self.theme.accent, 0.22)
            self.play(FadeIn(flaeche), run_time=0.7)
            senk = self.vecs[2]
            pfeil = self.arrow(senk, PALETTE[2])
            tag = self.tag(self.names[2], senk, PALETTE[2])
            self.pin(tag)
            self.play(Create(pfeil), FadeIn(tag), run_time=1.0)
            self.turn(5.0)

    class SpatScene(VectorScene):
        """Die drei Vektoren spannen einen Körper auf; sein Volumen ist der
        Betrag des Spatprodukts."""

        def construct(self):
            u, v, w = self.vecs[:3]
            self.open_view(extra=[u + v, u + w, v + w, u + v + w])
            self.caption("Spatprodukt", self.job.vector_note)
            self.show_vectors(3)
            # Sechs Flächen übereinander: einzeln sehr durchsichtig, sonst
            # verschwinden die Vektoren im Körper.
            faces = VGroup(
                self.parallelogram(u, v, self.theme.accent, 0.08),
                self.parallelogram(u, v, self.theme.accent, 0.08, start=w),
                self.parallelogram(u, w, self.theme.accent, 0.08),
                self.parallelogram(u, w, self.theme.accent, 0.08, start=v),
                self.parallelogram(v, w, self.theme.accent, 0.08),
                self.parallelogram(v, w, self.theme.accent, 0.08, start=u))
            self.play(FadeIn(faces), run_time=1.0)
            self.turn(5.5)

    class AngleScene(VectorScene):
        """Der Winkel zwischen u und v, als Bogen."""

        def construct(self):
            self.open_view()
            self.caption("Winkel zwischen u und v", self.job.vector_note)
            self.show_vectors(2)
            for radius in (0.3, 0.38, 0.45):
                bogen = self.arc(self.vecs[0], self.vecs[1],
                                 radius * self.span / 1.25, self.theme.accent)
                self.play(Create(bogen), run_time=0.45)
            self.turn(4.5)

    class ProjectionScene(VectorScene):
        """u fällt auf v herunter — das Lot zeigt, was Projektion heisst."""

        def construct(self):
            self.open_view(extra=[self.vecs[2]])
            self.caption("Projektion von u auf v", self.job.vector_note)
            self.show_vectors(2)
            p = self.vecs[2]
            schatten = self.arrow(p, PALETTE[2])
            lot = DashedLine(self.point(self.vecs[0]), self.point(p),
                             color=self.theme.muted, stroke_width=3)
            tag = self.tag(self.names[2], p, PALETTE[2])
            self.pin(tag)
            self.play(Create(lot), run_time=0.8)
            self.play(Create(schatten), FadeIn(tag), run_time=0.8)
            self.turn(4.5)

    return {"vec_dot": DotScene, "vec_cross": CrossScene,
            "vec_spat": SpatScene, "vec_angle": AngleScene,
            "vec_proj": ProjectionScene}


def _build_map_scenes():
    """
    Die Szene »abbild«: Einheitsgitter und Körper verformen sich stetig von
    der Identität zur Abbildung.

    Weil A linear ist, ist die Zwischenstufe (1−t)·E + t·A wieder eine
    Matrix; die Punkte wandern darum geradlinig von p nach A·p, und es genügt,
    Anfangs- und Endbild ineinander zu überführen. Auch hier ohne LaTeX.
    """
    mod = importlib.import_module(engine())
    Scene, VGroup, VMobject, Arrow, FadeIn, Transform, DOWN, UP, LEFT = (
        getattr(mod, n) for n in (
            "Scene VGroup VMobject Arrow FadeIn Transform DOWN UP LEFT".split()))
    Text = _swiss_text(mod.Text)
    import math
    import numpy as np

    class AbbildScene(Scene):
        """Gitter, Basisvektoren und Körper gehen von E zu A über."""

        def setup(self):
            self.job = AnimationJob.from_json(os.environ["LIALKIT_JOB"])
            self.theme = self.job.theme
            self.camera.background_color = self.theme.background

        def construct(self):
            th, job = self.theme, self.job
            M = np.array(job.matrix, dtype=float)
            A = M[:, :2]
            t = M[:, 2] if M.shape[1] == 3 else np.zeros(2)
            P = np.array(job.vectors, dtype=float).T            # 2×n
            bild = A @ P + t[:, None]
            ecken = np.hstack([P, bild, np.zeros((2, 1)), np.eye(2), A + t[:, None]])
            low, high = ecken.min(axis=1), ecken.max(axis=1)
            weite = np.maximum(high - low, 1.0)
            u = min(11.0 / weite[0], 5.0 / weite[1])            # Zeichenlänge je Einheit
            mitte = (low + high) / 2
            versatz = np.array([0.0, -0.5])

            def pt(x, y):
                return np.array([(x - mitte[0]) * u + versatz[0],
                                 (y - mitte[1]) * u + versatz[1], 0.0])

            def linie(punkte, farbe, breite, deckkraft=1.0, schluss=False):
                pts = [pt(*p) for p in punkte]
                if schluss:
                    pts.append(pts[0])
                m = VMobject(color=farbe, stroke_width=breite,
                             stroke_opacity=deckkraft)
                m.set_points_as_corners(pts)
                return m

            spanne = float(max(abs(ecken).max(), 1.0))
            schritt = next(s for s in (1, 2, 5, 10, 20, 50, 100, 1e3, 1e4)
                           if spanne / s <= 6)
            n = int(min(15, math.ceil(2.2 * spanne / schritt)))
            rand = n * schritt

            def gitter(f, farbe, breite, deckkraft):
                g = VGroup()
                for i in range(-n, n + 1):
                    w = i * schritt
                    for a, b in (((w, -rand), (w, rand)), ((-rand, w), (rand, w))):
                        g.add(linie([f(*a), f(*b)], farbe,
                                    breite * (2 if i == 0 else 1),
                                    1.0 if i == 0 else deckkraft))
                return g

            ident = lambda x, y: (x, y)                          # noqa: E731
            wirkung = lambda x, y: tuple(A @ np.array([x, y]) + t)   # noqa: E731
            start = gitter(ident, th.muted, 1.4, 0.45)
            ziel = gitter(wirkung, th.muted, 1.4, 0.45)
            koerper_0 = linie(P.T, th.accent, 5, schluss=True)
            koerper = linie(P.T, th.text, 4, schluss=True)
            koerper_1 = linie(bild.T, th.text, 4, schluss=True)

            kopf = VGroup(Text("Lineare Abbildung  x \u21a6 A \u00b7 x", font_size=30,
                               color=th.text))
            if job.vector_note:
                kopf.add(Text(job.vector_note, font_size=22, color=th.muted))
            kopf.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
            kopf.to_corner(UP + LEFT, buff=0.5)
            legende = VGroup(
                Text("Original", font_size=22, color=th.accent),
                Text("Bild", font_size=22, color=th.text))
            legende.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            legende.to_corner(UP + mod.RIGHT, buff=0.5)

            # Basisvektoren: e1, e2 gehen in die Spalten von A (ohne die
            # Verschiebung) über. Entartete Bilder (Länge 0) bleiben weg.
            pfeile0, pfeile1 = VGroup(), VGroup()
            for k in range(2):
                ende = A[:, k] + t
                if np.linalg.norm(A[:, k]) > 1e-6:
                    pfeile0.add(Arrow(pt(*t), pt(*(np.eye(2)[k] + t)),
                                      buff=0, color=th.accent, stroke_width=6))
                    pfeile1.add(Arrow(pt(*t), pt(*ende), buff=0,
                                      color=th.accent, stroke_width=6))

            self.play(FadeIn(start), FadeIn(koerper_0), FadeIn(kopf), FadeIn(legende),
                      run_time=1.0)
            self.add(koerper)
            if len(pfeile0):
                self.play(FadeIn(pfeile0), run_time=0.5)
            self.wait(0.6)
            self.play(Transform(start, ziel), Transform(koerper, koerper_1),
                      *( [Transform(pfeile0, pfeile1)] if len(pfeile0) else [] ),
                      run_time=3.2)
            self.wait(1.8)

    return {"abbild": AbbildScene}


SCENE_NAMES = {"step": "StepScene", "full": "FullScene", "pivots": "PivotScene",
               "vec_dot": "DotScene", "vec_cross": "CrossScene",
               "vec_spat": "SpatScene", "vec_angle": "AngleScene",
               "vec_proj": "ProjectionScene", "abbild": "AbbildScene"}


# --------------------------------------------------------------------------
# Rendern
# --------------------------------------------------------------------------

#: Wird vom Kindprozess ausgeführt. Lädt den Auftrag aus der Umgebung.
#: Die Engine sucht Szenenklassen nur in diesem Modul, darum wird __module__
#: der hier erzeugten Klassen umgehängt.
_RUNNER = '''
import importlib, importlib.util, os, sys
# Der Ordner ist das Paket, auch wenn er nicht so heisst.
if {pkg!r} not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        {pkg!r}, os.path.join({here!r}, "__init__.py"),
        submodule_search_locations=[{here!r}])
    _p = importlib.util.module_from_spec(_spec)
    sys.modules[{pkg!r}] = _p
    _spec.loader.exec_module(_p)
_mod = importlib.import_module({pkg!r} + ".grafik.anim")

_build = getattr(_mod, {builder!r})
for _key, _cls in _build().items():
    _cls.__module__ = __name__
    globals()[_mod.SCENE_NAMES[_key]] = _cls
'''

#: Welche Funktion die Szene baut; alles andere kommt aus _build_scenes.
_BUILDERS = {"abbild": "_build_map_scenes",
             **{k: "_build_vector_scenes" for k in SCENE_NAMES if k.startswith("vec_")}}

_PROGRESS = re.compile(r"(\d+)\s*%")


def _package() -> tuple[str, str]:
    """(Name des Pakets, Ordner, der es enthält) — damit der Kindprozess es importieren kann."""
    here = Path(__file__).resolve().parent.parent
    return (__package__ or here.name).split(".")[0], str(here.parent)


def render(job: AnimationJob, *, progress: Callable[[int, str], None] | None = None,
           out_dir: str | Path | None = None,
           timeout: float = 600) -> Path:
    """
    Rendert den Auftrag und gibt den Pfad der fertigen Datei zurück.

    Die Engine (manimgx, sonst Manim CE) läuft in einem eigenen Prozess: Sie
    schreibt viel in globale Konfiguration und lässt sich im selben Prozess
    nicht sauber mehrfach starten. progress bekommt (Prozent, Text). Fehlt
    die Engine, gibt es einen RuntimeError mit Installationshinweis.
    """
    ok, why = manim_available()
    if not ok:
        raise RuntimeError(f"{why}\nInstallation:\n{INSTALL_HINT}")
    if job.scene not in SCENE_NAMES:
        raise ValueError(f"Unbekannte Szene {job.scene!r}")

    pkg, root = _package()
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="lialkit-anim-"))
    work.mkdir(parents=True, exist_ok=True)
    script = work / "scene_runner.py"
    script.write_text(
        _RUNNER.format(pkg=pkg, here=str(Path(__file__).resolve().parent.parent),
                       builder=_BUILDERS.get(job.scene, "_build_scenes")),
        encoding="utf-8")

    folder, flag = QUALITIES.get(job.quality, QUALITIES["low"])
    name = SCENE_NAMES[job.scene]
    # Immer MP4 rendern. Manims eigener GIF-Export erzeugt eine fehlerhafte
    # Palette (schwarze Schrift wird gelb); das GIF entsteht darum unten in
    # einem zweiten Schritt mit palettegen.
    if engine() == "manimgx":
        size, fps = folder.split("p")
        res = {"480": "854x480", "720": "1280x720", "1080": "1920x1080",
               "1440": "2560x1440"}[size]
        out = work / "media" / "lialkit.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "manimgx", "render", str(script), name,
               "-o", str(out), "-r", res, "--fps", fps]
    else:
        cmd = [sys.executable, "-m", "manim", flag, "--disable_caching",
               "--media_dir", str(work / "media"), "-o", "lialkit",
               str(script), name]

    env = dict(os.environ)
    env["LIALKIT_JOB"] = job.to_json()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    if progress:
        progress(2, "Animation startet …")

    process = subprocess.Popen(cmd, cwd=str(work), env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=1)
    # Die Engine zählt je Animation von 0 bis 100. Für einen Balken, der nicht
    # zurückspringt, zählen wir die abgeschlossenen Animationen mit.
    # Die Vektorszenen haben keine Bilderfolge; dort ist die Zahl der
    # Teilanimationen ungefähr fest.
    expected = (9 if job.scene.startswith("vec_") else 4 if job.scene == "abbild"
                else max(1, len(job.frames) * 3))
    log: list[str] = []
    done = last = 0
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, stop)
    timer.start()
    try:
        for line in process.stdout:
            log.append(line)
            for match in _PROGRESS.finditer(line):
                within = int(match.group(1))
                if within >= 100:
                    done += 1
                overall = min(97, int((done + within / 100) / expected * 95) + 3)
                if overall > last and progress:
                    last = overall
                    progress(overall, "Rendere …")
        process.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise RuntimeError("Das Rendern hat zu lange gedauert.")

    output = "".join(log)
    # Manche Fehler stehen nur im Text, der Prozess endet trotzdem mit 0.
    if process.returncode != 0 or "Traceback" in output:
        lines = [l.rstrip() for l in log if l.strip()]
        tail = "\n".join(lines[-12:]) or "Keine Ausgabe."
        raise RuntimeError(f"Das Rendern ist fehlgeschlagen:\n{tail}")

    candidates = sorted((work / "media").rglob("lialkit*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Es wurde keine Datei erzeugt.")
    video = candidates[0]

    if job.fmt == "gif":
        if progress:
            progress(98, "Erzeuge GIF …")
        video = to_gif(video, work / "lialkit.gif",
                       width=job.gif_width, fps=job.gif_fps)
    if progress:
        progress(100, "Fertig.")
    return video


def to_gif(video: Path, target: Path, *, width: int = 760, fps: int = 15) -> Path:
    """
    Wandelt das Video in ein GIF mit eigener Palette um.

    ffmpeg erzeugt die Palette aus dem Material selbst (palettegen), sonst
    landet die Schrift auf einer Standardpalette und verfärbt sich.
    Hochskaliert wird nie: ist das Video schmaler als `width`, bleibt es so,
    sonst wäre das Ergebnis nur unscharf aufgeblasen.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg fehlt — nötig für die GIF-Ausgabe.")
    chain = (f"fps={fps},scale='min({width},iw)':-1:flags=lanczos,"
             f"split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
             f"[s1][p]paletteuse=dither=bayer:bayer_scale=3")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", chain, str(target)],
        capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"GIF-Umwandlung fehlgeschlagen:\n"
                           f"{result.stderr.strip()[-400:]}")
    return target


# --------------------------------------------------------------------------
# Beispiel von der Kommandozeile
# --------------------------------------------------------------------------

def _demo_job(scene: str = "full") -> AnimationJob:
    frames = [
        [["2", "1", "−1", "8"], ["−3", "−1", "2", "−11"], ["−2", "1", "2", "−3"]],
        [["2", "1", "−1", "8"], ["0", "1/2", "1/2", "1"], ["−2", "1", "2", "−3"]],
        [["2", "1", "−1", "8"], ["0", "1/2", "1/2", "1"], ["0", "2", "1", "5"]],
        [["2", "1", "−1", "8"], ["0", "1/2", "1/2", "1"], ["0", "0", "−1", "1"]],
    ]
    captions = ["Ausgangsmatrix", "R2 ← R2 + 3/2 · R1", "R3 ← R3 + R1",
                "R3 ← R3 − 4 · R2"]
    reasons = ["", "Eintrag unter dem Pivot in Spalte 1 zu 0 machen.",
               "Eintrag unter dem Pivot in Spalte 1 zu 0 machen.",
               "Eintrag unter dem Pivot in Spalte 2 zu 0 machen."]
    changed, kinds, targets = [], [], []
    for i in range(len(frames) - 1):
        changed.append([(r, c) for r in range(3) for c in range(4)
                        if frames[i + 1][r][c] != frames[i][r][c]])
        kinds.append("add")
        targets.append(i + 1 if i < 2 else 2)
    return AnimationJob(
        frames=frames, captions=captions, reasons=reasons, changed=changed,
        kinds=kinds, moved=[None] * 3, targets=targets,
        pivots=[(0, 0), (1, 1), (2, 2), None],
        solution=["x1 = 2", "x2 = 3", "x3 = −1"], scene=scene)


def _demo(argv: list[str]) -> int:
    scene = argv[0] if argv else "full"
    ok, why = manim_available()
    if not ok:
        print(why)
        print(INSTALL_HINT)
        return 1
    path = render(_demo_job(scene), progress=lambda p, t: print(f"{p:3d}%  {t}"))
    print("Fertig:", path)
    return 0


if __name__ == "__main__":
    if not __package__:
        # Direkt gestartet: das Paket über seinen Ordner importierbar machen.
        _here = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_here.parent))
        sys.exit(importlib.import_module(_here.name + ".grafik.anim")._demo(sys.argv[1:]))
    sys.exit(_demo(sys.argv[1:]))
