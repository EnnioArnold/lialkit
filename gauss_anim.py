#!/usr/bin/env python3
"""
Manim-Animationen für die Gauss-Elimination.

Bewusst ohne LaTeX: Die Matrix wird aus Text-Objekten und selbst gezeichneten
Klammern aufgebaut. Dadurch genügen Cairo und Pango, eine TeX-Installation
ist nicht nötig.

Installation:
    brew install cairo pango pkg-config ffmpeg      # macOS
    pip install manim

Verwendung aus der App:
    from gauss_anim import AnimationJob, render
    job = AnimationJob(frames=[...], captions=[...], ...)
    pfad = render(job, progress=print)

Direkt von der Kommandozeile (rendert ein Beispiel):
    python3 gauss_anim.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
}


@dataclass
class Theme:
    """Farben der Animation, aus dem Thema der App übernommen."""
    background: str = "#ffffff"
    text: str = "#1d1d1f"
    muted: str = "#86868b"
    accent: str = "#0071e3"
    highlight: str = "#fdeab0"
    pivot: str = "#d8e8fb"
    good: str = "#1d8a3c"


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
    scene     — 'step' | 'full' | 'pivots'
    """
    frames: list[list[list[str]]]
    captions: list[str]
    reasons: list[str] = field(default_factory=list)
    changed: list[list[tuple[int, int]]] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    moved: list[tuple[int, int] | None] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    pivots: list[tuple[int, int] | None] = field(default_factory=list)
    solution: list[str] = field(default_factory=list)
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
    from gauss_qt import fmt as fmt_value  # dieselbe Zahlendarstellung

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


# --------------------------------------------------------------------------
# Verfügbarkeit
# --------------------------------------------------------------------------

def manim_available() -> tuple[bool, str]:
    """(verfügbar, Begründung) — prüft Manim und ffmpeg ohne Import-Fehler."""
    try:
        import manim  # noqa: F401
    except ImportError:
        return False, "Manim ist nicht installiert."
    except Exception as exc:                       # defekte Installation
        return False, f"Manim lässt sich nicht laden: {exc}"
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg fehlt — nötig für die Videoausgabe."
    return True, ""


INSTALL_HINT = ("brew install cairo pango pkg-config ffmpeg\n"
                "pip install manim")


# --------------------------------------------------------------------------
# Die Szenen
# --------------------------------------------------------------------------

def _build_scenes():
    """
    Erzeugt die Szenenklassen erst beim Aufruf, damit das Modul auch ohne
    installiertes Manim importierbar bleibt.
    """
    from manim import (Scene, VGroup, VMobject, Text, Rectangle, FadeIn, FadeOut,
                       FadeTransform, Transform, Write, config, DOWN, UP, ORIGIN)

    class MatrixBoard(VGroup):
        """Die erweiterte Matrix: Zellen, Klammern, Trennlinie."""

        def __init__(self, rows: Sequence[Sequence[str]], theme: Theme,
                     cell_w: float, cell_h: float, font_size: int, **kw):
            super().__init__(**kw)
            self.theme = theme
            self.cell_w, self.cell_h = cell_w, cell_h
            self.font_size = font_size
            self.n_rows = len(rows)
            self.n_cols = len(rows[0])

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

            # Gestrichelte Linie vor der rechten Seite
            div_x = (self.cell_center(0, self.n_cols - 2)[0]
                     + self.cell_center(0, self.n_cols - 1)[0]) / 2
            self.divider = VGroup()
            y = height / 2 - 0.12
            while y > -height / 2:
                seg = VMobject(color=theme.muted, stroke_width=2)
                seg.set_points_as_corners([[div_x, y, 0], [div_x, y - 0.1, 0]])
                self.divider.add(seg)
                y -= 0.22

            self.add(self.brackets, self.divider, self.cells)

        def cell_center(self, r: int, c: int):
            x = (c - (self.n_cols - 1) / 2) * self.cell_w
            # Lücke vor der letzten Spalte
            if c == self.n_cols - 1:
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
            self.job = AnimationJob.from_json(os.environ["GAUSS_JOB"])
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
                               self.cell_w, self.cell_h, self.font_size)

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
                    box = board.cell_box(pivot[0], pivot[1], self.theme.pivot)
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


SCENE_NAMES = {"step": "StepScene", "full": "FullScene", "pivots": "PivotScene"}


# --------------------------------------------------------------------------
# Rendern
# --------------------------------------------------------------------------

#: Wird vom Kindprozess ausgeführt. Lädt den Auftrag aus der Umgebung.
#: Manim sucht Szenenklassen nur in diesem Modul, darum wird __module__
#: der hier erzeugten Klassen umgehängt.
_RUNNER = '''
import os, sys
sys.path.insert(0, {app_dir!r})
from gauss_anim import _build_scenes
_scenes = _build_scenes()
StepScene = _scenes["step"]
FullScene = _scenes["full"]
PivotScene = _scenes["pivots"]
for _cls in (StepScene, FullScene, PivotScene):
    _cls.__module__ = __name__
'''

_PROGRESS = re.compile(r"(\d+)\s*%")


def render(job: AnimationJob, *, progress: Callable[[int, str], None] | None = None,
           out_dir: str | Path | None = None,
           timeout: float = 300) -> Path:
    """
    Rendert den Auftrag und gibt den Pfad der fertigen Datei zurück.

    Manim läuft in einem eigenen Prozess: Es schreibt viel in globale
    Konfiguration und lässt sich im selben Prozess nicht sauber mehrfach
    starten. progress bekommt (Prozent, Text).
    """
    ok, why = manim_available()
    if not ok:
        raise RuntimeError(why)

    app_dir = str(Path(__file__).resolve().parent)
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="gauss-anim-"))
    work.mkdir(parents=True, exist_ok=True)
    script = work / "scene_runner.py"
    script.write_text(_RUNNER.format(app_dir=app_dir), encoding="utf-8")

    folder, flag = QUALITIES.get(job.quality, QUALITIES["low"])
    name = SCENE_NAMES[job.scene]
    # Immer MP4 rendern. Manims eigener GIF-Export erzeugt eine fehlerhafte
    # Palette (schwarze Schrift wird gelb); das GIF entsteht darum unten in
    # einem zweiten Schritt mit palettegen.
    cmd = [sys.executable, "-m", "manim", flag, "--disable_caching",
           "--media_dir", str(work / "media"), "-o", "gauss",
           str(script), name]

    env = dict(os.environ)
    env["GAUSS_JOB"] = job.to_json()
    env["PYTHONPATH"] = app_dir + os.pathsep + env.get("PYTHONPATH", "")

    if progress:
        progress(2, "Manim startet …")

    process = subprocess.Popen(cmd, cwd=str(work), env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=1)
    # Manim zählt je Animation von 0 bis 100. Für einen Balken, der nicht
    # zurückspringt, zählen wir die abgeschlossenen Animationen mit.
    expected = max(1, len(job.frames) * 3)
    log: list[str] = []
    done = 0
    last = 0
    try:
        for line in process.stdout:
            log.append(line)
            match = _PROGRESS.search(line)
            if match and progress:
                within = int(match.group(1))
                if within >= 100:
                    done += 1
                overall = min(97, int((done + within / 100) / expected * 95) + 3)
                if overall > last:
                    last = overall
                    progress(overall, "Rendere …")
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("Das Rendern hat zu lange gedauert.")

    output = "".join(log)
    # Manim meldet manche Fehler nur im Text und beendet trotzdem mit 0.
    if process.returncode != 0 or "ERROR" in output:
        lines = [l.rstrip() for l in log if l.strip()]
        tail = "\n".join(lines[-10:]) or "Keine Ausgabe."
        raise RuntimeError(f"Manim ist fehlgeschlagen:\n{tail}")

    candidates = sorted((work / "media").rglob("gauss*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Manim hat keine Datei erzeugt.")
    video = candidates[0]

    if job.fmt == "gif":
        if progress:
            progress(98, "Erzeuge GIF …")
        video = to_gif(video, work / "gauss.gif",
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


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "full"
    ok, why = manim_available()
    if not ok:
        print(why)
        print(INSTALL_HINT)
        sys.exit(1)
    path = render(_demo_job(scene), progress=lambda p, t: print(f"{p:3d}%  {t}"))
    print("Fertig:", path)
