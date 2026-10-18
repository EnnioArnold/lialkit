#!/usr/bin/env python3
"""
Darstellung: LaTeX für Jupyter, Text für die Konsole.

Jedes Ergebnis von lialkit ist ein Objekt, das sich selbst anzeigen kann.
Im Notebook erscheint es gesetzt, im Terminal als lesbarer Text — dieselbe
Rechnung, zwei Ausgaben.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Sequence

from .core import MINUS, Matrix, Num, Vector, fmt, tex_number


# ==========================================================================
# LaTeX
# ==========================================================================

def tex_matrix(M: Matrix, decimal: bool = False, digits: int = 4,
               highlight: Sequence[int] = ()) -> str:
    """
    Matrix als LaTeX. Eine erweiterte Matrix bekommt den senkrechten Strich
    vor der rechten Seite, damit man sieht, wo das Gleichheitszeichen stand.
    """
    highlight = set(highlight)
    body = []
    for r, row in enumerate(M.rows):
        cells = [tex_number(v, decimal, digits) for v in row]
        if r in highlight:
            cells = [r"\mathbf{" + c + "}" for c in cells]
        body.append(" & ".join(cells))
    inner = r" \\ ".join(body)
    if M.augmented:
        spec = "c" * M.n + "|c"
        return r"\left(\begin{array}{" + spec + "}" + inner + r"\end{array}\right)"
    return r"\begin{pmatrix}" + inner + r"\end{pmatrix}"


def tex_vector(v: Vector, decimal: bool = False, digits: int = 4,
               row: bool = False) -> str:
    cells = [tex_number(x, decimal, digits) for x in v.values]
    if row:
        return r"\begin{pmatrix}" + " & ".join(cells) + r"\end{pmatrix}"
    return r"\begin{pmatrix}" + r" \\ ".join(cells) + r"\end{pmatrix}"


def tex_any(obj, decimal: bool = False, digits: int = 4) -> str:
    if isinstance(obj, Matrix):
        return tex_matrix(obj, decimal, digits)
    if isinstance(obj, Vector):
        return tex_vector(obj, decimal, digits)
    if isinstance(obj, (int, float)):
        return tex_number(obj, decimal, digits)
    try:
        return tex_number(obj, decimal, digits)
    except Exception:
        return r"\text{" + str(obj) + "}"


# ==========================================================================
# HTML
# ==========================================================================

#: LaTeX-Schnipsel, die in der HTML-Anzeige als Text erscheinen sollen.
#: Die Notebook-Ausgabe wird nicht überall von MathJax gesetzt — in einer
#: nach HTML exportierten Datei zum Beispiel gar nicht. Darum zeigt die
#: HTML-Ansicht lesbare Zeichen statt roher Formeln; `latex()` liefert
#: weiterhin echtes LaTeX für die Abgabe.
_TEX_REPLACEMENTS = [
    (r"\leftrightarrow", "\u2194"), (r"\leftarrow", "\u2190"),
    (r"\Rightarrow", "\u21d2"), (r"\cdot", "\u00b7"), (r"\times", "\u00d7"),
    (r"\le", "\u2264"), (r"\ge", "\u2265"), (r"\neq", "\u2260"),
    (r"\lambda", "\u03bb"), (r"\mu", "\u03bc"), (r"\varphi", "\u03c6"),
    (r"\circ", "\u00b0"), (r"\in", "\u2208"), (r"\mathbb{R}", "\u211d"),
    (r"\operatorname{rg}", "rg"), (r"\operatorname{adj}", "adj"),
    (r"\det", "det"), (r"\quad", "   "), (r"\;", " "), (r"\,", " "),
    (r"\mid", "|"), (r"\\", "   "),
]

_SUBSCRIPTS = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084"
                                          "\u2085\u2086\u2087\u2088\u2089")
_SUPERSCRIPTS = str.maketrans("0123456789-n",
                              "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076"
                              "\u2077\u2078\u2079\u207b\u207f")


def tex_to_text(s: str) -> str:
    """Macht aus einem LaTeX-Schnipsel lesbaren Text für die HTML-Ansicht."""
    import re

    out = s
    for _ in range(4):                       # \frac von innen nach aussen
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", out)
        if new == out:
            break
        out = new
    out = re.sub(r"\\begin\{[a-z]*matrix\}.*?\\end\{[a-z]*matrix\}", "(…)",
                 out, flags=re.S)
    out = re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", out)
    for needle, replacement in _TEX_REPLACEMENTS:
        out = out.replace(needle, replacement)
    out = re.sub(r"_\{([0-9]+)\}",
                 lambda m: m.group(1).translate(_SUBSCRIPTS), out)
    out = re.sub(r"_([0-9])", lambda m: m.group(1).translate(_SUBSCRIPTS), out)
    out = out.replace("^°", "°")          # ^\circ war schon ersetzt
    out = re.sub(r"\^\{(-?[0-9n]+)\}",
                 lambda m: m.group(1).translate(_SUPERSCRIPTS), out)
    out = re.sub(r"\^(-?[0-9n])",
                 lambda m: m.group(1).translate(_SUPERSCRIPTS), out)
    out = out.replace("{", "").replace("}", "").replace("$", "")
    out = re.sub(r"\s{2,}", "   ", out)
    return out.strip()


def tex_row_op(target: int, source: int, factor, decimal: bool = False,
               digits: int = 4) -> str:
    """
    'R_i ← R_i + k · R_j' als LaTeX, mit dem Vorzeichen im Operator.

    Ein negativer Faktor darf nicht als 'R_i − (−3/2) · R_j' erscheinen —
    so schreibt es niemand auf dem Papier.
    """
    sign = "-" if factor < 0 else "+"
    magnitude = abs(factor)
    body = ("" if magnitude == 1
            else tex_number(magnitude, decimal, digits) + r" \cdot ")
    return (f"R_{{{target + 1}}} \\leftarrow R_{{{target + 1}}} {sign} "
            f"{body}R_{{{source + 1}}}")


def html_number(x, decimal: bool = False, digits: int = 4) -> str:
    """Zahl als HTML; echte Brüche übereinander mit Bruchstrich."""
    if isinstance(x, float) or decimal or getattr(x, "denominator", 1) == 1:
        return html.escape(fmt(x, decimal, digits))
    sign = MINUS if x < 0 else ""
    return (f"{sign}<span class='lk-fr'>"
            f"<span>{abs(x.numerator)}</span>"
            f"<span>{x.denominator}</span></span>")


def html_matrix(M: Matrix, decimal: bool = False, digits: int = 4) -> str:
    """Matrix als HTML-Tabelle mit gezeichneten Klammern."""
    rows = []
    for row in M.rows:
        parts = []
        for c, v in enumerate(row):
            klass = " class='lk-aug'" if (M.augmented and c == M.n) else ""
            parts.append(f"<td{klass}>{html_number(v, decimal, digits)}</td>")
        rows.append("<tr>" + "".join(parts) + "</tr>")
    return ("<span class='lk-mx'><span class='lk-br lk-brl'></span>"
            "<table>" + "".join(rows) + "</table>"
            "<span class='lk-br lk-brr'></span></span>")


def html_vector(v: Vector, decimal: bool = False, digits: int = 4) -> str:
    cells = "".join(f"<tr><td>{html_number(x, decimal, digits)}</td></tr>"
                    for x in v.values)
    return ("<span class='lk-mx'><span class='lk-br lk-brl'></span>"
            f"<table>{cells}</table>"
            "<span class='lk-br lk-brr'></span></span>")


def html_any(obj, decimal: bool = False, digits: int = 4) -> str:
    if isinstance(obj, Matrix):
        return html_matrix(obj, decimal, digits)
    if isinstance(obj, Vector):
        return html_vector(obj, decimal, digits)
    if isinstance(obj, str):
        return html.escape(tex_to_text(obj))
    try:
        return html_number(obj, decimal, digits)
    except Exception:
        return html.escape(str(obj))


def text_matrix(M: Matrix, decimal: bool = False) -> str:
    """Matrix als ausgerichteter Text, für die Konsole."""
    cells = [[fmt(v, decimal) for v in row] for row in M.rows]
    widths = [max(len(cells[r][c]) for r in range(M.m)) for c in range(M.width)]
    lines = []
    for row in cells:
        parts = [row[c].rjust(widths[c]) for c in range(M.n)]
        line = "  ".join(parts)
        if M.augmented:
            line += "  |  " + row[-1].rjust(widths[-1])
        lines.append("[ " + line + " ]")
    return "\n".join(lines)


# ==========================================================================
# Schritte
# ==========================================================================

@dataclass
class Step:
    """
    Ein Schritt im Rechenweg.

    `items` sind benannte Grössen, die zu diesem Schritt gehören —
    typischerweise die Matrix danach, manchmal mehrere nebeneinander.
    """
    title: str
    detail: str = ""
    items: list[tuple[str, object]] = field(default_factory=list)
    note: str = ""

    def tex_items(self, decimal: bool, digits: int) -> str:
        if not self.items:
            return ""
        parts = []
        for label, obj in self.items:
            piece = tex_any(obj, decimal, digits)
            parts.append(f"{label} = {piece}" if label else piece)
        return r"\qquad ".join(parts)

    def html_items(self, decimal: bool, digits: int) -> str:
        if not self.items:
            return ""
        parts = []
        for label, obj in self.items:
            piece = html_any(obj, decimal, digits)
            name = (f"<span class='lk-lbl'>{html.escape(tex_to_text(label))}"
                    f" =</span>" if label else "")
            parts.append(f"<span class='lk-item'>{name}{piece}</span>")
        return "".join(parts)


class Result:
    """
    Gemeinsame Grundlage aller Rechenergebnisse.

    Unterklassen setzen `title`, `summary`, `steps` und `highlights`
    (die Grössen, die am Ende zählen).
    """

    title = "Ergebnis"
    summary = ""

    def __init__(self, decimal: bool = False, digits: int = 4):
        self.decimal = decimal
        self.digits = digits
        self.steps: list[Step] = []
        self.highlights: list[tuple[str, object]] = []
        self.warnings: list[str] = []

    # -- Anzeige ----------------------------------------------------------

    def as_decimal(self, digits: int = 4) -> "Result":
        """Dieselbe Rechnung dezimal angezeigt."""
        self.decimal, self.digits = True, digits
        return self

    def as_exact(self) -> "Result":
        self.decimal = False
        return self

    def latex(self, with_steps: bool = True) -> str:
        """Der ganze Rechenweg als LaTeX — zum Einfügen in eine Abgabe."""
        out = [r"\textbf{" + self.title + "}"]
        if with_steps:
            for index, step in enumerate(self.steps, 1):
                out.append(r"\text{" + f"{index}. {step.title}" + "}")
                if step.detail:
                    out.append(r"\quad " + step.detail)
                items = step.tex_items(self.decimal, self.digits)
                if items:
                    out.append(items)
        for label, obj in self.highlights:
            out.append(f"{label} = {tex_any(obj, self.decimal, self.digits)}")
        return "\n\\\\[6pt]\n".join(out)

    def _repr_html_(self) -> str:
        return self._html(with_steps=bool(self.steps))

    #: Eigenes CSS statt MathJax: so sieht die Ausgabe in JupyterLab, in
    #: VS Code, in Colab und in einer exportierten HTML-Datei gleich aus.
    CSS = (
        "<style>"
        ".lk{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Helvetica,Arial,sans-serif;color:#1d1d1f;line-height:1.5;}"
        ".lk-h{font-weight:600;font-size:15px;margin:0 0 2px;}"
        ".lk-sum{color:#6e6e73;font-size:13px;margin:0 0 14px;}"
        ".lk-step{background:#f5f5f7;border-radius:12px;padding:12px 16px;"
        "margin:8px 0;}"
        ".lk-t{font-weight:600;font-size:13px;}"
        ".lk-d{color:#6e6e73;font-size:12.5px;margin-top:3px;"
        "font-variant-numeric:tabular-nums;}"
        ".lk-n{color:#86868b;font-size:11.5px;margin-top:5px;}"
        ".lk-res{background:#e3effc;border-radius:12px;padding:12px 16px;"
        "margin:12px 0 0;}"
        ".lk-warn{background:#fdf3d8;border-radius:10px;padding:8px 14px;"
        "margin:8px 0;color:#a86400;font-size:12px;}"
        ".lk-item{display:inline-flex;align-items:center;gap:7px;"
        "margin:8px 22px 4px 0;vertical-align:middle;}"
        ".lk-lbl{font-size:13px;font-weight:500;}"
        ".lk-mx{display:inline-flex;align-items:stretch;vertical-align:middle;}"
        ".lk-mx table{border-collapse:collapse;margin:0;border:none;}"
        ".lk-mx td{padding:3px 9px;text-align:center;border:none;"
        "font-size:13.5px;font-variant-numeric:tabular-nums;background:none;}"
        ".lk-mx tr{background:none !important;border:none;}"
        ".lk-mx td.lk-aug{border-left:1px dashed #b0b0b8;}"
        ".lk-br{width:6px;border:1.5px solid currentColor;}"
        ".lk-brl{border-right:none;border-radius:4px 0 0 4px;}"
        ".lk-brr{border-left:none;border-radius:0 4px 4px 0;}"
        ".lk-fr{display:inline-flex;flex-direction:column;align-items:center;"
        "line-height:1.1;vertical-align:middle;font-size:11px;}"
        ".lk-fr span:first-child{border-bottom:1.2px solid currentColor;"
        "padding:0 3px 1px;}"
        ".lk-fr span:last-child{padding-top:1px;}"
        "@media (prefers-color-scheme:dark){"
        ".lk{color:#f5f5f7;}.lk-sum,.lk-d{color:#a1a1a6;}"
        ".lk-step{background:#232325;}.lk-res{background:#16324f;}"
        ".lk-warn{background:#3a3113;color:#ffd60a;}"
        ".lk-mx td.lk-aug{border-left-color:#6e6e73;}}"
        "</style>")

    def _html(self, with_steps: bool) -> str:
        parts = [self.CSS, "<div class='lk'>",
                 f"<p class='lk-h'>{html.escape(self.title)}</p>"]
        if self.summary:
            parts.append(f"<p class='lk-sum'>{html.escape(self.summary)}</p>")
        for warning in self.warnings:
            parts.append(f"<div class='lk-warn'>{html.escape(warning)}</div>")
        if with_steps:
            for index, step in enumerate(self.steps, 1):
                parts.append("<div class='lk-step'>")
                parts.append(f"<div class='lk-t'>{index}. "
                             f"{html.escape(step.title)}</div>")
                if step.detail:
                    parts.append(f"<div class='lk-d'>"
                                 f"{html.escape(tex_to_text(step.detail))}</div>")
                items = step.html_items(self.decimal, self.digits)
                if items:
                    parts.append(f"<div>{items}</div>")
                if step.note:
                    parts.append(f"<div class='lk-n'>{html.escape(step.note)}</div>")
                parts.append("</div>")
        if self.highlights:
            parts.append("<div class='lk-res'>")
            for label, obj in self.highlights:
                piece = html_any(obj, self.decimal, self.digits)
                name = html.escape(tex_to_text(label))
                parts.append(f"<span class='lk-item'>"
                             f"<span class='lk-lbl'>{name} =</span>"
                             f"{piece}</span>")
            parts.append("</div>")
        parts.append("</div>")
        return "".join(parts)

    def __repr__(self) -> str:
        lines = [self.title]
        if self.summary:
            lines.append("  " + self.summary)
        for warning in self.warnings:
            lines.append("  ! " + warning)
        for index, step in enumerate(self.steps, 1):
            lines.append(f"  {index}. {step.title}")
            for label, obj in step.items:
                if isinstance(obj, Matrix):
                    block = text_matrix(obj, self.decimal)
                    head = f"     {label} =" if label else "     "
                    lines.append(head)
                    lines += ["       " + l for l in block.splitlines()]
                elif isinstance(obj, Vector):
                    lines.append(f"     {label} = "
                                 f"({', '.join(fmt(v, self.decimal) for v in obj)})")
                else:
                    lines.append(f"     {label} = {fmt(obj, self.decimal)}")
        for label, obj in self.highlights:
            if isinstance(obj, Matrix):
                lines.append(f"  {label} =")
                lines += ["    " + l
                          for l in text_matrix(obj, self.decimal).splitlines()]
            elif isinstance(obj, Vector):
                lines.append(f"  {label} = "
                             f"({', '.join(fmt(v, self.decimal) for v in obj)})")
            else:
                lines.append(f"  {label} = {fmt(obj, self.decimal)}")
        return "\n".join(lines)

    def show(self):
        """Im Notebook anzeigen, sonst drucken."""
        try:
            from IPython.display import HTML, display
            display(HTML(self._repr_html_()))
        except ImportError:
            print(self)


def show_matrix(M, label: str = "", decimal: bool = False):
    """Eine Matrix oder einen Vektor gesetzt anzeigen."""
    piece = tex_any(M, decimal)
    body = f"{label} = {piece}" if label else piece
    try:
        from IPython.display import Math, display
        display(Math(body))
    except ImportError:
        print(text_matrix(M, decimal) if isinstance(M, Matrix) else M)
