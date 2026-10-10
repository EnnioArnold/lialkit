#!/usr/bin/env python3
"""
Gauss- und Gauss-Jordan-Elimination — Desktop-App im Apple-Stil (Qt).

Installation:
    pip install PySide6

Start:
    python3 gauss_qt.py

Die Prüfung mit Wolfram|Alpha ist optional und braucht eine kostenlose AppID
von https://developer.wolframalpha.com/ (Umgebungsvariable WOLFRAM_APPID oder
einmalige Eingabe in der App).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import (QEasingCurve, QObject, QPoint, QPointF,
                            QPropertyAnimation, QRectF, QSize, Qt, QThread,
                            QTimer, QUrl, Property, Signal)
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QMovie,
                           QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame,
                               QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMenu, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
                               QVBoxLayout, QWidget)

# Manim ist freiwillig: fehlt es, sagt die Animationskarte, was zu tun ist.
# Dieser Ordner ist selbst das Paket `lialkit` — auch dann, wenn er nach dem
# GitHub-Download »lialkit-main« heisst oder man direkt `python3 gauss_qt.py`
# darin aufruft. Darum hier von Hand als Paket anmelden, vor allen Importen.
_HERE = os.path.dirname(os.path.abspath(__file__))
if "lialkit" not in sys.modules:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "lialkit", os.path.join(_HERE, "__init__.py"),
        submodule_search_locations=[_HERE])
    _pkg = _ilu.module_from_spec(_spec)
    sys.modules["lialkit"] = _pkg
    _spec.loader.exec_module(_pkg)
# gauss_anim liegt neben dieser Datei.
sys.path.insert(0, _HERE)
try:
    import gauss_anim
    ANIM_IMPORT_ERROR = ""
except Exception as _exc:                                    # pragma: no cover
    gauss_anim = None
    ANIM_IMPORT_ERROR = str(_exc)
from orb import ThinkingOrb

# QtMultimedia spielt das MP4 ab. Das ist der Normalfall; fehlt es, fällt der
# Abspieler auf das GIF zurück — sichtbar schlechter, weil ein GIF nur 256
# Farben kennt, aber besser als gar keine Animation.
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer  # noqa: F401
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_VIDEO = True
    VIDEO_IMPORT_ERROR = ""
except Exception as _exc:                                    # pragma: no cover
    HAS_VIDEO = False
    VIDEO_IMPORT_ERROR = str(_exc)

# ==========================================================================
# Rechenteil
# ==========================================================================
#
# Gerechnet wird in lialkit — dasselbe Modul, das auch im Notebook läuft.
# So gibt es nur eine Stelle, an der die Verfahren stehen, und ein Fehler
# kann sich nicht in einer der beiden Oberflächen verstecken.

from lialkit.core import (MINUS, Num, Operation, Vector, fmt, fmt_factor,
                          parse_number, tex_number)
from lialkit.core import Matrix as _CoreMatrix
from lialkit.determinant import det as _det


class Matrix(_CoreMatrix):
    """
    Die Matrix im Fenster ist immer eine erweiterte (A | b).

    Darum ist `augmented` hier voreingestellt; sonst verhält sie sich genau
    wie die aus lialkit.
    """

    def __init__(self, rows, augmented: bool = True):
        super().__init__(rows, augmented)

    def coefficients(self):
        """Nur A, als Liste von Zeilen — so erwartet es die Oberfläche."""
        return [row[:self.n] for row in self.rows]

    def determinant(self):
        """Determinante der Koeffizientenmatrix, None wenn nicht quadratisch."""
        if not self.is_square():
            return None
        return _det(_CoreMatrix(self.coefficients())).value

    def rank(self) -> int:
        return len(self.rref().pivots())

    def steps_to_rref(self):
        return self.steps_to(jordan=True)


PRESETS: dict[str, list[list]] = {
    "Eindeutige Lösung": [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]],
    "Mit Zeilentausch": [[0, 2, 1, 7], [1, -1, 2, 3], [3, 1, -1, 2]],
    "Unendlich viele Lösungen": [[1, 2, -1, 3], [2, 4, -2, 6], [1, 1, 1, 2]],
    "Keine Lösung": [[1, 1, 1, 3], [2, 2, 2, 7], [1, -1, 0, 1]],
    "Brüche": [[Fraction(1, 2), 1, -1, 2], [3, -2, Fraction(1, 3), 0],
               [1, 1, 1, Fraction(5, 2)]],
    "SW3 — LR-Zerlegung": [[2, 6, 2, 2], [-3, -8, 0, 2], [4, 9, 2, 3]],
    "SW4 — Inverse": [[4, 3, 0], [8, 7, 0]],
}


# ==========================================================================
# Wolfram|Alpha
# ==========================================================================

def ssl_context():
    """
    Der Zertifikatsspeicher für die Anfragen.

    Ein mit pyenv gebautes Python benutzt den Speicher der OpenSSL-Version,
    gegen die es gebaut wurde — der ist auf macOS oft leer. Ist das Paket
    certifi vorhanden, nehmen wir dessen Zertifikate; das ist derselbe Satz,
    den auch Browser verwenden, und erspart jede Bastelei am System.
    """
    import ssl
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()      # bewusst gesetzt, respektieren
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


WOLFRAM_URL = "https://api.wolframalpha.com/v2/query"
WOLFRAM_SITE = "https://www.wolframalpha.com/input"
APPID_FILE = Path.home() / ".gauss_wolfram_appid"
_CACHE: dict[str, str] = {}


def load_appid() -> str:
    key = os.environ.get("WOLFRAM_APPID", "").strip()
    if key:
        return key
    try:
        return APPID_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_appid(key: str) -> None:
    APPID_FILE.write_text(key.strip(), encoding="utf-8")
    try:
        APPID_FILE.chmod(0o600)
    except OSError:
        pass


def matrix_literal(rows: Sequence[Sequence[Num]]) -> str:
    return "{" + ",".join(
        "{" + ",".join(str(v) for v in row) + "}" for row in rows) + "}"


def system_literal(M: Matrix) -> str:
    """'{2x + y - z = 8, …}' mit x, y, z, w, v, u als Variablennamen."""
    names = VARIABLE_NAMES[:M.n]
    parts = []
    for row in M.rows:
        left = ""
        for c, k in enumerate(row[:M.n]):
            if k == 0:
                continue
            mag = abs(k)
            piece = names[c] if mag == 1 else f"{mag} {names[c]}"
            if left == "":
                left = piece if k > 0 else f"-{piece}"
            else:
                left += f" {'+' if k > 0 else '-'} {piece}"
        parts.append(f"{left or '0'} = {row[M.n]}")
    return "{" + ", ".join(parts) + "}"


VARIABLE_NAMES = ["x", "y", "z", "w", "v", "u"]

#: Beschriftung der Animationsarten für das Auswahlfeld.
ANIM_SCENES = {
    "Letzter Schritt": "step",
    "Ganzer Durchlauf": "full",
    "Pivot-Spur": "pivots",
}

#: Qualitätsstufen: (Manim-Stufe, Breite eines GIF, Bilder pro Sekunde, Faktor
#: für die geschätzte Rechenzeit). Die GIF-Werte greifen nur im Rückfall ohne
#: QtMultimedia; sonst entscheidet die Manim-Stufe über die Auflösung.
ANIM_QUALITY = {
    "Normal": ("medium", 1280, 25, 1.0),
    "Scharf": ("high", 1920, 30, 2.4),
    "Sehr scharf": ("ultra", 2560, 30, 5.0),
}

#: Ohne QtMultimedia bleibt nur das GIF — kleiner, aber nur 256 Farben.
ANIM_FORMAT = "mp4" if HAS_VIDEO else "gif"


def browser_url(query: str) -> str:
    return f"{WOLFRAM_SITE}?i={urllib.parse.quote(query)}"


def explain_network_error(reason) -> str:
    """
    Macht aus dem technischen Grund einer fehlgeschlagenen Anfrage einen
    Satz, der sagt, woran es liegt und was zu tun ist.
    """
    import ssl
    import socket

    text = str(reason)
    if isinstance(reason, ssl.SSLCertVerificationError) or \
            "CERTIFICATE_VERIFY_FAILED" in text:
        try:
            import certifi                                    # noqa: F401
            return ("Das Sicherheitszertifikat liess sich nicht prüfen, obwohl "
                    "certifi vorhanden ist. Vermutlich hängt ein Netzwerkfilter "
                    "dazwischen, der den Verkehr aufbricht — probier ein "
                    "anderes Netz, etwa den Hotspot vom Handy.")
        except ImportError:
            return ("Das Sicherheitszertifikat liess sich nicht prüfen: diesem "
                    "Python fehlen die Wurzelzertifikate. „pip install certifi“ "
                    "ausführen und die App neu starten — sie nimmt die "
                    "Zertifikate dann von dort.")
    if isinstance(reason, socket.gaierror) or "Name or service not known" in text \
            or "nodename nor servname" in text:
        return ("Der Server api.wolframalpha.com war nicht auffindbar. Das "
                "deutet auf eine fehlende Internetverbindung oder einen "
                "blockierenden DNS hin.")
    if "Tunnel connection failed" in text or "proxy" in text.lower():
        return ("Ein Proxy im Netzwerk hat die Anfrage abgewiesen. In einem "
                "Firmen- oder Schulnetz hilft meist ein anderes Netz, etwa der "
                "Hotspot vom Handy.")
    if isinstance(reason, ConnectionRefusedError) or "refused" in text.lower():
        return ("Die Verbindung wurde abgewiesen — vermutlich blockiert eine "
                "Firewall oder ein Netzwerkfilter den Zugriff.")
    if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in text:
        return ("Zeitüberschreitung beim Verbinden. Entweder ist die Leitung "
                "sehr langsam oder etwas blockiert die Anfrage.")
    return f"Keine Verbindung zu Wolfram|Alpha: {text}"


def cert_source() -> str:
    """Woher die Zertifikate kommen — gehört in die Diagnose."""
    if os.environ.get("SSL_CERT_FILE"):
        return "Zertifikat geprüft (SSL_CERT_FILE)."
    try:
        import certifi                                        # noqa: F401
        return "Zertifikat geprüft (certifi)."
    except ImportError:
        return "Zertifikat geprüft (Systemspeicher)."


def network_selftest(appid: str = "", timeout: float = 20) -> list[tuple[str, bool, str]]:
    """
    Prüft der Reihe nach: Namensauflösung, TLS-Verbindung, Antwort der API.
    Gibt je Schritt (Name, geklappt, Erläuterung) zurück.
    """
    import socket

    out: list[tuple[str, bool, str]] = []
    host = "api.wolframalpha.com"

    try:
        address = socket.gethostbyname(host)
        out.append(("Namensauflösung", True, f"{host} → {address}"))
    except OSError as exc:
        out.append(("Namensauflösung", False, explain_network_error(exc)))
        return out

    try:
        req = urllib.request.Request(f"https://{host}/",
                                     headers={"User-Agent": "gauss-app/2.0"})
        urllib.request.urlopen(req, timeout=timeout,
                               context=ssl_context()).read(64)
        out.append(("Verschlüsselte Verbindung", True, cert_source()))
    except urllib.error.HTTPError:
        # Eine Fehlerseite ist auch eine Antwort — die Leitung steht.
        out.append(("Verschlüsselte Verbindung", True, cert_source()))
    except urllib.error.URLError as exc:
        out.append(("Verschlüsselte Verbindung", False,
                    explain_network_error(exc.reason)))
        return out
    except OSError as exc:
        out.append(("Verschlüsselte Verbindung", False,
                    explain_network_error(exc)))
        return out

    if not appid:
        out.append(("AppID", False, "Noch keine AppID hinterlegt."))
        return out

    try:
        answer = wolfram_text(appid, "2+2", timeout=timeout)
        out.append(("Testabfrage 2+2", True, f"Antwort: {answer.splitlines()[0]}"))
    except RuntimeError as exc:
        out.append(("Testabfrage 2+2", False, str(exc)))
    return out


def wolfram_text(appid: str, query: str, timeout: float = 30) -> str:
    """Fragt die Full Results API und gibt den Klartext des Ergebnis-Pods."""
    if query in _CACHE:
        return _CACHE[query]
    params = urllib.parse.urlencode({
        "appid": appid, "input": query, "output": "json",
        "format": "plaintext", "podtimeout": "20",
    })
    req = urllib.request.Request(f"{WOLFRAM_URL}?{params}",
                                 headers={"User-Agent": "gauss-app/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "Die AppID wurde abgelehnt. Prüfe sie unter „AppID“ — eine "
                "frische AppID braucht manchmal ein paar Minuten.") from exc
        if exc.code == 429:
            raise RuntimeError("Zu viele Anfragen. Das kostenlose Kontingent "
                               "liegt bei 2000 Abfragen im Monat.") from exc
        raise RuntimeError(f"Wolfram antwortete mit HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(explain_network_error(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("Wolfram hat nicht rechtzeitig geantwortet. "
                           "Versuch es gleich noch einmal.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wolfram hat keine lesbare Antwort geschickt.") from exc

    qr = data.get("queryresult", {})
    if qr.get("error"):
        detail = qr.get("error")
        msg = detail.get("msg") if isinstance(detail, dict) else None
        raise RuntimeError(msg or "Wolfram meldet einen Fehler.")
    if not qr.get("success"):
        raise RuntimeError("Wolfram konnte die Anfrage nicht auswerten.")

    pods = qr.get("pods", [])
    preferred = [p for p in pods if p.get("id") in
                 ("Result", "Solution", "DecimalApproximation")]
    for pod in preferred + pods:
        if pod.get("id") in ("Input", "InputInterpretation"):
            continue
        for sub in pod.get("subpods", []):
            text = (sub.get("plaintext") or "").strip()
            if text:
                _CACHE[query] = text
                return text
    raise RuntimeError("Wolfram hat kein Ergebnis zurückgegeben.")


def parse_wolfram_matrix(text: str, shape: tuple[int, int]) -> Matrix:
    """Liest Wolframs Klartext-Matrix '(1 | 0 | 2\\n0 | 1 | 3)' ein."""
    rows = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("(").rstrip(")").strip()
        if not line or "|" not in line:
            continue
        rows.append([parse_number(p) for p in line.split("|")])
    if not rows:
        raise ValueError("Wolframs Antwort enthält keine Matrix.")
    if (len(rows), len(rows[0])) != shape or any(len(r) != shape[1] for r in rows):
        raise ValueError("Wolframs Matrix hat eine andere Form als erwartet.")
    return Matrix(rows)


@dataclass
class Check:
    """Ein einzelner Prüfpunkt im Bericht."""
    name: str
    status: str          # 'ok' | 'bad' | 'info' | 'error'
    ours: str = ""
    theirs: str = ""
    note: str = ""


class WolframReport:
    def __init__(self):
        self.checks: list[Check] = []
        self.their_matrix: Matrix | None = None
        self.solution_text: str = ""
        self.url: str = ""

    @property
    def verdict(self) -> str:
        if any(c.status == "bad" for c in self.checks):
            return "bad"
        if any(c.status == "error" for c in self.checks):
            return "warn"
        return "ok" if self.checks else "warn"

    @property
    def headline(self) -> str:
        return {
            "ok": "Wolfram|Alpha bestätigt das Ergebnis.",
            "bad": "Wolfram|Alpha kommt auf etwas anderes.",
            "warn": "Wolfram konnte nicht alles prüfen.",
        }[self.verdict]


def run_wolfram_checks(appid: str, original: Matrix, current: Matrix,
                       decimal: bool = False) -> WolframReport:
    """
    Vergleicht vier Grössen mit Wolfram: reduzierte Stufenform, Rang,
    Determinante und die Lösungsmenge. Geprüft wird die Ausgangsmatrix,
    damit auch ein Fehler in einem Zwischenschritt auffällt.
    """
    report = WolframReport()
    aug = original.rows
    coeff = original.coefficients()
    square = original.m == original.n

    queries = {
        "rref": f"RowReduce[{matrix_literal(aug)}]",
        "rank": f"MatrixRank[{matrix_literal(coeff)}]",
        "solve": f"solve {system_literal(original)}",
    }
    if square:
        queries["det"] = f"Det[{matrix_literal(coeff)}]"
    report.url = browser_url(queries["rref"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(wolfram_text, appid, q)
                   for key, q in queries.items()}
        answers: dict[str, str | Exception] = {}
        for key, fut in futures.items():
            try:
                answers[key] = fut.result()
            except (RuntimeError, ValueError) as exc:
                answers[key] = exc

    # Wenn schon die erste Anfrage scheitert, ist meist die AppID oder das
    # Netz schuld — dann lohnt kein Bericht mit vier gleichen Fehlern.
    if isinstance(answers["rref"], Exception) and all(
            isinstance(v, Exception) for v in answers.values()):
        raise RuntimeError(str(answers["rref"]))

    ours_rref = current.rref()

    # 1. Reduzierte Zeilenstufenform
    ans = answers["rref"]
    if isinstance(ans, Exception):
        report.checks.append(Check("Reduzierte Stufenform", "error",
                                   note=str(ans)))
    else:
        try:
            theirs = parse_wolfram_matrix(ans, (original.m, original.n + 1))
            report.their_matrix = theirs
            if ours_rref.same_as(theirs):
                report.checks.append(Check(
                    "Reduzierte Stufenform", "ok",
                    note="Zeile für Zeile identisch."))
            elif ours_rref.inconsistent() and theirs.inconsistent():
                report.checks.append(Check(
                    "Reduzierte Stufenform", "ok",
                    note="Beide finden denselben Widerspruch."))
            else:
                report.checks.append(Check(
                    "Reduzierte Stufenform", "bad",
                    note="Unterschiedlich — vermutlich steckt in einem "
                         "Schritt ein Fehler."))
        except ValueError as exc:
            report.checks.append(Check("Reduzierte Stufenform", "error",
                                       note=str(exc)))

    # 2. Rang der Koeffizientenmatrix
    ans = answers["rank"]
    our_rank = Matrix([row + [Fraction(0)] for row in coeff]).rank()
    if isinstance(ans, Exception):
        report.checks.append(Check("Rang", "error", ours=str(our_rank),
                                   note=str(ans)))
    else:
        try:
            their_rank = int(parse_number(ans.strip().splitlines()[0]))
            report.checks.append(Check(
                "Rang", "ok" if their_rank == our_rank else "bad",
                ours=str(our_rank), theirs=str(their_rank)))
        except (ValueError, IndexError):
            report.checks.append(Check("Rang", "error", ours=str(our_rank),
                                       note="Antwort nicht lesbar."))

    # 3. Determinante (nur quadratisch)
    if square:
        ans = answers["det"]
        our_det = original.determinant()
        if isinstance(ans, Exception):
            report.checks.append(Check("Determinante", "error",
                                       ours=fmt(our_det, decimal), note=str(ans)))
        else:
            try:
                their_det = parse_number(ans.strip().splitlines()[0])
                report.checks.append(Check(
                    "Determinante", "ok" if their_det == our_det else "bad",
                    ours=fmt(our_det, decimal), theirs=fmt(their_det, decimal),
                    note="" if our_det != 0 else
                    "Null, das System ist also nicht eindeutig lösbar."))
            except ValueError:
                report.checks.append(Check("Determinante", "error",
                                           ours=fmt(our_det, decimal),
                                           note="Antwort nicht lesbar."))

    # 4. Lösungsmenge im Klartext
    ans = answers["solve"]
    if isinstance(ans, Exception):
        report.checks.append(Check("Lösungsmenge", "error", note=str(ans)))
    else:
        report.solution_text = ans
        kind = original.analyze()["type"]
        lowered = ans.lower()
        no_solution = "no solution" in lowered or "false" == lowered.strip()
        if kind == "none":
            report.checks.append(Check(
                "Lösungsmenge", "ok" if no_solution else "bad",
                ours="keine Lösung",
                theirs="keine Lösung" if no_solution else ans.splitlines()[0]))
        elif no_solution:
            report.checks.append(Check("Lösungsmenge", "bad",
                                       ours=kind, theirs="keine Lösung"))
        else:
            report.checks.append(Check(
                "Lösungsmenge", "info",
                note="Wolframs Schreibweise steht unten — bei freien "
                     "Variablen wählt Wolfram andere Parameter."
                if kind == "infinite" else "Siehe unten."))
    return report


def wolfram_number(text: str) -> Num:
    """
    Die erste Zahl aus Wolframs Antwort.

    Wolfram hängt gern Einheiten oder Näherungen an — „12.9332 degrees“,
    „2 sqrt(3) ≈ 3.464“. Gesucht ist die erste Zahl in der ersten Zeile.
    """
    line = text.strip().splitlines()[0].replace(MINUS, "-")
    match = re.search(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", line)
    if not match:
        raise ValueError("Antwort nicht lesbar.")
    return parse_number(match.group(0).replace(" ", ""))


def parse_wolfram_vector(text: str, dim: int) -> list[Num]:
    """Liest '(-3 | 6 | -3)' oder '(-3, 6, -3)' als Vektor ein."""
    line = text.strip().splitlines()[0].strip().lstrip("(").rstrip(")")
    parts = line.split("|") if "|" in line else line.split(",")
    values = [parse_number(p) for p in parts if p.strip()]
    if len(values) != dim:
        raise ValueError("Wolframs Vektor hat eine andere Länge als erwartet.")
    return values


def _ask(appid: str, queries: dict[str, str]) -> dict[str, object]:
    """Stellt die Anfragen nebeneinander und sammelt Antworten wie Fehler."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(wolfram_text, appid, q)
                   for key, q in queries.items()}
        answers: dict[str, object] = {}
        for key, fut in futures.items():
            try:
                answers[key] = fut.result()
            except (RuntimeError, ValueError) as exc:
                answers[key] = exc
    if answers and all(isinstance(v, Exception) for v in answers.values()):
        raise RuntimeError(str(next(iter(answers.values()))))
    return answers


def _compare(report: WolframReport, name: str, ours, answer, read,
             fmt_one=str, note: str = ""):
    """Ein Prüfpunkt: lesen, vergleichen, eintragen — immer gleich."""
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=fmt_one(ours),
                                   note=str(answer)))
        return
    try:
        theirs = read(answer)
    except (ValueError, IndexError) as exc:
        report.checks.append(Check(name, "error", ours=fmt_one(ours),
                                   note=str(exc) or "Antwort nicht lesbar."))
        return
    same = theirs == ours
    report.checks.append(Check(name, "ok" if same else "bad",
                               ours=fmt_one(ours), theirs=fmt_one(theirs),
                               note=note if same else ""))


def run_methods_checks(appid: str, A, method: str,
                       decimal: bool = False) -> WolframReport:
    """
    Prüft die Verfahren auf einer quadratischen Matrix.

    Die LR-Zerlegung ist der Sonderfall: Wolfram pivotiert nach Betrag, die
    Vorlesung nur bei einer Null an der Pivotstelle. Zwei richtige
    Rechnungen kommen damit auf verschiedene L und R — ein Vergleich wäre
    irreführend. Stattdessen rechnet die App die Identität L · R = P · A
    selbst exakt nach, und Wolfram liefert mit Determinante und Stufenform
    eine unabhängige Gegenprobe.
    """
    from lialkit.determinant import det as _det_of
    from lialkit.factor import lu as _lu
    from lialkit.inverse import inverse as _inverse

    report = WolframReport()
    rows = [list(row) for row in A.rows]
    literal = matrix_literal(rows)
    queries = {"det": f"Det[{literal}]", "rank": f"MatrixRank[{literal}]"}
    if method == "inverse":
        queries["inv"] = f"Inverse[{literal}]"
    if method in ("lu", "rank"):
        queries["rref"] = f"RowReduce[{literal}]"
    report.url = browser_url(queries.get("inv") or queries["det"])

    answers = _ask(appid, queries)

    our_det = _det_of(A).value
    _compare(report, "Determinante", our_det, answers["det"], wolfram_number,
             lambda x: fmt(x, decimal),
             note="" if our_det != 0 else "Null — die Matrix ist singulär.")

    our_rank = Fraction(len(A.rref().pivots()))
    _compare(report, "Rang", our_rank, answers["rank"], wolfram_number,
             lambda x: str(int(x)))

    if method == "inverse":
        ans = answers["inv"]
        if our_det == 0:
            report.checks.append(Check(
                "Inverse", "info",
                note="Es gibt keine — die Determinante ist 0. Wolfram meldet "
                     "hier ebenfalls einen Fehler."))
        elif isinstance(ans, Exception):
            report.checks.append(Check("Inverse", "error", note=str(ans)))
        else:
            try:
                theirs = parse_wolfram_matrix(ans, (A.m, A.n))
                ours = _inverse(A).inverse
                same = all(theirs.rows[r][c] == ours.rows[r][c]
                           for r in range(A.m) for c in range(A.n))
                report.their_matrix = theirs
                report.checks.append(Check(
                    "Inverse", "ok" if same else "bad",
                    note="Eintrag für Eintrag gleich." if same else
                         "Unterschiedlich — Wolframs Matrix steht unten."))
            except ValueError as exc:
                report.checks.append(Check("Inverse", "error", note=str(exc)))

    if method == "lu":
        d = _lu(A)
        report.checks.append(Check(
            "L · R = P · A", "ok" if d.check() else "bad",
            note="Von der App exakt nachgerechnet, nicht von Wolfram: "
                 "Wolfram wählt die Pivots nach Betrag und käme auf ein "
                 "anderes, ebenso richtiges L und R."))

    if "rref" in answers:
        ans = answers["rref"]
        if isinstance(ans, Exception):
            report.checks.append(Check("Zeilenstufenform", "error",
                                       note=str(ans)))
        else:
            try:
                theirs = parse_wolfram_matrix(ans, (A.m, A.n))
                ours = A.rref() if hasattr(A, "rref") else None
                if ours is None:
                    report.their_matrix = theirs
                    report.checks.append(Check("Zeilenstufenform", "info",
                                               note="Wolframs Form steht unten."))
                else:
                    same = all(theirs.rows[r][c] == ours.rows[r][c]
                               for r in range(A.m) for c in range(A.n))
                    report.checks.append(Check(
                        "Zeilenstufenform", "ok" if same else "bad"))
            except ValueError as exc:
                report.checks.append(Check("Zeilenstufenform", "error",
                                           note=str(exc)))
    return report


def run_vector_checks(appid: str, op: str, u, v, w,
                      decimal: bool = False, k=None) -> WolframReport:
    """Prüft die Vektorrechnungen Eintrag für Eintrag gegen Wolfram."""
    report = WolframReport()
    lit = lambda vec: "{" + ",".join(str(x) for x in vec.values) + "}"  # noqa: E731
    dim = len(u.values)

    if op == "dot":
        query = f"Dot[{lit(u)},{lit(v)}]"
        report.url = browser_url(query)
        answers = _ask(appid, {"q": query})
        _compare(report, "Skalarprodukt", u.dot(v), answers["q"],
                 wolfram_number, lambda x: fmt(x, decimal))
    elif op == "cross":
        if dim != 3:
            raise ValueError("Das Vektorprodukt gibt es nur im Raum.")
        query = f"Cross[{lit(u)},{lit(v)}]"
        report.url = browser_url(query)
        answers = _ask(appid, {"q": query})
        ours = u.cross(v)
        _compare(report, "Vektorprodukt", list(ours.values), answers["q"],
                 lambda t: parse_wolfram_vector(t, 3),
                 lambda xs: "(" + ", ".join(fmt(x, decimal) for x in xs) + ")")
    elif op == "spat":
        rows = [list(u.values), list(v.values), list(w.values)]
        query = f"Det[{matrix_literal(rows)}]"
        report.url = browser_url(query)
        answers = _ask(appid, {"q": query})
        from lialkit.vectors import spat as _spat
        _compare(report, "Spatprodukt", _spat(u, v, w).value, answers["q"],
                 wolfram_number, lambda x: fmt(x, decimal))
    elif op == "angle":
        query = f"VectorAngle[{lit(u)},{lit(v)}] in degrees"
        report.url = browser_url(query)
        answers = _ask(appid, {"q": query})
        from lialkit.vectors import angle as _angle
        ours = float(_angle(u, v).degrees)
        ans = answers["q"]
        if isinstance(ans, Exception):
            report.checks.append(Check("Winkel", "error",
                                       ours=f"{ours:.4f}°", note=str(ans)))
        else:
            try:
                theirs = float(wolfram_number(ans))
                # Beide runden; auf die vierte Stelle genau zu vergleichen
                # wäre Haarspalterei.
                close = abs(theirs - ours) < 0.01
                report.checks.append(Check(
                    "Winkel", "ok" if close else "bad",
                    ours=f"{ours:.4f}°", theirs=f"{theirs:.4f}°"))
            except ValueError as exc:
                report.checks.append(Check("Winkel", "error",
                                           ours=f"{ours:.4f}°", note=str(exc)))
    elif op in ("add", "position", "scale"):
        if op == "add":
            name, query, ours = "Summe", f"{lit(u)}+{lit(v)}", u + v
        elif op == "position":          # u = A, v = B
            name, query, ours = ("Verbindungsvektor", f"{lit(v)}-{lit(u)}",
                                 v - u)
        else:
            name, query, ours = ("Vielfaches", f"({k})*{lit(u)}",
                                 u.scaled(k))
        report.url = browser_url(query)
        answers = _ask(appid, {"q": query})
        _compare(report, name, list(ours.values), answers["q"],
                 lambda t: parse_wolfram_vector(t, dim),
                 lambda xs: "(" + ", ".join(fmt(x, decimal) for x in xs) + ")")
    else:
        if v.is_zero():
            raise ValueError("Auf den Nullvektor lässt sich nicht "
                             "projizieren.")
        query = f"Projection[{lit(u)},{lit(v)}]"
        report.url = browser_url(query)
        answers = _ask(appid, {"q": query})
        ours = v.scaled(u.dot(v) / v.dot(v))
        _compare(report, "Projektion", list(ours.values), answers["q"],
                 lambda t: parse_wolfram_vector(t, dim),
                 lambda xs: "(" + ", ".join(fmt(x, decimal) for x in xs) + ")")
    return report


# ==========================================================================
# Farben und Schriften
# ==========================================================================

# Vercel-Palette: Schwarz/Weiss, Graustufen, ein Akzentblau für Links und Fokus.
LIGHT = dict(
    bg="#ffffff", nav="#fafafa", card="#ffffff", text="#171717", sec="#666666",
    ter="#8f8f8f", line="#ebebeb", hair="#f0f0f0", fill="#f2f2f2", fill2="#fafafa",
    seg="#ffffff", blue="#0070f3", blue_hi="#0761d1", blue_soft="#eaf3ff",
    green="#18794e", green_soft="#e6f4ec", red="#e5484d", red_soft="#fdecec",
    amber="#ab5200", amber_soft="#fff4e0", flash="#fff3c4",
    entry="#ffffff", shadow=20, dark=False)

DARK = dict(
    bg="#000000", nav="#0a0a0a", card="#0a0a0a", text="#ededed", sec="#a1a1a1",
    ter="#7d7d7d", line="#333333", hair="#1f1f1f", fill="#1a1a1a", fill2="#111111",
    seg="#282828", blue="#0070f3", blue_hi="#3291ff", blue_soft="#0b2a4f",
    green="#46a758", green_soft="#102a16", red="#e5484d", red_soft="#3a1516",
    amber="#ffb224", amber_soft="#3a2a0a", flash="#3d3210",
    entry="#0a0a0a", shadow=60, dark=True)

T: dict = dict(LIGHT)


def pick_font(candidates: Sequence[str], fallback: str) -> str:
    families = {f.lower(): f for f in QFontDatabase.families()}
    for name in candidates:
        if name.lower() in families:
            return families[name.lower()]
    return fallback


class Fonts:
    def __init__(self):
        ui = pick_font(["Geist", "Inter", "SF Pro Text", ".AppleSystemUIFont",
                        "Helvetica Neue", "Segoe UI Variable Text", "Segoe UI",
                        "DejaVu Sans"], "Sans Serif")
        display = pick_font(["Geist", "Inter", "SF Pro Display", "SF Pro Text",
                             "Helvetica Neue", "Segoe UI Variable Display",
                             "Segoe UI", "DejaVu Sans"], ui)
        mono = pick_font(["SF Mono", "Menlo", "Cascadia Mono", "Consolas",
                          "DejaVu Sans Mono"], "Monospace")
        self.family, self.display, self.mono_family = ui, display, mono

    def f(self, size: int, weight: int = QFont.Normal,
          display: bool = False, mono: bool = False) -> QFont:
        fam = self.mono_family if mono else (self.display if display else self.family)
        font = QFont(fam, size)
        font.setWeight(weight)
        if display:
            font.setLetterSpacing(QFont.PercentageSpacing, 97)
        return font


FONTS: Fonts | None = None


# ==========================================================================
# Bausteine
# ==========================================================================

def paint_value(p: QPainter, center: QPointF, value: Num, color: QColor,
                decimal: bool, *, size: int = 16, bold: bool = False):
    """Zeichnet eine Zahl; Brüche mit echtem Bruchstrich übereinander."""
    weight = QFont.DemiBold if bold else QFont.Normal
    main = FONTS.f(size, weight)
    p.setPen(color)
    if decimal or value.denominator == 1:
        p.setFont(main)
        p.drawText(QRectF(center.x() - 48, center.y() - size, 96, size * 2),
                   Qt.AlignCenter, fmt(value, decimal))
        return
    small = FONTS.f(max(8, size - 5), weight)
    num, den = str(abs(value.numerator)), str(value.denominator)
    fm_small = QFontMetricsF(small)
    fm_main = QFontMetricsF(main)
    tw = max(fm_small.horizontalAdvance(num), fm_small.horizontalAdvance(den))
    sign_w = fm_main.horizontalAdvance(MINUS) + 4 if value < 0 else 0
    left = center.x() - (tw + sign_w) / 2
    if value < 0:
        p.setFont(main)
        p.drawText(QRectF(left, center.y() - size + 3, sign_w, size * 2 - 6),
                   Qt.AlignLeft | Qt.AlignVCenter, MINUS)
    cx = left + sign_w + tw / 2
    lift = size * 0.62
    p.setFont(small)
    p.drawText(QRectF(cx - tw / 2 - 4, center.y() - lift - 7, tw + 8, size),
               Qt.AlignCenter, num)
    p.setPen(QPen(color, 1.3))
    p.drawLine(QPointF(cx - tw / 2 - 2, center.y()),
               QPointF(cx + tw / 2 + 2, center.y()))
    p.setPen(color)
    p.drawText(QRectF(cx - tw / 2 - 4, center.y() + 2, tw + 8, size),
               Qt.AlignCenter, den)


def qcolor(key: str) -> QColor:
    return QColor(T[key])


def rise_in(widget: QWidget, dy: int = 10, ms: int = 240):
    """
    Lässt ein Widget von unten einschweben.

    Kein Fade: Karten tragen schon einen Schatten-Effekt, und Qt erlaubt
    pro Widget nur einen Effekt. Der Layout-Pass läuft vorher, damit die
    Endposition stimmt.
    """
    if widget.parentWidget() and widget.parentWidget().layout():
        widget.parentWidget().layout().activate()
    end = widget.pos()
    anim = QPropertyAnimation(widget, b"pos", widget)
    anim.setDuration(ms)
    anim.setStartValue(end + QPoint(0, dy))
    anim.setEndValue(end)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.start(QPropertyAnimation.DeleteWhenStopped)


class Card(QFrame):
    """Fläche mit runden Ecken und weichem Schatten."""

    def __init__(self, parent=None, pad: int = 26):
        super().__init__(parent)
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(pad, pad, pad, pad)
        self.lay.setSpacing(0)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.setGraphicsEffect(self.shadow)
        self.restyle()

    def restyle(self):
        self.shadow.setBlurRadius(30)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(0, 0, 0, T["shadow"] // (3 if T["dark"] else 1)))


class Segmented(QWidget):
    """Segmentierte Umschaltleiste wie in iOS."""

    changed = Signal(int)

    def __init__(self, options: Sequence[str], value: int = 0,
                 seg_width: int | None = None, parent=None):
        super().__init__(parent)
        self.options = list(options)
        self.value = value
        self.enabled = True
        self.disabled_hint = ""
        self.font_ = FONTS.f(11)
        metrics = QFontMetricsF(self.font_)
        self.seg_w = seg_width or int(
            max(metrics.horizontalAdvance(o) for o in options)) + 30
        self.setFixedSize(self.seg_w * len(options) + 6, 30)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qcolor("fill2") if not self.enabled else qcolor("fill"))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 9, 9)
        p.setFont(self.font_)
        for i, label in enumerate(self.options):
            box = QRectF(3 + i * self.seg_w, 3, self.seg_w, self.height() - 6)
            if i == self.value:
                p.setBrush(qcolor("fill") if not self.enabled else qcolor("seg"))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(box, 7, 7)
            if not self.enabled:
                p.setPen(qcolor("ter"))
            else:
                p.setPen(qcolor("text") if i == self.value else qcolor("sec"))
            p.drawText(box, Qt.AlignCenter, label)

    def mousePressEvent(self, event):
        if not self.enabled:
            return
        i = int((event.position().x() - 3) // self.seg_w)
        if 0 <= i < len(self.options) and i != self.value:
            self.value = i
            self.update()
            self.changed.emit(i)

    def set_value(self, i: int):
        self.value = i
        self.update()

    def set_enabled(self, on: bool):
        """
        Schaltet die Leiste stumm.

        Sie bleibt stehen, damit die Kopfzeile nicht springt, nimmt aber
        keine Klicks mehr an und zeichnet sich blass.
        """
        if self.enabled == on:
            return
        self.enabled = on
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        self.setToolTip("" if on else self.disabled_hint)
        self.update()


class Stepper(QWidget):
    """Minus / Wert / Plus in einer Pille."""

    stepped = Signal(int)

    def __init__(self, value: int, parent=None):
        super().__init__(parent)
        self.value = value
        self.font_ = FONTS.f(12)
        self.setFixedSize(96, 30)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qcolor("fill"))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 15, 15)
        p.setFont(self.font_)
        p.setPen(qcolor("text"))
        h = self.height()
        p.drawText(QRectF(0, 0, 32, h), Qt.AlignCenter, MINUS)
        p.drawText(QRectF(32, 0, 32, h), Qt.AlignCenter, str(self.value))
        p.drawText(QRectF(64, 0, 32, h), Qt.AlignCenter, "+")

    def mousePressEvent(self, event):
        x = event.position().x()
        if x < 32:
            self.stepped.emit(-1)
        elif x > 64:
            self.stepped.emit(1)

    def set_value(self, value: int):
        self.value = value
        self.update()


class Dropdown(QPushButton):
    """Auswahlfeld mit Menü."""

    picked = Signal(str)

    def __init__(self, options: Sequence[str], value: str | None = None,
                 placeholder: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("dropdown")
        self.options = list(options)
        self.value = value if value is not None else self.options[0]
        self.placeholder = placeholder
        self.setText(f"  {placeholder or self.value}   ⌄")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.clicked.connect(self._open)

    def _open(self):
        menu = QMenu(self)
        menu.setFont(FONTS.f(11))
        for opt in self.options:
            menu.addAction(opt, lambda o=opt: self._choose(o))
        menu.exec(self.mapToGlobal(QPoint(0, self.height() + 2)))

    def _choose(self, opt: str):
        self.value = opt
        if not self.placeholder:
            self.setText(f"  {opt}   ⌄")
        self.picked.emit(opt)

    def set_value(self, value: str):
        self.value = value
        if not self.placeholder:
            self.setText(f"  {value}   ⌄")


def pill(text: str, kind: str = "primary", on_click: Callable | None = None,
         small: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName(kind)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedHeight(30 if small else 38)
    btn.setFont(FONTS.f(11 if small else 12))
    if on_click:
        btn.clicked.connect(on_click)
    return btn


def label(text: str, size: int = 12, color: str = "text", weight=QFont.Normal,
          display: bool = False, mono: bool = False, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setFont(FONTS.f(size, weight, display, mono))
    lb.setProperty("colorKey", color)
    lb.setStyleSheet(f"color: {T[color]}; background: transparent;")
    lb.setWordWrap(wrap)
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lb


def wrapped(text: str, width: int, size: int = 11, color: str = "sec") -> QLabel:
    """
    Umbrochener Text mit selbst berechneter Höhe.

    Qt bestimmt die Höhe eines Labels, bevor es den Umbruch kennt; in einem
    Rahmen mit fester Breite bleibt sonst nur die erste Zeile sichtbar.
    """
    lb = label(text, size, color, wrap=True)
    lb.setFixedWidth(width)
    metrics = QFontMetricsF(lb.font())
    box = metrics.boundingRect(QRectF(0, 0, width, 10000),
                               int(Qt.TextWordWrap), text)
    lb.setMinimumHeight(int(box.height()) + 4)
    return lb


class Badge(QLabel):
    def __init__(self, text: str, fg: str, bg: str, parent=None):
        super().__init__(text, parent)
        self.fg, self.bg = fg, bg
        self.setFont(FONTS.f(11, QFont.DemiBold))
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.restyle()

    def restyle(self):
        self.setStyleSheet(
            f"color: {T[self.fg]}; background: {T[self.bg]};"
            f"border-radius: 11px; padding: 4px 14px;")


def restyle_labels(root: QWidget):
    """Färbt alle mit label() erzeugten Beschriftungen neu ein."""
    for lb in root.findChildren(QLabel):
        if isinstance(lb, Badge):
            lb.restyle()
        elif lb.property("colorKey"):
            lb.setStyleSheet(
                f"color: {T[lb.property('colorKey')]}; background: transparent;")


# ==========================================================================
# Die Matrix als gezeichnete Fläche
# ==========================================================================

class MatrixView(QWidget):
    """Zeichnet die erweiterte Matrix mit Klammern, Brüchen und Markierungen."""

    row_clicked = Signal(int)
    submitted = Signal()
    edited = Signal()
    cell_edited = Signal(int, int, str)

    LABEL_W = 52
    BRACKET_W = 20
    CELL_W = 84
    CELL_H = 56
    HEADER_H = 28
    DIV_W = 24
    PAD = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editing = True
        self.matrix: Matrix | None = None
        self.draft: list[list[str]] = []
        self.rows_n = self.cols_n = 3
        self.pivots: set[tuple[int, int]] = set()
        self.flash_rows: list[int] = []
        self.target: int | None = None
        self.decimal = False
        self.fields: list[list[QLineEdit]] = []
        self.editor: QLineEdit | None = None
        self.editor_cell: tuple[int, int] | None = None
        self._flash = 0.0
        self._anim = QPropertyAnimation(self, b"flash", self)
        self._anim.setDuration(900)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.setMouseTracking(True)

    # -- Animationseigenschaft -------------------------------------------

    def get_flash(self) -> float:
        return self._flash

    def set_flash(self, value: float):
        self._flash = value
        self.update()

    flash = Property(float, get_flash, set_flash)

    def pulse(self, rows: Sequence[int]):
        self.flash_rows = list(rows)
        self._anim.stop()
        self._anim.start()

    # -- Geometrie --------------------------------------------------------

    def _geometry(self) -> tuple[int, int, int]:
        cols = self.cols_n if self.editing else self.matrix.n
        rows = self.rows_n if self.editing else self.matrix.m
        w = (self.PAD * 2 + self.LABEL_W + self.BRACKET_W * 2
             + self.CELL_W * (cols + 1) + self.DIV_W)
        h = self.PAD * 2 + self.HEADER_H + self.CELL_H * rows
        return w, h, cols

    def _cell_center(self, r: int, c: int, cols: int) -> QPointF:
        x0 = self.PAD + self.LABEL_W + self.BRACKET_W
        x = x0 + self.CELL_W * c + self.CELL_W / 2 + (self.DIV_W if c == cols else 0)
        y = self.PAD + self.HEADER_H + self.CELL_H * r + self.CELL_H / 2
        return QPointF(x, y)

    def _row_at(self, pos) -> int | None:
        y = pos.y() - self.PAD - self.HEADER_H
        if y < 0:
            return None
        r = int(y // self.CELL_H)
        rows = self.rows_n if self.editing else self.matrix.m
        return r if 0 <= r < rows else None

    def _cell_at(self, pos) -> tuple[int, int] | None:
        """Welche Zelle liegt unter dem Zeiger — None neben der Matrix."""
        row = self._row_at(pos)
        if row is None:
            return None
        _, _, cols = self._geometry()
        for c in range(cols + 1):
            centre = self._cell_center(row, c, cols)
            if abs(pos.x() - centre.x()) <= self.CELL_W / 2 - 2:
                return row, c
        return None

    # -- Aufbau -----------------------------------------------------------

    def configure(self, *, editing: bool, matrix: Matrix | None, draft, rows: int,
                  cols: int, pivots: set, target: int | None, decimal: bool):
        self.editing, self.matrix, self.draft = editing, matrix, draft
        self.rows_n, self.cols_n = rows, cols
        self.pivots, self.target, self.decimal = pivots, target, decimal
        self.close_editor()
        w, h, _ = self._geometry()
        self.setFixedSize(w, h)
        self._build_fields()
        self.setCursor(Qt.ArrowCursor if editing else Qt.PointingHandCursor)
        self.update()

    def _build_fields(self):
        for row in self.fields:
            for field in row:
                field.hide()
                field.setParent(None)
                field.deleteLater()
        self.fields = []
        if not self.editing:
            return
        _, _, cols = self._geometry()
        for r in range(self.rows_n):
            row_fields = []
            for c in range(cols + 1):
                center = self._cell_center(r, c, cols)
                field = QLineEdit(self.draft[r][c], self)
                field.setObjectName("cell")
                field.setAlignment(Qt.AlignCenter)
                field.setFont(FONTS.f(15))
                field.setFixedSize(66, 40)
                field.move(int(center.x() - 33), int(center.y() - 20))
                field.returnPressed.connect(self.submitted.emit)
                field.textChanged.connect(self.edited.emit)
                field.show()
                row_fields.append(field)
            self.fields.append(row_fields)

    def read(self) -> list[list[str]]:
        return [[f.text() for f in row] for row in self.fields]

    # -- Zeichnen ---------------------------------------------------------

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        w, h, cols = self._geometry()
        x0 = self.PAD + self.LABEL_W + self.BRACKET_W
        top = self.PAD + self.HEADER_H
        rows = self.rows_n if self.editing else self.matrix.m
        bottom = top + self.CELL_H * rows
        right_edge = x0 + self.CELL_W * (cols + 1) + self.DIV_W

        p.setFont(FONTS.f(10))
        p.setPen(qcolor("ter"))
        for c in range(cols + 1):
            center = self._cell_center(0, c, cols)
            name = "b" if c == cols else f"x{c + 1}"
            p.drawText(QRectF(center.x() - 40, self.PAD, 80, self.HEADER_H),
                       Qt.AlignCenter, name)

        for r in range(rows):
            cy = top + self.CELL_H * r + self.CELL_H / 2
            band = QRectF(x0 - 8, cy - self.CELL_H / 2 + 3,
                          right_edge - x0 + 16, self.CELL_H - 6)
            if r in self.flash_rows and self._flash > 0:
                color = qcolor("flash")
                color.setAlphaF(self._flash)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawRoundedRect(band, 14, 14)
            if self.target == r and not self.editing:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(qcolor("line"), 1.4))
                p.drawRoundedRect(band, 14, 14)

            active = self.target == r and not self.editing
            p.setFont(FONTS.f(12, QFont.DemiBold if active else QFont.Normal))
            p.setPen(qcolor("blue") if active else qcolor("ter"))
            p.drawText(QRectF(self.PAD, cy - 12, self.LABEL_W - 10, 24),
                       Qt.AlignRight | Qt.AlignVCenter, f"R{r + 1}")

            if self.editing:
                continue
            for c in range(cols + 1):
                center = self._cell_center(r, c, cols)
                is_pivot = (r, c) in self.pivots
                if is_pivot:
                    p.setPen(Qt.NoPen)
                    p.setBrush(qcolor("blue_soft"))
                    p.drawRoundedRect(
                        QRectF(center.x() - 32, center.y() - 20, 64, 40), 13, 13)
                self._draw_value(p, center, self.matrix.rows[r][c],
                                 qcolor("blue") if is_pivot else qcolor("text"),
                                 is_pivot)

        self._draw_bracket(p, self.PAD + self.LABEL_W + 5, top - 5, bottom + 5, True)
        self._draw_bracket(p, w - self.PAD - 5, top - 5, bottom + 5, False)
        div_x = x0 + self.CELL_W * cols + self.DIV_W / 2
        pen = QPen(qcolor("line"), 1.2)
        pen.setDashPattern([2, 4])
        p.setPen(pen)
        p.drawLine(QPointF(div_x, top), QPointF(div_x, bottom))

        # Im Eingabemodus zeichnen die QLineEdits ihren eigenen Rahmen.

    def _draw_value(self, p: QPainter, center: QPointF, value: Num,
                    color: QColor, bold: bool):
        paint_value(p, center, value, color, self.decimal, size=16,
                    bold=bold)

    def _draw_bracket(self, p: QPainter, x: float, y1: float, y2: float, left: bool):
        arm = 10
        d = 1 if left else -1
        path = QPainterPath(QPointF(x + d * arm, y1))
        path.lineTo(QPointF(x, y1))
        path.lineTo(QPointF(x, y2))
        path.lineTo(QPointF(x + d * arm, y2))
        pen = QPen(qcolor("text"), 2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def mousePressEvent(self, event):
        if self.editing:
            return
        self.close_editor()
        row = self._row_at(event.position())
        if row is not None:
            self.row_clicked.emit(row)

    def mouseDoubleClickEvent(self, event):
        """Doppelklick öffnet eine einzelne Zelle zum Überschreiben."""
        if self.editing:
            return
        cell = self._cell_at(event.position())
        if cell is None:
            return
        self.open_editor(*cell)

    def open_editor(self, r: int, c: int):
        self.close_editor()
        _, _, cols = self._geometry()
        centre = self._cell_center(r, c, cols)
        field = QLineEdit(str(self.matrix.rows[r][c]).replace("-", MINUS), self)
        field.setObjectName("cell")
        field.setAlignment(Qt.AlignCenter)
        field.setFont(FONTS.f(15))
        field.setFixedSize(68, 38)
        field.move(int(centre.x() - 34), int(centre.y() - 19))
        field.selectAll()
        field.show()
        field.setFocus()
        field.returnPressed.connect(self._commit_editor)
        field.editingFinished.connect(self.close_editor)
        self.editor, self.editor_cell = field, (r, c)

    def _commit_editor(self):
        if self.editor is None or self.editor_cell is None:
            return
        r, c = self.editor_cell
        text = self.editor.text()
        self.close_editor()
        self.cell_edited.emit(r, c, text)

    def close_editor(self):
        if self.editor is not None:
            editor, self.editor = self.editor, None
            self.editor_cell = None
            editor.blockSignals(True)
            editor.hide()
            editor.deleteLater()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.editor is not None:
            self.close_editor()
            return
        super().keyPressEvent(event)


# ==========================================================================
# Verlauf
# ==========================================================================

class HistoryRow(QFrame):
    picked = Signal(int)

    def __init__(self, index: int, text: str, auto: bool, current: bool,
                 parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("histrow_on" if current else "histrow")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 12, 6)
        lay.setSpacing(12)

        dot = QLabel(str(index))
        dot.setAlignment(Qt.AlignCenter)
        dot.setFixedSize(24, 24)
        dot.setFont(FONTS.f(10, QFont.DemiBold))
        dot.setStyleSheet(
            f"color: {'#ffffff' if current else T['sec']};"
            f"background: {T['blue'] if current else T['fill']};"
            f"border-radius: 12px;")
        lay.addWidget(dot)
        lay.addWidget(label(text, 11))
        lay.addStretch(1)
        if auto:
            lay.addWidget(label("Vorschlag", 10, "ter"))

    def mousePressEvent(self, _event):
        self.picked.emit(self.index)


# ==========================================================================
# Dialoge
# ==========================================================================

class Toast(QFrame):
    """Kurze Rückmeldung unten rechts, die sich selbst wieder ausblendet."""

    HEIGHT = 40

    def __init__(self, parent: QWidget, text: str, kind: str = "info"):
        super().__init__(parent)
        self.setObjectName("toast")
        tone = {"info": "text", "ok": "green", "bad": "red"}[kind]
        glyph = {"info": "", "ok": "✓  ", "bad": "✕  "}[kind]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 18, 0)
        self.label = label(glyph + text, 11, tone, weight=QFont.DemiBold)
        lay.addWidget(self.label)
        self.setFixedHeight(self.HEIGHT)
        self.setStyleSheet(
            f"#toast {{ background: {T['card']}; border-radius: {self.HEIGHT // 2}px;"
            f"border: 1px solid {T['line']}; }}")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 60 if not T["dark"] else 150))
        self.setGraphicsEffect(shadow)
        self.adjustSize()
        self.setFixedWidth(max(180, self.label.sizeHint().width() + 40))


class ToastArea(QObject):
    """Stapelt die Meldungen und räumt sie nach ein paar Sekunden weg."""

    def __init__(self, host: QWidget):
        super().__init__(host)
        self.host = host
        self.items: list[Toast] = []

    def show(self, text: str, kind: str = "info", ms: int = 2600):
        toast = Toast(self.host, text, kind)
        self.items.append(toast)
        if len(self.items) > 3:
            self._drop(self.items[0])
        toast.show()
        self._layout()
        QTimer.singleShot(ms, lambda: self._drop(toast))

    def _drop(self, toast: Toast):
        if toast not in self.items:
            return
        self.items.remove(toast)
        toast.hide()
        toast.deleteLater()
        self._layout()

    def _layout(self):
        bottom = self.host.height() - 24
        for toast in reversed(self.items):
            bottom -= Toast.HEIGHT + 8
            toast.move(self.host.width() - toast.width() - 24, bottom)
            toast.raise_()

    def reposition(self):
        self._layout()


class MiniMatrix(QWidget):
    """Kompakte, nicht bedienbare Matrix für das Rechenweg-Fenster."""

    CELL_W = 56
    CELL_H = 34
    PAD = 10
    DIV_W = 14

    def __init__(self, matrix: Matrix, decimal: bool, changed: Sequence[int] = (),
                 parent=None):
        super().__init__(parent)
        self.matrix, self.decimal = matrix, decimal
        self.changed = set(changed)
        # Eine erweiterte Matrix bekommt den Trennstrich, eine reine nicht.
        self.aug = bool(getattr(matrix, "augmented", False))
        self.cols = matrix.width
        gap = self.DIV_W if self.aug else 0
        self.setFixedSize(
            self.PAD * 2 + 26 + self.CELL_W * self.cols + gap,
            self.PAD * 2 + self.CELL_H * matrix.m)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        cols = self.cols
        gap = self.DIV_W if self.aug else 0
        split = self.matrix.n if self.aug else cols
        x0 = self.PAD + 13
        top = self.PAD
        bottom = top + self.CELL_H * self.matrix.m

        for r in range(self.matrix.m):
            cy = top + self.CELL_H * r + self.CELL_H / 2
            if r in self.changed:
                p.setPen(Qt.NoPen)
                p.setBrush(qcolor("flash"))
                p.drawRoundedRect(
                    QRectF(x0 - 4, cy - self.CELL_H / 2 + 2,
                           self.CELL_W * cols + gap + 8,
                           self.CELL_H - 4), 9, 9)
            for c in range(cols):
                x = x0 + self.CELL_W * c + self.CELL_W / 2
                if c >= split:
                    x += gap
                paint_value(p, QPointF(x, cy), self.matrix.rows[r][c],
                            qcolor("text"), self.decimal, size=12)

        pen = QPen(qcolor("text"), 1.6)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for x, d in ((self.PAD + 3, 1), (self.width() - self.PAD - 3, -1)):
            path = QPainterPath(QPointF(x + d * 7, top - 3))
            path.lineTo(QPointF(x, top - 3))
            path.lineTo(QPointF(x, bottom + 3))
            path.lineTo(QPointF(x + d * 7, bottom + 3))
            p.drawPath(path)

        if self.aug:
            div = QPen(qcolor("line"), 1)
            div.setDashPattern([2, 4])
            p.setPen(div)
            dx = x0 + self.CELL_W * split + gap / 2
            p.drawLine(QPointF(dx, top), QPointF(dx, bottom))


class WalkthroughDialog(QDialog):
    """Der komplette Rechenweg bis zur reduzierten Zeilenstufenform."""

    def __init__(self, parent, start: Matrix, decimal: bool):
        super().__init__(parent)
        self.setWindowTitle("Rechenweg")
        self.setModal(True)
        self.resize(640, 760)
        self.setStyleSheet(parent.styleSheet())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        head.setObjectName("nav")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(26, 18, 26, 16)
        hl.setSpacing(3)
        hl.addWidget(label("Rechenweg bis zur reduzierten Zeilenstufenform", 15,
                           weight=QFont.DemiBold))
        steps = start.steps_to_rref()
        hl.addWidget(label(
            "Vom aktuellen Stand aus, Schritt für Schritt."
            if steps else "Die Matrix steht bereits in reduzierter Zeilenstufenform.",
            11, "sec", wrap=True))
        outer.addWidget(head)

        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {T['line']};")
        outer.addWidget(line)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget()
        page.setObjectName("page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(26, 22, 26, 26)
        lay.setSpacing(14)

        previous = start
        lay.addWidget(self._card("Ausgangspunkt", "", start, decimal, ()))
        for index, (op, result) in enumerate(steps, 1):
            touched = ([op.i, op.j] if op.kind == "swap" else [op.i])
            lay.addWidget(self._card(f"Schritt {index}", str(op), result,
                                     decimal, touched, op.reason))
            previous = result
        if steps:
            info = previous.analyze()
            if info["type"] == "none":
                text = "Keine Lösung — eine Zeile ergibt einen Widerspruch."
            elif info["type"] == "unique":
                text = "Lösung:   " + "    ".join(previous.solution_lines(decimal))
            else:
                text = "Unendlich viele Lösungen:   " + "    ".join(
                    previous.solution_lines(decimal))
            lay.addWidget(label(text, 12, "green", weight=QFont.DemiBold, wrap=True))
        lay.addStretch(1)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

        foot = QWidget()
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(26, 14, 26, 18)
        fl.addWidget(label(f"{len(steps)} Schritt"
                           f"{'e' if len(steps) != 1 else ''}", 11, "ter"))
        fl.addStretch(1)
        fl.addWidget(pill("Schliessen", "primary", self.accept, small=True))
        outer.addWidget(foot)

    def _card(self, title: str, operation: str, matrix: Matrix, decimal: bool,
              touched: Sequence[int], reason: str = "") -> QWidget:
        card = QFrame()
        card.setObjectName("stepcard")
        card.setStyleSheet(
            f"#stepcard {{ background: {T['card']}; border-radius: 16px;"
            f"border: 1px solid {T['hair']}; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label(title, 11, "ter", weight=QFont.DemiBold))
        head.addStretch(1)
        if operation:
            head.addWidget(label(operation, 12, weight=QFont.DemiBold))
        lay.addLayout(head)
        if reason:
            lay.addWidget(label(reason, 10, "sec", wrap=True))
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(MiniMatrix(matrix, decimal, touched))
        holder.addStretch(1)
        lay.addLayout(holder)
        return card


# ==========================================================================
# Ergebnisse aus lialkit im Fenster darstellen
# ==========================================================================

class ResultView(QScrollArea):
    """
    Zeigt ein Ergebnis aus lialkit — Titel, Schritte, Endwerte.

    Die Matrizen werden mit demselben MiniMatrix gezeichnet wie im
    Rechenweg-Fenster, damit Notebook und App gleich aussehen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.host = QWidget()
        self.host.setObjectName("page")
        self.lay = QVBoxLayout(self.host)
        self.lay.setContentsMargins(0, 0, 8, 0)
        self.lay.setSpacing(10)
        self.setWidget(self.host)
        self.decimal = False
        self._placeholder()

    def _clear(self):
        while self.lay.count():
            item = self.lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().setParent(None)
                        sub.widget().deleteLater()

    def _placeholder(self, text: str = "Noch nichts gerechnet."):
        self._clear()
        self.lay.addWidget(label(text, 12, "ter", wrap=True))
        self.lay.addStretch(1)

    def show_error(self, message: str):
        self._clear()
        box = QFrame()
        box.setObjectName("errbox")
        box.setStyleSheet(f"#errbox {{ background: {T['red_soft']};"
                          f"border-radius: 12px; }}")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(16, 13, 16, 13)
        bl.addWidget(wrapped(message, 460, 12, "red"))
        self.lay.addWidget(box)
        self.lay.addStretch(1)

    def show_result(self, result, decimal: bool = False):
        """Baut die Anzeige aus einem lialkit-Result auf."""
        self._clear()
        result.decimal = decimal
        self.decimal = decimal

        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label(result.title, 15, weight=QFont.DemiBold))
        if result.summary:
            head.addWidget(wrapped(result.summary, 480, 11, "sec"))
        holder = QWidget()
        holder.setLayout(head)
        self.lay.addWidget(holder)

        for warning in result.warnings:
            box = QFrame()
            box.setObjectName("warnbox")
            box.setStyleSheet(f"#warnbox {{ background: {T['amber_soft']};"
                              f"border-radius: 10px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.addWidget(wrapped(warning, 460, 11, "amber"))
            self.lay.addWidget(box)

        for index, step in enumerate(result.steps, 1):
            self.lay.addWidget(self._step_card(index, step, decimal))

        if result.highlights:
            self.lay.addWidget(self._result_card(result.highlights, decimal))
        self.lay.addStretch(1)

    def _step_card(self, index: int, step, decimal: bool) -> QWidget:
        from lialkit.render import tex_to_text

        card = QFrame()
        card.setObjectName("stepcard")
        card.setStyleSheet(f"#stepcard {{ background: {T['fill2']};"
                           f"border-radius: 14px; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 13, 16, 13)
        lay.setSpacing(5)
        lay.addWidget(label(f"{index}. {step.title}", 12,
                            weight=QFont.DemiBold))
        if step.detail:
            lay.addWidget(wrapped(tex_to_text(step.detail), 450, 11, "sec"))
        if step.items:
            row = QHBoxLayout()
            row.setSpacing(18)
            for name, obj in step.items:
                row.addLayout(self._item(name, obj, decimal))
            row.addStretch(1)
            lay.addLayout(row)
        if step.note:
            lay.addWidget(wrapped(step.note, 450, 10, "ter"))
        return card

    def _result_card(self, highlights, decimal: bool) -> QWidget:
        card = QFrame()
        card.setObjectName("rescard")
        card.setStyleSheet(f"#rescard {{ background: {T['blue_soft']};"
                           f"border-radius: 14px; }}")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(16, 13, 16, 13)
        lay.setSpacing(20)
        for name, obj in highlights:
            lay.addLayout(self._item(name, obj, decimal, big=True))
        lay.addStretch(1)
        return card

    def _item(self, name: str, obj, decimal: bool, big: bool = False):
        from lialkit.core import Matrix as CoreMatrix, Vector as CoreVector
        from lialkit.render import tex_to_text

        row = QHBoxLayout()
        row.setSpacing(8)
        if name:
            row.addWidget(label(tex_to_text(name) + " =", 12 if big else 11,
                                weight=QFont.DemiBold if big else QFont.Normal),
                          0, Qt.AlignVCenter)
        if isinstance(obj, CoreVector):
            obj = CoreMatrix([[v] for v in obj.values])
        if isinstance(obj, CoreMatrix):
            row.addWidget(MiniMatrix(_as_app_matrix(obj), decimal), 0,
                          Qt.AlignVCenter)
        else:
            row.addWidget(label(fmt(obj, decimal), 13 if big else 12,
                                weight=QFont.DemiBold if big else QFont.Normal),
                          0, Qt.AlignVCenter)
        return row


def _as_app_matrix(M):
    """Eine lialkit-Matrix so verpacken, dass MiniMatrix sie zeichnen kann."""
    out = Matrix(M.rows, augmented=False)
    return out


# ==========================================================================
# Reiter: Verfahren
# ==========================================================================

METHODS = {
    "LR-Zerlegung": "lu",
    "Inverse": "inverse",
    "Determinante": "det",
    "Rang": "rank",
}


class MethodsPage(QWidget):
    """
    LR-Zerlegung, Inverse, Determinante und Rang für eine eigene Matrix.

    Die Matrix wird hier ohne rechte Seite eingegeben — für diese Verfahren
    gibt es keine.
    """

    #: Die Pivot-Spur fehlt mit Absicht: diese Rechenwege führen kein Pivot
    #: mit, das sich markieren liesse.
    ANIM_SCENES = {"Ganzer Durchlauf": "full", "Letzter Schritt": "step"}
    ANIM_SUBTITLE = ("Zeigt die Umformungen des gewählten Verfahrens als "
                     "kurzen Film.")

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.window = window
        self.size = 3
        self.draft = [["2", "1", "-1"], ["-3", "-1", "2"], ["-2", "1", "2"]]
        self.method = "lu"
        self.decimal = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 26, 32, 30)
        outer.setSpacing(0)
        outer.addWidget(label("Verfahren auf einer Matrix", 26,
                              weight=QFont.Bold, display=True))
        outer.addWidget(label("Zerlegen, invertieren, Determinante und Rang — "
                              "jeweils mit vollem Rechenweg.", 13, "sec",
                              display=True, wrap=True))
        outer.addSpacing(20)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        outer.addLayout(columns, 1)

        left = QVBoxLayout()
        left.setSpacing(18)
        columns.addLayout(left, 2)
        right = QVBoxLayout()
        right.setSpacing(18)
        columns.addLayout(right, 3)

        # -- Eingabe
        self.card_input = Card()
        left.addWidget(self.card_input)
        lay = self.card_input.lay
        lay.addWidget(label("Matrix", 15, weight=QFont.DemiBold))
        lay.addWidget(label("Quadratisch, 2×2 bis 6×6.", 11, "sec"))
        lay.addSpacing(14)
        self.grid_host = QWidget()
        self.grid = QVBoxLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.grid_host)
        lay.addSpacing(14)

        size_row = QHBoxLayout()
        size_row.addWidget(label("Grösse", 11, "sec"))
        self.stepper = Stepper(self.size)
        self.stepper.stepped.connect(self.resize_matrix)
        size_row.addWidget(self.stepper)
        size_row.addStretch(1)
        lay.addLayout(size_row)
        lay.addSpacing(10)

        pick_row = QHBoxLayout()
        drop = Dropdown(list(PRESETS), placeholder="Beispiel")
        drop.picked.connect(self.load_preset)
        pick_row.addWidget(drop)
        pick_row.addStretch(1)
        lay.addLayout(pick_row)
        self.lbl_err = label("", 11, "red", wrap=True)
        lay.addWidget(self.lbl_err)

        # -- Verfahren
        self.card_method = Card()
        left.addWidget(self.card_method)
        mlay = self.card_method.lay
        mlay.addWidget(label("Verfahren", 15, weight=QFont.DemiBold))
        mlay.addSpacing(12)
        self.seg_method = Segmented(list(METHODS), 0, seg_width=98)
        self.seg_method.changed.connect(self.set_method)
        mlay.addWidget(self.seg_method)
        mlay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(pill("Rechnen", "primary", self.compute))
        self.seg_fmt = Segmented(["Brüche", "Dezimal"], 0)
        self.seg_fmt.changed.connect(self.set_format)
        row.addWidget(self.seg_fmt)
        row.addStretch(1)
        mlay.addLayout(row)
        mlay.addSpacing(10)
        row_draw = QHBoxLayout()
        row_draw.setSpacing(8)
        row_draw.addWidget(pill("Zeichnen", "secondary", self.draw,
                                small=True))
        row_draw.addStretch(1)
        mlay.addLayout(row_draw)
        mlay.addSpacing(6)
        mlay.addWidget(label("Zeichnen zeigt, was die Matrix mit dem Raum "
                             "macht: bei 2×2 das verzerrte Einheitsquadrat, "
                             "dessen Fläche |det A| ist, bei 3×3 die drei "
                             "Spalten als Vektoren samt aufgespanntem Körper.",
                             10, "ter", wrap=True))

        # -- Prüfung
        self.wolfram = WolframPanel(
            self._wolfram_run,
            "Vergleicht Determinante, Rang und — je nach Verfahren — "
            "Inverse und Stufenform mit Wolfram|Alpha.")
        left.addWidget(self.wolfram)
        left.addStretch(1)

        # -- Ergebnis
        output_card(self, right)

        self.render_grid()

    def _wolfram_run(self, appid: str):
        """Gibt die Prüfung als Aufgabe zurück; ausgeführt wird sie im Faden."""
        A = self.matrix()
        method, decimal = self.method, self.decimal
        return lambda: run_methods_checks(appid, A, method, decimal)

    # -- Eingabe ----------------------------------------------------------

    def render_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().setParent(None)
                        sub.widget().deleteLater()
        self.fields = []
        for r in range(self.size):
            row = QHBoxLayout()
            row.setSpacing(6)
            line = []
            for c in range(self.size):
                field = QLineEdit(self.draft[r][c])
                field.setObjectName("cell")
                field.setAlignment(Qt.AlignCenter)
                field.setFont(FONTS.f(13))
                field.setFixedSize(58, 34)
                field.textChanged.connect(self.read_fields)
                row.addWidget(field)
                line.append(field)
            row.addStretch(1)
            self.fields.append(line)
            self.grid.addLayout(row)

    def read_fields(self):
        before = self.draft
        self.draft = [[f.text() for f in row] for row in self.fields]
        if before != self.draft and getattr(self, "wolfram", None):
            # Ein Bericht zu einer anderen Matrix wäre schlimmer als keiner.
            self.wolfram.invalidate()

    def resize_matrix(self, delta: int):
        self.read_fields()
        self.size = max(2, min(6, self.size + delta))
        self.draft = [[self.draft[r][c] if r < len(self.draft)
                       and c < len(self.draft[r]) else ("1" if r == c else "0")
                       for c in range(self.size)] for r in range(self.size)]
        self.stepper.set_value(self.size)
        self.render_grid()

    def load_preset(self, name: str):
        rows = [row[:-1] for row in PRESETS[name]]      # rechte Seite weg
        if len(rows) != len(rows[0]):
            self.lbl_err.setText("Dieses Beispiel ist nicht quadratisch.")
            return
        self.size = len(rows)
        self.draft = [[str(v) for v in row] for row in rows]
        self.stepper.set_value(self.size)
        self.render_grid()
        self.lbl_err.setText("")

    def set_method(self, index: int):
        self.method = list(METHODS.values())[index]
        self.wolfram.invalidate()
        self.anim.refresh()

    def set_format(self, index: int):
        self.decimal = bool(index)
        self.compute()

    # -- Rechnen ----------------------------------------------------------

    def matrix(self):
        from lialkit.core import Matrix as CoreMatrix
        self.read_fields()
        rows = []
        for r in self.draft:
            line = []
            for value in r:
                line.append(parse_number(value))
            rows.append(line)
        return CoreMatrix(rows)

    def compute(self):
        from lialkit.determinant import det
        from lialkit.factor import lu
        from lialkit.inverse import inverse
        from lialkit.solving import rank

        try:
            A = self.matrix()
        except ValueError as exc:
            self.lbl_err.setText(str(exc))
            self.view.show_error(str(exc))
            return
        self.lbl_err.setText("")
        try:
            if self.method == "lu":
                result = lu(A, decimal=self.decimal)
            elif self.method == "inverse":
                result = inverse(A, decimal=self.decimal)
            elif self.method == "det":
                result = det(A, decimal=self.decimal)
            else:
                result = rank(A, weg=True, decimal=self.decimal)
        except ValueError as exc:
            self.view.show_error(str(exc))
            show_view(self, 0)
            return
        self.view.show_result(result, self.decimal)
        show_view(self, 0)

    # -- Animation --------------------------------------------------------

    def anim_key(self):
        self.read_fields()
        return (tuple(tuple(row) for row in self.draft), self.method,
                self.decimal)

    def build_job(self, scene: str, quality: str):
        """
        Baut den Auftrag für das gerade gewählte Verfahren.

        Determinante und Inverse-Kurzformel haben keine Zwischenmatrizen;
        dort gibt es nichts zu animieren, und der Text sagt das statt einen
        Film zu zeigen, der nichts erklärt.
        """
        from lialkit.factor import lu
        from lialkit.inverse import inverse
        from lialkit.solving import rank

        if self.method == "det":
            return ("Die Determinante entsteht ohne Zwischenmatrizen — "
                    "Sarrus und Laplace rechnen in einer Summe. Anschaulich "
                    "wird sie unter »Zeichnung«: dort ist |det A| die Fläche "
                    "beziehungsweise das Volumen.")
        A = self.matrix()
        if self.method == "lu":
            result, picked = lu(A, decimal=self.decimal), "R"
            title = "LR-Zerlegung"
        elif self.method == "inverse":
            result, picked = inverse(A, decimal=self.decimal), None
            title = "Inverse über Gauss-Jordan"
        else:
            result, picked = rank(A, weg=True, decimal=self.decimal), None
            title = "Rang über die Zeilenstufenform"
        return gauss_anim.job_from_steps(
            result.steps, scene=scene, quality=quality,
            decimal=self.decimal, title=title, label=picked)

    # -- Zeichnung --------------------------------------------------------

    def draw(self):
        """
        Zeigt die Matrix als Abbildung.

        Bei 2×2 das Bild des Einheitsquadrats — seine Fläche ist |det A|,
        und bei einer singulären Matrix fällt es auf eine Strecke zusammen.
        Bei 3×3 die drei Spalten als Vektoren mit dem Spat, dessen Volumen
        |det A| ist. Grössere Matrizen lassen sich nicht zeichnen.
        """
        from lialkit import plots

        try:
            A = self.matrix()
        except ValueError as exc:
            self.lbl_err.setText(str(exc))
            return
        self.lbl_err.setText("")
        if not PlotCanvas.available():
            self.plot.show_message(
                "Zum Zeichnen fehlt Matplotlib. "
                "Mit »pip install matplotlib« nachinstallieren.")
            show_view(self, 1)
            return
        try:
            if self.size == 2:
                figure = plots.plot_transformation(A)
                hint = "Die Fläche des Parallelogramms ist |det A|."
            elif self.size == 3:
                cols = [Vector([A.rows[r][c] for r in range(3)])
                        for c in range(3)]
                figure = plots.plot_spat(
                    *cols, interactive=False,
                    labels=["a\u2081", "a\u2082", "a\u2083"])
                hint = ("Die Spalten a\u2081, a\u2082, a\u2083 von A. Das "
                        "Volumen des Körpers ist |det A| — mit der Maus "
                        "drehen.")
            else:
                self.plot.show_message(
                    f"Eine {self.size}×{self.size}-Matrix lässt sich nicht "
                    "zeichnen — anschaulich wird es nur in der Ebene und im "
                    "Raum, also bei 2×2 und 3×3.")
                show_view(self, 1)
                return
        except Exception as exc:
            self.plot.show_message(f"Zeichnen ging nicht: {exc}")
            show_view(self, 1)
            return
        self.plot.show_figure(figure, hint)
        show_view(self, 1)

    def restyle(self):
        for card in (self.card_input, self.card_method, self.card_out):
            card.restyle()
        self.view._placeholder("Nach dem Themawechsel neu rechnen.")
        self.plot.restyle()
        self.anim.restyle()
        self.wolfram.restyle()


# ==========================================================================
# Reiter: Vektoren
# ==========================================================================

VECTOR_OPS = {
    "Skalarprodukt": "dot",
    "Vektorprodukt": "cross",
    "Spatprodukt": "spat",
    "Winkel": "angle",
    "Projektion": "projection",
    "Addition": "add",
    "Ortsvektor / Verbindung": "position",
    "Skalar · Vektor": "scale",
}


# ==========================================================================
# Zeichnungen im Fenster
# ==========================================================================

def theme_figure(figure):
    """
    Passt eine Matplotlib-Figur an das gerade gewählte Thema an.

    Die Zeichnungen in lialkit sind für den hellen Hintergrund gebaut. Im
    dunklen Thema müssen Hintergrund, Achsen, Beschriftungen und Gitter
    nachgezogen werden, sonst steht ein weisser Block in einem schwarzen
    Fenster.
    """
    card, text, sec, line = T["card"], T["text"], T["sec"], T["line"]
    figure.patch.set_facecolor(card)
    for ax in figure.axes:
        spatial = hasattr(ax, "zaxis")
        ax.set_facecolor(card)
        ax.title.set_color(text)
        ax.tick_params(colors=sec, which="both")
        for spine in ax.spines.values():
            spine.set_color(line)
        axes_list = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if spatial else [])
        for axis in axes_list:
            axis.label.set_color(sec)
            if spatial:
                axis.set_pane_color((0, 0, 0, 0))
                axis.line.set_color(line)
                axis._axinfo["grid"]["color"] = line
                axis._axinfo["tick"]["color"] = line
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            gl.set_color(line)
        for child in ax.lines:
            # Die beiden Nulllinien der flachen Zeichnungen.
            if child.get_linestyle() == "-" and child.get_linewidth() <= 1.01:
                child.set_color(sec)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(card)
            legend.get_frame().set_edgecolor(line)
            for entry in legend.get_texts():
                entry.set_color(text)


class PlotCanvas(QWidget):
    """
    Eine Zeichnung im Fenster selbst, nicht im Browser.

    Steckt eine Matplotlib-Figur in eine Qt-Leinwand. Räumliche Bilder
    lassen sich damit von sich aus mit der Maus drehen — Matplotlib bringt
    das mit, sobald die Figur in einem Fenster hängt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self.figure = None
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(10)
        self.empty = label("Noch keine Zeichnung — »Zeichnen« drücken.",
                           13, "sec", wrap=True)
        self.lay.addWidget(self.empty, 1, Qt.AlignCenter)
        self.hint = label("", 11, "ter", wrap=True)
        self.lay.addWidget(self.hint)

    @staticmethod
    def available() -> bool:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvas  # noqa
        except Exception:
            return False
        return True

    def show_figure(self, figure, hint: str = ""):
        """Zeigt eine Matplotlib-Figur; die vorige wird freigegeben."""
        from matplotlib.backends.backend_qtagg import FigureCanvas

        self.clear()
        theme_figure(figure)
        self.figure = figure
        self.canvas = FigureCanvas(figure)
        self.canvas.setMinimumHeight(340)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.empty.hide()
        self.lay.insertWidget(0, self.canvas, 1)
        self.canvas.draw_idle()
        self.hint.setText(hint)

    def show_message(self, text: str):
        self.clear()
        self.empty.setText(text)
        self.empty.show()
        self.hint.setText("")

    def clear(self):
        if self.canvas is not None:
            self.canvas.hide()
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = None

    def restyle(self):
        if self.figure is not None and self.canvas is not None:
            theme_figure(self.figure)
            self.canvas.draw_idle()


class ChecksWorker(QObject):
    """Führt eine Prüffunktion aus, ohne das Fenster anzuhalten."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, run):
        super().__init__()
        self.run_checks = run

    def run(self):
        try:
            self.finished.emit(self.run_checks())
        except (RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))


class WolframPanel(QWidget):
    """
    Die Prüfung mit Wolfram|Alpha als eigene Karte.

    Sie kennt die Rechnung nicht: `build_checks()` liefert den Bericht, alles
    andere — AppID, Verbindungstest, Darstellung — ist für jede Seite gleich.
    """

    def __init__(self, build_checks, subtitle: str, parent=None):
        super().__init__(parent)
        self.build_checks = build_checks
        self.report: WolframReport | None = None
        self.status: tuple[str, str] | None = None
        self.thread = None
        self.worker = None
        self.decimal = False

        self.card = Card()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.card)
        lay = self.card.lay
        lay.addWidget(label("Prüfung mit Wolfram|Alpha", 15,
                            weight=QFont.DemiBold))
        lay.addWidget(label(subtitle, 11, "sec", wrap=True))
        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn = pill("Jetzt prüfen", "primary", self.start)
        row.addWidget(self.btn)
        self.btn_key = pill("AppID", "secondary", self.ask_appid, small=True)
        row.addWidget(self.btn_key)
        row.addStretch(1)
        lay.addLayout(row)
        self.body = QVBoxLayout()
        self.body.setSpacing(0)
        lay.addSpacing(6)
        lay.addLayout(self.body)
        self.render()

    # -- Bedienung --------------------------------------------------------

    def ask_appid(self):
        dialog = AppIDDialog(self.window(), load_appid())
        if dialog.exec() == QDialog.Accepted:
            save_appid(dialog.field.text().strip())
            self.render()

    def start(self):
        if self.thread is not None:
            return
        appid = load_appid()
        if not appid:
            self.ask_appid()
            appid = load_appid()
            if not appid:
                return
        try:
            run = self.build_checks(appid)
        except (ValueError, RuntimeError) as exc:
            self.status = ("bad", str(exc))
            self.render()
            return
        self.report = None
        self.status = ("busy", "Frage Wolfram|Alpha …")
        self.render()
        self.thread = QThread(self)
        self.worker = ChecksWorker(run)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def _stop_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None

    def _done(self, report):
        self._stop_thread()
        self.report, self.status = report, None
        self.render()

    def _failed(self, message: str):
        self._stop_thread()
        self.report, self.status = None, ("bad", message)
        self.render()

    def invalidate(self):
        """Nach einer Änderung an der Eingabe gilt der alte Bericht nicht mehr."""
        if self.report is not None or self.status is not None:
            self.report, self.status = None, None
            self.render()

    # -- Darstellung ------------------------------------------------------

    def render(self):
        while self.body.count():
            item = self.body.takeAt(0)
            # Nach setParent(None) gibt das Element den Verweis nicht mehr
            # heraus — darum vorher festhalten.
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.btn_key.setText("AppID ändern" if load_appid() else "AppID")

        if self.status:
            level, text = self.status
            color = {"busy": "sec", "bad": "red", "ok": "green"}[level]
            self.body.addSpacing(12)
            self.body.addWidget(label(text, 11, color, wrap=True))
            self.btn.setEnabled(level != "busy")
            self.btn.setText("Prüfe …" if level == "busy" else "Jetzt prüfen")
            return

        self.btn.setEnabled(True)
        self.btn.setText("Jetzt prüfen")
        if self.report is None:
            if not load_appid():
                self.body.addSpacing(12)
                self.body.addWidget(label(
                    "Dafür braucht es einmalig eine kostenlose AppID.",
                    11, "ter", wrap=True))
            return

        report = self.report
        self.body.addSpacing(14)
        tone = {"ok": ("green", "green_soft"), "bad": ("red", "red_soft"),
                "warn": ("amber", "amber_soft")}[report.verdict]
        mark = {"ok": "✓", "bad": "✕", "warn": "!"}[report.verdict]
        self.body.addWidget(Badge(f"{mark}  {report.headline}", *tone))
        self.body.addSpacing(10)

        for check in report.checks:
            glyph = {"ok": "✓", "bad": "✕", "info": "•",
                     "error": "!"}[check.status]
            color = {"ok": "green", "bad": "red", "info": "sec",
                     "error": "amber"}[check.status]
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(label(glyph, 11, color, weight=QFont.DemiBold))
            text = check.name
            if check.theirs and check.theirs != check.ours:
                text += f":  wir {check.ours}, Wolfram {check.theirs}"
            elif check.ours:
                text += f":  {check.ours}"
            row.addWidget(label(text, 11, wrap=True), 1)
            holder = QWidget()
            holder.setLayout(row)
            self.body.addWidget(holder)
            if check.note:
                self.body.addWidget(label("     " + check.note, 10, "ter",
                                          wrap=True))

        if report.their_matrix is not None:
            self.body.addSpacing(12)
            self.body.addWidget(label("Wolframs Matrix", 11, "sec"))
            cells = [[fmt(v, self.decimal) for v in row]
                     for row in report.their_matrix.rows]
            width = max(len(s) for row in cells for s in row)
            for row in cells:
                self.body.addWidget(label(
                    "  " + "  ".join(s.rjust(width) for s in row), 11,
                    mono=True))

        if report.url:
            self.body.addSpacing(10)
            btn = pill("Bei Wolfram öffnen", "plain", small=True)
            btn.clicked.connect(lambda: webbrowser.open(report.url))
            self.body.addWidget(btn)

    def restyle(self):
        self.card.restyle()
        self.render()


def manim_state() -> tuple[bool, str]:
    """(bereit, Begründung) — einmal zentral statt an jeder Karte neu."""
    if gauss_anim is None:
        return False, (ANIM_IMPORT_ERROR
                       or "Manim lässt sich nicht laden.")
    return gauss_anim.manim_available()


class AnimationPane(QWidget):
    """
    Die Animation als eigenständiger Bereich, für jede Seite verwendbar.

    Sie weiss nichts über die Rechnung. Zwei Rückrufe genügen ihr:
    `build_job(szene, qualität)` liefert einen Auftrag oder einen Text, der
    erklärt, warum es hier nichts zu zeigen gibt; `key_of()` liefert einen
    Fingerabdruck der Eingabe, damit ein veralteter Film als solcher
    gekennzeichnet wird statt stillschweigend falsch dazustehen.
    """

    def __init__(self, build_job, key_of, scenes: dict | None = None,
                 subtitle: str = "", parent=None):
        super().__init__(parent)
        self.build_job = build_job
        self.key_of = key_of
        self.scenes = scenes
        self.scene = next(iter(scenes.values())) if scenes else "full"
        self.quality = "Normal"
        self.thread = None
        self.worker = None
        self.out_dir: Path | None = None
        self.stamp = None
        self.ready, why = manim_state()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.sub = label(subtitle or "Zeigt die Rechnung als kurzen Film, "
                                     "gerendert mit Manim.", 11, "sec",
                         wrap=True)
        lay.addWidget(self.sub)

        if not self.ready:
            # Nicht verstecken: Wer die Animation sucht, soll lesen können,
            # warum sie fehlt und was dagegen hilft.
            hint = (gauss_anim.INSTALL_HINT if gauss_anim else
                    "brew install cairo pango pkg-config ffmpeg\n"
                    "pip install manim")
            self.sub.setText(f"{why}\n\nSo kommt sie dazu:\n{hint}")
            lay.addStretch(1)
            return

        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        if scenes:
            start = next(k for k, v in scenes.items() if v == self.scene)
            self.pick = Dropdown(list(scenes), value=start)
            self.pick.setFixedWidth(162)
            self.pick.picked.connect(self._set_scene)
            row.addWidget(self.pick)
        self.quality_pick = Dropdown(list(ANIM_QUALITY), value=self.quality)
        self.quality_pick.setFixedWidth(118)
        self.quality_pick.picked.connect(self._set_quality)
        row.addWidget(self.quality_pick)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn = pill("Rendern", "primary", self.render)
        row2.addWidget(self.btn)
        self.btn_big = pill("Gross anzeigen", "secondary", self.show_big,
                            small=True)
        self.btn_big.hide()
        row2.addWidget(self.btn_big)
        row2.addStretch(1)
        lay.addSpacing(10)
        lay.addLayout(row2)

        self.hint = label("", 10, "ter", wrap=True)
        lay.addWidget(self.hint)

        self.bar = QProgressBar()
        self.bar.setObjectName("progress")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        lay.addSpacing(10)
        lay.addWidget(self.bar)
        self.status = label("", 11, "sec", wrap=True)
        lay.addWidget(self.status)

        self.player = MoviePlayer()
        self.player.hide()
        self.player.clicked.connect(self.show_big)
        lay.addSpacing(12)
        lay.addWidget(self.player)
        lay.addStretch(1)
        self.refresh()

    # -- Auswahl ----------------------------------------------------------

    def _set_scene(self, name: str):
        self.scene = self.scenes[name]
        self.refresh()

    def _set_quality(self, name: str):
        self.quality = name
        self.refresh()

    def refresh(self):
        """Prüft, ob der gezeigte Film noch zur Eingabe passt."""
        if not self.ready:
            return
        busy = self.thread is not None
        self.btn.setEnabled(not busy)
        if busy:
            return
        if self.player.loaded and self.stamp is not None:
            if self.stamp != self._stamp_now():
                self.hint.setText("Zeigt einen früheren Stand. Neu rendern "
                                  "übernimmt die jetzige Eingabe.")
                return
            self.hint.setText("")
            return
        self.hint.setText(
            f"Ungefähr {self.estimate()} Sekunden Rechenzeit — das Rendern "
            "läuft im Hintergrund, die App bleibt bedienbar.")

    def estimate(self) -> int:
        """
        Grobe Schätzung der Rechenzeit.

        Gemessen auf einem Rechner ohne Grafikbeschleunigung: eine räumliche
        Szene mit Kamerafahrt braucht am längsten, eine Matrixfolge wächst
        mit der Zahl der Schritte.
        """
        factor = ANIM_QUALITY[self.quality][3]
        if self.scenes is None:            # eine Vektorszene
            return int(45 * factor)
        return int(25 * factor)

    def _stamp_now(self):
        return (self.key_of(), self.scene, self.quality, T["dark"])

    # -- Rendern ----------------------------------------------------------

    def render(self):
        if not self.ready or self.thread is not None:
            return
        quality, width, fps, _ = ANIM_QUALITY[self.quality]
        try:
            job = self.build_job(self.scene, quality)
        except Exception as exc:
            self._fail(str(exc))
            return
        if isinstance(job, str):              # nichts zu zeigen, mit Grund
            self._fail(job)
            return
        job.fmt = ANIM_FORMAT
        job.gif_width, job.gif_fps = width, fps
        job.theme = anim_theme()

        if self.out_dir is None:
            self.out_dir = Path(tempfile.mkdtemp(prefix="lial-anim-"))
        target = self.out_dir / f"{self.scene}-{self.quality}"
        shutil.rmtree(target, ignore_errors=True)

        self.player.stop()
        self.player.hide()
        self.btn_big.hide()
        self.btn.setEnabled(False)
        self.hint.setText("")
        self.bar.setRange(0, 100)
        self.bar.setValue(2)
        self.bar.show()
        self.status.setText("Manim startet …")

        self.thread = QThread(self)
        self.worker = AnimationWorker(job, target)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._fail)
        self.thread.start()

    def _progress(self, percent: int, text: str):
        self.bar.setValue(percent)
        self.status.setText(text)

    def _finish_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None
        self.bar.hide()
        self.btn.setEnabled(True)

    def _done(self, path: str):
        self._finish_thread()
        self.stamp = self._stamp_now()
        self.status.setText("")
        size = Path(path).stat().st_size / 1e6
        res = movie_size(path)
        self.player.load(path, f"{res.width()}×{res.height()}, "
                               f"{size:.1f} MB — zum Vergrössern anklicken")
        self.player.show()
        self.btn_big.show()

    def _fail(self, message: str):
        self._finish_thread()
        self.status.setText(message)
        self.player.stop()
        self.player.hide()
        self.btn_big.hide()

    def show_big(self):
        if self.player.loaded and self.player.path:
            AnimationWindow(self.window(), self.player.path,
                            "Animation").exec()

    # -- Thema und Aufräumen ----------------------------------------------

    def restyle(self):
        if self.ready:
            self.player.restyle()
            self.refresh()

    def shutdown(self):
        """Beendet einen laufenden Lauf, damit das Fenster schliessen kann."""
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
            self.thread = None
        if self.ready:
            self.player.stop()
        if self.out_dir is not None:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.out_dir = None


def anim_theme():
    """Die Farben der Animation aus dem gerade gewählten Thema."""
    return gauss_anim.Theme(
        background=T["card"], text=T["text"], muted=T["sec"],
        accent=T["blue"], highlight=T["flash"], pivot=T["blue_soft"],
        good=T["green"])


def output_card(page, parent_layout):
    """
    Legt die Ergebniskarte an: Rechenweg und Zeichnung als zwei Ansichten.

    Setzt auf `page` die Felder `card_out`, `seg_out`, `stack_out`, `view`
    und `plot` und gibt die Karte zurück.
    """
    page.card_out = Card()
    parent_layout.addWidget(page.card_out, 1)
    head = QHBoxLayout()
    page.seg_out = Segmented(["Rechenweg", "Zeichnung", "Animation"], 0,
                             seg_width=98)
    page.seg_out.changed.connect(lambda i: page.stack_out.setCurrentIndex(i))
    head.addWidget(page.seg_out)
    head.addStretch(1)
    page.card_out.lay.addLayout(head)
    page.card_out.lay.addSpacing(12)
    page.stack_out = QStackedWidget()
    # Mit Auswahl, nicht pauschal: eine Regel ohne Auswahl vererbt sich auf
    # alle Kinder und nähme den Knöpfen darin ihre Farbe.
    page.stack_out.setObjectName("outstack")
    page.stack_out.setStyleSheet("#outstack { background: transparent; }")
    page.view = ResultView()
    page.plot = PlotCanvas()
    page.anim = AnimationPane(page.build_job, page.anim_key,
                              scenes=getattr(page, "ANIM_SCENES", None),
                              subtitle=getattr(page, "ANIM_SUBTITLE", ""))
    page.stack_out.addWidget(page.view)
    page.stack_out.addWidget(page.plot)
    page.stack_out.addWidget(page.anim)
    page.card_out.lay.addWidget(page.stack_out, 1)
    return page.card_out


def show_view(page, index: int):
    """Schaltet die Ergebniskarte um, auch wenn der Nutzer nichts geklickt hat."""
    page.seg_out.set_value(index)
    page.stack_out.setCurrentIndex(index)


class VectorsPage(QWidget):
    """Skalarprodukt, Vektorprodukt, Spatprodukt, Winkel — mit Zeichnung."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.window = window
        self.dim = 3
        self.values = [["1", "2", "3"], ["4", "5", "6"], ["2", "-1", "4"]]
        self.op = "dot"
        self.decimal = False
        self.k = "2"
        self.k_field = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 26, 32, 30)
        outer.setSpacing(0)
        outer.addWidget(label("Vektoren", 26, weight=QFont.Bold, display=True))
        outer.addWidget(label("Skalar-, Vektor- und Spatprodukt, Winkel und "
                              "Projektion — mit Rechenweg und Zeichnung.",
                              13, "sec", display=True, wrap=True))
        outer.addSpacing(20)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        outer.addLayout(columns, 1)
        left = QVBoxLayout()
        left.setSpacing(18)
        columns.addLayout(left, 2)
        right = QVBoxLayout()
        right.setSpacing(18)
        columns.addLayout(right, 3)

        self.card_input = Card()
        left.addWidget(self.card_input)
        lay = self.card_input.lay
        lay.addWidget(label("Vektoren", 15, weight=QFont.DemiBold))
        lay.addWidget(label("u und v werden immer gebraucht, w nur beim "
                            "Spatprodukt.", 11, "sec", wrap=True))
        lay.addSpacing(14)
        self.grid_host = QWidget()
        self.grid = QVBoxLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        lay.addWidget(self.grid_host)
        lay.addSpacing(12)
        dim_row = QHBoxLayout()
        dim_row.addWidget(label("Einträge", 11, "sec"))
        self.seg_dim = Segmented(["2", "3"], 1, seg_width=48)
        self.seg_dim.changed.connect(self.set_dim)
        dim_row.addWidget(self.seg_dim)
        dim_row.addStretch(1)
        lay.addLayout(dim_row)
        self.lbl_err = label("", 11, "red", wrap=True)
        lay.addWidget(self.lbl_err)

        self.card_op = Card()
        left.addWidget(self.card_op)
        olay = self.card_op.lay
        olay.addWidget(label("Rechnung", 15, weight=QFont.DemiBold))
        olay.addSpacing(12)
        self.drop_op = Dropdown(list(VECTOR_OPS), value="Skalarprodukt")
        self.drop_op.picked.connect(self.set_op)
        olay.addWidget(self.drop_op)
        olay.addSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(pill("Rechnen", "primary", self.compute))
        row.addWidget(pill("Zeichnen", "secondary", self.draw, small=True))
        row.addStretch(1)
        olay.addLayout(row)
        olay.addSpacing(8)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(pill("Im Browser öffnen", "plain", self.draw_browser,
                            small=True))
        row2.addStretch(1)
        olay.addLayout(row2)
        olay.addSpacing(8)
        olay.addWidget(label("»Zeichnen« zeigt das Bild rechts im Fenster; "
                             "räumliche Bilder lassen sich dort mit der Maus "
                             "drehen. »Im Browser öffnen« nimmt Plotly — "
                             "dasselbe Bild, dafür mit Zoom und Beschriftung "
                             "beim Überfahren.", 10, "ter", wrap=True))

        self.wolfram = WolframPanel(
            self._wolfram_run,
            "Rechnet die gewählte Verknüpfung bei Wolfram|Alpha nach — "
            "Skalar-, Vektor- und Spatprodukt, Winkel, Projektion.")
        left.addWidget(self.wolfram)
        left.addStretch(1)

        output_card(self, right)
        self.render_grid()

    def _wolfram_run(self, appid: str):
        u, v, w = self.vectors()
        op, decimal = self.op, self.decimal
        k = self.scalar() if op == "scale" else None
        return lambda: run_vector_checks(appid, op, u, v, w, decimal, k)

    def render_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().setParent(None)
                        sub.widget().deleteLater()
        self.fields = []
        self.k_field = None
        names = ("A", "B", "w") if self.op == "position" else ("u", "v", "w")
        for i, name in enumerate(names):
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(label(name, 12, weight=QFont.DemiBold))
            line = []
            for c in range(self.dim):
                field = QLineEdit(self.values[i][c] if c < len(self.values[i])
                                  else "0")
                field.setObjectName("cell")
                field.setAlignment(Qt.AlignCenter)
                field.setFont(FONTS.f(13))
                field.setFixedSize(58, 34)
                field.textChanged.connect(self.read_fields)
                row.addWidget(field)
                line.append(field)
            row.addStretch(1)
            self.fields.append(line)
            self.grid.addLayout(row)
        if self.op == "scale":
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(label("k", 12, weight=QFont.DemiBold))
            self.k_field = QLineEdit(self.k)
            self.k_field.setObjectName("cell")
            self.k_field.setAlignment(Qt.AlignCenter)
            self.k_field.setFont(FONTS.f(13))
            self.k_field.setFixedSize(58, 34)
            self.k_field.textChanged.connect(self.read_fields)
            row.addWidget(self.k_field)
            row.addStretch(1)
            self.grid.addLayout(row)

    def read_fields(self):
        before = (self.values, self.k)
        self.values = [[f.text() for f in row] for row in self.fields]
        if self.k_field is not None:
            self.k = self.k_field.text()
        if before != (self.values, self.k) and getattr(self, "wolfram", None):
            self.wolfram.invalidate()

    def scalar(self):
        return parse_number(self.k)

    def set_dim(self, index: int):
        self.read_fields()
        self.dim = 2 + index
        self.values = [[row[c] if c < len(row) else "0"
                        for c in range(self.dim)] for row in self.values]
        self.render_grid()

    def set_op(self, name: str):
        self.read_fields()
        self.op = VECTOR_OPS[name]
        self.render_grid()          # Beschriftung A/B und Feld k hängen an op
        self.wolfram.invalidate()
        self.anim.refresh()

    def vectors(self):
        from lialkit.core import Vector as CoreVector
        self.read_fields()
        return [CoreVector([parse_number(v) for v in row])
                for row in self.values]

    def compute(self):
        from lialkit.vectors import (add, angle, cross, dot, position,
                                     projection, scale, spat)
        try:
            u, v, w = self.vectors()
            k = self.scalar() if self.op == "scale" else None
        except ValueError as exc:
            self.lbl_err.setText(str(exc))
            self.view.show_error(str(exc))
            return
        self.lbl_err.setText("")
        try:
            if self.op == "add":
                result = add(u, v, self.decimal)
            elif self.op == "position":
                result = position(u, v, self.decimal)
            elif self.op == "scale":
                result = scale(u, k, self.decimal)
            elif self.op == "dot":
                result = dot(u, v, self.decimal)
            elif self.op == "cross":
                result = cross(u, v, self.decimal)
            elif self.op == "spat":
                result = spat(u, v, w, self.decimal)
            elif self.op == "angle":
                result = angle(u, v)
            else:
                result = projection(u, v, self.decimal)
        except ValueError as exc:
            self.view.show_error(str(exc))
            show_view(self, 0)
            return
        self.view.show_result(result, self.decimal)
        show_view(self, 0)

    # -- Animation --------------------------------------------------------

    def anim_key(self):
        self.read_fields()
        return (tuple(tuple(row) for row in self.values), self.op, self.dim,
                self.k)

    def build_job(self, _scene: str, quality: str):
        """
        Die Szene ergibt sich hier aus der Rechnung, nicht aus einer Auswahl.

        Beim Vektorprodukt und bei der Projektion gehört das Ergebnis mit
        ins Bild — erst dann zeigt die Animation, worum es geht.
        """
        from lialkit.vectors import angle, dot, spat

        if self.op in ("add", "position", "scale"):
            return ("Für diese Rechnung gibt es keine Animation — "
                    "»Zeichnen« zeigt das Bild.")
        u, v, w = self.vectors()
        if self.op == "dot":
            note = f"u · v = {fmt(dot(u, v).value, self.decimal)}"
            return gauss_anim.job_from_vectors(
                "vec_dot", [u.values, v.values], ["u", "v"], note,
                quality=quality)
        if self.op == "cross":
            c = u.cross(v)
            note = ("u × v = (" +
                    ", ".join(fmt(x, self.decimal) for x in c.values) + ")")
            return gauss_anim.job_from_vectors(
                "vec_cross", [u.values, v.values, c.values],
                ["u", "v", "u × v"], note, quality=quality)
        if self.op == "spat":
            value = spat(u, v, w).value
            note = f"Volumen = {fmt(abs(value), self.decimal)}"
            return gauss_anim.job_from_vectors(
                "vec_spat", [u.values, v.values, w.values], ["u", "v", "w"],
                note, quality=quality)
        if self.op == "angle":
            return gauss_anim.job_from_vectors(
                "vec_angle", [u.values, v.values], ["u", "v"],
                angle(u, v).summary, quality=quality)
        if v.is_zero():
            return ("Auf den Nullvektor lässt sich nicht projizieren — er "
                    "gibt keine Richtung vor.")
        p = v.scaled(u.dot(v) / v.dot(v))
        note = ("Projektion = (" +
                ", ".join(fmt(x, self.decimal) for x in p.values) + ")")
        return gauss_anim.job_from_vectors(
            "vec_proj", [u.values, v.values, p.values],
            ["u", "v", "u auf v"], note, quality=quality)

    def figure(self, interactive: bool):
        """Baut die Zeichnung zur gerade gewählten Rechnung."""
        from lialkit import plots

        from lialkit.core import Vector as CoreVector

        u, v, w = self.vectors()
        zero = CoreVector([0] * self.dim)
        if self.op == "add":
            return plots.plot_vectors(u, v, u + v, origins=[zero, u, zero],
                                      labels=["u", "v", "u + v"],
                                      interactive=interactive)
        if self.op == "position":       # u = A, v = B
            return plots.plot_vectors(u, v, v - u, origins=[zero, zero, u],
                                      labels=["OA", "OB", "AB"],
                                      interactive=interactive)
        if self.op == "scale":
            return plots.plot_vectors(u, u.scaled(self.scalar()),
                                      labels=["u", "k · u"],
                                      interactive=interactive)
        if self.op == "spat":
            return plots.plot_spat(u, v, w, interactive=interactive)
        if self.op == "cross":
            return plots.plot_vectors(u, v, u.cross(v),
                                      labels=["u", "v", "u \u00d7 v"],
                                      interactive=interactive)
        if self.op == "projection" and not v.is_zero():
            # u auf v projiziert: (u\u00b7v / v\u00b7v) \u00b7 v
            scaled = v.scaled(u.dot(v) / v.dot(v))
            return plots.plot_vectors(u, v, scaled,
                                      labels=["u", "v", "u auf v"],
                                      interactive=interactive)
        return plots.plot_vectors(u, v, labels=["u", "v"],
                                  interactive=interactive)

    def draw(self):
        """Zeichnet in das Fenster selbst — räumlich mit der Maus drehbar."""
        try:
            u, v, w = self.vectors()
        except ValueError as exc:
            self.lbl_err.setText(str(exc))
            return
        self.lbl_err.setText("")
        if not PlotCanvas.available():
            self.plot.show_message(
                "Zum Zeichnen im Fenster fehlt Matplotlib. "
                "Mit »pip install matplotlib« nachinstallieren — "
                "oder »Im Browser öffnen« nehmen.")
            show_view(self, 1)
            return
        try:
            figure = self.figure(interactive=False)
        except Exception as exc:
            self.plot.show_message(f"Zeichnen ging nicht: {exc}")
            show_view(self, 1)
            return
        hint = ("Mit gedrückter Maustaste drehen."
                if self.dim == 3 else "")
        self.plot.show_figure(figure, hint)
        show_view(self, 1)

    def draw_browser(self):
        """Dasselbe Bild mit Plotly, dafür mit Zoom und Beschriftungen."""
        import tempfile
        import webbrowser

        try:
            self.vectors()
        except ValueError as exc:
            self.lbl_err.setText(str(exc))
            return
        self.lbl_err.setText("")
        try:
            figure = self.figure(interactive=True)
        except Exception as exc:
            self.view.show_error(f"Zeichnen ging nicht: {exc}")
            return

        path = Path(tempfile.mkdtemp(prefix="lial-plot-")) / "plot.html"
        if hasattr(figure, "write_html"):
            figure.write_html(str(path), include_plotlyjs="cdn")
        else:
            path = path.with_suffix(".png")
            figure.savefig(path, dpi=140, bbox_inches="tight")
        webbrowser.open(path.as_uri())

    def restyle(self):
        for card in (self.card_input, self.card_op, self.card_out):
            card.restyle()
        self.view._placeholder("Nach dem Themawechsel neu rechnen.")
        self.plot.restyle()
        self.anim.restyle()
        self.wolfram.restyle()


def movie_size(path: str) -> QSize:
    """
    Die Bildgrösse einer Animation, ohne sie abzuspielen.

    Beim GIF liest QMovie das erste Bild. Beim MP4 steht die Auflösung im
    Ordnernamen, den Manim anlegt — „1080p60“ —, das erspart ffprobe.
    """
    if path.lower().endswith(".mp4"):
        for part in reversed(Path(path).parts):
            match = re.fullmatch(r"(\d+)p(\d+)", part)
            if match:
                height = int(match.group(1))
                return QSize(int(round(height * 16 / 9)), height)
        return QSize(1920, 1080)
    probe = QMovie(path)
    probe.jumpToFrame(0)
    size = probe.currentImage().size()
    if size.isValid() and size.width() > 0:
        return size
    return QSize(860, 484)


class AnimationWindow(QDialog):
    """Die Animation in voller Grösse, mit Platz für die ganze Auflösung."""

    def __init__(self, parent, path: str, title: str):
        super().__init__(parent)
        self.setWindowTitle("Animation")
        self.setModal(True)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        head = QHBoxLayout()
        head.addWidget(label(title, 13, weight=QFont.DemiBold))
        head.addStretch(1)
        head.addWidget(pill("Schliessen", "plain", self.accept, small=True))
        lay.addLayout(head)

        self.player = MoviePlayer()
        lay.addWidget(self.player, 1)

        source = movie_size(path)
        screen = parent.screen().availableGeometry()
        dpr = self.devicePixelRatioF()
        # Nicht über die echte Auflösung hinaus und nicht über den Bildschirm.
        width = min(int(source.width() / dpr), int(screen.width() * 0.85), 1400)
        height = int(width * source.height() / max(1, source.width()))
        self.resize(width + 40, height + 110)
        self.player.load(path, title)


class HelpDialog(QDialog):
    """Kurze Erklärung der drei Zeilenoperationen und der beiden Formen."""

    SECTIONS = [
        ("Zeilen tauschen", "Rᵢ ↔ Rⱼ",
         "Zwei Zeilen vertauschen. Nötig, wenn an der Pivotstelle eine 0 steht "
         "und weiter unten eine Zeile mit einem Eintrag ungleich 0 wartet."),
        ("Zeile skalieren", "Rᵢ ← k · Rᵢ",
         "Alle Einträge einer Zeile mit k multiplizieren, k ≠ 0. Meist, um "
         "eine führende Zahl auf 1 zu bringen."),
        ("Vielfaches addieren", "Rᵢ ← Rᵢ + k · Rⱼ",
         "Das k-fache einer Zeile zu einer anderen addieren. Das ist der "
         "eigentliche Motor der Elimination: damit entstehen die Nullen."),
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Zeilenoperationen")
        self.setModal(True)
        self.setFixedWidth(540)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(12)
        lay.setSizeConstraint(QVBoxLayout.SetMinimumSize)

        lay.addWidget(label("Die drei Zeilenoperationen", 15,
                            weight=QFont.DemiBold))
        lay.addWidget(wrapped(
            "Keine davon ändert die Lösungsmenge des Gleichungssystems — "
            "deshalb darfst du sie beliebig anwenden.", 484))
        lay.addSpacing(4)

        for title, formula, text in self.SECTIONS:
            box = QFrame()
            box.setObjectName("helpbox")
            box.setStyleSheet(
                f"#helpbox {{ background: {T['fill2']}; border-radius: 14px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(16, 13, 16, 13)
            bl.setSpacing(4)
            row = QHBoxLayout()
            row.addWidget(label(title, 12, weight=QFont.DemiBold))
            row.addStretch(1)
            row.addWidget(label(formula, 11, "blue", mono=True))
            bl.addLayout(row)
            bl.addWidget(wrapped(text, 452))
            lay.addWidget(box)

        lay.addSpacing(6)
        lay.addWidget(label("Die beiden Zielformen", 12, weight=QFont.DemiBold))
        lay.addWidget(wrapped(
            "Zeilenstufenform: Nullzeilen stehen unten, und jede führende Zahl "
            "steht weiter rechts als die in der Zeile darüber. Daraus bekommst "
            "du die Lösung durch Rückwärtseinsetzen.", 484))
        lay.addWidget(wrapped(
            "Reduzierte Zeilenstufenform: zusätzlich ist jede führende Zahl "
            "eine 1 und in ihrer Spalte der einzige Eintrag ungleich 0. Dann "
            "steht die Lösung direkt in der letzten Spalte.", 484))

        lay.addSpacing(10)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(pill("Verstanden", "primary", self.accept, small=True))
        lay.addLayout(row)


class AppIDDialog(QDialog):
    def __init__(self, parent, current: str):
        super().__init__(parent)
        self.setWindowTitle("Wolfram|Alpha AppID")
        self.setModal(True)
        self.setFixedWidth(430)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24)
        lay.setSpacing(10)
        lay.addWidget(label("AppID hinterlegen", 15, weight=QFont.DemiBold))
        lay.addWidget(label(
            "Eine kostenlose AppID bekommst du unter developer.wolframalpha.com. "
            "Sie wird in deinem Benutzerordner gespeichert und nur für die "
            "Prüfanfragen verwendet.", 11, "sec", wrap=True))
        self.field = QLineEdit(current)
        self.field.setObjectName("cell")
        self.field.setFont(FONTS.f(12))
        self.field.setFixedHeight(38)
        self.field.setPlaceholderText("z. B. ABCDEF-GHIJKLMNOP")
        self.field.returnPressed.connect(self.accept)
        lay.addSpacing(6)
        lay.addWidget(self.field)
        lay.addSpacing(8)

        row = QHBoxLayout()
        link = pill("Seite öffnen", "plain", lambda: webbrowser.open(
            "https://developer.wolframalpha.com/access"), small=True)
        row.addWidget(link)
        row.addStretch(1)
        row.addWidget(pill("Abbrechen", "secondary", self.reject, small=True))
        row.addWidget(pill("Speichern", "primary", self.accept, small=True))
        lay.addLayout(row)

    def key(self) -> str:
        return self.field.text().strip()


# ==========================================================================
# Wolfram im Hintergrund
# ==========================================================================

class WolframWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, appid: str, original: Matrix, current: Matrix, decimal: bool):
        super().__init__()
        self.appid, self.original = appid, original
        self.current, self.decimal = current, decimal

    def run(self):
        try:
            self.finished.emit(run_wolfram_checks(
                self.appid, self.original, self.current, self.decimal))
        except (RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))


# ==========================================================================
# Animation im Hintergrund
# ==========================================================================

class AnimationWorker(QObject):
    """Rendert einen Manim-Auftrag, ohne das Fenster zu blockieren."""

    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, job, out_dir: Path):
        super().__init__()
        self.job, self.out_dir = job, out_dir

    def run(self):
        try:
            path = gauss_anim.render(
                self.job, out_dir=self.out_dir,
                progress=lambda p, t: self.progress.emit(p, t))
            self.finished.emit(str(path))
        except Exception as exc:                 # Manim wirft vielerlei
            self.failed.emit(str(exc))


class MoviePlayer(QWidget):
    """
    Spielt die gerenderte Animation ab.

    Zwei Wege, je nach Datei: ein MP4 über QtMultimedia — volle Farbtiefe und
    bis 1440p — oder ein GIF über QMovie als Rückfall. Beim GIF werden die
    Bilder selbst gesetzt statt über QLabel.setMovie: nur so lässt sich ihre
    Pixeldichte angeben, sonst zeigt ein Retina-Bildschirm es auf die
    doppelte Fläche aufgeblasen und damit unscharf.
    """

    clicked = Signal()

    #: Manim rendert immer im Breitbild; danach richtet sich die Höhe.
    ASPECT = 9 / 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self.movie: QMovie | None = None
        self.source_size = QSize()
        self.path = ""
        self.player = None
        self.audio = None
        self.aspect = self.ASPECT
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.stage = QStackedWidget()
        self.stage.setMinimumHeight(150)
        lay.addWidget(self.stage)

        self.screen = QLabel()
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setStyleSheet(
            f"background: {T['fill2']}; border-radius: 12px;")
        self.stage.addWidget(self.screen)

        self.video = None
        if HAS_VIDEO:
            self.video = QVideoWidget()
            self.video.setAspectRatioMode(Qt.KeepAspectRatio)
            self.video.setAutoFillBackground(True)
            self._paint_video_background()
            self.stage.addWidget(self.video)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.btn_play = pill("Pause", "secondary", self.toggle, small=True)
        controls.addWidget(self.btn_play)
        self.btn_restart = pill("Von vorn", "secondary", self.restart, small=True)
        controls.addWidget(self.btn_restart)
        controls.addStretch(1)
        self.lbl_info = label("", 10, "ter")
        controls.addWidget(self.lbl_info)
        lay.addLayout(controls)

    # -- Zustand ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self.movie is not None or self.player is not None

    def load(self, path: str, note: str = ""):
        self.stop()
        if path.lower().endswith(".mp4") and HAS_VIDEO:
            self._load_video(path)
        else:
            self._load_gif(path)
        self.lbl_info.setText(note)

    # -- MP4 --------------------------------------------------------------

    def _load_video(self, path: str):
        player = QMediaPlayer(self)
        # Ohne Tonausgabe meckert Qt auf manchen Rechnern; die Animation hat
        # keinen Ton, die Ausgabe bleibt darum stumm.
        self.audio = QAudioOutput(self)
        self.audio.setMuted(True)
        player.setAudioOutput(self.audio)
        player.setVideoOutput(self.video)
        player.setLoops(QMediaPlayer.Infinite)
        player.errorOccurred.connect(self._video_failed)
        player.setSource(QUrl.fromLocalFile(path))
        self.player = player
        self.path = path
        self.stage.setCurrentWidget(self.video)
        self.rescale()
        player.play()
        self.btn_play.setText("Pause")

    def _video_failed(self, _error, text: str):
        """Fällt auf das GIF neben dem Video zurück, wenn es eines gibt."""
        gif = Path(self.path).with_suffix(".gif") if self.path else None
        self._drop_player()
        if gif is not None and gif.exists():
            self._load_gif(str(gif))
            self.lbl_info.setText("Als GIF abgespielt.")
            return
        self.stage.setCurrentWidget(self.screen)
        self.screen.setText(f"Die Animation lässt sich nicht abspielen: {text}")

    def _paint_video_background(self):
        if self.video is None:
            return
        pal = self.video.palette()
        pal.setColor(self.video.backgroundRole(), QColor(T["fill2"]))
        self.video.setPalette(pal)

    # -- GIF --------------------------------------------------------------

    def _load_gif(self, path: str):
        movie = QMovie(path)
        if not movie.isValid():
            self.stage.setCurrentWidget(self.screen)
            self.screen.setText("Die Animation lässt sich nicht abspielen.")
            return
        movie.jumpToFrame(0)
        self.source_size = movie.currentImage().size()
        self.movie = movie
        self.path = path
        self.stage.setCurrentWidget(self.screen)
        movie.frameChanged.connect(self._show_frame)
        self.rescale()
        movie.start()
        self.btn_play.setText("Pause")

    def _show_frame(self):
        """Ein Bild übernehmen und ihm die Pixeldichte des Bildschirms geben."""
        if self.movie is None:
            return
        pix = self.movie.currentPixmap()
        if pix.isNull():
            return
        pix.setDevicePixelRatio(self.devicePixelRatioF())
        self.screen.setPixmap(pix)

    # -- Grösse -----------------------------------------------------------

    def rescale(self):
        if self.player is not None:
            # Die Höhe aus der Breite ableiten, sonst bleiben schwarze
            # Balken über und unter dem Bild stehen.
            width = max(220, self.width() - 4)
            self.stage.setFixedHeight(max(150, int(width * self.aspect)))
            return
        if self.movie is None or not self.source_size.isValid():
            return
        if self.source_size.width() <= 0:
            return
        dpr = self.devicePixelRatioF()
        available = max(220, self.width() - 4)
        # In echten Bildpunkten rechnen, dann in logische zurück.
        wanted = available * dpr
        scale = min(1.0, wanted / self.source_size.width())
        size = QSize(max(1, int(self.source_size.width() * scale)),
                     max(1, int(self.source_size.height() * scale)))
        self.movie.setScaledSize(size)
        self.stage.setFixedHeight(int(size.height() / dpr))
        self._show_frame()

    def mouseReleaseEvent(self, event):
        if self.loaded and self.stage.geometry().contains(
                event.position().toPoint()):
            self.clicked.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale()

    # -- Bedienung --------------------------------------------------------

    def toggle(self):
        if self.player is not None:
            running = self.player.playbackState() == QMediaPlayer.PlayingState
            self.player.pause() if running else self.player.play()
            self.btn_play.setText("Abspielen" if running else "Pause")
            return
        if self.movie is None:
            return
        running = self.movie.state() == QMovie.Running
        self.movie.setPaused(running)
        self.btn_play.setText("Abspielen" if running else "Pause")

    def restart(self):
        if self.player is not None:
            self.player.setPosition(0)
            self.player.play()
            self.btn_play.setText("Pause")
            return
        if self.movie is None:
            return
        self.movie.stop()
        self.movie.start()
        self.btn_play.setText("Pause")

    def _drop_player(self):
        if self.player is not None:
            try:
                self.player.errorOccurred.disconnect(self._video_failed)
            except (RuntimeError, TypeError):
                pass
            self.player.stop()
            self.player.setVideoOutput(None)
            self.player.deleteLater()
            self.player = None
        if self.audio is not None:
            self.audio.deleteLater()
            self.audio = None

    def stop(self):
        self._drop_player()
        if self.movie is not None:
            try:
                self.movie.frameChanged.disconnect(self._show_frame)
            except (RuntimeError, TypeError):
                pass
            self.movie.stop()
            self.movie = None
        self.path = ""
        self.screen.clear()
        self.stage.setCurrentWidget(self.screen)

    def restyle(self):
        self.screen.setStyleSheet(
            f"background: {T['fill2']}; border-radius: 12px;")
        self._paint_video_background()


# ==========================================================================
# Hauptfenster
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gauss-Elimination")
        self.resize(1240, 880)
        self.setMinimumSize(880, 560)

        self.jordan = False
        self.decimal = False
        self.editing = True
        self.running = False
        self.target: int | None = None
        self.op_kind = "add"
        self.rows_n, self.cols_n = 3, 3
        self.draft = [["2", "1", "-1", "8"], ["-3", "-1", "2", "-11"],
                      ["-2", "1", "2", "-3"]]
        self.history: list[tuple[Matrix, Operation | None, bool]] = []
        self.redo_stack: list[tuple[Matrix, Operation | None, bool]] = []
        self.matrix: Matrix | None = None
        self.wolfram: WolframReport | None = None
        self.wolfram_status: tuple[str, str] | None = None
        self.worker_thread = None
        self.op_i, self.op_j = 0, 1
        self.op_k = "1"
        self.anim_scene = "full"
        self.anim_ready = False
        self.anim_thread = None
        self.anim_dir: Path | None = None
        self.anim_theme_dark: bool | None = None
        self.anim_stamp: tuple[str, int, str] | None = None
        self.anim_quality = "Scharf"
        self.anim_path = ""

        self._build()
        self.toasts = ToastArea(self.centralWidget())
        self.apply_theme()
        self.render()

    # -- Aufbau -----------------------------------------------------------

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Navigationsleiste
        self.nav = QWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedHeight(56)
        nav_lay = QHBoxLayout(self.nav)
        nav_lay.setContentsMargins(32, 0, 32, 0)
        self.brand = label("lialkit", 13, weight=QFont.DemiBold)
        nav_lay.addWidget(self.brand)
        nav_lay.addSpacing(18)
        self.seg_page = Segmented(["Gauss", "Verfahren", "Vektoren"], 0,
                                  seg_width=84)
        self.seg_page.changed.connect(self.set_page)
        nav_lay.addWidget(self.seg_page)
        nav_lay.addStretch(1)
        self.seg_mode = Segmented(["Gauss", "Gauss-Jordan"], 0)
        self.seg_mode.disabled_hint = ("Gilt nur für die Gauss-Seite.")
        self.seg_mode.changed.connect(self.set_mode)
        nav_lay.addWidget(self.seg_mode)
        nav_lay.addSpacing(10)
        self.seg_theme = Segmented(["Hell", "Dunkel"], 0)
        self.seg_theme.changed.connect(self.set_theme)
        nav_lay.addWidget(self.seg_theme)
        nav_lay.addSpacing(6)
        nav_lay.addWidget(pill("Hilfe", "plain", self.show_help, small=True))
        outer.addWidget(self.nav)

        self.hairline = QFrame()
        self.hairline.setFixedHeight(1)
        outer.addWidget(self.hairline)

        # Drei Seiten im Stapel; die Leiste oben schaltet um.
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stack.addWidget(self.scroll)

        page = QWidget()
        page.setObjectName("page")
        self.scroll.setWidget(page)
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(32, 30, 32, 34)
        page_lay.setSpacing(0)

        self.hero = label("Lineare Gleichungssysteme, Zeile für Zeile.", 30,
                          weight=QFont.Bold, display=True, wrap=True)
        page_lay.addWidget(self.hero)
        self.lede = label("Matrix eingeben und selbst umformen – oder jeden "
                          "Schritt vorschlagen lassen.", 14, "sec", display=True,
                          wrap=True)
        page_lay.addWidget(self.lede)
        page_lay.addSpacing(24)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        page_lay.addLayout(columns, 1)

        left = QVBoxLayout()
        left.setSpacing(20)
        right = QVBoxLayout()
        right.setSpacing(20)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)

        self._build_matrix_card(left)
        self._build_result_card(left)
        left.addStretch(1)
        self._build_ops_card(right)
        self._build_anim_card(right)
        self._build_wolfram_card(right)
        self._build_history_card(right)
        right.addStretch(1)

        self.page_methods = MethodsPage(self)
        self.page_vectors = VectorsPage(self)
        for page in (self.page_methods, self.page_vectors):
            holder = QScrollArea()
            holder.setObjectName("page")
            holder.setWidgetResizable(True)
            holder.setFrameShape(QFrame.NoFrame)
            holder.setWidget(page)
            self.stack.addWidget(holder)

    def _build_matrix_card(self, parent_layout):
        self.card_matrix = Card()
        parent_layout.addWidget(self.card_matrix)
        lay = self.card_matrix.lay

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(3)
        stage_row = QHBoxLayout()
        stage_row.setSpacing(10)
        self.lbl_stage = label("", 15, weight=QFont.DemiBold)
        stage_row.addWidget(self.lbl_stage)
        self.badge_form = Badge("", "amber", "amber_soft")
        stage_row.addWidget(self.badge_form)
        stage_row.addStretch(1)
        titles.addLayout(stage_row)
        self.lbl_sub = label("", 11, "sec", wrap=True)
        titles.addWidget(self.lbl_sub)
        head.addLayout(titles, 1)
        self.seg_fmt = Segmented(["Brüche", "Dezimal"], 0)
        self.seg_fmt.changed.connect(self.set_format)
        head.addWidget(self.seg_fmt, 0, Qt.AlignTop)
        lay.addLayout(head)

        lay.addSpacing(18)
        self.view = MatrixView()
        self.view.row_clicked.connect(self.pick_row)
        self.view.submitted.connect(self.start)
        self.view.cell_edited.connect(self.edit_cell)
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(self.view)
        holder.addStretch(1)
        lay.addLayout(holder)
        lay.addSpacing(18)

        self.hint = QFrame()
        self.hint.setObjectName("hint")
        hint_lay = QHBoxLayout(self.hint)
        hint_lay.setContentsMargins(16, 13, 14, 13)
        hint_texts = QVBoxLayout()
        hint_texts.setSpacing(2)
        self.hint_op = label("", 12, weight=QFont.DemiBold, wrap=True)
        self.hint_why = label("", 11, "sec", wrap=True)
        hint_texts.addWidget(self.hint_op)
        hint_texts.addWidget(self.hint_why)
        hint_lay.addLayout(hint_texts, 1)
        self.btn_hint = pill("Übernehmen", "plain", self.use_hint, small=True)
        hint_lay.addWidget(self.btn_hint, 0, Qt.AlignVCenter)
        lay.addWidget(self.hint)
        lay.addSpacing(18)

        self.bar = QVBoxLayout()
        self.bar.setSpacing(12)
        lay.addLayout(self.bar)
        self.lbl_err = label("", 11, "red", wrap=True)
        lay.addWidget(self.lbl_err)
        self._build_bar()

    def _build_bar(self):
        self._clear_layout(self.bar)
        if self.editing:
            row1 = QHBoxLayout()
            row1.addWidget(label("Gleichungen", 11, "sec"))
            self.step_rows = Stepper(self.rows_n)
            self.step_rows.stepped.connect(lambda d: self.resize_matrix(d, 0))
            row1.addWidget(self.step_rows)
            row1.addSpacing(18)
            row1.addWidget(label("Unbekannte", 11, "sec"))
            self.step_cols = Stepper(self.cols_n)
            self.step_cols.stepped.connect(lambda d: self.resize_matrix(0, d))
            row1.addWidget(self.step_cols)
            row1.addStretch(1)
            self.bar.addLayout(row1)

            row2 = QHBoxLayout()
            drop = Dropdown(list(PRESETS) + ["Zufällig", "Leeren"],
                            placeholder="Beispiel laden")
            drop.picked.connect(self.load_preset)
            row2.addWidget(drop)
            row2.addStretch(1)
            row2.addWidget(pill("Umformen starten", "primary", self.start))
            self.bar.addLayout(row2)
        else:
            row = QHBoxLayout()
            row.setSpacing(8)
            self.btn_next = pill("Nächster Schritt", "primary", self.step_once)
            self.btn_all = pill("Alles lösen", "secondary", self.solve_all)
            self.btn_walk = pill("Rechenweg", "secondary", self.show_walkthrough)
            for b in (self.btn_next, self.btn_all, self.btn_walk):
                row.addWidget(b)
            row.addStretch(1)
            self.bar.addLayout(row)

            row2 = QHBoxLayout()
            row2.setSpacing(8)
            self.btn_undo = pill("Rückgängig", "secondary", self.undo, small=True)
            self.btn_redo = pill("Wiederholen", "secondary", self.redo, small=True)
            self.btn_reset = pill("Zurücksetzen", "secondary", self.reset_steps,
                                  small=True)
            self.btn_edit = pill("Matrix bearbeiten", "plain", self.back_to_edit,
                                 small=True)
            for b in (self.btn_undo, self.btn_redo, self.btn_reset):
                row2.addWidget(b)
            row2.addStretch(1)
            row2.addWidget(self.btn_edit)
            self.bar.addLayout(row2)

    def _clear_layout(self, lay):
        """Entfernt alle Kinder sofort — deleteLater allein lässt sie sichtbar."""
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
                item.layout().deleteLater()

    def _build_ops_card(self, parent_layout):
        self.card_ops = Card()
        parent_layout.addWidget(self.card_ops)
        lay = self.card_ops.lay
        lay.addWidget(label("Zeilenoperation", 15, weight=QFont.DemiBold))
        lay.addWidget(label("Ein Klick auf eine Zeile wählt sie als Zielzeile.",
                            11, "sec", wrap=True))
        lay.addSpacing(14)
        self.seg_op = Segmented(["Tauschen", "Skalieren", "Addieren"], 2,
                                seg_width=92)
        self.seg_op.changed.connect(self.set_op_kind)
        lay.addWidget(self.seg_op)
        lay.addSpacing(16)
        self.op_form = QHBoxLayout()
        self.op_form.setSpacing(6)
        lay.addLayout(self.op_form)
        lay.addSpacing(14)
        self.preview = QLabel("")
        self.preview.setObjectName("preview")
        self.preview.setFont(FONTS.f(12, QFont.DemiBold))
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        lay.addSpacing(14)
        self.btn_apply = pill("Anwenden", "primary", self.apply_manual)
        lay.addWidget(self.btn_apply)
        self.lbl_operr = label("", 11, "red", wrap=True)
        lay.addWidget(self.lbl_operr)
        self._build_op_form()

    def _build_op_form(self):
        self._clear_layout(self.op_form)
        rows = [f"R{i + 1}" for i in range(self.rows_n)]
        self.op_i = min(self.op_i, self.rows_n - 1)
        self.op_j = min(self.op_j, self.rows_n - 1)

        def rowpick(which: str):
            value = f"R{(self.op_i if which == 'i' else self.op_j) + 1}"
            d = Dropdown(rows, value=value)
            d.setFixedWidth(78)
            d.picked.connect(lambda v, w=which: self._set_row(w, v))
            return d

        def factor():
            field = QLineEdit(self.op_k)
            field.setObjectName("cell")
            field.setAlignment(Qt.AlignCenter)
            field.setFont(FONTS.f(12))
            field.setFixedSize(66, 30)
            field.textChanged.connect(self._set_factor)
            field.returnPressed.connect(self.apply_manual)
            return field

        if self.op_kind == "swap":
            widgets = [rowpick("i"), label("↔", 12, "sec"), rowpick("j")]
        elif self.op_kind == "scale":
            widgets = [rowpick("i"), label("←", 12, "sec"), factor(),
                       label(f"· R{self.op_i + 1}", 12, "sec")]
        else:
            widgets = [rowpick("i"), label(f"← R{self.op_i + 1} +", 12, "sec"),
                       factor(), label("·", 12, "sec"), rowpick("j")]
        for w in widgets:
            self.op_form.addWidget(w)
        self.op_form.addStretch(1)

    def _build_anim_card(self, parent_layout):
        self.card_anim = Card()
        parent_layout.addWidget(self.card_anim)
        lay = self.card_anim.lay
        lay.addWidget(label("Animation", 15, weight=QFont.DemiBold))
        self.anim_sub = label("", 11, "sec", wrap=True)
        lay.addWidget(self.anim_sub)

        ok, why = manim_state()
        self.anim_ready = ok
        if not ok:
            # Die Karte bleibt stehen: Wer die Animation sucht, soll lesen
            # können, warum sie fehlt und was dagegen hilft.
            self.anim_sub.setText(
                f"{why}\n\nSo kommt sie dazu:\n"
                + (gauss_anim.INSTALL_HINT if gauss_anim else
                   "pip install manim"))
            return

        self.anim_sub.setText("Zeigt die Umformung als kurzen Film, gerendert "
                              "mit Manim.")
        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        start = next(k for k, v in ANIM_SCENES.items() if v == self.anim_scene)
        self.anim_pick = Dropdown(list(ANIM_SCENES), value=start)
        self.anim_pick.setFixedWidth(162)
        self.anim_pick.picked.connect(self._set_anim_scene)
        row.addWidget(self.anim_pick)
        self.anim_quality_pick = Dropdown(list(ANIM_QUALITY),
                                          value=self.anim_quality)
        self.anim_quality_pick.setFixedWidth(118)
        self.anim_quality_pick.picked.connect(self._set_anim_quality)
        row.addWidget(self.anim_quality_pick)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_anim = pill("Rendern", "primary", self.render_animation)
        row2.addWidget(self.btn_anim)
        self.btn_anim_big = pill("Gross anzeigen", "secondary",
                                 self.show_animation_big, small=True)
        self.btn_anim_big.hide()
        row2.addWidget(self.btn_anim_big)
        row2.addStretch(1)
        lay.addSpacing(8)
        lay.addLayout(row2)

        self.anim_hint = label("", 10, "ter", wrap=True)
        lay.addWidget(self.anim_hint)

        self.anim_orb = ThinkingOrb(64)
        self.anim_orb.set_colors([T['blue'], T['text'], T['sec']])
        self.anim_orb.hide()
        lay.addSpacing(10)
        lay.addWidget(self.anim_orb, alignment=Qt.AlignLeft)
        self.anim_status = label("", 11, "sec", wrap=True)
        lay.addWidget(self.anim_status)

        self.player = MoviePlayer()
        self.player.hide()
        self.player.clicked.connect(self.show_animation_big)
        lay.addSpacing(12)
        lay.addWidget(self.player)

    def _set_anim_scene(self, name: str):
        self.anim_scene = ANIM_SCENES[name]
        self._update_anim_card()

    def _set_anim_quality(self, name: str):
        self.anim_quality = name
        self._update_anim_card()

    def _build_wolfram_card(self, parent_layout):
        self.card_wolfram = Card()
        parent_layout.addWidget(self.card_wolfram)
        lay = self.card_wolfram.lay
        head = QHBoxLayout()
        head.addWidget(label("Prüfung mit Wolfram|Alpha", 15, weight=QFont.DemiBold))
        head.addStretch(1)
        lay.addLayout(head)
        lay.addWidget(label(
            "Vergleicht Stufenform, Rang, Determinante und Lösungsmenge mit "
            "Wolfram|Alpha.", 11, "sec", wrap=True))
        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_wolfram = pill("Jetzt prüfen", "primary", self.check_wolfram)
        row.addWidget(self.btn_wolfram)
        self.btn_wolfram_key = pill("AppID", "secondary", self.ask_appid, small=True)
        row.addWidget(self.btn_wolfram_key)
        row.addStretch(1)
        lay.addLayout(row)
        self.wolfram_body = QVBoxLayout()
        self.wolfram_body.setSpacing(0)
        lay.addSpacing(6)
        lay.addLayout(self.wolfram_body)

    def _build_history_card(self, parent_layout):
        self.card_hist = Card()
        parent_layout.addWidget(self.card_hist)
        lay = self.card_hist.lay
        lay.addWidget(label("Verlauf", 15, weight=QFont.DemiBold))
        lay.addWidget(label("Ein Klick springt zu einem Schritt zurück.", 11, "sec"))
        lay.addSpacing(12)
        self.hist_scroll = QScrollArea()
        self.hist_scroll.setWidgetResizable(True)
        self.hist_scroll.setFrameShape(QFrame.NoFrame)
        self.hist_scroll.setMinimumHeight(150)
        self.hist_scroll.setMaximumHeight(280)
        self.hist_host = QWidget()
        self.hist_host.setObjectName("histhost")
        self.hist_lay = QVBoxLayout(self.hist_host)
        self.hist_lay.setContentsMargins(6, 6, 6, 6)
        self.hist_lay.setSpacing(2)
        self.hist_scroll.setWidget(self.hist_host)
        lay.addWidget(self.hist_scroll)
        lay.addSpacing(10)
        lay.addWidget(label("Leertaste: nächster Schritt · ⌘Z: zurück", 10, "ter"))

    def _build_result_card(self, parent_layout):
        self.card_result = Card()
        parent_layout.addWidget(self.card_result)
        self.result_lay = self.card_result.lay

    # -- Aussehen ---------------------------------------------------------

    def apply_theme(self):
        sheet = f"""
        QWidget {{ color: {T['text']}; }}
        QMainWindow, #page, QStackedWidget {{ background: {T['bg']}; }}
        QScrollArea {{ background: transparent; }}
        #nav {{ background: {T['nav']}; }}
        #card {{ background: {T['card']}; border-radius: 12px;
                 border: 1px solid {T['line']}; }}
        #hint, #preview {{ background: {T['fill2']}; border-radius: 14px; }}
        #preview {{ padding: 13px 15px; color: {T['text']}; }}
        #histhost {{ background: {T['fill2']}; border-radius: 14px; }}
        #histrow {{ background: transparent; border-radius: 10px; }}
        #histrow:hover {{ background: {T['fill']}; }}
        #histrow_on {{ background: {T['blue_soft']}; border-radius: 10px; }}
        QPushButton#primary {{ background: {T['text']}; color: {T['bg']};
            border: none; border-radius: 6px; padding: 0 22px; }}
        QPushButton#primary:hover {{ background: {T['sec']}; }}
        QPushButton#primary:disabled {{ background: {T['fill']};
            color: {T['ter']}; }}
        QPushButton#secondary {{ background: transparent; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 6px; padding: 0 18px; }}
        QPushButton#secondary:hover {{ background: {T['fill']}; }}
        QPushButton#secondary:disabled {{ color: {T['ter']}; }}
        QPushButton#plain {{ background: transparent; color: {T['blue']};
            border: none; padding: 0 10px; }}
        QPushButton#plain:hover {{ text-decoration: underline; }}
        QPushButton#plain:disabled {{ color: {T['ter']}; }}
        QPushButton#dropdown {{ background: {T['fill']}; color: {T['text']};
            border: none; border-radius: 9px; padding: 0 10px; text-align: left; }}
        QPushButton#dropdown:hover {{ background: {T['line']}; }}
        QLineEdit#cell {{ background: {T['entry']}; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 11px; padding: 0 6px;
            selection-background-color: {T['blue']}; }}
        QLineEdit#cell:focus {{ border: 2px solid {T['blue']}; }}
        QMenu {{ background: {T['card']}; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 10px; padding: 5px; }}
        QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
        QMenu::item:selected {{ background: {T['blue']}; color: #ffffff; }}
        QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {T['line']};
            border-radius: 5px; min-height: 40px; }}
        QScrollBar::handle:vertical:hover {{ background: {T['ter']}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {T['line']};
            border-radius: 5px; min-width: 40px; }}
        QDialog {{ background: {T['card']}; }}
        QProgressBar#progress {{ background: {T['fill']}; border: none;
            border-radius: 3px; }}
        QProgressBar#progress::chunk {{ background: {T['blue']};
            border-radius: 3px; }}
        """
        self.setStyleSheet(sheet)
        if hasattr(self, "anim_orb"):   # apply_theme läuft schon vor dem Bau der Karte
            self.anim_orb.set_colors([T['blue'], T['text'], T['sec']])
        self.hairline.setStyleSheet(f"background: {T['line']};")
        for card in (self.card_matrix, self.card_ops, self.card_hist,
                     self.card_result, self.card_wolfram, self.card_anim):
            card.restyle()
        if self.anim_ready:
            self.player.restyle()
        for name in ("page_methods", "page_vectors"):
            page = getattr(self, name, None)
            if page is not None:
                page.restyle()
        restyle_labels(self)

    # -- Zustandswechsel --------------------------------------------------

    def set_page(self, index: int):
        """
        Zwischen Gauss, Verfahren und Vektoren umschalten.

        Gauss gegen Gauss-Jordan betrifft nur die erste Seite; auf den
        beiden anderen bleibt die Leiste stehen, aber blass und stumm.
        """
        self.stack.setCurrentIndex(index)
        self.seg_mode.set_enabled(index == 0)

    def set_theme(self, index: int):
        T.clear()
        T.update(DARK if index else LIGHT)
        self.apply_theme()
        self.render()
        for widget in self.findChildren(QWidget):
            widget.update()

    def set_mode(self, index: int):
        self.jordan = bool(index)
        self.render()

    def set_format(self, index: int):
        self.decimal = bool(index)
        self.render()

    def set_op_kind(self, index: int):
        self.op_kind = ["swap", "scale", "add"][index]
        self._build_op_form()
        self.update_preview()

    def _set_row(self, which: str, value: str):
        index = int(value[1:]) - 1
        if which == "i":
            self.op_i = index
            self.target = index
        else:
            self.op_j = index
        self._build_op_form()
        self.render_matrix()
        self.update_preview()

    def _set_factor(self, text: str):
        self.op_k = text
        self.update_preview()

    def resize_matrix(self, dm: int, dn: int):
        if not self.editing:
            return
        self.read_fields()
        self.rows_n = max(2, min(8, self.rows_n + dm))
        self.cols_n = max(2, min(8, self.cols_n + dn))
        new = []
        for i in range(self.rows_n):
            old = self.draft[i] if i < len(self.draft) else None
            row = [old[c] if old and c < len(old) - 1 else "0"
                   for c in range(self.cols_n)]
            row.append(old[-1] if old else "0")
            new.append(row)
        self.draft = new
        self.step_rows.set_value(self.rows_n)
        self.step_cols.set_value(self.cols_n)
        self._build_op_form()
        self.render()

    def load_preset(self, name: str):
        if not self.editing:
            return
        if name == "Zufällig":
            import random
            self.draft = [[str(random.randint(-6, 6))
                           for _ in range(self.cols_n + 1)]
                          for _ in range(self.rows_n)]
        elif name == "Leeren":
            self.draft = [["0"] * (self.cols_n + 1) for _ in range(self.rows_n)]
        else:
            rows = PRESETS[name]
            self.rows_n, self.cols_n = len(rows), len(rows[0]) - 1
            self.draft = [[str(v) for v in row] for row in rows]
            self.step_rows.set_value(self.rows_n)
            self.step_cols.set_value(self.cols_n)
            self._build_op_form()
        self.lbl_err.setText("")
        self.render()
        self.toasts.show(f"Geladen: {name}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toasts"):
            self.toasts.reposition()

    def read_fields(self):
        if self.editing and self.view.fields:
            self.draft = self.view.read()

    def start(self):
        self.read_fields()
        rows, bad = [], False
        for r in self.draft:
            row = []
            for s in r:
                try:
                    row.append(parse_number(s))
                except ValueError:
                    bad = True
                    row.append(Fraction(0))
            rows.append(row)
        if bad:
            self.lbl_err.setText("Einige Felder enthalten keine gültige Zahl. "
                                 "Erlaubt sind z. B. 3, −2, 0.5 oder 3/4.")
            self.toasts.show("Einige Felder sind keine gültige Zahl.", "bad")
            return
        self.lbl_err.setText("")
        self.matrix = Matrix(rows)
        self.history = [(self.matrix, None, False)]
        self.redo_stack.clear()
        self.editing = False
        self.target = None
        self.wolfram = None
        self.wolfram_status = None
        self._clear_animation()
        self._build_bar()
        self.render()

    def _clear_animation(self):
        """Verwirft eine gezeigte Animation — ihr Rechenweg gilt nicht mehr."""
        if not self.anim_ready:
            return
        self.player.stop()
        self.player.hide()
        self.btn_anim_big.hide()
        self.anim_path = ""
        self.anim_stamp = None
        self.anim_theme_dark = None
        self.anim_status.setText("")

    def back_to_edit(self):
        if self.editing or not self.history:
            return
        self.draft = [[str(v) for v in row] for row in self.history[0][0].rows]
        self.editing = True
        self.target = None
        self.wolfram = None
        self.wolfram_status = None
        self._clear_animation()
        self._build_bar()
        self.render()

    # -- Schritte ---------------------------------------------------------

    def commit(self, op: Operation, auto: bool):
        try:
            self.matrix = self.matrix.apply(op)
        except ValueError as exc:
            self.lbl_operr.setText(str(exc))
            return
        self.history.append((self.matrix, op, auto))
        self.redo_stack.clear()
        self.lbl_operr.setText("")
        self.wolfram = None
        self.wolfram_status = None
        # Die Animation bleibt stehen und wird als veralteter Stand markiert —
        # sie wegzuwerfen hiesse, bei jedem Schritt die Rechenzeit zu verlieren.
        self.render()
        self.view.pulse([op.i, op.j] if op.kind == "swap" else [op.i])
        if not auto and not self.running:
            self.toasts.show(str(op), "ok")

    def step_once(self):
        if self.editing or self.running:
            return
        op = self.matrix.next_step(self.jordan)
        if op:
            self.commit(op, True)

    def solve_all(self):
        if self.editing or self.running:
            return
        self.running = True
        self.update_buttons()

        def tick():
            op = self.matrix.next_step(self.jordan) if self.matrix else None
            if op is None:
                self.running = False
                self.render()
                return
            self.commit(op, True)
            QTimer.singleShot(440, tick)

        tick()

    def undo(self):
        if self.editing or self.running or len(self.history) < 2:
            return
        self.redo_stack.append(self.history.pop())
        self.matrix = self.history[-1][0]
        self.wolfram = None
        self.wolfram_status = None
        self.render()
        self.toasts.show("Schritt zurückgenommen.")

    def redo(self):
        """Nimmt einen zurückgenommenen Schritt wieder auf."""
        if self.editing or self.running or not self.redo_stack:
            return
        entry = self.redo_stack.pop()
        self.history.append(entry)
        self.matrix = entry[0]
        self.wolfram = None
        self.wolfram_status = None
        self.render()
        self.toasts.show(f"Wiederholt: {entry[1]}" if entry[1] else "Wiederholt.")

    def reset_steps(self):
        """Zurück zur Ausgangsmatrix, ohne den Umformmodus zu verlassen."""
        if self.editing or self.running or len(self.history) < 2:
            return
        self.redo_stack = list(reversed(self.history[1:]))
        self.history = self.history[:1]
        self.matrix = self.history[0][0]
        self.wolfram = None
        self.wolfram_status = None
        self.render()
        self.toasts.show("Zurück zur Ausgangsmatrix.")

    def jump_to(self, index: int):
        if self.editing or self.running or index >= len(self.history):
            return
        self.redo_stack = list(reversed(self.history[index + 1:]))
        self.history = self.history[:index + 1]
        self.matrix = self.history[-1][0]
        self.wolfram = None
        self.wolfram_status = None
        self.render()

    def edit_cell(self, r: int, c: int, text: str):
        """Einzelne Zelle von Hand überschreiben, als eigener Schritt."""
        if self.editing or self.running:
            return
        try:
            value = parse_number(text)
        except ValueError:
            self.toasts.show(f"'{text}' ist keine gültige Zahl.", "bad")
            return
        if value == self.matrix.rows[r][c]:
            return
        rows = [row[:] for row in self.matrix.rows]
        rows[r][c] = value
        name = "b" if c == self.matrix.n else f"x{c + 1}"
        self.matrix = Matrix(rows)
        note = Operation(
            "edit", r, text=f"R{r + 1}, {name} = {fmt(value, self.decimal)}",
            reason="Von Hand gesetzt.")
        self.history.append((self.matrix, note, False))
        self.redo_stack.clear()
        self.wolfram = None
        self.wolfram_status = None
        self.render()
        self.view.pulse([r])
        self.toasts.show(f"R{r + 1}, {name} = {fmt(value, self.decimal)}", "ok")

    def show_walkthrough(self):
        if self.editing or self.matrix is None:
            return
        WalkthroughDialog(self, self.matrix, self.decimal).exec()

    def show_help(self):
        HelpDialog(self).exec()

    def build_op(self) -> tuple[Operation | None, str]:
        if self.op_kind == "swap":
            if self.op_i == self.op_j:
                return None, "Wähle zwei verschiedene Zeilen."
            return Operation("swap", self.op_i, j=self.op_j), ""
        try:
            k = parse_number(self.op_k)
        except ValueError:
            return None, "Der Faktor ist keine gültige Zahl."
        if k == 0:
            return None, ("Multiplizieren mit 0 ist keine erlaubte Umformung."
                          if self.op_kind == "scale"
                          else "Mit dem Faktor 0 ändert sich nichts.")
        if self.op_kind == "scale":
            return Operation("scale", self.op_i, k=k), ""
        if self.op_i == self.op_j:
            return None, "Wähle zwei verschiedene Zeilen."
        return Operation("add", self.op_i, j=self.op_j, k=k), ""

    def apply_manual(self):
        if self.editing or self.running:
            return
        op, err = self.build_op()
        if op is None:
            self.lbl_operr.setText(err)
            self.toasts.show(err, "bad")
            return
        self.commit(op, False)

    def use_hint(self):
        op = self.matrix.next_step(self.jordan) if self.matrix else None
        if not op:
            return
        self.op_kind = op.kind
        self.seg_op.set_value(["swap", "scale", "add"].index(op.kind))
        self.op_i = op.i
        if op.j is not None:
            self.op_j = op.j
        if op.k is not None:
            self.op_k = str(op.k)
        self.target = op.i
        self._build_op_form()
        self.render()

    def pick_row(self, index: int):
        if self.editing:
            return
        self.target = index
        self.op_i = index
        if self.op_j == index:
            self.op_j = (index + 1) % self.rows_n
        self._build_op_form()
        self.render_matrix()
        self.update_preview()

    def keyPressEvent(self, event):
        if isinstance(self.focusWidget(), QLineEdit):
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Space:
            self.step_once()
        elif event.key() == Qt.Key_Z and event.modifiers() & (
                Qt.ControlModifier | Qt.MetaModifier):
            self.undo()
        else:
            super().keyPressEvent(event)

    # -- Wolfram ----------------------------------------------------------

    def ask_appid(self):
        dialog = AppIDDialog(self, load_appid())
        if dialog.exec() == QDialog.Accepted and dialog.key():
            try:
                save_appid(dialog.key())
            except OSError as exc:
                self.wolfram_status = ("bad", f"AppID nicht speicherbar: {exc}")
                self.render_wolfram()
                return
            self.check_wolfram()

    def check_wolfram(self):
        if self.editing or self.running:
            self.wolfram_status = ("info", "Starte zuerst das Umformen.")
            self.render_wolfram()
            return
        appid = load_appid()
        if not appid:
            self.ask_appid()
            return
        self.wolfram = None
        self.wolfram_status = ("busy", "Frage Wolfram|Alpha …")
        self.render_wolfram()

        from PySide6.QtCore import QThread
        self.worker = WolframWorker(appid, self.history[0][0], self.matrix,
                                    self.decimal)
        self.worker_thread = QThread(self)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._wolfram_done)
        self.worker.failed.connect(self._wolfram_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _wolfram_done(self, report: WolframReport):
        self.wolfram = report
        self.wolfram_status = None
        self.render_wolfram()

    def _wolfram_failed(self, message: str):
        self.wolfram = None
        self.wolfram_status = ("bad", message)
        self.render_wolfram()

    # -- Animation --------------------------------------------------------

    def _anim_theme(self) -> "gauss_anim.Theme":
        """Die Farben der Animation folgen dem Thema der App."""
        return gauss_anim.Theme(
            background=T["card"], text=T["text"], muted=T["sec"],
            accent=T["blue"], highlight=T["flash"], pivot=T["blue_soft"],
            good=T["green"])

    def _update_anim_card(self):
        if not self.anim_ready:
            return
        steps = max(0, len(self.history) - 1)
        busy = self.anim_thread is not None
        if self.editing:
            self.anim_hint.setText("Starte zuerst das Umformen.")
            self.btn_anim.setEnabled(False)
            return
        if self.anim_scene == "step" and steps == 0:
            self.anim_hint.setText("Für den Einzelschritt braucht es "
                                   "mindestens eine Umformung.")
            self.btn_anim.setEnabled(False)
            return
        if steps == 0:
            self.anim_hint.setText("Noch keine Schritte zum Zeigen.")
            self.btn_anim.setEnabled(False)
            return
        if self.player.loaded and self.anim_stamp is not None:
            # Das GIF ist gebrannt: Stand, Qualität und Thema von damals
            # stecken darin. Je nachdem, was sich geändert hat, ein anderer Satz.
            old_scene, old_steps, old_quality = self.anim_stamp
            if (old_scene, old_steps) != (self.anim_scene, len(self.history)):
                self.anim_hint.setText("Zeigt einen früheren Stand. Neu rendern "
                                       "bringt die Animation auf den aktuellen.")
                self.btn_anim.setEnabled(not busy and not self.running)
                return
            if old_quality != self.anim_quality:
                self.anim_hint.setText(f"Gezeigt wird noch die Fassung in "
                                       f"„{old_quality}“. Neu rendern wechselt "
                                       f"die Qualität.")
                self.btn_anim.setEnabled(not busy and not self.running)
                return
            if self.anim_theme_dark is not None and self.anim_theme_dark != T["dark"]:
                self.anim_hint.setText("Mit dem anderen Thema gerendert. Ein neues "
                                       "Rendern übernimmt die aktuellen Farben.")
                self.btn_anim.setEnabled(not busy and not self.running)
                return

        # Gemessen: rund 2 s Anlauf plus 2,8 s je Schritt in „Normal“.
        shown = 1 if self.anim_scene == "step" else steps
        base = 6 if self.anim_scene == "step" else 2 + shown * 2.8
        factor = ANIM_QUALITY[self.anim_quality][3]
        self.anim_hint.setText(
            f"{shown} Schritt{'e' if shown != 1 else ''}, ungefähr "
            f"{int(base * factor)} Sekunden Rechenzeit.")
        self.btn_anim.setEnabled(not busy and not self.running)

    def render_animation(self):
        if not self.anim_ready or self.editing or self.anim_thread is not None:
            return
        history = self.history
        if self.anim_scene == "step":
            if len(history) < 2:
                return
            history = history[-2:]                 # nur der letzte Übergang

        solution = []
        if self.anim_scene == "full" and self.matrix.next_step(self.jordan) is None:
            solution = self.matrix.solution_lines(self.decimal)

        quality, width, fps, _ = ANIM_QUALITY[self.anim_quality]
        job = gauss_anim.job_from_history(
            history, scene=self.anim_scene, theme=self._anim_theme(),
            quality=quality, fmt=ANIM_FORMAT, decimal=self.decimal,
            solution=solution)
        job.gif_width, job.gif_fps = width, fps

        if self.anim_dir is None:
            self.anim_dir = Path(tempfile.mkdtemp(prefix="gauss-anim-"))
        target = self.anim_dir / self.anim_scene
        shutil.rmtree(target, ignore_errors=True)

        self.player.stop()
        self.player.hide()
        self.anim_orb.show()
        self.anim_status.setText("Manim startet …")
        self.btn_anim.setEnabled(False)
        self.anim_theme_dark = T["dark"]
        self.anim_stamp = (self.anim_scene, len(self.history), self.anim_quality)

        self.anim_worker = AnimationWorker(job, target)
        self.anim_thread = QThread(self)
        self.anim_worker.moveToThread(self.anim_thread)
        self.anim_thread.started.connect(self.anim_worker.run)
        self.anim_worker.progress.connect(self._anim_progress)
        self.anim_worker.finished.connect(self._anim_done)
        self.anim_worker.failed.connect(self._anim_failed)
        self.anim_worker.finished.connect(self.anim_thread.quit)
        self.anim_worker.failed.connect(self.anim_thread.quit)
        self.anim_thread.start()

    def _anim_progress(self, percent: int, text: str):
        self.anim_status.setText(text)

    def _anim_done(self, path: str):
        self.anim_thread = None
        self.anim_path = path
        self.anim_orb.hide()
        self.anim_status.setText("")
        name = [k for k, v in ANIM_SCENES.items() if v == self.anim_scene][0]
        self.player.show()
        self.player.load(path, name + " · Klick für gross")
        self.btn_anim_big.show()
        self._update_anim_card()

    def show_animation_big(self):
        """Zeigt die Animation in voller Grösse in einem eigenen Fenster."""
        if not self.anim_path:
            return
        name = [k for k, v in ANIM_SCENES.items() if v == self.anim_scene][0]
        AnimationWindow(self, self.anim_path, name).exec()

    def _anim_failed(self, message: str):
        self.anim_thread = None
        self.anim_path = ""
        self.btn_anim_big.hide()
        self.anim_orb.hide()
        first = message.strip().splitlines()[0] if message.strip() else "Unbekannt"
        self.anim_status.setText(f"Das Rendern ist fehlgeschlagen: {first}")
        self._update_anim_card()

    def closeEvent(self, event):
        """Temporäre Renderdateien beim Schliessen aufräumen."""
        if self.anim_ready:
            self.player.stop()
        if self.anim_dir is not None:
            shutil.rmtree(self.anim_dir, ignore_errors=True)
        for page in (self.page_methods, self.page_vectors):
            page.anim.shutdown()
        super().closeEvent(event)

    # -- Zeichnen ---------------------------------------------------------

    def render(self):
        self.render_matrix()
        self.render_hint()
        self.render_history()
        self.render_result()
        self.render_wolfram()
        self._update_anim_card()
        self.update_preview()
        self.update_buttons()
        if self.editing:
            self.lbl_stage.setText("Matrix eingeben")
            self.lbl_sub.setText("Eine Zeile pro Gleichung, die letzte Spalte ist "
                                 "die rechte Seite. Erlaubt sind z. B. 3, −2, 0.5 "
                                 "oder 3/4.")
            self.badge_form.hide()
        else:
            self.lbl_stage.setText(f"Schritt {len(self.history) - 1}")
            self.lbl_sub.setText(
                "Ziel: reduzierte Zeilenstufenform. Die Lösung steht dann in der "
                "letzten Spalte. Doppelklick auf eine Zahl ändert sie von Hand."
                if self.jordan else
                "Ziel: Zeilenstufenform, danach Rückwärtseinsetzen. Doppelklick "
                "auf eine Zahl ändert sie von Hand.")
            state = self.matrix.form_state()
            text, fg, bg = {
                "standard": ("Allgemeine Form", "amber", "amber_soft"),
                "ref": ("Zeilenstufenform", "blue", "blue_soft"),
                "rref": ("Reduzierte Zeilenstufenform", "green", "green_soft"),
            }[state]
            self.badge_form.setText(text)
            self.badge_form.fg, self.badge_form.bg = fg, bg
            self.badge_form.restyle()
            self.badge_form.show()

    def render_matrix(self):
        pivots: set[tuple[int, int]] = set()
        if not self.editing:
            nxt = self.matrix.next_step(self.jordan)
            if nxt is None:
                pivots = set(self.matrix.pivots())
            elif nxt.kind == "swap":
                pivots = {(nxt.i, self.matrix._leading_col(nxt.j))}
            elif nxt.kind == "scale":
                pivots = {(nxt.i, self.matrix._leading_col(nxt.i))}
            else:
                pivots = {(nxt.j, self.matrix._leading_col(nxt.j))}
            pivots = {p for p in pivots if p[1] >= 0}
        self.view.configure(
            editing=self.editing, matrix=self.matrix, draft=self.draft,
            rows=self.rows_n if self.editing else self.matrix.m,
            cols=self.cols_n if self.editing else self.matrix.n,
            pivots=pivots, target=self.target, decimal=self.decimal)

    def render_hint(self):
        if self.editing:
            self.hint.setStyleSheet(
                f"#hint {{ background: {T['fill2']}; border-radius: 14px; }}")
            self.hint_op.setText("So geht es weiter")
            self.hint_op.setStyleSheet(f"color: {T['text']}; background: transparent;")
            self.hint_why.setText("Trage die Zahlen ein und starte das Umformen. "
                                  "Danach schlägt die App jeden Schritt vor.")
            self.btn_hint.hide()
            return
        nxt = self.matrix.next_step(self.jordan)
        if nxt is None:
            self.hint.setStyleSheet(
                f"#hint {{ background: {T['green_soft']}; border-radius: 14px; }}")
            form = "Reduzierte Zeilenstufenform" if self.jordan else "Zeilenstufenform"
            self.hint_op.setText(f"✓  {form} erreicht")
            self.hint_op.setStyleSheet(
                f"color: {T['green']}; background: transparent;")
            self.hint_why.setText("Das Ergebnis steht unten.")
            self.btn_hint.hide()
            return
        self.hint.setStyleSheet(
            f"#hint {{ background: {T['fill2']}; border-radius: 14px; }}")
        self.hint_op.setText(f"Vorschlag:  {nxt}")
        self.hint_op.setStyleSheet(f"color: {T['text']}; background: transparent;")
        self.hint_why.setText(nxt.reason)
        self.btn_hint.show()

    def render_history(self):
        self._clear_layout(self.hist_lay)
        if self.editing or not self.history:
            self.hist_lay.addWidget(label("Noch keine Schritte.", 11, "ter"))
            self.hist_lay.addStretch(1)
            return
        for index, (_, op, auto) in enumerate(self.history):
            text = "Ausgangsmatrix" if op is None else str(op)
            row = HistoryRow(index, text, auto, index == len(self.history) - 1)
            row.picked.connect(self.jump_to)
            self.hist_lay.addWidget(row)
        self.hist_lay.addStretch(1)
        QTimer.singleShot(0, lambda: self.hist_scroll.verticalScrollBar().setValue(
            self.hist_scroll.verticalScrollBar().maximum()))

    def render_result(self):
        self._clear_layout(self.result_lay)
        if self.editing:
            self.card_result.hide()
            return
        if not self.card_result.isVisible():
            self.card_result.show()
            rise_in(self.card_result)
        lay = self.result_lay
        lay.addWidget(label("Ergebnis", 15, weight=QFont.DemiBold))

        if self.matrix.next_step(self.jordan):
            form = "reduzierter Zeilenstufenform" if self.jordan else "Zeilenstufenform"
            lay.addSpacing(6)
            lay.addWidget(label(f"Die Lösung erscheint, sobald die Matrix in "
                                f"{form} ist.", 11, "sec", wrap=True))
            return

        if not self.jordan:
            lay.addSpacing(12)
            lay.addWidget(label("Gestaffeltes System", 11, "sec"))
            lay.addSpacing(4)
            for eq in self.matrix.equation_lines(self.decimal):
                lay.addWidget(label("   " + eq, 12, mono=True))

        info = self.matrix.analyze()
        lay.addSpacing(14)
        if info["type"] == "none":
            lay.addWidget(Badge("Keine Lösung", "red", "red_soft"))
            lay.addSpacing(6)
            lay.addWidget(label(
                f"Eine Zeile ergibt 0 = {fmt(info['value'], self.decimal)}. "
                f"Das ist ein Widerspruch.", 11, "sec", wrap=True))
        else:
            if info["type"] == "unique":
                lay.addWidget(Badge("Eindeutige Lösung", "green", "green_soft"))
            else:
                params = ", ".join(f"t{i + 1}" for i in range(len(info["free"])))
                lay.addWidget(Badge("Unendlich viele Lösungen", "blue", "blue_soft"))
                lay.addSpacing(6)
                lay.addWidget(label(f"Frei wählbar: {params} ∈ ℝ",
                                    11, "sec"))
            lay.addSpacing(8)
            for line in self.matrix.solution_lines(self.decimal):
                lay.addWidget(label("   " + line, 13, mono=True))

        # Kennzahlen gehören zur Ausgangsmatrix: Zeilenumformungen ändern die
        # Determinante, der Rang bleibt gleich.
        original = self.history[0][0]
        det = original.determinant()
        parts = [f"Rang: {original.rank()}"]
        if det is not None:
            parts.insert(0, f"Determinante der Ausgangsmatrix: "
                            f"{fmt(det, self.decimal)}")
        lay.addSpacing(12)
        lay.addWidget(label("   ·   ".join(parts), 11, "ter"))

    def render_wolfram(self):
        self._clear_layout(self.wolfram_body)
        lay = self.wolfram_body
        self.btn_wolfram_key.setText("AppID ändern" if load_appid() else "AppID")

        if self.wolfram_status:
            level, text = self.wolfram_status
            color = {"busy": "sec", "bad": "red", "info": "sec",
                     "ok": "green"}[level]
            lay.addSpacing(12)
            lay.addWidget(wrapped(text, 280, 11, color))
            self.btn_wolfram.setEnabled(level != "busy")
            self.btn_wolfram.setText("Prüfe …" if level == "busy" else "Jetzt prüfen")
            return

        self.btn_wolfram.setEnabled(not self.editing and not self.running)
        self.btn_wolfram.setText("Jetzt prüfen")

        if self.wolfram is None:
            if not load_appid():
                lay.addSpacing(12)
                lay.addWidget(label("Dafür braucht es einmalig eine kostenlose "
                                    "AppID.", 11, "ter", wrap=True))
            return

        report = self.wolfram
        lay.addSpacing(14)
        tone = {"ok": ("green", "green_soft"), "bad": ("red", "red_soft"),
                "warn": ("amber", "amber_soft")}[report.verdict]
        mark = {"ok": "✓", "bad": "✕", "warn": "!"}[report.verdict]
        lay.addWidget(Badge(f"{mark}  {report.headline}", *tone))
        lay.addSpacing(10)

        for check in report.checks:
            row = QHBoxLayout()
            row.setSpacing(8)
            glyph = {"ok": "✓", "bad": "✕", "info": "•",
                     "error": "!"}[check.status]
            color = {"ok": "green", "bad": "red", "info": "sec",
                     "error": "amber"}[check.status]
            row.addWidget(label(glyph, 11, color, weight=QFont.DemiBold))
            text = check.name
            if check.theirs and check.theirs != check.ours:
                text += f":  wir {check.ours}, Wolfram {check.theirs}"
            elif check.ours:
                text += f":  {check.ours}"
            row.addWidget(label(text, 11, wrap=True), 1)
            holder = QWidget()
            holder.setLayout(row)
            lay.addWidget(holder)
            if check.note:
                note = label("     " + check.note, 10, "ter", wrap=True)
                lay.addWidget(note)

        if report.their_matrix is not None:
            lay.addSpacing(12)
            lay.addWidget(label("Wolframs reduzierte Stufenform", 11, "sec"))
            lay.addSpacing(4)
            cells = [[fmt(v, self.decimal) for v in row]
                     for row in report.their_matrix.rows]
            width = max(len(s) for row in cells for s in row)
            for row in cells:
                line = "  ".join(s.rjust(width) for s in row[:-1])
                line += "  |  " + row[-1].rjust(width)
                lay.addWidget(label("  " + line, 11, mono=True))

        if report.solution_text:
            lay.addSpacing(12)
            lay.addWidget(label("Wolframs Lösung", 11, "sec"))
            lay.addSpacing(4)
            for line in report.solution_text.splitlines()[:6]:
                lay.addWidget(label("  " + line.replace("-", MINUS), 11, mono=True,
                                    wrap=True))

        if report.url:
            lay.addSpacing(10)
            row = QHBoxLayout()
            row.addWidget(pill("Im Browser öffnen", "plain",
                               lambda: webbrowser.open(report.url), small=True))
            row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            lay.addWidget(holder)

    def update_preview(self):
        op, err = self.build_op()
        if op is None:
            self.preview.setText(err)
            self.preview.setStyleSheet(
                f"#preview {{ background: {T['fill2']}; border-radius: 14px;"
                f"padding: 13px 15px; color: {T['ter']}; }}")
        else:
            self.preview.setText(str(op))
            self.preview.setStyleSheet(
                f"#preview {{ background: {T['fill2']}; border-radius: 14px;"
                f"padding: 13px 15px; color: {T['text']}; }}")

    def update_buttons(self):
        self.btn_apply.setEnabled(not self.editing and not self.running)
        self.seg_op.setEnabled(not self.editing)
        if self.editing or not hasattr(self, "btn_next"):
            return
        done = self.matrix.next_step(self.jordan) is None
        self.btn_next.setEnabled(not done and not self.running)
        self.btn_all.setEnabled(not done and not self.running)
        self.btn_walk.setEnabled(not self.running)
        self.btn_undo.setEnabled(len(self.history) > 1 and not self.running)
        self.btn_redo.setEnabled(bool(self.redo_stack) and not self.running)
        self.btn_reset.setEnabled(len(self.history) > 1 and not self.running)
        self.btn_edit.setEnabled(not self.running)


def _prewarm_matplotlib():
    try:
        import matplotlib.pyplot  # noqa: F401
        from matplotlib.backends.backend_qtagg import FigureCanvas  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass            # fehlt Matplotlib, meldet »Zeichnen« das selbst


def main():
    global FONTS
    app = QApplication(sys.argv)
    app.setApplicationName("Gauss-Elimination")
    FONTS = Fonts()
    window = MainWindow()
    window.show()
    # Matplotlib im Hintergrund vorladen, damit das erste »Zeichnen« nicht
    # das Fenster anhält. Imports sind thread-sicher; die Figur selbst wird
    # weiter im GUI-Thread gebaut (ponytail: Millisekunden, kein Worker nötig).
    threading.Thread(target=_prewarm_matplotlib, daemon=True).start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
