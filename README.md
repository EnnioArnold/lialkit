# lialkit

Lineare Algebra mit nachvollziehbarem Rechenweg, gebaut zum Modul
I.BA_LIAL_MM.20 der HSLU. Deckt die Semesterwochen 1 bis 4 ab.

## Installieren

Nötig ist nur Python. Für die Zeichnungen und die App kommt dazu:

```bash
pip install matplotlib plotly        # Zeichnen
pip install PySide6                  # das Fenster
brew install cairo pango pkg-config ffmpeg   # Unterbau für Manim
pip install manim                    # Animationen, freiwillig
```

Der Ordner `lialkit/` liegt neben deinen Wochen-Ordnern. Aus einem
Notebook in `Week 3/` heraus:

```python
import sys
sys.path.insert(0, '..')
import lialkit as lk
```

## Aufbau

![Notebook und Fenster rechnen mit demselben Modul](docs/architektur.svg)

Notebook und Fenster teilen sich den Rechenkern, darum können ihre Ergebnisse
nicht auseinanderlaufen.

## Genauigkeit

Gerechnet wird exakt mit Brüchen. `1/3` bleibt `1/3` und wird nie zu
`0.33333`. Das ist in der linearen Algebra kein Luxus: der Unterschied
zwischen `0` und `1e-17` entscheidet darüber, ob eine Matrix singulär ist.

Wo Wurzeln auftreten — Vektorlängen, Winkel — schaltet lialkit von selbst
auf Gleitkomma um und sagt es im Rechenweg. Geht die Wurzel auf, bleibt
sie exakt: `|(3,4)| = 5`, nicht `5.0000`.

## Die Übungsaufgaben

```python
from lialkit.exercises import liste, aufgabe, loese

liste('SW3')          # welche es gibt
aufgabe('SW3.1a')     # die Angabe
loese('SW3.1a')       # durchgerechnet
```

Die Notebooks in `notebooks/` enthalten alle Aufgaben der Wochen 1 bis 4,
je eine Datei pro Woche.

**Eine Aufgabe ist bewusst offen gelassen:** SW3 Aufgabe 3 verlangt, dass
du den LR-Algorithmus selbst programmierst. Dafür gibt es ein Gerüst mit
Prüfstand:

```python
from lialkit.uebung import vorlage, pruefe_lr
print(vorlage())

def meine_lr(A):
    ...
    return L, R

pruefe_lr(meine_lr)
```

Der Prüfstand rechnet die Aufgaben vom Blatt und ein paar Fälle, die man
leicht übersieht — Einheitsmatrix, schon dreieckige Matrix, Null an der
Pivotstelle, singuläre Matrix.

## Das Fenster

Ordner von GitHub laden, entpacken, dann im Terminal:

```bash
cd lialkit-main                  # wie der Ordner bei dir heisst
pip install -r requirements.txt  # einmalig
python3 gauss_qt.py
```

Der Ordnername ist egal. Drei Ansichten:

- **Gauss** — Elimination von Hand oder Schritt für Schritt, mit Animation
  und Prüfung über Wolfram|Alpha
- **Verfahren** — LR-Zerlegung, Inverse, Determinante und Rang
- **Vektoren** — Skalar-, Vektor- und Spatprodukt, Winkel, Projektion

Auf den beiden letzten Seiten steht die Ergebniskarte in drei Ansichten:
**Rechenweg**, **Zeichnung** und **Animation**.

Die **Zeichnung** zeigt das Bild im Fenster selbst; räumliche Bilder lassen
sich dort mit gedrückter Maustaste drehen. Auf der Vektorenseite führt
»Im Browser öffnen« dieselbe Zeichnung mit Plotly vor — mit Zoom und
Beschriftung beim Überfahren. Auf der Verfahrensseite zeigt sie, was die
Matrix mit dem Raum macht: bei 2×2 das verzerrte Einheitsquadrat, dessen
Fläche |det A| ist, bei 3×3 die drei Spalten mit dem aufgespannten Spat.

Die **Animation** rendert Manim: bei den Verfahren die Umformungen von
LR-Zerlegung, Inverse und Rang, bei den Vektoren eine räumliche Szene je
Rechnung — beim Vektorprodukt fährt u × v senkrecht aus der Fläche, beim
Spatprodukt füllt sich der Körper, bei der Projektion fällt das Lot. Die
Kamera dreht sich dabei, denn erst die Bewegung zeigt, dass etwas wirklich
senkrecht steht. Die Determinante hat keine Animation: Sarrus und Laplace
rechnen ohne Zwischenmatrizen, anschaulich wird sie unter »Zeichnung«.

Gerendert wird als MP4 in drei Stufen — 720p30, 1080p60, 1440p60 — und über
QtMultimedia abgespielt. Fehlt QtMultimedia, fällt der Abspieler auf ein GIF
zurück; das kennt nur 256 Farben und sieht entsprechend aus.

Die **Prüfung mit Wolfram|Alpha** gibt es auf allen drei Seiten. Bei den
Verfahren vergleicht sie Determinante, Rang und — je nach Verfahren —
Inverse und Stufenform, bei den Vektoren die gewählte Verknüpfung. Eine
Ausnahme ist die LR-Zerlegung: Wolfram wählt die Pivots nach Betrag, die
Vorlesung nur bei einer Null an der Pivotstelle, darum kommen beide auf
verschiedene, gleichermassen richtige L und R. Die App rechnet dort
stattdessen `L · R = P · A` selbst exakt nach und sagt das im Bericht.

Der Regler Gauss / Gauss-Jordan oben rechts gilt nur für die Gauss-Seite
und ist auf den beiden anderen ausgegraut.

## Geprüft gegen

Die Ergebnisse sind Zahl für Zahl mit `lial-sw-04-loesungen.pdf` verglichen
worden: Determinanten und Inversen der Aufgaben 1 und 2, einschliesslich
der beiden singulären Matrizen. Die LR-Zerlegungen aus SW3 sind über
`L · R = P · A` und `A · x = b` nachgerechnet.
