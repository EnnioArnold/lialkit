# lialkit

Lineare Algebra mit nachvollziehbarem Rechenweg, gebaut zum Modul
I.BA_LIAL_MM.20 der HSLU. Deckt die Semesterwochen 1 bis 4 ab.

## Installieren

Nötig ist nur Python. Für die Zeichnungen und die App kommt dazu:

```bash
pip install matplotlib plotly        # Zeichnen
pip install PySide6                  # das Fenster
pip install manim                    # Animationen, freiwillig
```

Der Ordner `lialkit/` liegt neben deinen Wochen-Ordnern. Aus einem
Notebook in `Week 3/` heraus:

```python
import sys
sys.path.insert(0, '..')
import lialkit as lk
```

## Was es kann

```python
A = lk.Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = lk.Vector([8, -11, -3])

lk.gauss(A, b)                 # nur das Ergebnis
lk.gauss(A, b, weg=True)       # jeder Schritt mit Begründung
lk.gauss_jordan(A, b, weg=True)
lk.solve(A, b, method='lu')    # oder 'gauss', 'inverse'

lk.lu(A)                       # LR-Zerlegung, bei Bedarf P·A = L·R
lk.inverse(A)                  # über Gauss-Jordan
lk.det(A)                      # Verfahren nach Grösse gewählt
lk.det_sarrus(A)               # nur 3×3
lk.det_laplace(A)              # Entwicklung nach einer Zeile
lk.det_triangular(A)           # über die Dreiecksform
lk.rank(A, weg=True)

u, v, w = lk.Vector([1, 2, 3]), lk.Vector([4, 5, 6]), lk.Vector([2, -1, 4])
lk.dot(u, v)
lk.cross(u, v)
lk.spat(u, v, w)
lk.angle(u, v)
lk.projection(u, v)
lk.triangle_inequality(u, v)
lk.cauchy_schwarz(u, v)

g = lk.Line([1, 1], [2, 1])
h = lk.Line([1, 4], [1, -1])
g.intersect(h)

E = lk.Plane([1, 2, 4], u=[1, 0, 1], v=[0, 1, -2])
E.intersect_line(lk.Line([1, 1, 7], [2, 3, 4]))

lk.plot_vectors(u, v)          # 2D und 3D
lk.plot_spat(u, v, w)          # drehbar, mit Plotly
lk.plot_transformation(A2)     # Determinante als Flächenfaktor
```

Jedes Ergebnis zeigt sich im Notebook von selbst und kennt:

- `.as_decimal()` — vier Stellen, wie das Übungsblatt es verlangt
- `.as_exact()` — zurück zu Brüchen
- `.latex()` — der Rechenweg zum Einfügen in eine Abgabe
- `.show()` — ausdrücklich anzeigen

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

`python3 -m lialkit.gauss_qt` aus dem Ordner über `lialkit/`
startet die App. Drei Ansichten:

- **Gauss** — Elimination von Hand oder Schritt für Schritt, mit Animation
  und Prüfung über Wolfram|Alpha
- **Verfahren** — LR-Zerlegung, Inverse, Determinante und Rang
- **Vektoren** — Skalar-, Vektor- und Spatprodukt, Winkel, Projektion

Auf den beiden letzten Seiten steht neben dem Rechenweg eine zweite
Ansicht: **Zeichnung**. Sie zeigt das Bild im Fenster selbst, und
räumliche Bilder lassen sich dort mit gedrückter Maustaste drehen. Auf der
Vektorenseite führt »Im Browser öffnen« dieselbe Zeichnung mit Plotly vor —
mit Zoom und Beschriftung beim Überfahren. Auf der Verfahrensseite zeigt
die Zeichnung, was die Matrix mit dem Raum macht: bei 2×2 das verzerrte
Einheitsquadrat, dessen Fläche |det A| ist, bei 3×3 die drei Spalten mit
dem aufgespannten Spat.

Fenster und Notebook rechnen mit demselben Modul, darum können sie nicht
auseinanderlaufen.

## Geprüft gegen

Die Ergebnisse sind Zahl für Zahl mit `lial-sw-04-loesungen.pdf` verglichen
worden: Determinanten und Inversen der Aufgaben 1 und 2, einschliesslich
der beiden singulären Matrizen. Die LR-Zerlegungen aus SW3 sind über
`L · R = P · A` und `A · x = b` nachgerechnet.
