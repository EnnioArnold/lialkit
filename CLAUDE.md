# lialkit

Lern-Kit für LIAL (HSLU, Lineare Algebra), Sprache Deutsch (Docstrings, Rechenwege, UI).
Der Repo-Ordner heisst nicht immer `lialkit`, importiert wird aber `lialkit`:
darum muss ein Ordner `lialkit` (oder Symlink darauf) im Elternordner liegen / auf
`PYTHONPATH` stehen. Tests: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python3 tests/tests_<modul>.py`.

## Aufbau
`rechnen/` Rechenkern (exakt mit Brüchen, Ergebnis-Klassen `*Result` mit `steps`/`highlights`),
`grafik/` Matplotlib-Zeichnungen (`plots*.py`, Swiss-Style: Rot Original, Schwarz Bild) und Manim-Animationen (`anim.py`),
`aufgaben/` Übungsaufgaben (`exercises.py`, Schlüssel `SW<n>.<aufgabe>`), `app/lialkit_qt.py` Fenster, `tests/` ein Skript je Modul.

## Woche hinzufügen (Checkliste)
1. Material lesen (Folien, Übungen, Lösungen, Notiz-md in `Week n/`). Mit Vorwoche per Textvergleich prüfen: manchmal ist der Stoff identisch (SW6 = SW5).
2. Rechnen: neue Funktionen in `rechnen/<thema>.py` (Name deutsch, Docstring, Rückgabe `*Result` mit `Step`s, `decimal=`-Argument, keine Breaking Changes).
3. Export: in `__init__.py` importieren, in `__all__`, in `info()` und im Modul-Docstring eintragen.
4. Grafik/Animation: `grafik/plots_*.py` (Dekorator `@_swiss`), ggf. Szene in `grafik/anim.py`.
5. Aufgaben: in `aufgaben/exercises.py` als `Exercise("SW<n>.<x>", "SW<n>", ...)` mit Daten aus dem Blatt; `loese()` um neue `method` erweitern; Lösungen nur aus `lial-sw-0n-loesungen.pdf`, Unklares als TODO.
6. App: Rechnung/Seite in `app/lialkit_qt.py` registrieren (z. B. `MAP_OPS`, Hilfedialog), bestehende Indizes nicht verschieben.
7. Doku: README (Wochenzahl, Abschnitt, Fenster, »Geprüft gegen«, Tests), Docstring-Kopf `__init__.py`.
8. Tests: neues `tests/tests_<thema>.py` mit Beispielen aus Folien/Lösungen; ganze Suite laufen lassen (alle `tests/tests_*.py`), App headless prüfen.
9. Git: kleine Commits `Add week n: <Thema>`, Branch `week-n`.
