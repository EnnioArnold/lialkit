#!/usr/bin/env python3
"""
Tests für app/lialkit_qt.py (Fenster ohne Bildschirm). Start: python3 tests/tests_app.py

Läuft mit QT_QPA_PLATFORM=offscreen und braucht kein Netz. Ohne PySide6 wird
übersprungen (SKIP, Exit-Code 0).
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("MPLBACKEND", "Agg")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # Elternordner des Pakets lialkit
sys.path.insert(0, os.path.join(os.path.dirname(_here), "app"))

try:
    import PySide6  # noqa: F401
except ImportError:
    print("SKIP: PySide6 nicht installiert")
    sys.exit(0)

import lialkit_qt as Q  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

_app = QApplication.instance() or QApplication([])
if Q.FONTS is None:                                   # main() setzt das sonst
    Q.FONTS = Q.Fonts()


def test_import():
    assert hasattr(Q, "MainWindow") and issubclass(Q.MainWindow, QMainWindow)
    assert callable(Q.main)


def test_main_window():
    w = Q.MainWindow()
    try:
        assert w.windowTitle() and w.width() > 0
        assert w.rows_n == 3 and w.cols_n == 3 and w.editing
        assert w.draft[0] == ["2", "1", "-1", "8"]     # Beispiel aus __init__.py
        w.show()
        _app.processEvents()
        assert w.isVisible()
    finally:
        w.close()
        w.deleteLater()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print("FAIL", name, repr(e))
                traceback.print_exc()
    print("alle Tests bestanden" if not fails else f"{fails} Test(s) fehlgeschlagen")
    sys.exit(1 if fails else 0)
