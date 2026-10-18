#!/usr/bin/env python3
"""
Tests für grafik/anim.py: Aufträge der Szene »abbild« (ohne Render).
Start: python3 tests/tests_anim.py

Gerendert wird hier nicht — das braucht manimgx und dauert; der Auftrag, die
Szenentabelle und der Runner sind aber ohne Engine prüfbar.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # Elternordner des Pakets lialkit

from lialkit.grafik import anim  # noqa: E402
from lialkit.rechnen.abbildung import (BUCHSTABE_F_FOLIEN, drehung,  # noqa: E402
                                       homogen, scherung)

QUADRAT = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]


def test_scene_registered():
    assert anim.SCENE_NAMES["abbild"] == "AbbildScene"
    assert anim._BUILDERS["abbild"] == "_build_map_scenes"
    assert anim._BUILDERS["vec_dot"] == "_build_vector_scenes"
    assert "full" not in anim._BUILDERS                      # Rest kommt aus _build_scenes
    assert callable(anim._build_map_scenes)


def test_job_matrix_2x2():
    job = anim.job_from_abbildung(scherung(1).rows, QUADRAT, "A = (1 1 ; 0 1)")
    assert job.scene == "abbild" and job.matrix == [[1.0, 1.0], [0.0, 1.0]]
    assert job.vectors == [[float(x), float(y)] for x, y in QUADRAT]
    assert job.vector_note.startswith("A =")


def test_job_homogeneous_drops_last_row():
    H = homogen(drehung(90), (3, 1))
    job = anim.job_from_abbildung(H.rows, QUADRAT)
    assert job.matrix == [[0.0, -1.0, 3.0], [1.0, 0.0, 1.0]]


def test_job_letter_and_irrational():
    pts = list(zip(*BUCHSTABE_F_FOLIEN.rows))
    job = anim.job_from_abbildung(drehung(30).rows, pts)
    assert len(job.vectors) == 10 and abs(job.matrix[0][0] - 0.8660254) < 1e-6


def test_job_json_roundtrip():
    job = anim.job_from_abbildung(scherung(2).rows, QUADRAT)
    again = anim.AnimationJob.from_json(job.to_json())
    assert again.matrix == job.matrix and again.scene == "abbild"
    assert again.vectors == job.vectors


def test_job_rejects_bad_input():
    for bad in ([[1, 0, 0], [0, 1, 0], [0, 0, 2]],     # 3×3 ohne Zeile 0 0 1
                [[1, 2, 3]]):                          # keine Matrix der Ebene
        try:
            anim.job_from_abbildung(bad, QUADRAT)
        except ValueError:
            continue
        raise AssertionError(f"ValueError erwartet für {bad}")
    try:
        anim.job_from_abbildung([[1, 0], [0, 1]], [(0, 0, 0)])
    except ValueError:
        pass
    else:
        raise AssertionError("Punkte mit drei Koordinaten müssen abgelehnt werden")


def test_old_jobs_still_load():
    """Ältere Aufträge ohne Feld `matrix` laden weiter (Standardwert)."""
    job = anim.job_from_vectors("vec_dot", [[1, 0], [0, 1]], ["u", "v"])
    import json
    data = json.loads(job.to_json())
    data.pop("matrix")
    assert anim.AnimationJob.from_json(json.dumps(data)).matrix == []


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
