"""Deutscher Sprach-Spezialist: Regeln als Daten, nicht als Gewicht.

Der Skill hinter atomic_gpt.py: formt rohen Text in regelgerechtes
Deutsch. Die Wortliste liegt als Daten (data/deutsch/woerter.txt, aus
hunspell de_DE / igerman98, frei), die Regeln daneben (regeln.md).
Nichts davon sitzt in einem Modellgewicht - der Harvester kann beides
mitfuettern, die Dateien bleiben die Wahrheit.
"""

import re
from pathlib import Path

WURZEL = Path(__file__).parent.parent
WOERTER_DATEI = WURZEL / "data" / "deutsch" / "woerter.txt"
MAX_UNBEKANNT = 20

_WORT = re.compile(r"[A-Za-zÄÖÜäöüß-]+", re.UNICODE)
_SATZENDE = re.compile(r"([.!?]+)\s*$")
_SATZANFANG = re.compile(r"(^|[.!?]\s+|\n\s*)([a-zäöü])")

_woerter_cache = None


def woerter_laden(pfad=WOERTER_DATEI):
    """Laedt die Wortliste als Menge (kleingeschrieben). Fehlt sie: leere Menge."""
    global _woerter_cache
    if _woerter_cache is not None and pfad == WOERTER_DATEI:
        return _woerter_cache
    pfad = Path(pfad)
    menge = set()
    if pfad.exists():
        try:
            for zeile in pfad.read_text(encoding="utf-8",
                                        errors="replace").splitlines():
                w = zeile.strip().lower()
                if w:
                    menge.add(w)
        except OSError:
            pass
    if pfad == WOERTER_DATEI:
        _woerter_cache = menge
    return menge


def grossschreibung(text):
    """Gross am Satzanfang und nach Satzzzeichen. Sonst unangetastet."""
    if not text:
        return text
    return _SATZANFANG.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def satzzzeichen(text):
    """Setzt am Ende einen Punkt, wenn kein Satzzzeichen steht."""
    text = (text or "").rstrip()
    if not text:
        return text
    if not _SATZENDE.search(text):
        return text + "."
    return text


def woerter_pruefen(text, limit=MAX_UNBEKANNT):
    """Woerter, die nicht in der Wortliste stehen. Gibt eine Liste zurueck."""
    menge = woerter_laden()
    if not menge:
        return []
    unbekannt = []
    for w in _WORT.findall(text or ""):
        if len(w) <= 3:
            continue
        klein = w.lower()
        if klein in menge:
            continue
        if klein not in unbekannt:
            unbekannt.append(klein)
        if len(unbekannt) >= limit:
            break
    return unbekannt


def nachputzen(text):
    """Der Weg fuer atomic_gpt: Regeln anwenden, Text zurueck. Wirft nie."""
    try:
        text = (text or "").strip()
        if not text:
            return ""
        text = grossschreibung(text)
        text = satzzzeichen(text)
        return text
    except Exception:
        return text or ""


if __name__ == "__main__":
    assert grossschreibung(
        "hallo welt. das ist ein test") == "Hallo welt. Das ist ein test"
    assert grossschreibung("") == ""
    assert satzzzeichen("Hallo Welt") == "Hallo Welt."
    assert satzzzeichen("Hallo Welt.") == "Hallo Welt."
    assert satzzzeichen("Fertig!") == "Fertig!"
    assert satzzzeichen("") == ""
    assert nachputzen("  hallo welt") == "Hallo welt."
    assert nachputzen("") == ""
    unbekannt = woerter_pruefen("Hallo Welt xyzqwd")
    assert "xyzqwd" in unbekannt, unbekannt
    assert "hallo" not in unbekannt and "welt" not in unbekannt, unbekannt
    assert woerter_pruefen("") == []
    print("deutsch_sprache ok")
