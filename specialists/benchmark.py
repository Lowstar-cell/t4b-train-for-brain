"""Benchmark-Auswertung fuer Phase 5.

Haelt eine feste Menge von Aufgaben mit Erwartungen (JSONL) und wertet
Antworten dagegen aus. Ruft selbst kein Modell auf: die zu pruefende
Sache wird als callable uebergeben bzw. als fertiges Antwort-Dictionary.
"""

import json
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
DATENSATZ = WURZEL / "data" / "benchmark.jsonl"
MINDESTQUOTE = 0.8


def datensatz_lesen(pfad=DATENSATZ):
    """Liest die JSONL-Datei und gibt eine Liste der Aufgaben zurueck.

    Jede gueltige Zeile ist ein Woerterbuch mit "frage" (String) und
    "enthaelt" (Liste von Zeichenketten). Kaputte Zeilen werden
    uebersprungen. Fehlt die Datei oder ist sie unlesbar, kommt []
    zurueck. Diese Funktion schreibt nie.
    """
    pfad = Path(pfad)
    if not pfad.exists():
        return []
    try:
        text = pfad.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    aufgaben = []
    for zeile in text.splitlines():
        zeile = zeile.strip()
        if not zeile:
            continue
        try:
            eintrag = json.loads(zeile)
        except Exception:
            continue
        if not isinstance(eintrag, dict):
            continue
        frage = eintrag.get("frage")
        enthaelt = eintrag.get("enthaelt")
        if not isinstance(frage, str) or not frage.strip():
            continue
        if not isinstance(enthaelt, list):
            continue
        if not all(isinstance(e, str) for e in enthaelt):
            continue
        aufgaben.append({"frage": frage, "enthaelt": enthaelt})
    return aufgaben


def aufgabe_bewerten(antwort, aufgabe):
    """Zaehlt, wie viele Stuecke aus "enthaelt" in der Antwort stehen.

    Gibt (treffer, fehlende) zurueck: treffer die Zahl der gefundenen
    Stuecke, fehlende die Liste der nicht gefundenen. Gross-/Klein-
    schreibung wird nicht beachtet, Whitespace getrimmt. Leere oder
    nicht-Text-Antwort: (0, alle Stuecke).
    """
    enthaelt = aufgabe.get("enthaelt", []) if isinstance(aufgabe, dict) else []
    if not isinstance(antwort, str) or not antwort.strip():
        return 0, [str(e) for e in enthaelt]
    antwort = antwort.strip().lower()
    treffer = 0
    fehlende = []
    for stueck in enthaelt:
        if str(stueck).strip().lower() in antwort:
            treffer += 1
        else:
            fehlende.append(str(stueck))
    return treffer, fehlende


def korrektur_text(treffer, fehlende):
    """Der nette Korrektur-Satz: erst loben, was gut war, dann berichtigen."""
    if not fehlende:
        return "Sehr gut, das ist richtig."
    fehlt = ", ".join(fehlende)
    if treffer:
        return ("Das war schon sehr gut, aber nicht ganz richtig: "
                "%s waere richtig." % fehlt)
    return "Das war nicht ganz richtig: %s waere richtig." % fehlt


def aufgabe_pruefen(antwort, aufgabe):
    """Prueft eine Antwort gegen eine Aufgabe. Gibt True oder False.

    Bestanden ist der Fall, wenn ALLE Zeichenketten aus "enthaelt"
    irgendwo in der Antwort stehen. Gross-/Kleinschreibung wird nicht
    beachtet, Whitespace wird getrimmt. Leere Antwort -> False.
    """
    if not isinstance(antwort, str):
        return False
    antwort = antwort.strip().lower()
    if not antwort:
        return False
    enthaelt = aufgabe.get("enthaelt", []) if isinstance(aufgabe, dict) else []
    for stueck in enthaelt:
        if str(stueck).strip().lower() not in antwort:
            return False
    return True


def benchmark_laufen(antworten, pfad=DATENSATZ):
    """Wertet ein Antwort-Dictionary gegen den Datensatz aus.

    "antworten" ist ein Dictionary {"frage": antwort}. Verglichen wird
    nur, was in beiden vorkommt; Aufgaben ohne Antwort zaehlen als
    "fehlt". Gibt ein Ergebnis-Dictionary zurueck und wirft niemals.
    """
    pfad = Path(pfad)
    if not isinstance(antworten, dict):
        antworten = {}
    aufgaben = datensatz_lesen(pfad)
    uebersprungen = 0
    try:
        roh = pfad.read_text(encoding="utf-8", errors="replace") if pfad.exists() else ""
        zeilen = [z for z in roh.splitlines() if z.strip()]
        uebersprungen = max(0, len(zeilen) - len(aufgaben))
    except Exception:
        uebersprungen = 0
    geprueft = 0
    bestanden = 0
    fehlt = 0
    schwach = []
    for aufgabe in aufgaben:
        frage = aufgabe["frage"]
        if frage not in antworten:
            fehlt += 1
            continue
        geprueft += 1
        try:
            ok = aufgabe_pruefen(antworten[frage], aufgabe)
        except Exception:
            ok = False
        if ok:
            bestanden += 1
        else:
            if len(schwach) < 5:
                schwach.append(frage)
    quote = round(bestanden / geprueft, 2) if geprueft else 0.0
    return {
        "geprueft": geprueft,
        "bestanden": bestanden,
        "quote": quote,
        "uebersprungen": uebersprungen,
        "fehlt": fehlt,
        "bestanden_quote": quote,
        "schwach": schwach,
    }


if __name__ == "__main__":
    import tempfile

    # Fehlende Datei: [] und nichts wird angelegt
    with tempfile.TemporaryDirectory() as tmp:
        fehlt_pfad = Path(tmp) / "gibt_es_nicht.jsonl"
        assert datensatz_lesen(fehlt_pfad) == []
        assert not fehlt_pfad.exists()

    with tempfile.TemporaryDirectory() as tmp:
        wegwerf = Path(tmp) / "benchmark.jsonl"
        wegwerf.write_text(
            '{"frage": "Was ist 2+2?", "enthaelt": ["4"]}\n'
            '{"frage": "3*3?", "enthaelt": ["9"]}\n'
            'das ist keine json zeile\n',
            encoding="utf-8",
        )
        aufgaben = datensatz_lesen(wegwerf)
        assert len(aufgaben) == 2, aufgaben

        aufgabe9 = aufgaben[1]
        assert aufgabe_pruefen("Es sind 9.", aufgabe9) is True
        assert aufgabe_pruefen("3 mal 3 ist 8", aufgabe9) is False
        assert aufgabe_pruefen("", aufgabe9) is False
        treffer, fehlende = aufgabe_bewerten("Es sind 9.", aufgabe9)
        assert treffer == 1 and fehlende == [], (treffer, fehlende)
        treffer, fehlende = aufgabe_bewerten("3 mal 3 ist 8", aufgabe9)
        assert treffer == 0 and fehlende == ["9"], (treffer, fehlende)
        treffer, fehlende = aufgabe_bewerten("", aufgabe9)
        assert treffer == 0 and fehlende == ["9"], (treffer, fehlende)
        assert korrektur_text(1, []) == "Sehr gut, das ist richtig."
        assert korrektur_text(
            1, ["9"]) == ("Das war schon sehr gut, aber nicht ganz richtig: "
                          "9 waere richtig.")
        assert korrektur_text(
            0, ["9"]) == "Das war nicht ganz richtig: 9 waere richtig."

        erg = benchmark_laufen(
            {"Was ist 2+2?": "4", "Nenne die Hauptstadt": "Paris"}, wegwerf
        )
        assert erg["geprueft"] == 1, erg
        assert erg["fehlt"] == 1, erg
        assert erg["quote"] == 1.0, erg
        assert erg["uebersprungen"] == 1, erg

        erg = benchmark_laufen({}, wegwerf)
        assert erg["geprueft"] == 0, erg
        assert erg["quote"] == 0.0, erg

        def absturz(frage):
            raise RuntimeError("absichtlich")

        try:
            antworten = {a["frage"]: absturz(a["frage"]) for a in aufgaben}
        except RuntimeError:
            antworten = {}
        erg = benchmark_laufen(antworten, wegwerf)
        assert erg["quote"] == 0.0, erg

    print("benchmark ok")
