"""Verwaltet die LoRA-Adapter-Registry (data/adapter.json).

Kein Modell, kein torch - nur die Standardbibliothek.
Alle Funktionen geben Strings zurueck und werfen nicht.
"""

import json
import tempfile
import time
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
REGISTRY = WURZEL / "data" / "adapter.json"
ZEITFORMAT = "%Y-%m-%d %H:%M:%S"


def registry_lesen(pfad=REGISTRY):
    """Liest die Registry. Gibt {"adapter": [...]} zurueck.

    Fehlt die Datei oder ist sie kaputt, kommt {"adapter": []} zurueck.
    Schreibt niemals. Eintraege, die keine Woerterbuecher sind, werden
    uebersprungen.
    """
    pfad = Path(pfad)
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            daten = json.load(f)
    except Exception:
        return {"adapter": []}
    if not isinstance(daten, dict):
        return {"adapter": []}
    eintraege = daten.get("adapter")
    if not isinstance(eintraege, list):
        return {"adapter": []}
    return {"adapter": [e for e in eintraege if isinstance(e, dict)]}


def _registry_schreiben(pfad, daten):
    """Schreibt die Registry als Ganzes, UTF-8, legt den Ordner an."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(daten, f, ensure_ascii=False, indent=2)


def adapter_anmelden(name, pfad, metrik=None):
    """Meldet einen neuen Adapter an. Gibt einen String zurueck."""
    if not name:
        return "Kein Name angegeben."
    daten = registry_lesen(REGISTRY)
    for eintrag in daten["adapter"]:
        if eintrag.get("name") == name:
            return f"Adapter schon da: {name}"
    daten["adapter"].append({
        "name": name,
        "pfad": str(pfad),
        "metrik": metrik,
        "zeit": time.strftime(ZEITFORMAT),
        "aktiv": False,
    })
    _registry_schreiben(REGISTRY, daten)
    return f"Adapter angemeldet: {name}"


def adapter_aktivieren(name):
    """Setzt genau einen Adapter auf aktiv, alle anderen auf inaktiv."""
    daten = registry_lesen(REGISTRY)
    namen = [e.get("name") for e in daten["adapter"]]
    if name not in namen:
        return f"Unbekannter Adapter: {name}"
    for eintrag in daten["adapter"]:
        eintrag["aktiv"] = eintrag.get("name") == name
    _registry_schreiben(REGISTRY, daten)
    return f"Aktiver Adapter: {name}"


def adapter_liste(pfad=REGISTRY):
    """Gibt die Eintraege als Text zurueck, eine Zeile je Adapter."""
    daten = registry_lesen(pfad)
    if not daten["adapter"]:
        return "Keine Adapter."
    zeilen = []
    for e in daten["adapter"]:
        zustand = "aktiv" if e.get("aktiv") else "inaktiv"
        zeilen.append(
            f"{e.get('name')} | {e.get('zeit')} | metrik={e.get('metrik')} | {zustand}"
        )
    return "\n".join(zeilen)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fehlt = tmp / "gibtsnicht.json"
        assert registry_lesen(fehlt) == {"adapter": []}
        assert not fehlt.exists()

        kaputt = tmp / "kaputt.json"
        kaputt.write_text("{{{ kaputt", encoding="utf-8")
        assert registry_lesen(kaputt) == {"adapter": []}

        test_reg = tmp / "data" / "adapter.json"
        # REGISTRY temporaer auf den Wegwerf-Ordner umbiegen
        alt = REGISTRY
        REGISTRY = test_reg
        try:
            antwort = adapter_anmelden("test", "p", 0.5)
            assert isinstance(antwort, str) and "test" in antwort
            daten = registry_lesen(test_reg)
            assert len(daten["adapter"]) == 1
            assert daten["adapter"][0]["name"] == "test"

            antwort = adapter_anmelden("test", "p")
            assert "schon da" in antwort
            assert len(registry_lesen(test_reg)["adapter"]) == 1

            assert "Kein Name" in adapter_anmelden("", "p")

            liste = adapter_liste(test_reg)
            assert "test" in liste and "aktiv" not in liste.replace("inaktiv", "")

            groesse_vorher = test_reg.read_text(encoding="utf-8")
            assert "Unbekannter Adapter" in adapter_aktivieren("gibtsnicht")
            assert test_reg.read_text(encoding="utf-8") == groesse_vorher

            adapter_aktivieren("test")
            daten = registry_lesen(test_reg)
            aktive = [e for e in daten["adapter"] if e.get("aktiv")]
            assert len(aktive) == 1 and aktive[0]["name"] == "test"
            assert "aktiv" in adapter_liste(test_reg)
        finally:
            mod.REGISTRY = alt

    assert not REGISTRY.exists(), "echte Registry wurde angelegt!"
    print("lora_verwaltung ok")
