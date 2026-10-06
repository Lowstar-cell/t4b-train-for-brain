import json
import time
from pathlib import Path

from specialists.lora_verwaltung import adapter_aktivieren, registry_lesen

WURZEL = Path(__file__).resolve().parent
VERLAUF = WURZEL / "data" / "metriken.jsonl"
ZEITFORMAT = "%Y-%m-%d %H:%M:%S"


def metrik_zahl(wert):
    """Wandelt eine Metrik in eine Zahl. Gibt None zurueck, wenn es keine ist."""
    try:
        return float(wert)
    except (TypeError, ValueError):
        return None


def verlauf_lesen(pfad=VERLAUF):
    """Liest den Messverlauf als Liste von Woertern. Fehlt die Datei: []. Wirft nie."""
    pfad = Path(pfad)
    zeilen = []
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            for roh in f:
                roh = roh.strip()
                if not roh:
                    continue
                try:
                    eintrag = json.loads(roh)
                except Exception:
                    continue
                if isinstance(eintrag, dict):
                    zeilen.append(eintrag)
    except OSError:
        return []
    return zeilen


def verlauf_buchen(adapter, metrik, pfad=VERLAUF):
    """Bucht eine Messung als JSONL-Zeile an. Gibt einen String zurueck."""
    zahl = metrik_zahl(metrik)
    if not adapter:
        return "Kein Adapter angegeben."
    if zahl is None:
        return "Keine Metrik angegeben."
    pfad = Path(pfad)
    eintrag = {"adapter": adapter, "metrik": zahl,
               "zeit": time.strftime(ZEITFORMAT)}
    try:
        pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(pfad, "a", encoding="utf-8") as f:
            f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
    except OSError as e:
        return f"Nicht gebucht: {e}"
    return f"Gebucht: {adapter} {zahl}"


def verschlechtert(neu, alt):
    """True, wenn neu schlechter als alt ist. Fehlt eine Zahl, gibt es nichts zu entscheiden."""
    n = metrik_zahl(neu)
    a = metrik_zahl(alt)
    if n is None or a is None:
        return False
    return n < a


def vortages_eintrag(verlauf, heute):
    """Der neueste Eintrag mit Zahlenmetrik von einem Tag vor heute. Sonst None."""
    kandidaten = [e for e in verlauf
                  if str(e.get("zeit", ""))[:10] < heute
                  and metrik_zahl(e.get("metrik")) is not None]
    if not kandidaten:
        return None
    return max(kandidaten, key=lambda e: str(e.get("zeit", "")))


def aktueller_adapter(daten=None):
    """Der aktive Adapter aus der Registry, sonst der neueste. Sonst None."""
    if daten is None:
        daten = registry_lesen()
    eintraege = [e for e in daten.get("adapter", []) if e.get("name")]
    if not eintraege:
        return None
    for eintrag in eintraege:
        if eintrag.get("aktiv"):
            return eintrag
    return max(eintraege, key=lambda e: str(e.get("zeit", "")))


def rollback_durchfuehren(metrik, pfad=VERLAUF, daten=None,
                          aktivieren=adapter_aktivieren):
    """Bucht die Metrik, vergleicht sie mit dem Vortages-Adapter und schaltet
    zurueck, wenn sie schlechter ist. Gibt einen String zurueck, der immer
    sagt, was passiert ist."""
    eintrag = aktueller_adapter(daten)
    if eintrag is None:
        return "Kein Adapter angemeldet."
    name = eintrag["name"]
    gebucht = verlauf_buchen(name, metrik, pfad)
    if gebucht.startswith("Nicht gebucht"):
        return gebucht
    vortag = vortages_eintrag(verlauf_lesen(pfad), time.strftime("%Y-%m-%d"))
    if vortag is None:
        return f"{gebucht} | kein Vortages-Adapter, {name} bleibt aktiv."
    if not verschlechtert(metrik, vortag.get("metrik")):
        return f"{gebucht} | Metrik nicht schlechter, {name} bleibt aktiv."
    ziel = vortag.get("adapter")
    if not ziel:
        return f"{gebucht} | Vortag ohne Namen, {name} bleibt aktiv."
    return f"{gebucht} | Metrik schlechter, {aktivieren(ziel)}"


if __name__ == "__main__":
    import datetime
    import tempfile
    gestern = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    heute = datetime.date.today().isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pfad = tmp / "metriken.jsonl"
        # verlauf_lesen auf einen Pfad, den es nicht gibt: [] und die Datei
        # wird nicht angelegt
        assert verlauf_lesen(pfad) == []
        assert not pfad.exists()
        # eine kaputte Zeile wird uebersprungen, nicht gewertet
        pfad.write_text("kaputt\n\n" + json.dumps(
            {"adapter": "a-alt", "metrik": 0.9, "zeit": gestern + " 09:00:00"})
            + "\n", encoding="utf-8")
        assert len(verlauf_lesen(pfad)) == 1
        # metrik_zahl und verschlechtert
        assert metrik_zahl("0.75") == 0.75
        assert metrik_zahl("keine Zahl") is None
        assert verschlechtert(0.4, 0.9) is True
        assert verschlechtert(0.9, 0.4) is False
        assert verschlechtert(None, 0.9) is False
        assert verschlechtert(0.4, None) is False
        # verlauf_buchen bucht, und eine Nichtzahl wird abgewiesen
        assert "Keine Metrik" in verlauf_buchen("b-neu", "keine Zahl", pfad)
        assert "Gebucht" in verlauf_buchen("b-neu", 0.4, pfad)
        assert len(verlauf_lesen(pfad)) == 2
        # vortages_eintrag nimmt den neuesten von gestern, nicht den von heute
        pfad2 = tmp / "nur_heute.jsonl"
        pfad2.write_text(json.dumps(
            {"adapter": "b-neu", "metrik": 0.1, "zeit": heute + " 08:00:00"})
            + "\n", encoding="utf-8")
        assert vortages_eintrag(verlauf_lesen(pfad2), heute) is None
        # aktueller_adapter: aktiv gewinnt, sonst der neueste, sonst None
        assert aktueller_adapter({"adapter": []}) is None
        assert aktueller_adapter({"adapter": [
            {"name": "alt", "zeit": "2026-09-26 10:00:00"},
            {"name": "neu", "zeit": "2026-09-27 10:00:00"}]})["name"] == "neu"
        assert aktueller_adapter({"adapter": [
            {"name": "alt", "zeit": "2026-09-26 10:00:00", "aktiv": True},
            {"name": "neu", "zeit": "2026-09-27 10:00:00"}]})["name"] == "alt"
        # der Rollback: schlechter -> zurueck auf den Vortages-Adapter
        daten = {"adapter": [{"name": "b-neu", "zeit": heute + " 10:00:00"}]}
        aufgerufen = []
        def fake(ziel):
            aufgerufen.append(ziel)
            return f"Aktiver Adapter: {ziel}"
        r = rollback_durchfuehren(0.4, pfad, daten, fake)
        assert aufgerufen == ["a-alt"], aufgerufen
        assert "a-alt" in r and "schlechter" in r, r
        # besser -> nichts zurueckgeschaltet
        aufgerufen = []
        r = rollback_durchfuehren(0.95, pfad, daten, fake)
        assert aufgerufen == [], aufgerufen
        assert "bleibt aktiv" in r, r
        # keine Registry -> kein Rollback, aber ein String
        assert "Kein Adapter" in rollback_durchfuehren(0.4, pfad, {"adapter": []}, fake)
        assert not (tmp / "data").exists()
    print("rollback ok")
