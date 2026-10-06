"""Automatischer Sammler: erntet die Themenliste im Hintergrund.

Fuehrt fuer jedes Thema den Lernmini-Zyklus aus (ernten: Harvester ->
Schwarm -> Keim). Die Themenliste liegt als Daten (data/themen.jsonl,
eine Zeile {"gebiet", "thema"}), der Sammler liesst sie und arbeitet
der Reihe nach. Das 200-MB-Tageslimit greift im Harvester selbst.
"""

import json
import threading
import time
from pathlib import Path

THAMEN = Path(__file__).parent.parent / "data" / "themen.jsonl"
ERGEBNIS_KUERZEN = 300


def themen_lesen(pfad=THAMEN):
    """Liest die Themenliste. Liste von {"gebiet", "thema"}. Wirft nie."""
    pfad = Path(pfad)
    themen = []
    if not pfad.exists():
        return themen
    try:
        text = pfad.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return themen
    for zeile in text.splitlines():
        zeile = zeile.strip()
        if not zeile:
            continue
        try:
            eintrag = json.loads(zeile)
        except ValueError:
            continue
        if isinstance(eintrag, dict) and eintrag.get("thema"):
            themen.append({"gebiet": str(eintrag.get("gebiet", "")),
                           "thema": str(eintrag["thema"]).strip()})
    return themen


class Sammler:
    """Erntet die Themenliste der Reihe nach in einem Hintergrund-Thread."""

    def __init__(self, ernten, pause=2.0, wiederholung_pause=14400, loggen=None):
        """ernten: callable(thema) -> Ergebnis (der Lernmini-Zyklus).
        wiederholung_pause: Sekunden zwischen zwei Durchlaeufen (dauerhaft).
        loggen: callable(zeile) oder None - wird nach jedem Thema gerufen."""
        self.ernten = ernten
        self.pause = float(pause)
        self.wiederholung_pause = float(wiederholung_pause)
        self.loggen = loggen
        self.status = {"laufend": False, "thema": "", "fortschritt": 0,
                       "gesamt": 0, "ergebnis": ""}

    def starten(self, themen=None, dauerhaft=False):
        """Startet den Sammel-Thread. Gibt einen Status-String zurueck."""
        if self.status.get("laufend"):
            return "Sammler laeuft bereits."
        liste = themen if themen else themen_lesen()
        if not liste:
            return "Keine Themenliste gefunden: " + str(THAMEN)
        self.status = {"laufend": True, "thema": "", "fortschritt": 0,
                       "gesamt": len(liste), "ergebnis": ""}
        thread = threading.Thread(target=self._arbeiten,
                                  args=(liste, bool(dauerhaft)), daemon=True)
        thread.start()
        return "Sammler gestartet: %d Themen%s." % (
            len(liste), ", dauerhaft" if dauerhaft else "")

    def _arbeiten(self, liste, dauerhaft=False):
        while True:
            for i, eintrag in enumerate(liste):
                self.status["thema"] = "%s: %s" % (eintrag.get("gebiet", ""),
                                                   eintrag["thema"])
                self.status["fortschritt"] = i
                try:
                    self.status["ergebnis"] = str(self.ernten(eintrag["thema"]))[:ERGEBNIS_KUERZEN]
                except Exception as e:
                    self.status["ergebnis"] = "Fehler: %s" % e
                if self.loggen:
                    try:
                        self.loggen("%s | %s | %s" % (eintrag.get("gebiet", ""),
                                                      eintrag["thema"],
                                                      self.status["ergebnis"]))
                    except Exception:
                        pass
                time.sleep(self.pause)
            self.status["fortschritt"] = len(liste)
            self.status["thema"] = ""
            if not dauerhaft:
                self.status["laufend"] = False
                break
            time.sleep(self.wiederholung_pause)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert themen_lesen(Path(tmp) / "gibtsnicht.jsonl") == []
        datei = Path(tmp) / "themen.jsonl"
        datei.write_text(
            '{"gebiet": "Netzwerke", "thema": "TCP/IP"}\n'
            'keine json zeile\n'
            '{"gebiet": "Medizin", "thema": "  Immunsystem  "}\n'
            '{"thema": ""}\n', encoding="utf-8")
        themen = themen_lesen(datei)
        assert len(themen) == 2, themen
        assert themen[0]["thema"] == "TCP/IP", themen
        assert themen[1]["thema"] == "Immunsystem", themen

        gesammelt = []
        s = Sammler(lambda thema: gesammelt.append(thema) or
                    "gelernt: %s" % thema, pause=0.01)
        meldung = s.starten(themen)
        assert "2 Themen" in meldung, meldung
        for _ in range(100):
            if not s.status["laufend"] and s.status["fortschritt"] == 2:
                break
            time.sleep(0.01)
        assert gesammelt == ["TCP/IP", "Immunsystem"], gesammelt
        assert s.status["laufend"] is False
        assert "gelernt: Immunsystem" in s.status["ergebnis"], s.status

        absturz = Sammler(lambda thema: (_ for _ in ()).throw(
            RuntimeError("absichtlich")), pause=0.01)
        absturz.starten(themen)
        for _ in range(100):
            if not absturz.status["laufend"]:
                break
            time.sleep(0.01)
        assert "Fehler" in absturz.status["ergebnis"], absturz.status

        dauer = Sammler(lambda thema: "d: %s" % thema,
                        pause=0.01, wiederholung_pause=0.01)
        dauer.starten(themen, dauerhaft=True)
        war_fertig = False
        zweiter = False
        for _ in range(600):
            if dauer.status["fortschritt"] == 2:
                war_fertig = True
            if war_fertig and dauer.status["fortschritt"] < 2:
                zweiter = True
                break
            time.sleep(0.01)
        assert war_fertig and zweiter, (war_fertig, zweiter, dauer.status)
        assert dauer.status["laufend"] is True

        zeilen = []
        s2 = Sammler(lambda thema: "gelernt: %s" % thema, pause=0.01,
                     loggen=zeilen.append)
        s2.starten(themen)
        for _ in range(100):
            if not s2.status["laufend"] and s2.status["fortschritt"] == 2:
                break
            time.sleep(0.01)
        assert len(zeilen) == 2, zeilen
        assert "Netzwerke | TCP/IP" in zeilen[0], zeilen
    print("sammler ok")
