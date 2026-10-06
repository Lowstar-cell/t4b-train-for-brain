"""Laesst den Trainingszyklus dauerhaft laufen.

Ein Durchlauf = trainingslauf.py als eigener Prozess (Ernten, Schwarm,
Distillation, Training einer Runde, Benchmark, Chat). Danach 4 Stunden
Pause, wieder von vorn. Neue Themen/Lehrer-Antworten werden pro Tag nur
einmal gemacht; neue Texte liefert der Wiki-Sammler (immer eine Runde im
Vorlauf). Status: data/trainingszyklus.log. Beenden: Dienst t4b-training
stoppen (Pi5).
"""
import subprocess
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).parent

LOG = WURZEL / "data" / "trainingszyklus.log"
PAUSE = 4 * 3600
FEHLER_PAUSE = 600


def loggen(text):
    zeile = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(zeile + "\n")
    except OSError:
        pass


def main():
    n = 0
    while True:
        n += 1
        loggen("== Zyklus %d startet" % n)
        try:
            # Eigener Prozess: danach ist der RAM (Megatron, Schwarm) wieder
            # frei - sonst haelt die Pause 1,5 GB fest.
            subprocess.run([sys.executable, "-u", "-X", "utf8",
                            str(WURZEL / "trainingslauf.py")], check=True)
        except Exception as e:
            loggen("Zyklus %d fehlgeschlagen: %s" % (n, e))
            time.sleep(FEHLER_PAUSE)
            continue
        loggen("== Zyklus %d fertig, naechster in %d Stunden"
               % (n, PAUSE // 3600))
        time.sleep(PAUSE)


if __name__ == "__main__":
    main()
