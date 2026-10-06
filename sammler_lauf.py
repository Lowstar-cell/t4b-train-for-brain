"""Laesst den Sammler dauerhaft laufen.

Erntet die Themenliste (data/themen.jsonl) der Reihe nach, nach einem
Durchlauf 4 Stunden Pause, dann wieder von vorn. Status landet in
data/sammler_lauf.log. Beenden: Prozess killen (PID steht im Log) oder
Task-Manager.
"""
import os
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).parent
sys.path.insert(0, str(WURZEL))

from brain_core_v2 import BrainCoreV2

LOG = WURZEL / "data" / "sammler_lauf.log"


def loggen(text):
    zeile = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(zeile + "\n")
    except OSError:
        pass


def main():
    brain = BrainCoreV2()
    brain.sammler.loggen = loggen
    meldung = brain.sammler.starten(dauerhaft=True)
    loggen("PID %d | %s" % (os.getpid(), meldung))
    while True:
        time.sleep(300)
        s = brain.sammler.status
        loggen("%d/%d %s | %s" % (s["fortschritt"], s["gesamt"],
                                  s["thema"], s["ergebnis"][:120]))


if __name__ == "__main__":
    main()
