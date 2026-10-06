class LoopVerbrochen(Exception):
    pass


class AktionsFehler(Exception):
    pass


class LoopWaechter:
    def __init__(self, grenze=3):
        self.grenze = grenze
        self.letzte = None
        self.zaehler = 0
        self.fehler = 0

    def beobachte(self, aktion):
        if aktion == self.letzte:
            self.zaehler += 1
        else:
            self.zaehler = 1
        self.letzte = aktion
        if self.zaehler >= self.grenze and self.zaehler >= 3:
            self.zaehler = 0
            raise LoopVerbrochen(f"{self.zaehler} mal hintereinander: {aktion}")

    def fehler_zaehlen(self, aktion):
        self.fehler += 1
        if self.fehler >= self.grenze:
            self.fehler = 0
            raise AktionsFehler(aktion)

    def zuruecksetzen(self):
        self.letzte = None
        self.zaehler = 0
        self.fehler = 0

    def alsogaenge(self):
        return self.zaehler


if __name__ == "__main__":
    w = LoopWaechter(grenze=3)
    w.beobachte("datei_lesen")
    w.beobachte("datei_lesen")
    if w.alsogaenge() != 2:
        raise SystemExit("Selbstpruefung FAIL: alsogaenge != 2")
    try:
        w.beobachte("datei_lesen")
    except LoopVerbrochen:
        pass
    else:
        raise SystemExit("Selbstpruefung FAIL: kein LoopVerbrochen")
    if w.alsogaenge() != 0:
        raise SystemExit("Selbstpruefung FAIL: Zaehler nicht zurueckgesetzt")
    f = LoopWaechter(grenze=2)
    f.fehler_zaehlen("bash")
    try:
        f.fehler_zaehlen("bash")
    except AktionsFehler:
        pass
    else:
        raise SystemExit("Selbstpruefung FAIL")
    print("supervisor ok")
