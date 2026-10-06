import re

_LEERZEILEN = re.compile(r"\n{3,}")
_RABLAUF = re.compile(r"[ \t]{2,}")
_ENDE = re.compile(r"[\s.!?…]+$")

MAX_OUT = 4000
MAX_IN = 2000

TONS = {
    "ruhig": "",
    "knapp": "\n- ",
    "freundlich": "",
}


class EloquenceExpert:
    """Formt eine fertige Antwort, bevor sie in die Oberflaeche geht."""

    def __init__(self, ton="ruhig", max_zeichen=MAX_OUT):
        self.ton = ton if ton in TONS else "ruhig"
        self.max_zeichen = int(max_zeichen)

    def verfeinern(self, text, max_zeichen=None):
        """Putzt einen Text. Gibt immer einen String zurueck, nie None."""
        text = (text or "").replace("\r\n", "\n").strip()
        if not text:
            return "Keine Antwort."
        text = text[:max_zeichen or self.max_zeichen]
        text = _RABLAUF.sub(" ", text)
        text = _LEERZEILEN.sub("\n\n", text)
        text = _ENDE.sub(".", text)
        return text

    def kuerzen(self, text, max_zeichen=MAX_IN):
        """Kuerzt auf max_zeichen an einer Wortgrenze, haengt '...' an."""
        text = self.verfeinern(text, max_zeichen + 1)
        if len(text) <= max_zeichen:
            return text
        schnitt = text[:max_zeichen].rsplit(" ", 1)[0].rstrip()
        return schnitt + "..."


if __name__ == "__main__":
    e = EloquenceExpert()
    assert e.verfeinern("Hallo   Welt.") == "Hallo Welt.", e.verfeinern("Hallo   Welt.")
    assert e.verfeinern("") == "Keine Antwort."
    assert e.verfeinern("a\n\n\n\n\nb") == "a\n\nb"
    assert e.verfeinern("frag?") == "frag."
    assert len(e.verfeinern("x" * 9000)) <= MAX_OUT
    lang = " ".join(["Wort"] * 900)
    assert len(e.kuerzen(lang)) <= MAX_IN + 3
    assert e.kuerzen(lang).endswith("...")
    assert isinstance(EloquenceExpert(ton="knapp").verfeinern("hi"), str)
    assert EloquenceExpert(ton="gibtsnicht").ton == "ruhig"
    print("eloquence ok")
