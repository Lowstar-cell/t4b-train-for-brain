"""Wiki-Sammler: deutsche Fachtexte (Technik, Medizin, Naturwissenschaft)
aus der deutschsprachigen Wikipedia - Lizenz CC BY-SA 4.0, abgerufen ueber
die offizielle Schnittstelle (API), so wie Wikimedia es erlaubt.

Holt die Artikel der Kategorien unten abwechselnd (jede Kategorie kommt
reihum dran, danach ihre Unterkategorien bis TIEFE), saeubert sie (Quellen,
Ueberschriften, Links raus) und haengt sie an data/harvest.jsonl an - dort
nimmt der Trainingszyklus sie auf (Schwarm + Megatron).

Laeuft dauerhaft (Pi5, Dienst wiki-sammler) und haelt immer eine Runde im
Vorlauf: Der Trainingslauf lernt je Zyklus hoechstens RUNDE Texte, der
Sammler holt, bis VORRAT ungelernte Texte im Archiv liegen (die Runde, die
gerade trainiert wird, plus die naechste) und wartet dann. Gelernte Texte
wirft der Trainingslauf mit aufraeumen() wieder aus dem Archiv.
Geholte Titel stehen in data/wiki_titel.txt - nichts wird zweimal geholt.
Sammeln und Aufraeumen nur unter Linux (fcntl-Sperre).

Tempo: Wikimedia erlaubt Clients ohne Kontaktangabe im User-Agent 10 Anfragen
pro Minute (mediawiki.org/wiki/Wikimedia_APIs/Rate_limits) -> eine Anfrage
alle 6 s, ca. 400 Artikel pro Stunde. Bei 429 wird so lange gewartet, wie
Wikipedia verlangt.

Laufen (Pi5, im Ordner T4Bv2):  python3 -u wiki_sammler.py
Selbsttest:                      python3 wiki_sammler.py test
"""
import contextlib
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WURZEL = Path(__file__).parent
sys.path.insert(0, str(WURZEL))

from specialists.harvester import saeubern

ARCHIV = WURZEL / "data" / "harvest.jsonl"
GELERNT = WURZEL / "data" / "gelernt.txt"
TITEL = WURZEL / "data" / "wiki_titel.txt"
API = "https://de.wikipedia.org/w/api.php"
UA = "T4Bv2-Sammler/1.0 (privates Lernprojekt) python-urllib"
PAUSE = 6.0
MAX_TEXT = 30000
MIN_TEXT = 1500   # Stubs und Listen bringen Megatron nichts
RUNDE = 1500      # Texte je Trainingszyklus
VORRAT = 2 * RUNDE  # die laufende Runde + eine im Vorlauf
TIEFE = 2         # Unterkategorien so tief mitnehmen

KATEGORIEN = {
    "Technik": [
        "Digitaltechnik", "Halbleiterbauelement", "Elektronik",
        "Elektrisches Bauelement", "Elektrische Maschine", "Sensor",
        "Mikrocontroller", "Messtechnik", "Nachrichtentechnik",
        "Energietechnik", "Maschinenbau", "Werkstoffkunde",
        "Fertigungsverfahren",
    ],
    "Informatik": [
        "Informatik", "Rechnerarchitektur", "Betriebssystem",
        "Netzwerkprotokoll", "Programmiersprache", "Softwaretechnik",
        "Datenbank", "IT-Sicherheit", "Kryptologie", "Künstliche Intelligenz",
    ],
    "Medizin": [
        "Anatomie", "Physiologie", "Immunologie", "Pharmakologie",
        "Arzneistoff", "Hormon", "Infektionskrankheit",
        "Krankheitsbild in der Inneren Medizin",
        "Krankheitsbild in der Kardiologie",
        "Krankheitsbild in der Neurologie", "Therapie", "Chirurgie",
        "Klinische Psychologie", "Psychologie", "Neurowissenschaften",
    ],
    "Naturwissenschaft": [
        "Physik", "Chemie", "Biochemie", "Biologie", "Zellbiologie",
        "Genetik", "Mathematik",
    ],
}


def anfrage(params):
    """Eine API-Anfrage im erlaubten Tempo. JSON zurueck, None bei Fehler."""
    url = API + "?" + urllib.parse.urlencode(dict(params, format="json"))
    for _ in range(5):
        time.sleep(PAUSE)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                return None
            warten = int(e.headers.get("Retry-After") or 30)
            print("   Wikipedia bremst (%d), warte %d s" % (e.code, warten))
            time.sleep(min(300, max(5, warten)))
        except (OSError, ValueError):
            time.sleep(30)
    return None


def artikel_titel(kategorie, gesehen, tiefe=TIEFE):
    """Alle Artikel der Kategorie (500 je Anfrage), danach die ihrer
    Unterkategorien bis tiefe. gesehen: Kategorien, die schon dran waren."""
    unter, weiter = [], {}
    while True:
        d = anfrage(dict({"action": "query", "list": "categorymembers",
                          "cmtitle": "Kategorie:" + kategorie,
                          "cmnamespace": "0|14", "cmtype": "page|subcat",
                          "cmlimit": "500"}, **weiter))
        if not d:
            break
        for m in d.get("query", {}).get("categorymembers", []):
            if m["ns"] == 14:
                unter.append(m["title"].split(":", 1)[1])
            else:
                yield m["title"]
        if "continue" not in d:
            break
        weiter = d["continue"]
    for k in unter if tiefe else []:
        if k not in gesehen:
            gesehen.add(k)
            yield from artikel_titel(k, gesehen, tiefe - 1)


def artikel_text(titel):
    """Ganzer Artikel als reiner Text, gesaeubert und gekuerzt."""
    d = anfrage({"action": "query", "prop": "extracts", "explaintext": "1",
                 "redirects": "1", "titles": titel})
    for seite in (d or {}).get("query", {}).get("pages", {}).values():
        return saeubern(seite.get("extract") or "")[:MAX_TEXT]
    return ""


def kennung(text):
    """Dieselbe Kennung, die trainingslauf.py in data/gelernt.txt schreibt."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def bekannte_titel():
    """Schon geholte Titel. Beim ersten Mal aus dem Archiv uebernommen."""
    if not TITEL.exists():
        alt = set()
        if ARCHIV.exists():
            with ARCHIV.open("r", encoding="utf-8", errors="replace") as f:
                for zeile in f:
                    try:
                        d = json.loads(zeile)
                    except ValueError:
                        continue
                    if d.get("art") == "wikipedia" and d.get("thema"):
                        alt.add(d["thema"])
        TITEL.write_text("".join(t + "\n" for t in alt), encoding="utf-8")
    return set(TITEL.read_text(encoding="utf-8").splitlines())


def offen_zaehlen(archiv=ARCHIV, gelernt=None):
    """Texte im Archiv, die Megatron noch nicht gelernt hat (ohne Arbeitsweisen,
    die lernt er nicht)."""
    if gelernt is None:
        gelernt = set(GELERNT.read_text().split()) if GELERNT.exists() else set()
    n = 0
    with archiv.open("r", encoding="utf-8", errors="replace") as f:
        for zeile in f:
            try:
                d = json.loads(zeile)
            except ValueError:
                continue
            if d.get("art") != "arbeitsweise" and kennung(d.get("text", "")) not in gelernt:
                n += 1
    return n


@contextlib.contextmanager
def sperre(archiv=ARCHIV):
    """Nur einer schreibt: der Sammler haengt an, der Trainingslauf raeumt auf."""
    import fcntl  # nur Linux - so laesst sich das Modul auch unter Windows laden
    with open(archiv.with_suffix(".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def ablegen(eintrag):
    """Eine Zeile ans Archiv - ein einziger write, damit sich der
    Trainingslauf und der Sammler nicht in die Zeile schreiben."""
    zeile = (json.dumps(eintrag, ensure_ascii=False) + "\n").encode("utf-8")
    with sperre():
        fd = os.open(ARCHIV, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, zeile)
        finally:
            os.close(fd)


def aufraeumen(gelernt, archiv=ARCHIV):
    """Gelernte Texte aus dem Archiv werfen - Megatron und Schwarm haben sie.
    Bleiben: Ungelerntes, Arbeitsweisen und die Harvester-Eintraege von heute
    (an denen haengt 'heute schon geholt/distilliert'). Sammler-Eintraege
    erkennt man an "kategorie". Gibt die Zahl der geloeschten zurueck."""
    heute = time.strftime("%Y-%m-%d")
    behalten, weg = [], 0
    with sperre(archiv):
        with archiv.open("rb") as f:
            for zeile in f:
                try:
                    d = json.loads(zeile)
                except ValueError:
                    behalten.append(zeile)
                    continue
                if (kennung(d.get("text", "")) in gelernt
                        and ("kategorie" in d or not d.get("zeit", "").startswith(heute))):
                    weg += 1
                else:
                    behalten.append(zeile)
        tmp = archiv.with_suffix(".tmp")
        tmp.write_bytes(b"".join(behalten))
        tmp.replace(archiv)
    return weg


def reihum(quellen):
    """Nimmt abwechselnd aus jeder Quelle eins, bis alle leer sind."""
    quellen = list(quellen)
    while quellen:
        for q in list(quellen):
            try:
                yield next(q[1]), q[0]
            except StopIteration:
                quellen.remove(q)


def main():
    bekannt = bekannte_titel()
    print("Wiki-Sammler: %d Titel schon geholt, Vorrat-Ziel %d" % (len(bekannt), VORRAT))
    t0, neu, kb = time.time(), 0, 0.0
    while True:
        gesehen = {k for liste in KATEGORIEN.values() for k in liste}
        quellen = [((gebiet, k), artikel_titel(k, gesehen))
                   for gebiet, liste in KATEGORIEN.items() for k in liste]
        offen = offen_zaehlen()
        for titel, (gebiet, kategorie) in reihum(quellen):
            if titel in bekannt:
                continue
            while offen >= VORRAT:
                print("Vorrat voll: %d ungelernte Texte, warte 10 min" % offen)
                time.sleep(600)
                offen = offen_zaehlen()
            text = artikel_text(titel)
            if len(text) >= MIN_TEXT:
                ablegen({"quelle": "https://de.wikipedia.org/wiki/" + urllib.parse.quote(titel.replace(" ", "_")),
                         "art": "wikipedia", "thema": titel, "gebiet": gebiet,
                         "kategorie": kategorie, "lizenz": "CC BY-SA 4.0",
                         "text": text, "zeit": time.strftime("%Y-%m-%d %H:%M:%S")})
                neu += 1
                offen += 1
                kb += len(text) / 1024
                print("%5d %-12s %-40s %5.1f KB | Vorrat %d/%d | gesamt %.1f MB, %.0f min"
                      % (neu, gebiet, titel[:40], len(text) / 1024, offen, VORRAT,
                         kb / 1024, (time.time() - t0) / 60))
            bekannt.add(titel)
            with TITEL.open("a", encoding="utf-8") as f:
                f.write(titel + "\n")
        print("Alle Kategorien durch (%d neue Artikel) - neuer Durchgang in 24 h" % neu)
        time.sleep(86400)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        q = reihum([("a", iter([1, 2, 3])), ("b", iter([9]))])
        assert list(q) == [(1, "a"), (9, "b"), (2, "a"), (3, "a")]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "h.jsonl"
            jetzt = time.strftime("%Y-%m-%d %H:%M:%S")
            zeilen = [{"art": "wikipedia", "kategorie": "Physik", "text": "gelernt", "zeit": jetzt},
                      {"art": "wikipedia", "kategorie": "Physik", "text": "neu", "zeit": jetzt},
                      {"art": "lehrer", "text": "gelernt heute", "zeit": jetzt},
                      {"art": "lehrer", "text": "gelernt alt", "zeit": "2026-01-01 00:00:00"},
                      {"art": "arbeitsweise", "text": "befehl"}]
            a.write_text("".join(json.dumps(z) + "\n" for z in zeilen) + "kaputt\n")
            g = {kennung(t) for t in ("gelernt", "gelernt heute", "gelernt alt")}
            assert offen_zaehlen(a, g) == 1
            assert aufraeumen(g, a) == 2
            rest = a.read_text().splitlines()
            assert [json.loads(z)["text"] for z in rest[:-1]] == ["neu", "gelernt heute", "befehl"], rest
            assert rest[-1] == "kaputt", rest
        print("wiki_sammler ok")
    else:
        main()
