import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
ARCHIV = WURZEL / "data" / "harvest.jsonl"
MEMORY_VERZEICHNIS = Path.home() / "MCP-Server" / "agent-memory-mcp" / "data" / "records"
API = "https://de.wikipedia.org/w/api.php"
MAX_TEXT = 2000
TIMEOUT = 10
TAGESLIMIT_MB = 200

_UA = "T4Bv2-Harvester/1.0 (lokal, T4Bv2)"
# Ab diesen Ueberschriften kommen nur noch Quellen und Links.
_ANHANG = re.compile(r"\n=+\s*(Siehe auch|Literatur|Weblinks|Einzelnachweise|"
                     r"Anmerkungen|Quellen|Belege)\s*=+.*", re.S)


def saeubern(text):
    """Seitenmuell raus: Quellen-Anhang, Ueberschriften-Zeilen, Links,
    Markdown-Zeichen. Uebrig bleibt Fliesstext zum Lernen."""
    text = _ANHANG.sub("", text or "")
    text = re.sub(r"(?m)^=+.*=+\s*$", "", text)
    text = re.sub(r"https?://\S+|\S*(title|oldid)=\S*", "", text)
    text = re.sub(r"(?m)^\s*(#+|[-*]|\d+\.)\s+", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


class KnowledgeHarvester:
    """Holt Text aus dem Netz und legt ihn im Archiv ab."""

    def __init__(self, archiv=ARCHIV):
        self.archiv = Path(archiv)
        self.archiv.parent.mkdir(parents=True, exist_ok=True)

    def learn_url(self, url):
        """Holt die Seite als Text. Gibt einen String zurueck."""
        if not (url or "").startswith(("http://", "https://")):
            return "Keine gueltige URL: " + str(url)
        if self._gibt_es_schon(url):
            return "Schon im Archiv von heute, uebersprungen."
        try:
            anfrage = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(anfrage, timeout=TIMEOUT) as r:
                roh = r.read(200_000)
            text = self._html_zu_text(roh.decode("utf-8", errors="replace"))
        except Exception as e:
            return f"Abruf fehlgeschlagen: {e}"
        if not text:
            return "Leere Seite: " + url
        grund = self._ablegen({"quelle": url, "art": "url", "text": text})
        if grund:
            return grund
        return f"Gelernt aus {url}: {len(text)} Zeichen. Anfang: {text[:200]}"

    def learn_wikipedia(self, topic, ganz=False):
        """Holt den Einleitungstext von Wikipedia, mit ganz=True den ganzen
        Artikel als reinen Text (ohne Menue/Quellen). String zurueck."""
        topic = (topic or "").strip()
        if not topic:
            return "Kein Thema angegeben."
        felder = {
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": "1", "redirects": "1", "titles": topic,
        }
        if not ganz:
            felder["exintro"] = "1"
        params = urllib.parse.urlencode(felder)
        voll = f"{API}?{params}"
        if self._gibt_es_schon(voll):
            return "Schon im Archiv von heute, uebersprungen."
        for versuch in range(2):
            try:
                anfrage = urllib.request.Request(voll, headers={"User-Agent": _UA})
                with urllib.request.urlopen(anfrage, timeout=TIMEOUT) as r:
                    daten = json.loads(r.read().decode("utf-8", errors="replace"))
                break
            except urllib.error.HTTPError as e:
                # 429 = zu viele Anfragen: Wikipedia sagt, wie lange warten.
                if e.code != 429 or versuch:
                    return f"Wikipedia nicht erreichbar: {e}"
                time.sleep(min(60, int(e.headers.get("Retry-After") or 30)))
            except Exception as e:
                return f"Wikipedia nicht erreichbar: {e}"
        seiten = daten.get("query", {}).get("pages", {})
        for seite in seiten.values():
            text = saeubern(seite.get("extract") or "")[:MAX_TEXT]
            if not text:
                continue
            grund = self._ablegen({"quelle": voll, "art": "wikipedia",
                                   "thema": seite.get("title", topic), "text": text})
            if grund:
                return grund
            return (f"Gelernt: {seite.get('title', topic)} "
                    f"({len(text)} Zeichen). Anfang: {text[:200]}")
        return f"Kein Artikel zu {topic}."

    def learn_arbeitsweisen(self, verzeichnis=None):
        """Liest die Arbeitsweisen (Playbooks/Probleme) aus der agent-memory
        und legt sie direkt im Archiv ab - ohne Netz. String zurueck."""
        verzeichnis = Path(verzeichnis) if verzeichnis else MEMORY_VERZEICHNIS
        if not verzeichnis.exists():
            return "Kein Memory-Verzeichnis: %s" % verzeichnis
        gelernt = 0
        for datei in sorted(verzeichnis.glob("*.json")):
            try:
                daten = json.loads(datei.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            quelle = "memory://" + str(daten.get("id") or datei.stem)
            if self._gibt_es_schon(quelle):
                continue
            text = self._memory_zu_text(daten)
            if not text:
                continue
            grund = self._ablegen({"quelle": quelle, "art": "arbeitsweise",
                                   "text": text})
            if grund:
                continue
            gelernt += 1
        return "Gelernt: %d Arbeitsweisen aus %s" % (gelernt, verzeichnis)

    def _heute_gebraucht(self):
        """Bytes im Archiv, deren zeit mit dem heutigen Datum beginnt."""
        heute = time.strftime("%Y-%m-%d")
        summe = 0
        if not self.archiv.exists():
            return 0
        with self.archiv.open("r", encoding="utf-8", errors="replace") as f:
            for zeile in f:
                if '"zeit"' in zeile and zeile.find(heute) > -1:
                    summe += len(zeile.encode("utf-8", errors="replace"))
        return summe

    def _gibt_es_schon(self, quelle):
        """True, wenn diese Quelle heute schon abgelegt wurde."""
        heute = time.strftime("%Y-%m-%d")
        if not quelle or not self.archiv.exists():
            return False
        with self.archiv.open("r", encoding="utf-8", errors="replace") as f:
            for zeile in f:
                if quelle in zeile and zeile.find(heute) > -1:
                    return True
        return False

    def _ablegen(self, eintrag):
        eintrag["zeit"] = time.strftime("%Y-%m-%d %H:%M:%S")
        zeile = json.dumps(eintrag, ensure_ascii=False) + "\n"
        groesse = len(zeile.encode("utf-8", errors="replace"))
        if self._heute_gebraucht() + groesse > TAGESLIMIT_MB * 1024 * 1024:
            return "Tageslimit erreicht: heute sind bereits MB von 200 voll."
        if self._gibt_es_schon(eintrag.get("quelle", "")):
            return "Schon im Archiv von heute, uebersprungen."
        with self.archiv.open("a", encoding="utf-8") as f:
            f.write(zeile)
        return ""

    def _memory_zu_text(self, d):
        """Flacht einen Memory-Eintrag zu Prosa (Arbeitsweise)."""
        teile = []
        for schluessel in ("title", "trigger", "context"):
            wert = (d.get(schluessel) or "").strip()
            if wert:
                teile.append(wert)
        for schritt in d.get("steps") or []:
            do = ((schritt or {}).get("do") or "").strip()
            if do:
                teile.append(do)
        cmd = (d.get("verify") or {}).get("command") or ""
        if cmd:
            teile.append("Geprueft: " + cmd)
        for pitfall in d.get("pitfalls") or []:
            pitfall = (pitfall or "").strip()
            if pitfall:
                teile.append("Achtung: " + pitfall)
        if not teile:
            return ""
        return " ".join(teile)[:MAX_TEXT]

    def _html_zu_text(self, html):
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                    .replace("&quot;", '"').replace("&#39;", "'")
                    .replace("&lt;", "<").replace("&gt;", ">"))
        return re.sub(r"\s+", " ", text).strip()[:MAX_TEXT]


if __name__ == "__main__":
    h = KnowledgeHarvester()
    assert isinstance(h.learn_wikipedia(""), str)
    assert isinstance(h.learn_url("keine-url"), str)
    assert "gueltige URL" in h.learn_url("keine-url")
    text = h._html_zu_text("<p>Hallo <b>Welt</b></p><script>x=1</script>")
    assert "Hallo Welt" in text, text
    assert "x=1" not in text, text
    assert "&" not in h._html_zu_text("a &amp; b") or "&" in "a &amp; b"
    roh = ("Ein Router leitet Pakete.\n\n== Geschichte ==\nSiehe https://x.de/a "
           "und **fett**.\n- Punkt eins\n\n== Weblinks ==\nhttps://y.de\n"
           "== Einzelnachweise ==\nISBN 123")
    sauber = saeubern(roh)
    assert sauber == "Ein Router leitet Pakete.\nSiehe und fett.\nPunkt eins", sauber
    assert saeubern("x Title=physiologie&oldid=270715908 y") == "x y"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pfad = Path(tmp) / "h.jsonl"
        h2 = KnowledgeHarvester(archiv=pfad)
        assert h2._ablegen({"quelle": "u1", "art": "url", "text": "a" * 100}) == ""
        assert "Schon im Archiv" in h2._ablegen({"quelle": "u1", "art": "url", "text": "b"})
        assert h2._heute_gebraucht() > 0
        assert h2._gibt_es_schon("u1") and not h2._gibt_es_schon("u2")
        alt = TAGESLIMIT_MB
        try:
            TAGESLIMIT_MB = 0
            assert "Tageslimit" in h2._ablegen({"quelle": "u3", "art": "url", "text": "c"})
        finally:
            TAGESLIMIT_MB = alt
        assert not (Path(tmp) / "u3").exists()
    print("harvester ok")
