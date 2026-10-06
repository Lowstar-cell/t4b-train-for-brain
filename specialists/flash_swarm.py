"""Der Schwarm: das Textgedaechtnis hinter brain_core_v2.py.

Liegt als SQLite-Datei auf der Platte (data/schwarm/gedaechtnis.sqlite), nicht
im RAM. Gesucht wird ueber den Volltextindex FTS5 (Rangfolge BM25).
Darf bis GRENZE (10 GB) wachsen. Darueber fliegen die am wenigsten nuetzlichen
Eintraege raus (ausduennen): kleinstes Gewicht x (1 + Treffer), bei Gleichstand
das Aelteste zuerst. Gewicht kommt aus dem Feedback, Treffer zaehlt jede Suche.
Eine alte gedaechtnis.jsonl wird beim ersten Oeffnen uebernommen (mit Gewicht)
und in gedaechtnis.jsonl.vor_sqlite umbenannt.
"""
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

BASIS = Path(__file__).parent.parent / "data" / "schwarm"
DATEI = "gedaechtnis.sqlite"
ALT = "gedaechtnis.jsonl"
GRENZE = 10 * 1024 ** 3
MAX_TREFFER = 3

_WORT = re.compile(r"\w+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS eintraege(
    id INTEGER PRIMARY KEY, kennung TEXT UNIQUE, text TEXT, quelle TEXT,
    gewicht REAL DEFAULT 1.0, treffer INTEGER DEFAULT 0, zeit TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS suche USING fts5(
    text, content='eintraege', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS rein AFTER INSERT ON eintraege BEGIN
    INSERT INTO suche(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS raus AFTER DELETE ON eintraege BEGIN
    INSERT INTO suche(suche, rowid, text) VALUES ('delete', old.id, old.text); END;
"""


def _abfrage(text):
    """Suchwoerter als FTS5-Abfrage (ODER). Kurze Woerter (der, ist, ein)
    treffen fast alles und bremsen die Suche - die fallen weg."""
    # ponytail: Laengenfilter statt Stoppwortliste
    worte = set(_WORT.findall((text or "").lower()))
    lang = {w for w in worte if len(w) > 3} or worte
    return " OR ".join('"%s"' % w for w in sorted(lang))


class FlashSwarm:
    def __init__(self, basis=BASIS):
        self.basis = Path(basis)
        self.basis.mkdir(parents=True, exist_ok=True)
        self.pfad = self.basis / DATEI
        # Gradio (interface.py) ruft aus anderen Threads - SQLite serialisiert selbst.
        self.db = sqlite3.connect(self.pfad, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._letzte = []
        alt = self.basis / ALT
        if alt.exists():
            self.import_archiv(alt)
            alt.rename(alt.with_name(ALT + ".vor_sqlite"))

    def __len__(self):
        return self.db.execute("SELECT count(*) FROM eintraege").fetchone()[0]

    def _rein(self, text, quelle="", gewicht=1.0):
        """Ein Text rein, ohne Commit. Nummer zurueck, 0 = leer oder schon da."""
        text = (text or "").strip()
        if not text:
            return 0
        c = self.db.execute(
            "INSERT OR IGNORE INTO eintraege(kennung, text, quelle, gewicht, zeit)"
            " VALUES (?, ?, ?, ?, ?)",
            (hashlib.sha1(text.encode("utf-8")).hexdigest(), text, quelle or "",
             float(gewicht), time.strftime("%Y-%m-%d %H:%M:%S")))
        return c.lastrowid if c.rowcount else 0

    def learn(self, text, quelle=""):
        """Legt einen Text im Gedaechtnis ab, gibt seine Nummer zurueck."""
        with self.db:
            return self._rein(text, quelle)

    def import_archiv(self, pfad, quelle_feld="quelle"):
        """Uebernimmt die Texte eines JSONL-Archivs (Vorhandenes nicht nochmal).
        Gibt die Zahl der neuen zurueck."""
        pfad = Path(pfad)
        if not pfad.exists():
            return 0
        vorher = len(self)
        with self.db, pfad.open("r", encoding="utf-8", errors="replace") as f:
            for zeile in f:
                try:
                    e = json.loads(zeile)
                except ValueError:
                    continue
                self._rein(e.get("text"), e.get(quelle_feld, ""), e.get("gewicht", 1.0))
        return len(self) - vorher

    def swarm_search(self, query, limit=MAX_TREFFER):
        """Die naechsten Treffer, der beste zuerst. Zaehlt sie als genutzt."""
        abfrage = _abfrage(query)
        if not abfrage:
            return []
        zeilen = self.db.execute(
            "SELECT e.id, e.text, e.quelle, t.rank FROM"
            " (SELECT rowid, rank FROM suche WHERE suche MATCH ? ORDER BY rank LIMIT ?) t"
            " JOIN eintraege e ON e.id = t.rowid ORDER BY t.rank",
            (abfrage, max(1, int(limit)))).fetchall()
        with self.db:
            self.db.executemany("UPDATE eintraege SET treffer = treffer + 1 WHERE id = ?",
                                [(z[0],) for z in zeilen])
        self._letzte = [z[0] for z in zeilen]
        # BM25 ist negativ, je kleiner desto besser -> Abstand 0..1
        return [{"id": i, "text": t, "quelle": q, "_distance": round(1 / (1 - r), 6)}
                for i, t, q, r in zeilen]

    def distribute_feedback(self, liked):
        """Gewichtet die Treffer der letzten Suche. Gibt einen String zurueck."""
        faktor = 1.2 if liked else 0.8
        with self.db:
            self.db.executemany("UPDATE eintraege SET gewicht = round(gewicht * ?, 6) WHERE id = ?",
                                [(faktor, i) for i in self._letzte])
        zurueck = f"Schwarm: {len(self._letzte)} Treffer x{faktor}"
        self._letzte = []
        return zurueck

    def belegt(self):
        """Wirklich belegte Bytes - geloeschte Seiten nutzt SQLite wieder,
        die Datei schrumpft nicht, waechst aber auch nicht weiter."""
        groesse, seiten, frei = (self.db.execute("PRAGMA " + p).fetchone()[0]
                                 for p in ("page_size", "page_count", "freelist_count"))
        return groesse * (seiten - frei)

    def ausduennen(self, grenze=GRENZE):
        """Ueber der Grenze: die am wenigsten nuetzlichen Eintraege loeschen,
        bis 5 % Luft ist. Gibt die Zahl der geloeschten zurueck."""
        weg = 0
        if self.belegt() <= grenze:
            return 0
        while self.belegt() > grenze * 0.95 and len(self):
            with self.db:
                weg += self.db.execute(
                    "DELETE FROM eintraege WHERE id IN (SELECT id FROM eintraege"
                    " ORDER BY gewicht * (1 + treffer), id LIMIT ?)",
                    (max(100, len(self) // 100),)).rowcount
        return weg

    def drop(self):
        """Leert das Gedaechtnis. Gibt die Zahl der geloeschten Eintraege zurueck."""
        anzahl = len(self)
        with self.db:
            self.db.execute("DELETE FROM eintraege")
        return anzahl


_standard = None


def swarm_search(query, limit=MAX_TREFFER):
    """Suche im Standard-Schwarm (data/schwarm) - so ruft megatron.denken ab."""
    global _standard
    if _standard is None:
        _standard = FlashSwarm()
    return _standard.swarm_search(query, limit)


if __name__ == "__main__":
    import tempfile
    # Windows: offene SQLite-Dateien lassen sich nicht loeschen
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        s = FlashSwarm(basis=Path(tmp))
        s.learn("Der Hund bellt laut.", "test1")
        s.learn("Katzen schlafen auf dem Sofa.", "test2")
        assert s.learn("Der Hund bellt laut.") == 0, "doppelt abgelegt"
        treffer = s.swarm_search("Hund bellen")
        assert treffer and "Hund" in treffer[0]["text"], treffer
        assert 0.0 <= treffer[0]["_distance"] < 1.0, treffer[0]
        assert isinstance(s.distribute_feedback(True), str)
        assert s.db.execute("SELECT gewicht, treffer FROM eintraege WHERE id = 1").fetchone() == (1.2, 1)
        s.db.close()
        neu = FlashSwarm(basis=Path(tmp))
        assert len(neu) == 2, len(neu)
        assert neu.swarm_search("Katzen"), "nicht gespeichert"

        archiv = Path(tmp) / "harvest.jsonl"
        archiv.write_text(
            '{"quelle": "u1", "art": "url", "text": "Ein neuer Satz."}\n'
            '{"quelle": "u1", "art": "url", "text": "Ein neuer Satz."}\n'
            'kaputte zeile ohne json\n'
            '{"quelle": "u2", "art": "url", "text": "   "}\n'
            '{"quelle": "u3", "art": "url", "text": "Ein zweiter Satz."}\n',
            encoding="utf-8")
        frisch = FlashSwarm(basis=Path(tmp) / "leer")
        assert frisch.import_archiv(archiv) == 2
        assert frisch.import_archiv(archiv) == 0
        assert frisch.swarm_search("Satz"), "nichts im Gedaechtnis"
        assert frisch.import_archiv(Path(tmp) / "gibtsnicht.jsonl") == 0
        assert frisch.drop() == 2 and len(frisch) == 0

        # alte jsonl wird uebernommen, Gewicht bleibt
        alt = Path(tmp) / "alt"
        alt.mkdir()
        (alt / ALT).write_text('{"id": 1, "text": "Altes Wissen.", "quelle": "q",'
                               ' "gewicht": 2.5, "vektor": {"3": 1.0}}\n', encoding="utf-8")
        a = FlashSwarm(basis=alt)
        assert len(a) == 1 and (alt / (ALT + ".vor_sqlite")).exists()
        assert a.db.execute("SELECT gewicht FROM eintraege").fetchone()[0] == 2.5

        # ausduennen: Genutztes bleibt, Ungenutztes (Aeltestes zuerst) fliegt
        g = FlashSwarm(basis=Path(tmp) / "grenze")
        g.learn("Der Transistor verstaerkt Strom.")
        for i in range(400):
            g.learn("Fuelltext %d %s" % (i, "wort " * 100))
        assert g.swarm_search("Transistor")[0]["id"] == 1
        grenze = g.belegt() // 2
        weg = g.ausduennen(grenze)
        assert weg > 0 and g.belegt() <= grenze * 0.95, (weg, g.belegt(), grenze)
        assert g.db.execute("SELECT count(*) FROM eintraege WHERE id = 1").fetchone()[0] == 1
        assert g.ausduennen(grenze) == 0
    print("flash_swarm ok")
