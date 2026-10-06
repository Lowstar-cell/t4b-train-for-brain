import json
from pathlib import Path

from specialists.flash_swarm import BASIS, FlashSwarm

ZIEL = BASIS.parent / "training.jsonl"
MIN_GEWICHT = 0.5


def exportieren(basis=BASIS, ziel=ZIEL, min_gewicht=MIN_GEWICHT):
    """Schreibt das Gedaechtnis als JSONL nach ziel. Gibt die Zahl der Zeilen zurueck."""
    zeilen = FlashSwarm(basis).db.execute(
        "SELECT text, quelle, gewicht FROM eintraege WHERE gewicht >= ?"
        " ORDER BY gewicht DESC, id", (min_gewicht,)).fetchall()
    ziel = Path(ziel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with ziel.open("w", encoding="utf-8") as f:
        for text, quelle, gewicht in zeilen:
            f.write(json.dumps({"text": text, "quelle": quelle, "gewicht": gewicht},
                               ensure_ascii=False) + "\n")
    return len(zeilen)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        ziel = tmp / "training.jsonl"
        s = FlashSwarm(tmp)
        s.learn("Der Hund bellt laut.", "test1")
        s.learn("Ein kurzer Satz.", "test2")
        s.learn("   ", "test3")
        with s.db:
            s.db.execute("UPDATE eintraege SET gewicht = 1.5 WHERE id = 1")
            s.db.execute("UPDATE eintraege SET gewicht = 0.2 WHERE id = 2")
        assert exportieren(tmp, ziel) == 1
        Zeilen = ziel.read_text(encoding="utf-8").splitlines()
        assert len(Zeilen) == 1, Zeilen
        satz = json.loads(Zeilen[0])
        assert satz == {"text": "Der Hund bellt laut.", "quelle": "test1", "gewicht": 1.5}, satz
        assert exportieren(tmp, ziel, min_gewicht=0.0) == 2
        assert not (tmp / "data").exists()
    print("export ok")
