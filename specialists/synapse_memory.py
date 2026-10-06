import json
from pathlib import Path


class SynapseMemory:
    """Gewichte je Spezialist, aus Feedback gelernt, als JSON gespeichert."""

    ROUTEN = ("search", "learn", "system", "vision")

    def __init__(self, memory_file, start=1.0, schritt=0.1, max_w=3.0):
        self.pfad = Path(memory_file)
        self.start = float(start)
        self.schritt = float(schritt)
        self.max_w = float(max_w)
        self.weights = {r: self.start for r in self.ROUTEN}
        self._laden()

    def _laden(self):
        if not self.pfad.exists():
            return
        try:
            daten = json.loads(self.pfad.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        if isinstance(daten, dict):
            for r in self.ROUTEN:
                wert = daten.get(r)
                if isinstance(wert, (int, float)) and wert > 0:
                    self.weights[r] = float(wert)

    def _speichern(self):
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        self.pfad.write_text(json.dumps(self.weights, indent=2),
                             encoding="utf-8")

    def _anheben(self, route, faktor):
        wert = self.weights.get(route, self.start) * faktor
        self.weights[route] = min(self.max_w, round(wert, 4))

    def learn_feedback(self, score=0.0, tool_correct=None, answer_correct=None):
        """Hebt die Gewichte der Module, die richtig lagen. Gibt einen Text zurueck."""
        gelernt = []
        if tool_correct is not None:
            self._anheben("search" if tool_correct else "system", 1.0 + self.schritt)
            gelernt.append("werkzeug" if tool_correct else "werkzeug-verworfen")
        if answer_correct is not None:
            self._anheben("learn" if answer_correct else "vision", 1.0 + self.schritt)
            gelernt.append("antwort" if answer_correct else "antwort-verworfen")
        self._speichern()
        return f"synapse: score={score} {', '.join(gelernt) or 'nichts'}"

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ziel = Path(tmp) / "synapse_weights.json"
        m = SynapseMemory(memory_file=str(ziel))
        assert m.weights == {"search": 1.0, "learn": 1.0, "system": 1.0,
                             "vision": 1.0}, m.weights
        m.learn_feedback(score=100, tool_correct=True, answer_correct=True)
        assert m.weights["search"] > 1.0 and m.weights["learn"] > 1.0
        assert m.weights["system"] == 1.0 and m.weights["vision"] == 1.0
        assert ziel.exists(), "nichts gespeichert"
        m2 = SynapseMemory(memory_file=str(ziel))
        assert m2.weights == m.weights, (m2.weights, m.weights)
        for _ in range(200):
            m2.learn_feedback(score=100, tool_correct=True, answer_correct=True)
        assert all(w <= m2.max_w for w in m2.weights.values()), m2.weights
        kaputt = Path(tmp) / "kaputt.json"
        kaputt.write_text("{kein json", encoding="utf-8")
        m3 = SynapseMemory(memory_file=str(kaputt))
        assert m3.weights["search"] == 1.0, m3.weights
        print("synapse_memory ok")
