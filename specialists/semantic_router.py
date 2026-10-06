class SemanticRouter:
    """Entscheidet, welcher Spezialist eine Anfrage bekommt."""

    def __init__(self):
        self.regeln = (
            # (route, schluesselwoerter, punkte)
            ("learn",   ("lerne", "lern ", "wikipedia", "merke dir", "url"), 3),
            ("vision",  ("bild", "foto", "screenshot", "kamera", "video"), 3),
            ("system",  ("shutdown", "neustart", "status", "speicher",
                         "prozess", "systeminfo", "beende"), 3),
            ("search",  ("was weiss", "wisse", "finde", "suche", "gibt es",
                         "kontext", "was steht"), 2),
        )

    def decide_route(self, query, weights=None):
        """Gibt (route, score) zurueck. Leerer Text -> ("vision", 0.0)."""
        text = (query or "").lower()
        if not text.strip():
            return "vision", 0.0
        best = ("vision", 0.0)
        for route, woerter, punkte in self.regeln:
            treffer = sum(1 for w in woerter if w in text)
            if treffer == 0:
                continue
            score = float(treffer * punkte)
            if weights:
                score *= float(weights.get(route, 1.0))
            if score > best[1]:
                best = (route, score)
        return best

if __name__ == "__main__":
    r = SemanticRouter()
    faelle = [
        ("Lerne Wikipedia ueber Vulkan", "learn"),
        ("Bitte ein Bild hochladen", "vision"),
        ("Wie ist der Speicherstand?", "system"),
        ("Finde etwas ueber Vulkan", "search"),
        ("Wie geht es dir?", "vision"),
        ("", "vision"),
    ]
    for frage, erwartet in faelle:
        route, score = r.decide_route(frage)
        assert route == erwartet, f"{frage!r} -> {route}, erwartet {erwartet}"
        assert isinstance(score, float), score
    gewichte = {"learn": 2.0}
    route, score = r.decide_route("Lerne etwas", weights=gewichte)
    assert score == 6.0, score
    print("semantic_router ok")
