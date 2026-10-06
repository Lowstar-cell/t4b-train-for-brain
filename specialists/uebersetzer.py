import json
from pathlib import Path


def yolo_zu_text(treffer):
    """Aus YOLO-Erkennungen einen deutschen Text. Gibt einen String zurueck.

    Jeder Treffer ist ein DICT mit "klasse" (str) und "bbox":
    [x1, y1, x2, y2] als Ecken in Zahlen von 0 bis 1. Ist die Liste leer,
    kommt "Keine Objekte erkannt." zurueck.
    """
    if not treffer:
        return "Keine Objekte erkannt."
    teile = []
    for t in treffer:
        x1, y1, x2, y2 = [float(w) for w in t["bbox"]]
        links = x1 < 0.34
        rechts = x2 > 0.66
        if links and rechts:
            breite = "breit"
        elif links:
            breite = "links"
        elif rechts:
            breite = "rechts"
        else:
            breite = "mittig"
        if y2 < 0.34:
            hoehe = "oben"
        elif y1 > 0.66:
            hoehe = "unten"
        else:
            hoehe = "mittig"
        mitte_x = (x1 + x2) / 2.0
        mitte_y = (y1 + y2) / 2.0
        teile.append(
            f"{t['klasse']} {breite} {hoehe}, "
            f"Mitte bei x={round(mitte_x * 100)}%, y={round(mitte_y * 100)}%.")
    return " ".join(teile)


def whisper_zu_text(eintrag):
    """Aus einer Whisper-Ausgabe (DICT) flachen Text machen.

    "segments" mit einer Liste von DICTs mit "text" wird zu einem
    Satz. Gibt es kein "segments", zaehlt "text". Fehlt beides, ist das
    Ergebnis "". Wirft nichts.
    """
    if not isinstance(eintrag, dict):
        return ""
    segmente = eintrag.get("segments")
    if isinstance(segmente, list) and segmente:
        texte = [str(s.get("text", "")).strip() for s in segmente
                 if isinstance(s, dict)]
        return " ".join(t for t in texte if t).strip()
    return str(eintrag.get("text", "") or "").strip()


if __name__ == "__main__":
    assert yolo_zu_text([]) == "Keine Objekte erkannt."
    mittig = [{"klasse": "stuhl", "bbox": [0.4, 0.4, 0.6, 0.6]}]
    text = yolo_zu_text(mittig)
    assert "stuhl" in text and "mittig" in text, text
    assert "x=50%" in text and "y=50%" in text, text
    oben = [{"klasse": "lampe", "bbox": [0.8, 0.05, 0.95, 0.3]}]
    assert "oben" in yolo_zu_text(oben) and "rechts" in yolo_zu_text(oben)
    zwei = yolo_zu_text(mittig + oben)
    assert zwei.count(".") == 2, zwei
    assert whisper_zu_text({"segments": [{"text": "Hallo Welt."},
                                         {"text": "Zweiter Satz."}]}) \
        == "Hallo Welt. Zweiter Satz."
    assert whisper_zu_text({"text": "  nur text  "}) == "nur text"
    assert whisper_zu_text({}) == ""
    assert whisper_zu_text(None) == ""
    assert whisper_zu_text({"segments": []}) == ""
    assert whisper_zu_text({"segments": [{"text": "  "}]}) == ""
    print("uebersetzer ok")
