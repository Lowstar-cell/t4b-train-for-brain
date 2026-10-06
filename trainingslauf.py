"""Trainingslauf: Daten ernten, Schwarm fuellen, Megatron trainieren.

Alles mit Zeiten. Erntet alle Themen aus data/themen.jsonl (Wikipedia,
Einleitung + Volltext, 30 KB/Artikel), legt alles im Schwarm ab (SQLite-
Volltextindex, max. 10 GB, darueber wird ausgeduennt), trainiert Megatron
mehrfach ueber eine Runde NEUER Texte (Imitation: Uebergange + Hebb-
Verdrahtung) und laesst den Benchmark laufen. Laufen: python3 trainingslauf.py
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).parent
sys.path.insert(0, str(WURZEL))

from brain_core_v2 import BrainCoreV2, LEHRER, lehrer_fragen
import specialists.harvester as harvester_mod
from specialists.flash_swarm import GRENZE
from wiki_sammler import RUNDE, aufraeumen

ARCHIV = WURZEL / "data" / "harvest.jsonl"
# Kennungen der Texte, die Megatron schon gelernt hat (eine je Zeile).
GELERNT = WURZEL / "data" / "gelernt.txt"
DURCHLAEUFE = 5
MAX_KB = 30
# Ab so vielen Wortfolgen vergisst Megatron die nur einmal gesehenen
# (Zaehler <= DURCHLAEUFE). Pi5 4 GB: 2,2 Mio. ~ 1,5 GB im Trainingsprozess.
MAX_FOLGEN = 3_000_000
# Fortschritt je Zyklus: Lehrernote + Worttreffer, eine Zeile je Zyklus.
FORTSCHRITT = WURZEL / "data" / "fortschritt.jsonl"
# Feste Pruefungsfragen - in jedem Zyklus gleich, damit die Note vergleichbar ist.
FRAGEN = ("Wie lernt ein Modell?", "Was ist ein Netzwerk?", "Was ist Gedächtnis?",
          "Was ist ein Transistor?", "Wozu dient eine Firewall?", "Was macht das Herz?",
          "Was ist ein Betriebssystem?", "Wie funktioniert Verschlüsselung?",
          "Was ist ein Mikrocontroller?", "Was macht das Immunsystem?")
PRUEFUNG = ("Du bewertest ein kleines Sprachmodell. Gib jeder Antwort eine Note von 0 bis 10: "
            "0 = Wortsalat ohne Bezug zur Frage, 5 = verstaendliche Saetze mit Bezug zur Frage, "
            "10 = richtig, klar und vollstaendig. Antworte NUR mit den %d Zahlen, durch Komma "
            "getrennt, ohne Nummerierung. Die Antworten: %s")


def zeit(t0):
    return "%.1fs" % (time.time() - t0)


def lehrernote(paare):
    """Der Lehrer benotet alle Antworten in einem Aufruf, 0 (Gruetze) bis 10.
    Gibt (noten, rohtext) zurueck - noten leer, wenn er nicht sauber antwortet."""
    # Einzeilig: opencode.cmd (Windows) schneidet am ersten Zeilenumbruch ab.
    liste = " ".join("(%d) Frage: %s Antwort: %s" % (i, f, (a or "(nichts)").replace("\n", " "))
                     for i, (f, a) in enumerate(paare, 1))
    try:
        text = lehrer_fragen(PRUEFUNG % (len(paare), liste))
    except Exception as e:
        return [], str(e)
    noten = [min(10, int(z)) for z in re.findall(r"\d+", text)]
    return (noten if len(noten) == len(paare) else []), text


def themen_lesen():
    themen = []
    with (WURZEL / "data" / "themen.jsonl").open("r", encoding="utf-8") as f:
        for zeile in f:
            try:
                themen.append(json.loads(zeile)["thema"])
            except (ValueError, KeyError):
                pass
    return themen


def main():
    brain = BrainCoreV2()

    # ---- Phase 1: Ernten ----
    # Mehr Text pro Artikel: 30 KB statt 2 KB - hoehere Grundintelligenz.
    harvester_mod.MAX_TEXT = MAX_KB * 1000
    # Arbeitsweisen aus der agent-memory zuerst - direkt, ohne Netz.
    t0 = time.time()
    print("== Phase 1a: Arbeitsweisen ernten (agent-memory)")
    print("   %s, %s" % (brain.harvester.learn_arbeitsweisen(), zeit(t0)))
    themen = themen_lesen()
    print("== Phase 1: Ernten (%d Themen, %d KB/Artikel)" % (len(themen), MAX_KB))
    t0 = time.time()
    gelernt = 0
    for n, thema in enumerate(themen, 1):
        m1 = brain.harvester.learn_wikipedia(thema)
        if m1.startswith("Gelernt"):
            gelernt += 1
        m2 = brain.harvester.learn_wikipedia(thema, ganz=True)
        time.sleep(2)  # Wikipedia bremst schnelle Volltext-Abrufe (429)
        if m2.startswith("Gelernt"):
            gelernt += 1
        print("   %2d/%d %-30s %s" % (n, len(themen), thema[:30], zeit(t0)))
    print("   Ernten fertig: %d neue Quellen, %s | Archiv: %.1f MB"
          % (gelernt, zeit(t0), ARCHIV.stat().st_size / 1048576))

    # ---- Phase 2: Schwarm vektorisieren (Embedding -> Vector-DB) ----
    print("== Phase 2: Schwarm (Volltextindex, max. %d GB)" % (GRENZE // 1024 ** 3))
    t0 = time.time()
    schwarm = brain.flash_swarm
    uebernommen = schwarm.import_archiv(ARCHIV)
    weg = schwarm.ausduennen()
    print("   Schwarm: %d neue Eintraege, %d am wenigsten genutzte geloescht, %s"
          " | Bestand: %d, %.2f GB" % (uebernommen, weg, zeit(t0), len(schwarm),
                                       schwarm.belegt() / 1024 ** 3))

    # ---- Phase 3: Distillation (LM-Studio-Lehrer -> Archiv + Keim) ----
    print("== Phase 3: Distillation (%d Themen, Lehrer: %s)" % (len(themen), LEHRER))
    t0 = time.time()
    distilliert = 0
    for n, thema in enumerate(themen, 1):
        m = brain.distillieren(thema)
        if m.startswith("Distilliert"):
            distilliert += 1
        print("   %2d/%d %-30s %s | %s"
              % (n, len(themen), thema[:30], zeit(t0), m[:70]))
    print("   Distillation fertig: %d neue Lehrer-Antworten, %s"
          % (distilliert, zeit(t0)))

    # ---- Phase 4: Training (Imitation: Uebergange + Verdrahtung) ----
    print("== Phase 4: Training (%d Durchlaeufe ueber die neuen Texte)" % DURCHLAEUFE)
    texte = []
    with ARCHIV.open("r", encoding="utf-8", errors="replace") as f:
        for zeile in f:
            try:
                eintrag = json.loads(zeile)
            except ValueError:
                continue
            # Arbeitsweisen (agent-memory) sind Befehle und Code - die bleiben
            # im Schwarm zum Nachschlagen, Sprache lernt Megatron aus Fliesstext.
            if eintrag.get("art") != "arbeitsweise":
                texte.append(eintrag.get("text", ""))
    # Jeden Tag wird neu geerntet - gleicher Artikel = gleicher Text, nur einmal lernen.
    texte = list(dict.fromkeys(texte))
    # Nur Neues lernen: Gelerntes steckt schon in den Zaehlern, nochmal
    # zaehlen aendert die Verhaeltnisse nicht - kostet nur Stunden.
    gelernt = set(GELERNT.read_text().split()) if GELERNT.exists() else set()
    kennung = {t: hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texte}
    gesamt = len(texte)
    texte = [t for t in texte if kennung[t] not in gelernt]
    neu = len(texte)
    texte = texte[:RUNDE]  # eine Runde je Zyklus, der Rest ist Vorrat
    print("   Material: %d neue von %d Texten, diese Runde %d (%.0f KB)"
          % (neu, gesamt, len(texte), sum(len(t) for t in texte) / 1024))
    if not texte:
        print("   Nichts Neues - Training uebersprungen.")
    worttreffer = None
    if texte:
        # Vor dem Lernen: wie gut sagt Megatron Text voraus, den er noch nie sah?
        t0 = time.time()
        worttreffer, woerter = brain.keim.megatron.worttreffer(texte)
        print("   Worttreffer: %.1f %% von %d Woertern neuem Text (vor dem Lernen), %s"
              % (100 * worttreffer, woerter, zeit(t0)))
    for durchlauf in range(1, DURCHLAEUFE + 1 if texte else 0):
        t0 = time.time()
        verluste = []
        for text in texte:
            v = brain.keim.megatron.lernen_text(text)
            if v:
                verluste.append(v[-1])
        mittel = sum(verluste) / max(1, len(verluste))
        print("   Durchlauf %d: Loss %.3f, %s | Erregung %.4f"
              % (durchlauf, mittel, zeit(t0), brain.keim.megatron.erregung()))
        if durchlauf % 2 == 0 or durchlauf == DURCHLAEUFE:
            pfad = brain.keim.megatron.speichern()
            print("      Zustand gesichert: %.1f MB"
                  % (pfad.stat().st_size / 1048576))
    with GELERNT.open("a", encoding="utf-8") as f:
        f.writelines(kennung[t] + "\n" for t in texte)
    # Gelerntes steckt in Megatron und im Schwarm - Rohtext raus aus dem Archiv.
    print("   Archiv: %d gelernte Texte geloescht"
          % aufraeumen(gelernt | {kennung[t] for t in texte}))
    megatron = brain.keim.megatron
    if megatron.eintraege() > MAX_FOLGEN:
        weg = megatron.vergessen(DURCHLAEUFE)
        pfad = megatron.speichern()
        print("   Vergessen: %d einmal gesehene Wortfolgen geloescht, %d bleiben, %.1f MB"
              % (weg, megatron.eintraege(), pfad.stat().st_size / 1048576))

    # ---- Phase 5: Benchmark (Lernzyklus mit Bestrafung) ----
    print("== Phase 5: Benchmark (3 Durchlaeufe)")
    t0 = time.time()
    erg = brain.lernen_zyklus(3)
    print("   %s | %s" % (erg, zeit(t0)))

    # ---- Pruefung: Megatron antwortet, der Lehrer benotet ----
    print("== Chat")
    paare = []
    for frage in FRAGEN:
        t0 = time.time()
        antwort = brain.keim.generate(frage)
        paare.append((frage, antwort))
        print("   %s -> %s (%s)" % (frage, antwort[:150], zeit(t0)))
    t0 = time.time()
    noten, roh = lehrernote(paare)
    note = sum(noten) / len(noten) if noten else None
    if noten:
        print("   Lehrernote: %.1f / 10 | einzeln %s, %s" % (note, noten, zeit(t0)))
    else:
        print("   Lehrernote: keine - Lehrer sagte: %s" % roh[:150].replace("\n", " "))
    with FORTSCHRITT.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"zeit": time.strftime("%Y-%m-%d %H:%M"), "note": note,
                            "noten": noten, "worttreffer": worttreffer}) + "\n")

    brain.keim.megatron.lora_speichern()
    print("== Fertig: Licht an")


if __name__ == "__main__":
    main()
