"""
Megatron: der nicht-Transformer-Kern von Train4brain.

Keine Attention, kein QKV, keine Positional-Embeddings, kein fremdes
Transformer-Netz. Stattdessen:

- Markov-Kern: Uebergangswahrscheinlichkeiten zwischen Woertern.
  Wissen lebt hier UND im FlashSwarm (Abruf), ins Gewicht kommt nur Verhalten.
- Analog-Vollverbund: N Nodes, jede mit ALLEM verbunden. Signale sind keine
  1/0 sondern eine Bandbreite (-1..1) - denken = Propagation, wie
  Superposition: jeder Node empfaengt von allen, entlaedt weich.
- Parameter = die internen Verbindungen (Gewichte N x N). Je mehr Nodes,
  desto breiter das Aktivierungsprofil - denken/reasoning als interner Loop.
- Vertrauen-Schicht (LoRA-artig): nur Delta-Gewichte aus Feedback, austauschbar.
"""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

ZUSTAND = Path(__file__).parent / "data" / "megatron_zustand.json"
LORA = Path(__file__).parent / "data" / "megatron_vertrauen.json"
SATZZEICHEN = ".,!?;:"
# Wie weit eine Verbindung pro Treffer zum neuen Muster wandert.
# Klein = lange Erinnerung (Mittel ueber ~1/LERNRATE Treffer).
LERNRATE = 0.02
# Wie stark die Frage die Wortwahl zieht (Reaktion in Streuungen -> Faktor).
VERSTAERKER = 0.5


def tokenisieren(text):
    """Worte + Satzzeichen getrennt, kleingeschrieben. Zahlen bleiben ganz."""
    tokens = []
    wort = ""
    for z in text:
        if z.isspace():
            if wort:
                tokens.append(wort.lower())
                wort = ""
        elif z in SATZZEICHEN:
            if wort:
                tokens.append(wort.lower())
                wort = ""
            tokens.append(z)
        else:
            wort += z
    if wort:
        tokens.append(wort.lower())
    return tokens


class Megatron:
    """Nicht-Transformer: Markov-Kern + analoger Vollverbund-Node-Graph.

    Die Nodes leben als Signalfeld (signals[i] = analoges Signal der Node i
    im Band (-1..1), kein diskretes 1/0). Die Verdrahtung ist ein
    numpy-Matmul - analoges Computing auf der CPU, schnell genug fuer
    Masse-Training."""

    def __init__(self, groesse=1024, band=(-1.0, 1.0)):
        self.groesse = groesse
        self.band = band
        self.signals = np.zeros(groesse, dtype=np.float64)
        # Parameter: interne Verbindungen. Jede Node mit ALLEM verbunden.
        self.gewichte = np.random.uniform(-0.03, 0.03, (groesse, groesse))
        # Markov-Kern: {wort: {folge: zaehler}}
        self.markov = {}
        # 2 Woerter Rueckblick: {"wort1 wort2": {folge: zaehler}}
        self.markov2 = {}
        # Satzanfaenge: {wort: zaehler}
        self.anfaenge = {}
        # LoRA-artige Vertrauen-Schicht: {wort: {folge: delta}}
        self.vertrauen = {}

    # ---------- Signale ----------
    def _node_index(self, token):
        """Stabiler Hash eines Tokens -> Node-Index."""
        h = hashlib.sha1(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.groesse

    def _einpraegen(self, token, staerke=1.0):
        """Token als analoges Signal einpragen, Band einhalten."""
        i = self._node_index(token)
        self.signals[i] = min(self.band[1], max(self.band[0],
                                                self.signals[i] + staerke))

    def _propagieren(self):
        """Ein Denkschritt: jeder Node empfaengt von ALLEN (Matmul),
        entlaedt linear im Band (kein tanh - das nuellierte die
        Unterschiede weg).
        Beruhigen: was ALLE Nodes gleich bekommen, hemmt sich weg (nur der
        Unterschied zaehlt), und schiesst eine Node uebers Band, wird das
        ganze Feld heruntergeregelt statt abgeschnitten - das Muster bleibt,
        nicht alles steht auf Anschlag."""
        summe = self.gewichte @ self.signals / math.sqrt(self.groesse)
        summe -= summe.mean()
        neu = self.signals * 0.6 + summe * 0.5
        spitze = np.abs(neu).max()
        if spitze > self.band[1]:
            neu /= spitze
        self.signals = neu

    def _aktivierung(self, token):
        """Aktuelles Signal des Tokens (0, wenn unbeteiligt)."""
        return float(self.signals[self._node_index(token)])

    def _frage_reaktion(self, quelle_tokens):
        """Direkte Frage-Verdrahtung: nur die Frage-Knoten schicken Signal,
        jede Node bekommt ihre Reaktion. Der globale Rausch des Vollverbunds
        zaehlt nicht - sonst versinkt der Frage-Unterschied.
        Beruhigt: gezaehlt wird nur, wie weit eine Node ueber dem
        Durchschnitt liegt (in Streuungen) - was ALLE bekommen, hebt keine
        Antwort heraus."""
        idx = sorted({self._node_index(k) for k in quelle_tokens})
        if not idx:
            return np.zeros(self.groesse)
        v = self.gewichte[idx, :].T @ self.signals[idx]
        v[idx] -= self.gewichte[idx, idx] * self.signals[idx]
        return (v - v.mean()) / (v.std() + 1e-9)

    def erregung(self):
        """Wie viel Licht die Nodes gerade tragen - fuer Dashboard/Anzeige."""
        return round(float(np.abs(self.signals).mean()), 4)

    def eintraege(self):
        """Gelernte Wortfolgen (1- und 2-Wort-Liste) - daran haengt der RAM."""
        return (sum(map(len, self.markov.values()))
                + sum(map(len, self.markov2.values())))

    def vergessen(self, grenze):
        """Wortfolgen mit Zaehler <= grenze loeschen (nur einmal gesehen),
        leere Eintraege mit. Gibt die Zahl der geloeschten Folgen zurueck."""
        # ponytail: Seltenes, das erst ueber mehrere Runden wiederkommt, fliegt
        # auch raus - Zaehler halbieren statt loeschen, wenn das spuerbar fehlt.
        vorher = self.eintraege()
        for tabelle in (self.markov, self.markov2):
            for kontext in list(tabelle):
                folge = {w: c for w, c in tabelle[kontext].items() if c > grenze}
                if folge:
                    tabelle[kontext] = folge
                else:
                    del tabelle[kontext]
        return vorher - self.eintraege()

    def worttreffer(self, texte):
        """Anteil der Woerter, die Megatron in Text, den er noch nicht kennt,
        richtig vorhersagt: wahrscheinlichste Folge aus der 2-Wort-, sonst
        der 1-Wort-Liste (wie beim Sprechen). Lernt dabei nichts.
        Gibt (anteil, woerter) zurueck."""
        beste = {}

        def tipp(tabelle, kontext):
            if kontext not in beste:
                folge = tabelle.get(kontext)
                beste[kontext] = max(folge, key=folge.get) if folge else None
            return beste[kontext]

        richtig = gesamt = 0
        for text in texte:
            tokens = tokenisieren(text)
            for i in range(1, len(tokens)):
                t = ((i > 1 and tipp(self.markov2, tokens[i - 2] + " " + tokens[i - 1]))
                     or tipp(self.markov, tokens[i - 1]))
                richtig += t == tokens[i]
                gesamt += 1
        return richtig / max(1, gesamt), gesamt

    # ---------- Lernen ----------
    def lernen_text(self, text, knoten=True):
        """Imitation: Markov-Uebergange zaehlen + aktive Verbindungen staerken.
        Rueckgabe: mittlere Ueberraschung der Uebergange (Loss)."""
        tokens = tokenisieren(text)
        if len(tokens) < 2:
            return []
        self.anfaenge[tokens[0]] = self.anfaenge.get(tokens[0], 0) + 1
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            folge2 = self.markov2.setdefault(a + " " + b, {})
            folge2[c] = folge2.get(c, 0) + 1
        ueberraschung = 0.0
        for a, b in zip(tokens, tokens[1:]):
            folge = self.markov.setdefault(a, {})
            folge[b] = folge.get(b, 0) + 1
            summe = sum(folge.values())
            p = folge[b] / summe
            ueberraschung += -math.log2(max(p, 1e-9))
            if knoten:
                self._einpraegen(a, 0.6)
                self._einpraegen(b, 0.6)
                # Hebb mit Vergessen: die Verbindungen ZUM Ziel-Wort wandern
                # ein Stueck zum aktuellen Signalmuster - was mit b feuert,
                # wird staerker, was nicht mitfeuert, wird wieder schwaecher.
                # Bleibt von selbst im Band (-0.5..0.5), kein Anschlag.
                # Selbstverbindung bleibt: Reflexion ist kein Lernen.
                j_b = self._node_index(b)
                spalte = self.gewichte[:, j_b]
                ziel = 0.5 * self.signals
                ziel[j_b] = spalte[j_b]
                spalte += LERNRATE * (ziel - spalte)
                self._propagieren()
        return [round(ueberraschung / (len(tokens) - 1), 4)]

    def lernen_paar(self, gut, schlecht, schritt=0.4):
        """DPO-artig: Uebergange aus gut heben, aus schlecht senken.
        Schreibt nur die Delta-Schicht (vertrauen) - austauschbar wie ein LoRA."""
        for wort, folge, faktor in (
            _uebergangspaare(gut, schritt),
            _uebergangspaare(schlecht, -schritt),
        ):
            ziel = self.vertrauen.setdefault(wort, {})
            for f, d in folge.items():
                ziel[f] = max(-2.0, min(2.0, ziel.get(f, 0.0) + d * faktor))

    def lernen_belohnung(self, gut, schlecht):
        """DPO-artig: gute Uebergange heben (Vertrauen), schlechte aus der
        Verdrahtung ABKOPPELN - die Frage-Kontexte verlieren die Verbin-
        dung zur falschen Antwort, die richtige bleibt verdrahtet."""
        self.lernen_paar(gut, schlecht)
        # Bestrafung direkt im Markov-Kern: die Zaehler der falschen
        # Uebergange halbieren - die falschen Antworten verschwinden
        # aus der Wahl, die richtigen bleiben.
        tokens = tokenisieren(schlecht)
        for a, b in zip(tokens, tokens[1:]):
            folge = self.markov.get(a)
            if folge and b in folge:
                folge[b] = max(0, folge[b] // 2)
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            folge = self.markov2.get(a + " " + b)
            if folge and c in folge:
                folge[c] = max(0, folge[c] // 2)
        if len(tokens) < 2:
            return
        for t in tokens:
            self._einpraegen(t, 0.8)
        self._propagieren()
        for a, b in zip(tokens, tokens[1:]):
            j_b = self._node_index(b)
            aktive = np.where(self.signals > 0.05)[0]
            aktive = aktive[aktive != j_b]
            if aktive.size:
                self.gewichte[aktive, j_b] = np.clip(
                    self.gewichte[aktive, j_b] - 0.3 * self.signals[aktive],
                    -0.5, 0.5)

    # ---------- Denken + Sprechen ----------
    def denken(self, text, runden=3, erinnerung=True):
        """Interner Loop (thinking/reasoning): Prompt einpragen, runden x
        propagieren. Die Frage-Signale werden nach jeder Runde aufgefrisiert
        (kontinuierliche Eingangskopplung) - die Frage bleibt klar und
        positiv im Band, die Propagation kann sie nicht negativ druecken.
        FlashSwarm (Embedding-Abruf) ist nativ verdrahtet: gefundene
        Erinnerungen werden mit eingepraegt."""
        prompt_tokens = tokenisieren(text)
        for t in prompt_tokens:
            self._einpraegen(t, 0.8)
        if erinnerung:
            try:
                from specialists.flash_swarm import swarm_search
                for treffer in swarm_search(text, limit=3):
                    for t in tokenisieren(treffer.get("text", "")):
                        self._einpraegen(t, 0.2)
            except Exception:
                pass
        for _ in range(max(1, runden)):
            self._propagieren()
            for t in prompt_tokens:
                self._einpraegen(t, 0.5)

    def _start_wort(self, quelle_tokens, temperatur=0.6):
        """Satzanfang aus den gelernten Anfaengen - Kontext-Reaktion traegt."""
        if not self.anfaenge:
            return None
        folge = self.anfaenge
        summe = sum(folge.values())
        punkte = []
        schluessel = list(folge.keys())
        reaktion = self._frage_reaktion(quelle_tokens or [])
        for f in schluessel:
            p = folge[f] / summe
            akt = max(0.0, reaktion[self._node_index(f)])
            punkte.append(max(1e-9, p * math.exp(VERSTAERKER * akt)))
        if temperatur <= 0:
            return schluessel[punkte.index(max(punkte))]
        logits = [math.log(p) / temperatur for p in punkte]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        zufall = random.random() * sum(exps)
        for f, e in zip(schluessel, exps):
            zufall -= e
            if zufall <= 0:
                return f
        return schluessel[-1]

    def _uebergang(self, wort, quelle_tokens=None, temperatur=0.6, vorher=None):
        """Naechstes Wort aus der Verteilung - Superposition, keine harte Wahl.
        Aktivierung + Vertrauen modulieren die Wahrscheinlichkeiten.
        Mit vorher: erst die 2-Wort-Liste, nur wenn sie nichts kennt die 1-Wort-Liste."""
        folge = (vorher is not None and self.markov2.get(vorher + " " + wort)) or self.markov.get(wort) or (
            self.markov.get(None) if wort is None else None)
        if not folge:
            return None
        vertrauen = self.vertrauen.get(wort, {})
        punkte = []
        reaktion = self._frage_reaktion(quelle_tokens or [])
        for f, zaehler in folge.items():
            p = zaehler / sum(folge.values())
            # Analog-Verstaerker: die Kontext-Reaktion (Frage -> Kandidat)
            # wird exponentiell eingearbeitet. Der Kontext dominiert,
            # das Sampling bleibt nur als analoge Streuung.
            akt = max(0.0, reaktion[self._node_index(f)])
            punkte.append(max(1e-9, p * math.exp(VERSTAERKER * akt) + vertrauen.get(f, 0.0)))
        if temperatur <= 0:
            return list(folge.keys())[punkte.index(max(punkte))]
        logits = [math.log(p) / temperatur for p in punkte]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        zufall = random.random() * sum(exps)
        for f, e in zip(folge.keys(), exps):
            zufall -= e
            if zufall <= 0:
                return f
        return list(folge.keys())[-1]

    def generieren(self, prompt, max_woerter=40):
        """Sprechen: denken(), dann Markov-Uebergange. Satzzeichen bleiben."""
        self.denken(prompt)
        # Satzzeichen am Fragenende weg: weiter geht es ab dem letzten Wort.
        tokens = [t for t in tokenisieren(prompt) if t not in SATZZEICHEN]
        kontext = list(dict.fromkeys(tokens))
        wort = tokens[-1] if tokens else None
        vorher = tokens[-2] if len(tokens) > 1 else None
        if wort not in self.markov:
            wort, vorher = self._start_wort(kontext), None
            if wort is None:
                return ""
        ausgabe = []
        for _ in range(max(1, max_woerter)):
            naechstes = self._uebergang(wort, kontext, vorher=vorher)
            if naechstes is None or naechstes == wort:
                break
            ausgabe.append(naechstes)
            vorher, wort = wort, naechstes
        text = " ".join(ausgabe)
        try:
            from specialists.deutsch_sprache import nachputzen
            text = nachputzen(text)
        except Exception:
            pass
        return text

    # ---------- Speichern / Laden ----------
    def speichern(self, pfad=None):
        pfad = Path(pfad) if pfad else ZUSTAND
        pfad.parent.mkdir(parents=True, exist_ok=True)
        daten = {
            "groesse": self.groesse,
            "band": list(self.band),
            "anfaenge": self.anfaenge,
            "markov": self.markov,
            "markov2": self.markov2,
            "gewichte": np.round(self.gewichte, 6).tolist(),
        }
        tmp = pfad.with_suffix(pfad.suffix + ".tmp")
        tmp.write_text(json.dumps(daten), encoding="utf-8")
        tmp.replace(pfad)
        return pfad

    def laden(self, pfad=None):
        pfad = Path(pfad) if pfad else ZUSTAND
        if not pfad.exists():
            return False
        try:
            daten = json.loads(pfad.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        try:
            groesse = int(daten["groesse"])
            band = tuple(daten["band"])
            markov = daten["markov"]
            gewichte = daten["gewichte"]
            anfaenge = daten.get("anfaenge", {})
        except (KeyError, TypeError, ValueError):
            return False
        if len(gewichte) != groesse or any(len(r) != groesse for r in gewichte):
            return False
        self.groesse = groesse
        self.band = band
        self.signals = np.zeros(groesse, dtype=np.float64)
        self.markov = markov
        self.markov2 = daten.get("markov2", {})
        self.anfaenge = anfaenge
        self.gewichte = np.array([[float(w) for w in row] for row in gewichte],
                                 dtype=np.float64)
        return True

    def lora_speichern(self, pfad=None):
        pfad = Path(pfad) if pfad else LORA
        pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = pfad.with_suffix(pfad.suffix + ".tmp")
        tmp.write_text(json.dumps(self.vertrauen), encoding="utf-8")
        tmp.replace(pfad)
        return pfad

    def lora_laden(self, pfad=None):
        pfad = Path(pfad) if pfad else LORA
        if not pfad.exists():
            return False
        try:
            self.vertrauen = json.loads(pfad.read_text(encoding="utf-8"))
            return True
        except (ValueError, OSError):
            return False


def _uebergangspaare(text, schritt):
    """Uebergange eines Texts mit ihrer relativen Haeufigkeit als Paare."""
    tokens = tokenisieren(text)
    if len(tokens) < 2:
        return None, {}, schritt
    folge = {}
    for a, b in zip(tokens, tokens[1:]):
        if a == tokens[0]:
            folge[b] = folge.get(b, 0) + 1
    n = max(1, sum(folge.values()))
    return tokens[0], {f: round(c / n, 4) for f, c in folge.items()}, schritt


class MegatronExpert:
    """Train4brain-Kern. Gleiche Schnittstelle wie AtomicGPTExpert
    (train_on_text/generate), aber kein Transformer: Markov-Kern +
    analoger Vollverbund. Whisper/Yolo/SAM3-Sensorik lazily verdrahtet."""

    def __init__(self, data_path=None):
        print("[Megatron] Initialisiere nicht-Transformer (Markov + Vollverbund)...")
        self.megatron = Megatron()
        geladen = self.megatron.laden()
        self.megatron.lora_laden()
        if geladen:
            print("[Megatron] Zustand geladen: %d Uebergangs-Quellen, %d Nodes."
                  % (len(self.megatron.markov), self.megatron.groesse))

    def train_on_text(self, text, steps=1):
        """Kompatibel zu AtomicGPTExpert: Rueckgabe = Verlustliste (floats).
        steps = wie oft der Text geimpft wird."""
        verluste = []
        for _ in range(max(1, int(steps))):
            verlust = self.megatron.lernen_text(text)
            if not verlust:
                break
            verluste.append(verlust[-1])
        if verluste:
            self.megatron.speichern()
        return verluste

    def generate(self, prompt="", max_woerter=40):
        return self.megatron.generieren(prompt, max_woerter=max_woerter)

    def aufnahme_einlesen(self, pfad):
        """Sensorik der Schnittstelle: Audio -> Whisper, Bild -> Yolo,
        Rueckgabe = Text. Fehlt ein Paket, wird nur der Textkanal genutzt.
        SAM3: Hook bleibt, kommt spaeter."""
        pfad = Path(pfad)
        if not pfad.exists():
            return ""
        suffix = pfad.suffix.lower()
        if suffix in (".wav", ".mp3", ".m4a", ".flac"):
            try:
                import whisper
                modell = whisper.load_model("base")
                erg = modell.transcribe(str(pfad))
                from specialists.uebersetzer import whisper_zu_text
                return whisper_zu_text(erg)
            except Exception:
                return ""
        if suffix in (".jpg", ".jpeg", ".png", ".bmp"):
            try:
                from ultralytics import YOLO
                modell = YOLO("yolov8n.pt")
                erg = modell(str(pfad))
                from specialists.uebersetzer import yolo_zu_text
                return yolo_zu_text([r.boxes for r in erg])
            except Exception:
                return ""
        return ""


if __name__ == "__main__":
    expert = MegatronExpert()
    test_text = "Hallo Welt. Das ist ein Test. Der Test ist kurz."
    verluste = expert.train_on_text(test_text, steps=3)
    print("Verlust:", verluste[0] if verluste else None,
          "->", verluste[-1] if verluste else None)
    print("Erregung:", expert.megatron.erregung())
    print("Generated:", expert.generate("Hallo"))
    expert.megatron.lernen_paar("hallo welt", "hallo test")
    expert.megatron.lora_speichern()
    print("Speichern:", expert.megatron.speichern())
