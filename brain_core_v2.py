import os, sys, json, time, threading, subprocess, atexit, signal, requests, math
import shutil, tempfile
import numpy as np
from pathlib import Path

T4BV1 = str(Path(__file__).parent.parent / "T4Bv1")
if T4BV1 not in sys.path:
    sys.path.insert(0, T4BV1)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
LLAMA_BIN = Path(T4BV1) / "llama-bin"
MODEL_DIR = Path(__file__).parent / "Modelle" / "Qwen3.6-35B-A3B-GGUF"
MODEL_PATH = MODEL_DIR / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
MMPROJ_PATH = MODEL_DIR / "mmproj-Qwen3.6-35B-A3B-BF16.gguf"
LLAMA_PORT = 8188
OAI_URL = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
# Lehrer: kostenloses OpenCode-Modell. space-bunny-free zieht nichts vom
# Gratis-Kontingent (das teilen sich mehrere Projekte) - kein anderes nehmen.
LEHRER = "opencode/space-bunny-free"

from specialists.flash_swarm import FlashSwarm
from specialists.harvester import KnowledgeHarvester, saeubern
from specialists.synapse_memory import SynapseMemory
from specialists.semantic_router import SemanticRouter
from specialists.eloquence_expert import EloquenceExpert
from specialists.tool_belt import ToolBelt
from specialists.sammler import Sammler
from specialists.benchmark import (datensatz_lesen, benchmark_laufen,
                                   aufgabe_bewerten, korrektur_text)
from supervisor import LoopWaechter, LoopVerbrochen
from specialists import mcp_tools
from megatron import MegatronExpert


def lehrer_fragen(frage, modell=LEHRER, timeout=180):
    """Eine Frage an das OpenCode-Modell (opencode run, Agent plan = darf
    nichts aendern). Gibt den Antworttext zurueck, leer wenn nichts kam."""
    exe = shutil.which("opencode.cmd" if os.name == "nt" else "opencode") or str(Path.home() / ".opencode" / "bin" / "opencode")
    erg = subprocess.run([exe, "run", "--agent", "plan", "--format", "json",
                          "-m", modell, frage],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=timeout,
                         cwd=tempfile.gettempdir())
    teile = []
    for zeile in erg.stdout.splitlines():
        try:
            d = json.loads(zeile)
        except ValueError:
            continue
        if d.get("type") == "text":
            teile.append((d.get("part") or {}).get("text", ""))
    return "\n".join(teile).strip()


def token_geschaetzt(text):
    return len(text) // 4


def verlauf_einpassen(verlauf, grenze=1000, behalten=4):
    """Zu langer Verlauf: die aeltesten Eintraege fliegen raus."""
    verlauf = [dict(m) for m in verlauf]
    while len(verlauf) > behalten and token_geschaetzt(
            "\n".join(str(m.get("content", "")) for m in verlauf)) > grenze:
        verlauf.pop(0)
    return verlauf


AKTIONEN = ("antwort", "terminal", "datei", "web", "lern", "kontext")


def react_aktion(text):
    """Liest die Antwort des Modells und gibt (aktion, argument) zurueck.

    Gesucht wird die erste Zeile der Form "AKTION: argument", wherein
    AKTION in AKTIONEN steht. Zeilen wie "Gedanke: ..." werden uebersprungen.
    Steht keine solche Zeile drin, ist die Antwort selbst die Aktion:
    ("antwort", letzte_nichtleere_zeile). Auch eine leere Eingabe gibt
    ("antwort", "") zurueck - es wirft nichts.
    """
    letzte = ""
    for zeile in str(text).strip().splitlines():
        zeile = zeile.strip()
        if not zeile:
            continue
        letzte = zeile
        if ":" not in zeile:
            continue
        kopf, rest = zeile.split(":", 1)
        kopf = kopf.strip().lower()
        if kopf == "aktion" and ":" in rest:
            kopf2, rest2 = rest.split(":", 1)
            kopf2 = kopf2.strip().lower()
            if kopf2 in AKTIONEN:
                return (kopf2, rest2.strip())
        if kopf in AKTIONEN:
            return (kopf, rest.strip())
    return ("antwort", letzte)


class BrainCoreV2:
    def __init__(self):
        self.llama_proc = None
        self.module_activity = {"swarm": 0, "harvester": 0, "vision": 0,
                                "qwen": 0, "eloquence": 0, "whisper": 0,
                                "keim": 0, "system": 0}
        self.module_log = []
        self.harvester_speedys = {i: {"status": "idle", "progress": 0, "task": ""} for i in range(5)}

        print("[BrainV2] Initialisiere Spezialisten...")
        self.router = SemanticRouter()
        self.synapse = SynapseMemory(memory_file=str(DATA_DIR / "synapse_weights.json"))
        self.flash_swarm = FlashSwarm()
        self.harvester = KnowledgeHarvester()
        self.eloquence = EloquenceExpert()
        self.tools = ToolBelt()
        self.keim = MegatronExpert()
        self.sammler = Sammler(self.ernten)
        self.waechter = LoopWaechter(grenze=3)
        self.verlauf = []
        self.last_module = "system"

        self._llama_started = False
        atexit.register(self._stop_llama)

    def _ensure_llama(self):
        if not self._llama_started:
            self._start_llama()
            self._llama_started = True
        return self._llama_ready()

    def _llama_ready(self):
        try:
            r = requests.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2)
            return r.status_code == 200
        except: return False

    def _start_llama(self):
        if self._llama_ready():
            print(f"[BrainV2] llama-server läuft bereits auf Port {LLAMA_PORT}")
            return
        if not MODEL_PATH.exists():
            print(f"[BrainV2] Modell nicht gefunden: {MODEL_PATH}")
            return
        cmd = [str(LLAMA_BIN / "llama-server.exe"),
               "-m", str(MODEL_PATH),
               "--port", str(LLAMA_PORT),
               "--host", "127.0.0.1",
               "-ngl", "99",
               "-c", "8192",
               "--no-mmap"]
        try:
            self.llama_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[BrainV2] llama-server gestartet (PID {self.llama_proc.pid})")
        except Exception as e:
            print(f"[BrainV2] Fehler beim Start: {e}")

    def _stop_llama(self):
        if self.llama_proc:
            self.llama_proc.kill()
            self.llama_proc.wait(timeout=5)
            print("[BrainV2] llama-server beendet")

    def qwen_chat(self, messages, max_tokens=1024, temperature=0.7):
        if not self._ensure_llama():
            return "Der Gehirnkern (Qwen3.6) ist noch nicht bereit."
        try:
            messages = self.verlauf_einpassen(messages)
            r = requests.post(OAI_URL, json={
                "model": "qwen3.6",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }, timeout=60)
            antwort = r.json()["choices"][0]["message"]["content"]
            self.verlauf.append({"role": "assistant", "content": antwort})
            return antwort
        except Exception as e:
            return f"[Fehler: {e}]"

    def verlauf_einpassen(self, messages):
        """Haelt den Verlauf unter der Token-Grenze, gibt die Liste fuer den Aufruf zurueck."""
        for m in messages:
            if m.get("role") == "user":
                self.verlauf.append(dict(m))
        self.verlauf = verlauf_einpassen(self.verlauf)
        return messages[:1] + self.verlauf[-4:]

    def _set_activity(self, module, value=1.0):
        self.module_activity[module] = value
        self.last_module = module
        ts = time.strftime("%H:%M:%S")
        self.module_log.append((ts, module, value))
        if len(self.module_log) > 200:
            self.module_log = self.module_log[-100:]

    def _decay_activity(self):
        for k in self.module_activity:
            self.module_activity[k] = max(0, self.module_activity[k] * 0.92)

    def process(self, query):
        self._decay_activity()
        try:
            self.waechter.beobachte(query)
        except LoopVerbrochen:
            self.waechter.zuruecksetzen()
            self._set_activity("system")
            return "Abbruch: drei Mal dieselbe Frage. Der Waechter hat gestoppt."
        weights = self.synapse.weights
        route, score = self.router.decide_route(query, weights=weights)
        self._set_activity("qwen")
        self.verlauf.append({"role": "user", "content": query})

        system_msg = {"role": "system", "content":
            "Du bist Train4brain, ein modulares KI-System auf AMD ROCm. "
            "Antworte praezise auf Deutsch. Nutze Kontext wenn vorhanden."}
        reakt_msg = {"role": "system", "content":
            "Du arbeitest in einem ReAct-Loop. Antworte mit GENAU EINER "
            "Zeile in der Form 'AKTION: argument'. Erlaubte Aktionen: "
            "antwort (deine fertige Antwort), terminal (Name eines Befehls: "
            "ipconfig, systeminfo, dir, nvidia-smi), datei (Aktion und Pfad, "
            "z.B. 'liste .' oder 'lesen start.bat'), web (Suchbegriff), "
            "lern (Thema fuer Wikipedia), kontext (Frage fuer den eigenen "
            "Speicher). Keine Erklaerung, kein Codeblock, nur diese eine "
            f"Zeile. Hinweis vom Router: {route}"}

        beobachtung = "noch keine"
        for _runde in range(3):
            roh = self.qwen_chat([system_msg, reakt_msg, {"role": "user",
                "content": f"{query}\n\nBeobachtung: {beobachtung}"}])
            aktion, argument = react_aktion(roh)
            if aktion == "antwort":
                return argument
            beobachtung = self._beobachte(aktion, argument)
        self._set_activity("system")
        return f"Loop nach drei Runden abgebrochen. Letzte Beobachtung: {beobachtung}"

    def _beobachte(self, aktion, argument):
        """Fuehrt eine ReAct-Aktion aus und gibt die Beobachtung als String zurueck."""
        if aktion == "terminal":
            self._set_activity("terminal")
            return str(mcp_tools.terminal_befehl(argument.strip()))
        if aktion == "datei":
            self._set_activity("datei")
            teile = argument.strip().split(None, 1)
            if len(teile) < 2:
                return "datei braucht Aktion und Pfad, z.B. 'liste .'"
            return str(mcp_tools.datei_aktion(teile[0], teile[1].strip().strip('"')))
        if aktion == "web":
            self._set_activity("web")
            return str(mcp_tools.websuche(argument.strip()))
        if aktion == "lern":
            return self.ernten(argument.strip())
        if aktion == "kontext":
            self._set_activity("swarm")
            treffer = self.flash_swarm.swarm_search(argument.strip())
            if not treffer:
                return "Nichts im eigenen Speicher gefunden."
            return " ".join(str(t.get("text", "")) for t in treffer[:3])
        return "Unbekannte Aktion: " + str(aktion)

    def ernten(self, topic):
        """Erntet ein Thema: Archiv, Schwarm (vektorisiert), Keim (1 Step-Training)."""
        self._set_activity("harvester")
        thema = (topic or "").strip()
        if thema.startswith(("http://", "https://")):
            meldungen = [self.harvester.learn_url(thema)]
        else:
            meldungen = [self.harvester.learn_wikipedia(thema)]
        uebernommen = self.flash_swarm.import_archiv(self.harvester.archiv)
        if uebernommen:
            self._set_activity("swarm")
            meldungen.append("Schwarm: %d Eintraege vektorisiert." % uebernommen)
        verluste = self.keim.train_on_text(thema, steps=1)
        if verluste:
            self._set_activity("keim")
            meldungen.append("Keim: Verlust %.3f nach 1 Step." % verluste[-1])
        return " | ".join(meldungen)

    def distillieren(self, thema, modell=LEHRER, max_token=300):
        """Distillation: der OpenCode-Lehrer erklaert das Thema, die Antwort
        landet im Archiv (art: lehrer) und der Keim lernt die Uebergange.
        Der Lehrer-Abruf wird pro Tag nur einmal gemacht (quelle lehrer://)."""
        self._set_activity("qwen")
        thema = (thema or "").strip()
        if not thema:
            return "Kein Thema angegeben."
        quelle = "lehrer://" + thema.replace(" ", "_")
        if self.harvester._gibt_es_schon(quelle):
            return "Schon heute distilliert: " + thema
        try:
            antwort = saeubern(lehrer_fragen(
                "Antworte kurz, sachlich und auf Deutsch in vollstaendigen "
                "Saetzen, ohne Aufzaehlungen und ohne Werkzeuge. Erklaere: %s"
                % thema, modell))
        except Exception as e:
            return "Lehrer nicht erreichbar: %s" % e
        if not antwort:
            return "Leere Antwort zu: " + thema
        self.harvester._ablegen({"quelle": quelle, "art": "lehrer",
                                 "thema": thema, "text": antwort})
        verluste = self.keim.train_on_text("%s %s" % (thema, antwort), steps=1)
        self.keim.megatron.speichern()
        return "Distilliert: %s (%d Zeichen, Loss %.3f)" % (
            thema, len(antwort), verluste[-1] if verluste else -1.0)

    def verstaerken(self, frage, antwort):
        """Bewertet eine Antwort, korrigiert nett und trainiert den Keim.

        Richtig: Lob + Training auf der eigenen Antwort (positive
        Verstaerkung). Knapp daneben: der nette Korrektur-Satz + Training
        in Richtung richtige Antwort.
        """
        aufgaben = {a["frage"]: a for a in datensatz_lesen()}
        aufgabe = aufgaben.get((frage or "").strip())
        if not aufgabe:
            return "Keine Aufgabe zu: " + str(frage)
        treffer, fehlende = aufgabe_bewerten(antwort, aufgabe)
        msg = korrektur_text(treffer, fehlende)
        if not fehlende:
            self.keim.train_on_text("%s %s" % (frage.strip(),
                                               (antwort or "").strip()), steps=1)
        else:
            self.keim.train_on_text("%s %s." % (frage.strip(),
                                                " ".join(aufgabe["enthaelt"])),
                                    steps=2)
            gut = "%s %s." % (frage.strip(), " ".join(aufgabe["enthaelt"]))
            schlecht = "%s %s." % (frage.strip(), (antwort or "").strip())
            self.keim.megatron.lernen_belohnung(gut, schlecht)
        return msg

    def lernen_zyklus(self, durchlaeufe=3):
        """Lernzyklus: Keim beantwortet die Benchmark-Aufgaben, verstaerken
        korrigiert nett und trainiert nach. Gibt die Quote je Durchlauf zurueck."""
        aufgaben = datensatz_lesen()
        if not aufgaben:
            return "Keine Benchmark-Aufgaben gefunden."
        quoten = []
        for _durchlauf in range(max(1, int(durchlaeufe))):
            antworten = {}
            for aufgabe in aufgaben:
                antwort = self.keim.generate(aufgabe["frage"])
                antworten[aufgabe["frage"]] = antwort
                self.verstaerken(aufgabe["frage"], antwort)
            erg = benchmark_laufen(antworten)
            quoten.append(erg["quote"])
        return "Lernzyklus: Quote je Durchlauf %s" % " -> ".join(
            str(q) for q in quoten)

    def give_feedback(self, liked):
        self.waechter.zuruecksetzen()
        msg = self.synapse.learn_feedback(
            score=100 if liked else 30,
            tool_correct=liked,
            answer_correct=liked)
        if self.last_module == "swarm":
            msg += " | " + self.flash_swarm.distribute_feedback(liked)
        return msg

    def get_dashboard(self):
        return {
            "activity": dict(self.module_activity),
            "weights": dict(self.synapse.weights),
            "last_module": self.last_module,
            "log": list(self.module_log[-20:]),
            "speedys": dict(self.harvester_speedys),
            "sammler": dict(self.sammler.status),
            "llama_ready": self._llama_ready(),
        }
