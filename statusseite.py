"""Statusseite: zeigt im Browser, was der T4Bv2-Trainingszyklus gerade macht.

Liest nur die Logs in data/, steuert nichts und laedt kein Gehirn.
Laeuft auf dem Pi5 als Dienst t4b-ui (systemctl --user), Adresse
http://<PI_IP>:8088 - Seite aktualisiert sich alle 10 s selbst.
Pruefen ohne Server: python3 statusseite.py pruefen
"""
import html
import json
import os
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trainingszyklus import FEHLER_PAUSE, PAUSE

WURZEL = Path(__file__).parent
DATA = WURZEL / "data"
PORT = 8088
# Letzte Zeile mit diesem Anfang im Ausgabe-Log = letztes Ergebnis der Phase.
ERGEBNISSE = {
    "Ernten": "Ernten fertig:",
    "Schwarm": "Schwarm:",
    "Lehrer": "Distillation fertig:",
    "Material": "Material:",
    "Training": "Durchlauf ",
    "Aufräumen": "Archiv:",
    "Vergessen": "Vergessen:",
    "Benchmark": "Lernzyklus:",
}


def ende(pfad, kb=256):
    """Letzte kb Kilobyte einer Datei als Zeilen - die Logs wachsen endlos."""
    try:
        with pfad.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - kb * 1024))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def dienst(name):
    return subprocess.run(["systemctl", "--user", "is-active", name],
                          capture_output=True, text=True).stdout.strip()


def uhr(t):
    return time.strftime("%H:%M", time.localtime(t))


def dauer(s):
    s = int(max(0, s))
    return "%d h %02d min" % (s // 3600, s % 3600 // 60)


def stand():
    zyklus = ende(DATA / "trainingszyklus.log")
    aus = ende(DATA / "trainingszyklus_ausgabe.log")
    letzte = zyklus[-1] if zyklus else ""
    seit = time.mktime(time.strptime(letzte[:19], "%Y-%m-%d %H:%M:%S")) if letzte else 0
    phase = next((z[3:] for z in reversed(aus) if z.startswith("== ")), "")

    if dienst("t4b-training") != "active":
        art, jetzt = "aus", "Dienst t4b-training läuft nicht"
    elif "startet" in letzte:
        art = "laeuft"
        jetzt = "%s seit %s · %s · %s" % (letzte[23:-8], uhr(seit), phase,
                                          aus[-1].strip() if aus else "")
    elif "fehlgeschlagen" in letzte:
        art, jetzt = "fehler", "%s – neuer Versuch %s" % (letzte[20:], uhr(seit + FEHLER_PAUSE))
    else:
        naechst = seit + PAUSE
        art, jetzt = "pause", "nächster Zyklus %s (in %s)" % (
            uhr(naechst), dauer(naechst - time.time()))

    # Antworten aus dem letzten "== Chat"-Block: "   Frage -> Antwort (0.4s)"
    i = max((n for n, z in enumerate(aus) if z.startswith("== Chat")), default=len(aus))
    chat = []
    for z in aus[i + 1:]:
        if z.startswith("=="):
            break
        if " -> " not in z:
            continue
        frage, _, antwort = z.strip().partition(" -> ")
        chat.append((frage, antwort.rsplit(" (", 1)[0]))

    erg = {}
    for z in aus:
        z = z.strip()
        for name, anfang in ERGEBNISSE.items():
            if z.startswith(anfang):
                erg[name] = z

    fortschritt = []
    for z in ende(DATA / "fortschritt.jsonl", 8)[-8:]:
        try:
            fortschritt.append(json.loads(z))
        except ValueError:
            pass
    mem = dict(z.split(":", 1) for z in Path("/proc/meminfo").read_text().splitlines())
    zustand = (DATA / "megatron_zustand.json").stat()
    wiki = ende(DATA / "wiki_sammler.log", 4)
    return {
        "art": art,
        "jetzt": jetzt,
        "chat": chat,
        "noten": [f["note"] for f in fortschritt if f.get("note") is not None],
        "worttreffer": [100 * f["worttreffer"] for f in fortschritt
                        if f.get("worttreffer") is not None],
        "ergebnisse": [(n, erg.get(n, "–")) for n in ERGEBNISSE],
        "zyklen": zyklus[-8:][::-1],
        "ausgabe": aus[-25:],
        "archiv_mb": (DATA / "harvest.jsonl").stat().st_size / 1048576,
        "gelernt": (DATA / "gelernt.txt").read_bytes().count(b"\n"),
        "zustand_mb": zustand.st_size / 1048576,
        "schwarm_gb": (DATA / "schwarm" / "gedaechtnis.sqlite").stat().st_size / 1024 ** 3,
        "zustand_zeit": time.strftime("%d.%m. %H:%M", time.localtime(zustand.st_mtime)),
        "last": "%.2f" % os.getloadavg()[0],
        "ram_frei_mb": int(mem["MemAvailable"].split()[0]) // 1024,
        "temp": int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000,
        "platte_frei_gb": shutil.disk_usage(WURZEL).free / 1e9,
        "wiki_dienst": dienst("wiki-sammler"),
        "wiki_zeile": wiki[-1] if wiki else "–",
    }


STIL = """
:root{--bg:#f5f6f8;--karte:#fff;--text:#1c2230;--leise:#667085;--rand:#e2e5ea;
--laeuft:#15803d;--pause:#b45309;--fehler:#b91c1c;--aus:#667085}
@media (prefers-color-scheme:dark){:root{--bg:#10131a;--karte:#191e27;--text:#e5e8ee;
--leise:#8a93a3;--rand:#29303b;--laeuft:#4ade80;--pause:#fbbf24;--fehler:#f87171;--aus:#9aa3b2}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--leise);margin:22px 0 8px}
.kopf{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.karte{background:var(--karte);border:1px solid var(--rand);border-radius:10px;padding:12px 14px}
.jetzt{font-size:17px}.jetzt b{margin-right:8px}
.laeuft{color:var(--laeuft)}.pause{color:var(--pause)}.fehler{color:var(--fehler)}.aus{color:var(--aus)}
.kacheln{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.zahl{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.klein{color:var(--leise);font-size:13px}
dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:6px 14px}
dt{color:var(--leise)}dd{margin:0;overflow-wrap:anywhere}
pre{margin:0;font:12px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
"""
ART = {"laeuft": "● läuft", "pause": "● Pause", "fehler": "● Fehler", "aus": "● aus"}


def urteil(note):
    return ("Grütze" if note < 2 else "erste Spuren" if note < 5
            else "brauchbar" if note < 8 else "intelligent")


def verlauf(werte):
    return " → ".join("%.1f" % w for w in werte[-6:]) or "kommt mit dem nächsten Zyklus"


def seite(s):
    e = html.escape
    kachel = '<div class="karte"><div class="klein">%s</div><div class="zahl">%s</div><div class="klein">%s</div></div>'
    kacheln = "".join(kachel % k for k in (
        ("Archiv", "%.1f MB" % s["archiv_mb"], "Rohtexte, Gelerntes fliegt raus"),
        ("Gelernt", "%d Texte" % s["gelernt"], "schon in Megatron"),
        ("Megatron-Zustand", "%.0f MB" % s["zustand_mb"], "gesichert " + s["zustand_zeit"]),
        ("Schwarm", "%.2f GB" % s["schwarm_gb"], "von 10 GB, darüber wird ausgedünnt"),
        ("Pi5", "%.0f °C" % s["temp"], "Last %s · RAM frei %d MB · Platte %.0f GB"
         % (s["last"], s["ram_frei_mb"], s["platte_frei_gb"])),
    ))
    noten, wt = s["noten"], s["worttreffer"]
    fortschritt = "".join(kachel % k for k in (
        ("Lehrernote (space-bunny, 10 feste Fragen)",
         "%.1f / 10 · %s" % (noten[-1], urteil(noten[-1])) if noten else "–",
         "Verlauf: " + verlauf(noten)),
        ("Worttreffer bei neuem Text (vor dem Lernen)",
         "%.1f %%" % wt[-1] if wt else "–", "Verlauf: " + verlauf(wt)),
    ))
    chat = "".join("<dt>%s</dt><dd>%s</dd>" % (e(f), e(a)) for f, a in s["chat"]) or "<dd>–</dd>"
    erg = "".join("<dt>%s</dt><dd>%s</dd>" % (e(n), e(t)) for n, t in s["ergebnisse"])
    return """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T4Bv2 Status</title><style>%s</style></head><body><main>
<div class="kopf"><h1>T4Bv2 · Megatron auf dem Pi5</h1><span class="klein">Stand %s</span></div>
<h2>Jetzt</h2><div class="karte jetzt"><b class="%s">%s</b>%s</div>
<h2>Fortschritt</h2><div class="kacheln">%s</div>
<h2>Zahlen</h2><div class="kacheln">%s</div>
<h2>Megatron antwortet (letzter Zyklus)</h2><div class="karte"><dl>%s</dl></div>
<h2>Letzte Ergebnisse je Phase</h2><div class="karte"><dl>%s</dl></div>
<h2>Zyklen</h2><div class="karte"><pre>%s</pre></div>
<h2>Wiki-Sammler (%s)</h2><div class="karte"><pre>%s</pre></div>
<h2>Live-Ausgabe</h2><div class="karte"><pre>%s</pre></div>
</main><script>
setInterval(()=>fetch(location.href).then(r=>r.text()).then(t=>{document.querySelector("main").innerHTML=
new DOMParser().parseFromString(t,"text/html").querySelector("main").innerHTML}).catch(()=>{}),10000)
</script></body></html>""" % (
        STIL, time.strftime("%H:%M:%S"), s["art"], ART[s["art"]], e(s["jetzt"]),
        fortschritt, kacheln, chat, erg, e("\n".join(s["zyklen"])), e(s["wiki_dienst"]),
        e(s["wiki_zeile"]), e("\n".join(s["ausgabe"])))


class Seite(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        try:
            body = seite(stand()).encode("utf-8")
        except Exception as fehler:  # Seite soll nie ganz wegbrechen
            body = ("<pre>Fehler beim Lesen: %s</pre>" % html.escape(repr(fehler))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # kein Zugriffslog alle 10 s ins Journal
        pass


if __name__ == "__main__":
    if sys.argv[1:] == ["pruefen"]:
        print(json.dumps(stand(), indent=1, ensure_ascii=False))
    else:
        ThreadingHTTPServer(("", PORT), Seite).serve_forever()
