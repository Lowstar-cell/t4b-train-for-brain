import os
import platform
import subprocess
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent

# Nur diese Kommandos duerfen ausgefuehrt werden, und nur so, wie sie hier
# stehen. Kein Zusammenbauen aus der Frage heraus - das waere eine
# Kommando-Injektion ueber den Chat.
ERLAUBT = {
    "status":   ["cmd", "/c", "ver"],
    "speicher": ["cmd", "/c", "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value"],
    "prozess":  ["cmd", "/c", "tasklist /fo csv /nh"],
    "systeminfo": ["cmd", "/c", "systeminfo"],
}


class ToolBelt:
    """Fuehrt die erlaubten Systemabfragen aus. Nichts sonst."""

    def __init__(self, wurzel=WURZEL):
        self.wurzel = Path(wurzel)

    def handle_request(self, query):
        """Gibt einen String zurueck. Unbekanntes -> Liste der Aktionen."""
        text = (query or "").lower()
        if "datei" in text:
            return self._datei(text)
        for name, kommando in ERLAUBT.items():
            if name in text:
                return self._kaufen(name, kommando)
        return self._hilfe()

    def _kaufen(self, name, kommando):
        try:
            raus = subprocess.run(kommando, capture_output=True, text=True,
                                  timeout=15, encoding="utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError) as e:
            return f"{name} fehlgeschlagen: {e}"
        text = (raus.stdout or raus.stderr or "").strip()
        if not text:
            return f"{name}: keine Ausgabe (Code {raus.returncode})"
        if len(text) > 1500:
            text = text[:1500] + " ... [gekuerzt]"
        return f"{name} (Code {raus.returncode}):\n{text}"

    def _datei(self, text):
        rest = text.split("datei", 1)[-1].replace(":", " ").replace(",", " ")
        namen = [w for w in rest.split() if w and "\\" not in w]
        if not namen:
            return "Welche Datei? So: datei <name>"
        ziel = (self.wurzel / namen[0]).resolve()
        if self.wurzel.resolve() not in ziel.parents:
            return "Nur Dateien im Projektordner."
        if not ziel.exists():
            return f"Nicht gefunden: {ziel.name}"
        if ziel.is_dir():
            eintraege = sorted(p.name + ("/" if p.is_dir() else "")
                               for p in ziel.iterdir())
            return f"{ziel.name}/: " + ", ".join(eintraege[:50])
        return ziel.read_text(encoding="utf-8", errors="replace")[:1500]

    def _hilfe(self):
        return ("Ich kann: status, speicher, prozess, systeminfo, "
                "datei <name> (nur lesen). Neustart und beende machen wir nicht.")

if __name__ == "__main__":
    b = ToolBelt()
    assert isinstance(b.handle_request(""), str)
    assert "kann" in b.handle_request("wie geht es")
    assert isinstance(b.handle_request("status"), str)
    pfad = b._datei("datei Roadmap.md")
    assert "atomic_gpt" in pfad, pfad
    assert "Projektordner" in b._datei("datei ../../../../Windows/win.ini")
    print("tool_belt ok")
