"""Lokale MCP-Werkzeuge: Terminal, Dateisystem, Websuche.

Alle Funktionen geben einen String zurueck und werfen niemals.
"""
import subprocess
from pathlib import Path

from specialists.harvester import KnowledgeHarvester

WURZEL = Path(__file__).resolve().parent.parent
MAX_ZEICHEN = 2000

BEFEHLE = {
    "ipconfig": ["ipconfig"],
    "systeminfo": ["systeminfo"],
    "dir": ["cmd", "/c", "dir"],
    "nvidia-smi": ["nvidia-smi"],
}


def terminal_befehl(name, timeout=10):
    """Fuehrt einen erlaubten Befehl aus BEFEHLE aus. String zurueck."""
    if name not in BEFEHLE:
        return f"Unbekannter Befehl: {name}"
    try:
        erg = subprocess.run(
            BEFEHLE[name],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        ausgabe = erg.stdout or erg.stderr or ""
    except subprocess.TimeoutExpired:
        return f"Zeitueberschreitung nach {timeout}s: {name}"
    except Exception as e:
        return f"Befehl fehlgeschlagen ({name}): {e}"
    return ausgabe[:MAX_ZEICHEN]


def datei_aktion(aktion, pfad, inhalt=""):
    """Liest, schreibt oder listet innerhalb von WURZEL. String zurueck."""
    try:
        ziel = (WURZEL / pfad).resolve()
    except Exception as e:
        return f"Ungueltiger Pfad ({pfad}): {e}"
    try:
        ziel.relative_to(WURZEL.resolve())
    except ValueError:
        return f"Pfad liegt ausserhalb des Projekts: {pfad}"
    try:
        if aktion == "lesen":
            return ziel.read_text(encoding="utf-8", errors="replace")[:MAX_ZEICHEN]
        if aktion == "schreiben":
            ziel.write_text(inhalt, encoding="utf-8")
            return f"Geschrieben: {len(inhalt)} Zeichen nach {pfad}"
        if aktion == "liste":
            if not ziel.is_dir():
                return f"Kein Ordner: {pfad}"
            return "\n".join(sorted(e.name for e in ziel.iterdir()))
        return f"Unbekannte Aktion: {aktion}"
    except Exception as e:
        return f"Dateiaktion fehlgeschlagen ({aktion} {pfad}): {e}"


def websuche(begriff):
    """Sucht ueber den KnowledgeHarvester bei Wikipedia. String zurueck."""
    try:
        return KnowledgeHarvester().learn_wikipedia(begriff)
    except Exception as e:
        return f"Websuche fehlgeschlagen ({begriff}): {e}"


if __name__ == "__main__":
    assert "Unbekannter Befehl" in terminal_befehl("gibtsnicht")
    assert isinstance(terminal_befehl("ipconfig"), str)
    assert isinstance(datei_aktion("lesen", "pyproject.toml"), str)
    fremd = datei_aktion("schreiben", r"C:\Windows\System32\test.txt", "x")
    assert isinstance(fremd, str) and "ausserhalb" in fremd
    assert not Path(r"C:\Windows\System32\test.txt").exists()
    assert isinstance(websuche(""), str)
    print("mcp_tools ok")
