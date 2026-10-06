"""Isolierte Erstellung und Ausfuehrung von Code in einer eigenen venv."""

import subprocess
import sys
from pathlib import Path

# Zielverzeichnis der isolierten Umgebung
WERKBANK = Path(__file__).parent / "werkbank"


def venv_bereit():
    """True, wenn unter WERKBANK bereits eine venv liegt."""
    return (WERKBANK / "pyvenv.cfg").is_file()


def venv_erzeugen():
    """Erzeugt die venv, falls sie fehlt, und gibt den Python-Pfad zurueck.

    Gibt None zurueck, wenn der venv-Aufruf fehlschlaegt.
    """
    if not venv_bereit():
        ergebnis = subprocess.run(
            [sys.executable, "-m", "venv", str(WERKBANK)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if ergebnis.returncode != 0:
            return None
    if sys.platform == "win32":
        python = WERKBANK / "Scripts" / "python.exe"
    else:
        python = WERKBANK / "bin" / "python"
    if not python.is_file():
        return None
    return python


def code_ausfuehren(code, timeout=30):
    """Fuehrt Python-Code isoliert in der Werkbank-venv aus.

    Gibt stdout (gestrippt) zurueck oder None bei Fehler/Timeout.
    """
    python = venv_erzeugen()
    if python is None:
        return None
    datei = WERKBANK / "code.py"
    datei.write_text(code, encoding="utf-8")
    try:
        ergebnis = subprocess.run(
            [str(python), str(datei)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if ergebnis.returncode != 0:
        return None
    return ergebnis.stdout.strip()


if __name__ == "__main__":
    print(venv_bereit())
    print(code_ausfuehren("print('werkbank ok')"))
