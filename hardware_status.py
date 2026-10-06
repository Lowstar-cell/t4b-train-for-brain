# -*- coding: utf-8 -*-
"""Hardware-Monitoring via rocm-smi.

Liest GPU-Status (Name, Auslastung, Temperatur, VRAM) ueber rocm-smi
und stellt ihn als einzeiligen Text fuer die System-Prompt-Injektion bereit.
Fehlt rocm-smi oder schlaegt der Aufruf fehl, wird das sauber gemeldet -
es wird nichts erfunden und nichts installiert.
"""

import json
import subprocess


def rocm_smi_roh(timeout=3):
    """Fuehrt rocm-smi aus und gibt stdout als String zurueck, sonst None.

    Fehlt das Programm (FileNotFoundError) oder laeuft es nicht
    (Timeout, sonstiger Fehler), wird None zurueckgegeben - kein Absturz.
    """
    try:
        ergebnis = subprocess.run(
            ["rocm-smi", "--showuse", "--showtemp", "--showmemuse",
             "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=timeout
        )
        return ergebnis.stdout
    except Exception:
        # Programm fehlt, Timeout oder sonstiger Fehler: None melden.
        return None


def hardware_status():
    """Gibt einen dict mit dem GPU-Status zurueck.

    Schluessel: gpu, auslastung, temperatur, vram_genutzt_mb,
    vram_gesamt_mb, quelle. Ohne rocm-smi kommen die Fallback-Werte
    ("keine AMD-GPU" bzw. None) und quelle="nicht verfuegbar".
    """
    roh = rocm_smi_roh()
    if roh is None:
        return {
            "gpu": "keine AMD-GPU",
            "auslastung": None,
            "temperatur": None,
            "vram_genutzt_mb": None,
            "vram_gesamt_mb": None,
            "quelle": "nicht verfuegbar",
        }

    try:
        daten = json.loads(roh)
    except Exception:
        # Ungueltiges JSON: ebenfalls sauber als nicht verfuegbar melden.
        return {
            "gpu": "keine AMD-GPU",
            "auslastung": None,
            "temperatur": None,
            "vram_genutzt_mb": None,
            "vram_gesamt_mb": None,
            "quelle": "nicht verfuegbar",
        }

    # rocm-smi liefert pro GPU einen Eintrag unter "card0" (oder aehnlich).
    karte = None
    if isinstance(daten, dict):
        for schluessel in sorted(daten.keys()):
            wert = daten[schluessel]
            if isinstance(wert, dict):
                karte = wert
                break

    gpu = "keine AMD-GPU"
    auslastung = None
    temperatur = None
    vram_genutzt_mb = None
    vram_gesamt_mb = None

    if karte is not None:
        # GPU-Name: bevorzugt "gpu_id"/"name", sonst erster String-Wert.
        for feld in ("gpu_id", "name", "card"):
            if feld in karte and isinstance(karte[feld], str) and karte[feld]:
                gpu = karte[feld]
                break
        if gpu == "keine AMD-GPU":
            for wert in karte.values():
                if isinstance(wert, str) and wert:
                    gpu = wert
                    break

        # Auslastung in Prozent: Felder wie "GPU use (%)"
        for feld, wert in karte.items():
            if "use" in feld.lower() and "%" in feld:
                zahl = _als_zahl(wert)
                if zahl is not None:
                    auslastung = zahl
                    break

        # Temperatur in Grad: Felder wie "Temperature (C)"
        for feld, wert in karte.items():
            if "temperature" in feld.lower():
                zahl = _als_zahl(wert)
                if zahl is not None:
                    temperatur = zahl
                    break

        # VRAM: genutzt und gesamt in MB aus den meminfo-Feldern.
        for feld, wert in karte.items():
            f = feld.lower()
            if "vram" in f and "used" in f and "total" not in f:
                zahl = _als_zahl(wert)
                if zahl is not None:
                    vram_genutzt_mb = zahl
            if "vram" in f and "total" in f:
                zahl = _als_zahl(wert)
                if zahl is not None:
                    vram_gesamt_mb = zahl

    return {
        "gpu": gpu,
        "auslastung": auslastung,
        "temperatur": temperatur,
        "vram_genutzt_mb": vram_genutzt_mb,
        "vram_gesamt_mb": vram_gesamt_mb,
        "quelle": "rocm-smi",
    }


def _als_zahl(wert):
    """Wandelt einen Wert in eine Zahl um, sonst None.

    rocm-smi liefert Zahlen teils als String mit Einheit ("8120 MB",
    "61.0"). Es wird nur der Zahlenteil gelesen, nichts erfunden.
    """
    if isinstance(wert, (int, float)):
        return wert
    if isinstance(wert, str):
        text = wert.strip()
        # Zahlenteil am Anfang herausnehmen (Ziffern, Punkt, Vorzeichen).
        stueck = ""
        for zeichen in text:
            if zeichen.isdigit() or zeichen in ".-+":
                stueck += zeichen
            else:
                break
        if stueck in ("", ".", "-", "+", "-.", "+."):
            return None
        try:
            zahl = float(stueck)
        except ValueError:
            return None
        if zahl == int(zahl):
            return int(zahl)
        return zahl
    return None


def system_prompt():
    """Gibt den Hardware-Status als einzeiligen String zurueck.

    Beispiel: "HW: RX 7900 XTX, 54% Auslastung, 61 C, 8120/24560 MB VRAM".
    Ohne rocm-smi: "HW: nicht verfuegbar".
    """
    status = hardware_status()
    if status["quelle"] == "nicht verfuegbar":
        return "HW: nicht verfuegbar"

    teile = ["HW: " + str(status["gpu"])]

    if status["auslastung"] is not None:
        teile.append(str(status["auslastung"]) + "% Auslastung")
    if status["temperatur"] is not None:
        temperatur = status["temperatur"]
        if isinstance(temperatur, float):
            temperatur = int(temperatur)
        teile.append(str(temperatur) + " C")

    if (status["vram_genutzt_mb"] is not None
            and status["vram_gesamt_mb"] is not None):
        teile.append(str(status["vram_genutzt_mb"]) + "/"
                     + str(status["vram_gesamt_mb"]) + " MB VRAM")
    elif status["vram_genutzt_mb"] is not None:
        teile.append(str(status["vram_genutzt_mb"]) + " MB VRAM genutzt")

    return ", ".join(teile)


if __name__ == "__main__":
    print(system_prompt())
