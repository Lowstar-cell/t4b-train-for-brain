alles beruht auf dieser datei atomic_gpt.py





Phase 1: Fundament & Hardware
ROCm/PyTorch für AMD-GPU installieren.
Hardware-Monitoring via rocm-smi einrichten und als dynamische System-Prompt-Injektion anbinden.
Unzensiertes Basismodell (1B-3B) laden.
Token-Router für Context Window Management integrieren (automatische Summarization bei >1000 Tokens).

Phase 2: Sicherheit & Tool-Integration (MCP)
Isolierte erstellung von code in einer python venv
Lokale MCP-Server (Model Context Protocol) für Terminal, Dateisystem und Websuche bereitstellen.
Basismodell strikt auf ReAct-Loop (Thought, Action, Observation) trimmen.
Supervisor-Skript zur Loop-Erkennung aktivieren (Hash-Abgleich der Aktionen, harter Abbruch nach 3 identischen Fehlern).

Phase 3: Sensorik & Datenbank
Lokale Vector-DB (Milvus/Qdrant) initiieren.
Embedding-Modelle laden: BGE-M3 (Text), jina-embeddings-v2-code (Code), SigLIP (Vision), CLAP (Audio).
Translatoren-Skripte für Rohdaten implementieren (YOLO-Koordinaten zu Text, Whisper-Audio zu Text, Web-Scraping-Bereinigung).

Phase 4: Frontend & Datensammlung
User Interface (Gradio/React) mit Chat, System-Metriken, LoRA-Modulverwaltung und Datei-Drop-Zone aufbauen.
Automatisierte Agenten zur Datensuche und Filterung aktivieren (Limit: 200 MB Tagesvolumen).
Gefilterte und übersetzte Daten über Embedding-Modelle in die Vector-DB schreiben.

Phase 5: Lernzyklus & Rollback (Automatisiert)
Daten aus Vector-DB in Trainingsformat (JSONL) exportieren.
LoRA-Finetuning auf der AMD-Karte durchführen (neuen Adapter mit Zeitstempel speichern).
Automatisierte Evaluierung via Benchmark-Datensatz starten.
Rollback-Mechanismus anwenden: Bei Verschlechterung der Metriken sofortigen Wechsel zum Vortages-Adapter erzwingen.
Vector-DB durch DROP-Befehl vollständig leeren.
## Stand 04.10.2026 (Megatron beruhigt)
- megatron.py: Knoten stehen nicht mehr alle auf 1.0 (Erregung jetzt ~0.008).
  Verbindungen koennen wieder schwaecher werden, verschiedene Fragen loesen
  verschiedene Muster aus. Stellschrauben oben in megatron.py: LERNRATE 0.02,
  VERSTAERKER 0.5 (1.0 = Antworten wiederholen sich in Schleifen).
- Alter Zustand gesichert: data/megatron_zustand_vor_beruhigen_2026-10-04.json
- Antworten sind trotzdem noch Wortsalat. Naechste Schritte, in dieser Reihenfolge:
  1. Seitenmuell aus data/harvest.jsonl filtern ("quelltext bearbeiten",
     "title=...&oldid=", Links, PowerShell-Befehle) - schnellster Gewinn.
  2. Mehr Knoten: 38.000 Woerter teilen sich 256 Knoten (149 je Knoten).
  3. Wortliste schaut nur 1 Wort zurueck - auf 2 Woerter erweitern.
- Trainingszyklus laeuft nach einem Neustart des PCs NICHT von selbst wieder an.
  Starten (im Ordner T4Bv2):
  werkbank\Scripts\python.exe -u -X utf8 trainingszyklus.py > data\trainingszyklus_ausgabe.log

## Stand 04.10.2026 abends (Training laeuft auf dem Pi5)
- Die drei Punkte von oben sind eingebaut:
  1. Seitenmuell: Wikipedia-Volltext kommt jetzt ueber die Wikipedia-Schnittstelle
     als reiner Text (harvester.learn_wikipedia(thema, ganz=True)), nicht mehr als
     HTML-Seite. saeubern() in specialists/harvester.py wirft Quellen-Anhang,
     Ueberschriften, Links und Markdown raus. Archiv und Schwarm einmal gesaeubert,
     Sicherungen: data/harvest_vor_saeubern_2026-10-04.jsonl,
     data/schwarm_vor_saeubern_2026-10-04.jsonl.
     Arbeitsweisen (agent-memory, Befehle und Code) lernt Megatron nicht mehr als
     Sprache - sie bleiben im Schwarm. Rueckgaengig: eine Zeile in trainingslauf.py.
  2. 1024 Knoten statt 256. Alter Zustand: data/megatron_zustand_256knoten_2026-10-04.json
  3. Wortliste schaut 2 Woerter zurueck (markov2), sonst 1 Wort.
  Dazu: Satzzeichen am Fragenende werden beim Antworten uebergangen.
- Lehrer ist jetzt opencode/space-bunny-free (zieht nichts vom Gratis-Kontingent,
  KEIN anderes OpenCode-Modell nehmen). LM Studio wird nicht mehr gebraucht.
- Training laeuft auf dem Pi5 (<PI_IP>, ~/Projekte/T4Bv2) als Dienst
  t4b-training (startet nach Neustart von selbst). Ein Zyklus ca. 15-20 Minuten,
  dann 4 Stunden Pause. Auf dem PC NICHT mehr trainingszyklus.py starten.
  Zustand ansehen:  ssh admin@<PI_IP> "tail -30 Projekte/T4Bv2/data/trainingszyklus_ausgabe.log"
  Zustand zum PC holen (fuer interface.py):
    scp admin@<PI_IP>:Projekte/T4Bv2/data/megatron_zustand.json data/
  Code-Aenderungen vom PC auf den Pi bringen (Git Bash, im Ordner T4Bv2):
    tar czf - *.py specialists/*.py | ssh admin@<PI_IP> "tar xzf - -C Projekte/T4Bv2"
    ssh admin@<PI_IP> "systemctl --user restart t4b-training"

## Stand 05.10.2026 (Wiki-Sammler)
- wiki_sammler.py holt deutsche Wikipedia-Artikel (CC BY-SA 4.0) aus 45 Kategorien
  (Technik, Informatik, Medizin, Naturwissenschaft, ca. 9.500 Artikel), saeubert sie
  und haengt sie an data/harvest.jsonl an. Kategorien stehen oben im Skript.
  Tempo: 10 Anfragen/Minute (Wikimedia-Grenze ohne Kontakt im User-Agent),
  ca. 250-600 Artikel/Stunde. Schon vorhandene Artikel werden uebersprungen.
  Mit Kontakt (Mail oder Wiki-Benutzerseite) im User-Agent waeren 200/Minute erlaubt.
  Laufen (Pi5):  systemd-run --user --unit=wiki-sammler --nice=10 \
    -p WorkingDirectory=/home/admin/Projekte/T4Bv2 \
    -p StandardOutput=append:/home/admin/Projekte/T4Bv2/data/wiki_sammler.log \
    /usr/bin/python3 -u -X utf8 wiki_sammler.py 1500
  Fortschritt:   ssh admin@<PI_IP> "tail Projekte/T4Bv2/data/wiki_sammler.log"
- Trainingslauf lernt nur noch NEUE Texte (Kennungen in data/gelernt.txt).
  Nochmal zaehlen aendert die Wahrscheinlichkeiten nicht, kostete nur Zeit.
  Alles neu lernen: data/gelernt.txt loeschen.
- Arbeitsspeicher beobachten: 1 MB Text ~ 90 MB im Megatron-Zustand (markov2).
  Der Pi hat 4 GB und teilt sie mit MI6-2.

## Stand 05.10.2026 abends (Statusseite)
- Im Browser sehen, was der Pi macht: http://<PI_IP>:8088
  (Jetzt-Zustand, naechster Zyklus, Megatron-Antworten, Ergebnis je Phase,
  Wiki-Sammler, Live-Ausgabe; aktualisiert sich alle 10 s).
- statusseite.py liest nur data/*.log, laedt kein Gehirn (wenige MB RAM).
  Dienst t4b-ui (systemctl --user, startet nach Neustart von selbst).
  Pruefen ohne Server: python3 statusseite.py pruefen
- Auf dem Pi-Desktop (kein Monitor, Bildschirm per VNC) oeffnet sich die Seite nach
  der Anmeldung als eigenes Fenster: ~/.config/autostart/t4b-status.desktop
  (Chromium im App-Modus, eigenes Profil ~/.config/t4b-status-chromium, Uebersetzen
  dort abgeschaltet). Kostet ca. 300-400 MB RAM. Weg damit: die .desktop-Datei loeschen.

## Stand 05.10.2026 nachts (Sammler immer eine Runde voraus)
- Beobachtet: die 1500 Wiki-Artikel waren nach EINEM Zyklus gelernt (2 h 15 min),
  danach liefen die Zyklen leer.
- wiki_sammler.py laeuft jetzt dauerhaft als Dienst wiki-sammler (Pi5). Er holt,
  bis 3000 ungelernte Texte im Archiv liegen (laufende Runde + eine im Vorlauf),
  und wartet dann. Geholte Titel: data/wiki_titel.txt. Unterkategorien bis Tiefe 2.
- Trainingslauf lernt je Zyklus hoechstens 1500 Texte (RUNDE in wiki_sammler.py).
  Danach fliegen gelernte Texte aus data/harvest.jsonl (Harvester-Eintraege von
  heute bleiben bis morgen, Arbeitsweisen bleiben immer).
- Ueber 3 Mio. Wortfolgen vergisst Megatron die nur einmal gesehenen
  (MAX_FOLGEN in trainingslauf.py). Test am Zustand vom 05.10.: 2,17 Mio. -> 0,42 Mio.
  Sicherungen vor dem ersten Loeschen:
  data/megatron_zustand_vor_vergessen_2026-10-05.json,
  data/harvest_vor_aufraeumen_2026-10-05.jsonl
- trainingslauf.py laeuft als eigener Prozess je Zyklus: in der Pause belegt der
  Dienst t4b-training nur noch ~12 MB statt 1,5 GB.
- Schwarm liegt seit 05.10.2026 in SQLite (data/schwarm/gedaechtnis.sqlite,
  Volltextsuche FTS5) statt komplett im RAM (vorher 22 MB Datei = 46 MB RAM, haette
  bei 10 GB ~21 GB RAM gebraucht). Alte Datei: gedaechtnis.jsonl.vor_sqlite.
  Grenze 10 GB (GRENZE in specialists/flash_swarm.py). Darueber fliegt das am
  wenigsten Nuetzliche raus: kleinstes Gewicht x (1 + Treffer), dann das Aelteste.
  Gewicht = Feedback, Treffer = wie oft eine Suche den Text gefunden hat.
- Megatrons Erinnerungsabruf (megatron.py, denken) war kaputt: swarm_search gab es
  nicht als Modulfunktion, der Fehler wurde verschluckt. Seit 06.10.2026 repariert
  (flash_swarm.swarm_search). Jede Antwort holt jetzt 3 Texte aus dem Schwarm und
  zaehlt sie als Treffer - daran misst das Ausduennen, was nuetzlich ist.
- Lehrer bleibt opencode/space-bunny-free (Wunsch des Users, 06.10.2026).

## Stand 06.10.2026 (Fortschritt messen: Intelligenz oder Gruetze)
- Jeder Zyklus schreibt eine Zeile in data/fortschritt.jsonl, die Statusseite zeigt
  sie oben unter "Fortschritt" mit Verlauf:
  - Lehrernote 0-10: Megatron beantwortet 10 feste Fragen (FRAGEN in
    trainingslauf.py), space-bunny benotet alle in einem Aufruf.
    0-2 Gruetze, 2-5 erste Spuren, 5-8 brauchbar, 8-10 intelligent.
  - Worttreffer %: Anteil der Woerter, die Megatron in der neuen Runde VOR dem
    Lernen richtig vorhersagt (megatron.worttreffer). Steigt, wenn er Sprache lernt,
    auch solange die Note noch bei 0 steht.
- Ausgangswert (Stand 06.10. 03:30, vor Runde 2): Note 0,1 / 10, Worttreffer 11,9 %.
- Der alte Benchmark (Phase 5, 8 Rechenaufgaben) sagt nichts: richtig ist, wenn die
  Ziffer irgendwo im Wortsalat steht.
- Prompt an opencode einzeilig halten: opencode.cmd (Windows) schneidet am ersten
  Zeilenumbruch ab.
