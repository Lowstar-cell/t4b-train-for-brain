# T4B – Train for Brain v2

A self-built language model that trains itself, around the clock, on a
Raspberry Pi 5. No transformer, no pretrained weights: the core
("Megatron") is written from scratch in Python/NumPy, and a fully automated
cycle collects text, cleans it, trains, evaluates and reports progress —
without anyone touching it.

The point of the project is the **automation and the measuring**, not a
finished chatbot: the answers are still word salad, and the system says so
itself on its status page.

## Megatron – the model

Deliberately *not* a transformer — no attention, no QKV, no positional
embeddings:

- **Markov core** – transition probabilities between words, looking back two
  words (markov-2).
- **Analog fully-connected net** – N nodes (currently 1024), each connected to
  all others. Signals are continuous (−1…1) instead of 0/1; "thinking" is
  propagation through the net, trained with Hebbian wiring.
- **Trust layer** (LoRA-like) – only delta weights from feedback, swappable.
- **Swarm memory** – knowledge lives in a SQLite full-text index (FTS5,
  capped at 10 GB). Each answer pulls three texts from it and counts them as
  hits; when the cap is reached, the least useful texts (lowest weight ×
  hits, then oldest) are dropped.
- Forgetting: above 3 million word sequences, sequences seen only once are
  dropped (2.17 M → 0.42 M in a test).

`atomic_gpt.py` is the starting point: a tiny autograd + GPT written from
scratch, from which the project grew.

## The automated cycle (Raspberry Pi 5)

```
wiki_sammler.py (service)        trainingszyklus.py (service t4b-training)
  German Wikipedia, 45 categories    every 4 h: trainingslauf.py as its own process
  cleaned plain text                   1. harvest topics
  always one round ahead               2. fill the swarm (SQLite FTS5)
  (≤ 3000 unlearned texts)             3. distil answers from a teacher model
          │                            4. train Megatron on ≤ 1500 NEW texts
          └──▶ data/harvest.jsonl ──▶  5. benchmark + progress
                                       6. drop learned texts, forget rare sequences
                                     statusseite.py (service t4b-ui, port 8088)
                                       live status page, refreshes every 10 s
```

- Runs as `systemd --user` services and survives reboots.
- A cycle takes ~15–20 min of work, then a 4-hour pause. Between cycles the
  service uses ~12 MB RAM instead of 1.5 GB, because each run is a separate
  process.
- Only new texts are learned (IDs in `data/gelernt.txt`) — counting the same
  text again changes no probabilities, it only costs time.
- Wikipedia is fetched through the API as plain text, within the Wikimedia
  rate limit (10 requests/min without contact info).

## Measuring progress

Every cycle appends one line to `data/fortschritt.jsonl`, shown on the status
page:

- **Teacher grade 0–10** – Megatron answers 10 fixed questions, a teacher
  model grades them in one call (0–2 garbage, 2–5 first traces, 5–8 usable,
  8–10 intelligent).
- **Word hit rate** – share of words Megatron predicts correctly in a new
  round *before* learning it. Rises as it learns language, even while the
  grade is still 0.

Baseline (6 Oct 2026, before round 2): grade 0.1 / 10, word hit rate 11.9 %.

## Other parts

| File | Purpose |
|------|---------|
| `brain_core_v2.py` | orchestration: swarm, router, teacher distillation, tools, benchmark |
| `supervisor.py` | loop guard – stops after repeated identical actions |
| `rollback.py` | metric history per adapter, switch back when a new one is worse |
| `hardware_status.py` | GPU status via rocm-smi, injected into the system prompt |
| `interface.py` | Gradio chat UI (PC, port 7865) |
| `specialists/` | harvester, swarm memory, semantic router, MCP tools, LoRA management, benchmark |
| `docs/DEVLOG.de.md` | development log (German) |

## Run

```bash
pip install numpy requests gradio
python3 trainingslauf.py            # one cycle
python3 trainingszyklus.py          # endless: cycle + 4 h pause
python3 wiki_sammler.py 1500        # collect Wikipedia texts
python3 statusseite.py              # status page on :8088
python3 statusseite.py pruefen      # self-check without server
```

The teacher model is called through the [opencode](https://opencode.ai) CLI.
Code and comments are in German.

## Tech

Python · NumPy · SQLite FTS5 · Gradio · systemd · Raspberry Pi 5 · Wikipedia API
