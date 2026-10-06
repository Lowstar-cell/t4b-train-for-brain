import gradio as gr
import time, threading, subprocess, os, json, sys, tempfile
from pathlib import Path
from brain_core_v2 import BrainCoreV2
from specialists import mcp_tools

brain = BrainCoreV2()
DATA_DIR = Path(__file__).parent / "data"

# ─── Equalizer HTML ───────────────────────────────────────────────
COLORS = {"swarm":"#f43f5e","harvester":"#22c55e","vision":"#3b82f6",
          "qwen":"#a855f7","eloquence":"#f59e0b","whisper":"#06b6d4","keim":"#84cc16",
          "system":"#64748b"}

def render_dashboard():
    d = brain.get_dashboard()
    bars = ""
    for name, val in d["activity"].items():
        pct = min(100, round(val * 100))
        c = COLORS.get(name, "#64748b")
        bars += f"""<div style="margin:6px 0">
            <div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8">
                <span>{name}</span><span>{pct}%</span>
            </div>
            <div style="height:16px;background:#1e293b;border-radius:8px;overflow:hidden">
                <div style="height:100%;width:{pct}%;background:{c};border-radius:8px;transition:width .3s"></div>
            </div>
        </div>"""

    spd = ""
    for i in range(5):
        s = d["speedys"].get(str(i), d["speedys"].get(i, {"status":"idle","progress":0,"task":""}))
        run = s["status"] == "running"
        spd += f"""<div style="margin:4px 0;display:flex;align-items:center;gap:8px">
            <span style="color:{'#22c55e' if run else '#64748b'}">{'🏃' if run else '⏸️'}</span>
            <span style="width:70px;font-size:11px;color:#94a3b8">Speedy#{i+1}</span>
            <div style="flex:1;height:10px;background:#1e293b;border-radius:5px;overflow:hidden">
                <div style="height:100%;width:{min(100,s['progress'])}%;background:{'#22c55e' if run else '#475569'};border-radius:5px;transition:width .5s"></div>
            </div>
            <span style="width:130px;font-size:10px;color:#64748b">{str(s['task'])[:25] or s['status']}</span>
        </div>"""

    sm = d.get("sammler", {})
    sm_run = sm.get("laufend", False)
    sm_gesamt = max(1, int(sm.get("gesamt", 0)))
    sm_pct = min(100, round(100 * int(sm.get("fortschritt", 0)) / sm_gesamt))
    sm_txt = sm.get("thema", "") or ("l\u00e4uft" if sm_run else "aus")
    sm_line = f"""<div style="margin:6px 0;display:flex;align-items:center;gap:8px">
        <span style="color:{'#22c55e' if sm_run else '#64748b'}">{'🌾' if sm_run else '⏸️'}</span>
        <span style="width:70px;font-size:11px;color:#94a3b8">Sammler</span>
        <div style="flex:1;height:10px;background:#1e293b;border-radius:5px;overflow:hidden">
            <div style="height:100%;width:{sm_pct}%;background:{'#22c55e' if sm_run else '#475569'};border-radius:5px;transition:width .5s"></div>
        </div>
        <span style="width:130px;font-size:10px;color:#64748b">{sm_txt[:25]}</span>
    </div>"""

    w = d["weights"]
    wh = "".join(f"<tr><td style='color:#94a3b8;padding:2px 8px'>{k}</td><td style='color:#e2e8f0'>{v:.2f}</td></tr>" for k,v in sorted(w.items()))
    lh = "".join(f"<tr><td style='color:#64748b;font-size:10px'>{t}</td><td style='color:{COLORS.get(m,"#64748b")};font-size:11px'>{m}</td></tr>" for t,m,_ in d["log"][-12:])

    ready = d.get("llama_ready", False)
    status_color = "#22c55e" if ready else "#f43f5e"
    status_txt = "ONLINE" if ready else "OFFLINE"

    return f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-family:monospace">
        <div style="background:#0f1117;padding:14px;border-radius:10px;border:1px solid #1e293b">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <span style="color:#e2e8f0;font-weight:bold">⚡ Aktivität</span>
                <span style="color:{status_color};font-size:11px">Qwen3.6: {status_txt}</span>
            </div>
            {bars}
        </div>
        <div style="background:#0f1117;padding:14px;border-radius:10px;border:1px solid #1e293b">
            <div style="color:#e2e8f0;font-weight:bold;margin-bottom:10px">🧠 Synapse</div>
            <table style="width:100%;font-size:12px">{wh}</table>
        </div>
        <div style="background:#0f1117;padding:14px;border-radius:10px;border:1px solid #1e293b">
            <div style="color:#e2e8f0;font-weight:bold;margin-bottom:10px">🏃 Runner</div>
            {sm_line}
            {spd}
        </div>
        <div style="background:#0f1117;padding:14px;border-radius:10px;border:1px solid #1e293b">
            <div style="color:#e2e8f0;font-weight:bold;margin-bottom:10px">📋 Log</div>
            <table style="width:100%">{lh}</table>
        </div>
    </div>"""

# ─── Chat ─────────────────────────────────────────────────────────
def chat_fn(msg, hist):
    resp = brain.process(msg)
    hist.append((msg, resp))
    return "", hist

def feedback_fn(data: gr.LikeData):
    brain.give_feedback(data.liked)
    return

# ─── Data Upload: nativ verdrahtet - Sensoren fuettern Megatron ──
def handle_upload(file):
    if file is None: return "Keine Datei."
    p = Path(file.name)
    brain._set_activity("vision")
    text = brain.keim.aufnahme_einlesen(str(p))
    if text:
        brain.keim.megatron.denken(text)
        brain._set_activity("keim")
        return (f"Datei empfangen: {p.name} ({p.stat().st_size/1024:.0f} KB)"
                f" - Sensoren: {text[:120]}")
    return f"Datei empfangen: {p.name} ({p.stat().st_size/1024:.0f} KB)"

def handle_camera(img):
    if img is None: return "Kein Bild."
    brain._set_activity("vision")
    tmp = Path(tempfile.gettempdir()) / "megatron_kamera.png"
    try:
        from PIL import Image
        Image.fromarray(img).save(tmp)
    except Exception:
        return f"Kamera-Bild empfangen ({img.shape[1]}x{img.shape[0]})"
    text = brain.keim.aufnahme_einlesen(str(tmp))
    if text:
        brain.keim.megatron.denken(text)
        brain._set_activity("keim")
        return (f"Kamera-Bild empfangen ({img.shape[1]}x{img.shape[0]})"
                f" - Sensoren: {text[:120]}")
    return f"Kamera-Bild empfangen ({img.shape[1]}x{img.shape[0]})"

# ─── Terminal ─────────────────────────────────────────────────────
def terminal_cmd(cmd, history):
    if not cmd.strip(): return "", history or ""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "[Timeout]"
    except Exception as e:
        out = str(e)
    new = (history or "") + f"\n$ {cmd}\n{out}"
    return "", new

# ─── MCP / Tools ──────────────────────────────────────────────────
MCP_TOOLS = {
    "Terminal": "terminal",
    "Dateisystem": "datei",
    "Websuche": "websuche",
}

def mcp_run(tool, prompt):
    brain._set_activity("system")
    text = (prompt or "").strip()
    if tool == "terminal":
        name = text.split()[0] if text else ""
        return mcp_tools.terminal_befehl(name)
    elif tool == "datei":
        teile = text.split(" ", 2)
        if len(teile) < 2:
            return "So: lesen <pfad> | schreiben <pfad> <text> | liste <ordner>"
        return mcp_tools.datei_aktion(teile[0], teile[1],
                                       teile[2] if len(teile) > 2 else "")
    elif tool == "websuche":
        return mcp_tools.websuche(text)
    return f"Unbekanntes Tool: {tool}"

# ─── Harvester ─────────────────────────────────────────────────────
def harvester_go(topic, depth):
    if not topic.strip(): return "Bitte ein Thema eingeben."
    brain._set_activity("harvester")
    for i in range(min(5, depth)):
        brain.harvester_speedys[i] = {"status":"running","progress":i*20,"task":topic[:25]}
    result = brain.ernten(topic)
    for i in range(min(5, depth)):
        brain.harvester_speedys[i] = {"status":"idle","progress":100,"task":""}
    return result

# ─── Gradio App ────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue="teal", secondary_hue="slate", neutral_hue="slate"
).set(body_background_fill="#0f1117", block_background_fill="#1e293b",
       body_text_color="#e2e8f0", border_color_accent="#2dd4bf")

CSS = """footer{display:none!important}
.dashboard-refresh{margin:0!important;padding:0!important}
.terminal-input textarea{font-family:monospace!important;background:#0a0c10!important;color:#22c55e!important}
"""

with gr.Blocks(title="Train4brain 2.0") as demo:
    gr.Markdown("# 🧠 Train4brain 2.0 · Qwen3.6-35B-A3B")
    gr.Markdown("Modulare Schwarm-Intelligenz auf AMD ROCm")

    with gr.Tabs():
        # ─── TAB 1: Dashboard ───
        with gr.TabItem("Dashboard"):
            dash_html = gr.HTML(render_dashboard())
            def refresh(): return render_dashboard()
            gr.Timer(1).tick(refresh, outputs=dash_html)

        # ─── TAB 2: Chat ───
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(label="Neuraler Uplink", height=400)
            msg_in = gr.Textbox(label="Nachricht", placeholder="Was möchtest du wissen?", scale=5)
            with gr.Row():
                send = gr.Button("Senden", variant="primary", scale=1)
                clear = gr.Button("Leeren", scale=1)
            send.click(chat_fn, [msg_in, chatbot], [msg_in, chatbot])
            msg_in.submit(chat_fn, [msg_in, chatbot], [msg_in, chatbot])
            clear.click(lambda: ([], ""), outputs=[chatbot, msg_in])
            chatbot.like(feedback_fn, None, None)

        # ─── TAB 3: Data ───
        with gr.TabItem("Data"):
            gr.Markdown("### Datenquellen")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Dateien**")
                    file_up = gr.File(label="Datei hochladen", file_types=[".txt",".json",".jsonl",".csv",".pdf"])
                    file_out = gr.Textbox(label="Result", interactive=False)
                    file_up.upload(handle_upload, file_up, file_out)
                with gr.Column():
                    gr.Markdown("**Bilder**")
                    img_up = gr.Image(label="Bild hochladen", type="numpy")
                    img_out = gr.Textbox(label="Result", interactive=False)
                    img_up.upload(lambda i: handle_upload(i) if hasattr(i,'name') else "Bild empfangen", img_up, img_out)
                with gr.Column():
                    gr.Markdown("**Kamera**")
                    cam = gr.Image(label="Kamera", type="numpy", sources=["webcam"])
                    cam_out = gr.Textbox(label="Result", interactive=False)
                    cam.change(handle_camera, cam, cam_out)

        # ─── TAB 4: Run ───
        with gr.TabItem("Run"):
            gr.Markdown("### 🏃 Harvester Speedys")
            gr.Markdown("Parallele Wissenssammler. Starte bis zu 5 Harvesting-Jobs gleichzeitig.")
            with gr.Row():
                topic_in = gr.Textbox(label="Thema / URL", placeholder="Wikipedia-Artikel oder URL...", scale=4)
                depth_in = gr.Slider(1, 5, value=3, step=1, label="Speedys", scale=1)
            harv_btn = gr.Button("🍯 Ernten!", variant="primary")
            harv_out = gr.Textbox(label="Ergebnis", lines=4)
            harv_btn.click(harvester_go, [topic_in, depth_in], harv_out)
            sammler_btn = gr.Button("🌾 Alle Gebiete ernten (Hintergrund)", variant="primary")
            sammler_btn.click(lambda: brain.sammler.starten(), None, harv_out)
            lern_btn = gr.Button("🧠 Lernzyklus: Benchmark + positive Verstärkung", variant="primary")
            lern_btn.click(lambda: brain.lernen_zyklus(), None, harv_out)

        # ─── TAB 5: Terminal ───
        with gr.TabItem("Terminal"):
            gr.Markdown("### 💻 Terminal")
            gr.Markdown("Führe Shell-Befehle aus (Vorsicht: volle System-API).")
            term_out = gr.Textbox(label="Ausgabe", lines=15, interactive=False, elem_classes="terminal-input")
            term_in = gr.Textbox(label="$", placeholder="Befehl eingeben...", elem_classes="terminal-input")
            term_in.submit(terminal_cmd, [term_in, term_out], [term_in, term_out])

        # ─── TAB 6: MCP ───
        with gr.TabItem("MCP"):
            gr.Markdown("### 🔧 MCP Tools")
            gr.Markdown("Lokale API-Tools für Bildgenerierung, Transkription und mehr.")
            tool_dd = gr.Dropdown(list(MCP_TOOLS.keys()), label="Tool", value="Terminal")
            mcp_prompt = gr.Textbox(label="Prompt", placeholder="Was soll generiert/verarbeitet werden?")
            mcp_btn = gr.Button("Ausführen", variant="primary")
            mcp_out = gr.Textbox(label="Result", lines=4)
            mcp_btn.click(mcp_run, [tool_dd, mcp_prompt], mcp_out)

    gr.Markdown("---\n*Train4brain 2.0 · Qwen3.6-35B-A3B · AMD ROCm*")

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:7865")
    demo.launch(server_name="127.0.0.1", server_port=7865, show_error=True, theme=theme, css=CSS)
