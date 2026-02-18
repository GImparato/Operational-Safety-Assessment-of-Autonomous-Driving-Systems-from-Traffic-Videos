#!/usr/bin/env python3
import streamlit as st
import subprocess
import os
import time
import json
import pandas as pd
import plotly.graph_objects as go
import shutil
import socket
from pathlib import Path
import yaml

# === NEW IMPORT (ANALYSIS) ===
from scripts.analysis.analyze_scenario import analyze_scenario_from_json


# ============================================================================
# SESSION STATE (PATCH MINIMA E CORRETTA)
# ============================================================================
if "last_scenario_path" not in st.session_state:
    st.session_state.last_scenario_path = None

if "last_video_name" not in st.session_state:
    st.session_state.last_video_name = None


# === BASE PATHS ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "outputs" / "raw_scenarios"
MOD_DIR = BASE_DIR / "outputs" / "modified_scenarios"
RESULTS_DIR = BASE_DIR / "results"
SIM_SCRIPT = BASE_DIR / "run_svl_from_json.py"

DEFAULT_BRIDGE_IP = "100.68.170.116"
DEFAULT_BRIDGE_PORT = 9090

for p in [DATA_DIR, RAW_DIR, MOD_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILITIES
# ============================================================================
def run_command(cmd, cwd=None):
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True
    )

    output = ""
    log_box = st.empty()

    for line in iter(process.stdout.readline, ''):
        output += line
        log_box.code(output[-3000:], language="bash")

    process.wait()
    return process.returncode, output


def check_bridge(ip, port, timeout=2):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        return True
    except:
        return False


# ============================================================================
# FIX JSON SORA-SVL
# ============================================================================
def fix_sora_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    if "agents" not in data:
        data["agents"] = []

    ego_exists = any(a.get("type", "").upper() == "EGO" for a in data["agents"])
    if not ego_exists:
        data["agents"].insert(0, {
            "id": "ego",
            "name": "EgoVehicle",
            "type": "EGO",
            "vehicleModel": "Jaguar2015XE",
            "spawn": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 90.0}
        })
    else:
        for a in data["agents"]:
            if a.get("type", "").upper() == "EGO":
                a["vehicleModel"] = "Jaguar2015XE"

    for a in data["agents"]:
        if a.get("type", "").upper() == "NPC":
            a.setdefault("vehicleModel", "Sedan")
            a.setdefault("spawn", {"x": 5.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
            a.setdefault("behavior", {
                "controller": "FollowWaypoints",
                "waypoints": [
                    {"x": a["spawn"]["x"], "y": a["spawn"]["y"], "z": 0.0, "speed": 5},
                    {"x": a["spawn"]["x"] + 10, "y": a["spawn"]["y"], "z": 0.0, "speed": 5}
                ]
            })

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    st.success("Scenario aggiornato (compatibile con SORA-SVL 2021.3)")
    return data


# ============================================================================
# VISUALIZATION
# ============================================================================
def validate_agents_table(data):
    rows = []
    for a in data.get("agents", []):
        s = a.get("spawn", {})
        rows.append({
            "ID": a.get("id", ""),
            "Nome": a.get("name", ""),
            "Tipo": a.get("type", ""),
            "Modello": a.get("vehicleModel", ""),
            "Spawn": f"x={s.get('x',0):.1f}, y={s.get('y',0):.1f}, yaw={s.get('yaw',0):.0f}"
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def plot_agents_map(data):
    ego_x, ego_y, npc_x, npc_y = [], [], [], []

    for a in data.get("agents", []):
        s = a.get("spawn", {})
        if a.get("type", "").upper() == "EGO":
            ego_x.append(s.get("x", 0))
            ego_y.append(s.get("y", 0))
        else:
            npc_x.append(s.get("x", 0))
            npc_y.append(s.get("y", 0))

    fig = go.Figure()

    if ego_x:
        fig.add_trace(go.Scatter(
            x=ego_x, y=ego_y, mode="markers+text",
            text=["EGO"], marker=dict(size=12, color="red")
        ))

    if npc_x:
        fig.add_trace(go.Scatter(
            x=npc_x, y=npc_y, mode="markers",
            marker=dict(size=8, color="blue")
        ))

    fig.update_layout(
        title="Spawn Points (EGO/NPC)",
        xaxis_title="X",
        yaxis_title="Y",
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True,
                    key=f"plot_{int(time.time()*1e6)}")


def find_scenarios(folder):
    return sorted(folder.glob("scenario_svl*.json"),
                  key=os.path.getmtime, reverse=True)


# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="Video2Scenario",
                   page_icon="🎬", layout="wide")

st.title("🎬 Video → Scenario → Simulazione (SORA-SVL + Apollo 6)")

tabs = st.tabs([
    "🎥 Genera Scenario",
    "🚗 Simulazione",
    "🛠 Modifica Scenario",
    "📊 Risultati"
])

# ============================================================================
# TAB 1 – GENERA SCENARIO
# ============================================================================
with tabs[0]:
    st.header("🎥 Genera scenario dal video")

    uploaded_video = st.file_uploader(
        "Carica video (.mp4)", type=["mp4"],
        key="upload_video_gen"
    )

    if uploaded_video:
        save_path = DATA_DIR / uploaded_video.name
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.session_state.last_video_name = uploaded_video.name
        st.success(f"Video salvato in {save_path}")

        if st.button("⚙️ Genera Scenario JSON", key="generate_json_btn"):
            code, out = run_command(
                ["bash", "run_pipeline.sh", str(save_path)],
                cwd=BASE_DIR
            )

            if code == 0:
                # === PRENDI LO SCENARIO SVL PIÙ RECENTE ===
                candidates = sorted(
                    (BASE_DIR / "outputs").glob("scenario_svl*.json"),
                    key=os.path.getmtime,
                    reverse=True
                )

                if candidates:
                    src = candidates[0]
                    dst = RAW_DIR / f"{src.stem}_{int(time.time())}.json"
                    shutil.move(str(src), str(dst))

                    st.session_state.last_scenario_path = str(dst)

                    # === SALVA PATH DEL FILE DI ANALISI (TRAJECTORIES) ===
                    st.session_state.last_trajectories_path = str(
                        BASE_DIR / "outputs" / "trajectories.json"
                    )

                    data = fix_sora_json(dst)
                    st.success(f"Scenario JSON creato: {dst.name}")

                    validate_agents_table(data)
                    plot_agents_map(data)
                else:
                    st.error("Nessuno scenario generato.")
            else:
                st.error("Errore pipeline.")
                st.code(out)

    # ==================================================
    # ANALISI SCENARIO (NO SIMULAZIONE)
    # ==================================================
    if (
        "last_trajectories_path" in st.session_state
        and st.session_state.last_trajectories_path is not None
    ):
        st.divider()
        st.caption("Analisi basata sui dati estratti dal video (trajectories.json)")

        if st.button("📄 Analizza scenario (no simulazione)", key="analyze_no_sim"):
            scenario_id = analyze_scenario_from_json(
                scenario_json_path=st.session_state.last_trajectories_path,
                source_video=st.session_state.last_video_name,
                csv_path=str(RESULTS_DIR / "scenario_metrics.csv")
            )

            st.success(
                f"CSV aggiornato correttamente – Scenario ID: {scenario_id}"
            )





# ============================================================================
# TAB 2 – SIMULAZIONE
# ============================================================================
with tabs[1]:
    st.header("🚗 Avvia simulazione")

    conf = yaml.safe_load(open(BASE_DIR / "config.yaml"))
    sora_conf = conf.get("sora", {})
    sora_conf.setdefault("maps", ["BorregasAve"])

    scenarios = find_scenarios(RAW_DIR) + find_scenarios(MOD_DIR)

    if scenarios:
        scenario_choice = st.selectbox(
            "Scenario", scenarios,
            format_func=lambda x: x.name,
            key="scenario_select_sim"
        )

        duration = st.slider("Durata (s)", 5, 300, 30)
        selected_map = st.selectbox("Mappa SVL", sora_conf["maps"])

        bridge_ip = st.text_input("IP Apollo", DEFAULT_BRIDGE_IP)
        bridge_port = st.number_input("Porta Apollo", value=DEFAULT_BRIDGE_PORT)

        if st.button("🚀 Avvia simulazione"):
            cmd = [
                "python3", str(SIM_SCRIPT),
                "--scenario", str(scenario_choice),
                "--duration", str(duration),
                "--bridge", bridge_ip,
                "--port", str(int(bridge_port)),
                "--map", selected_map
            ]

            code, out = run_command(cmd, cwd=BASE_DIR)

            if code == 0:
                st.success("Simulazione completata!")
            else:
                st.error("Errore simulazione")
                st.code(out)


# ============================================================================
# TAB 3 – MODIFICA SCENARIO
# ============================================================================
with tabs[2]:
    st.header("🛠 Modifica Scenario")

    all_s = find_scenarios(RAW_DIR) + find_scenarios(MOD_DIR)
    if all_s:
        sel = st.selectbox(
            "Scenario",
            all_s,
            format_func=lambda x: x.name,
            key="scenario_select_modify"
        )

        data = fix_sora_json(sel)
        validate_agents_table(data)
        plot_agents_map(data)


# ============================================================================
# TAB 4 – RISULTATI
# ============================================================================
with tabs[3]:
    st.header("📊 Risultati")

    csv_path = RESULTS_DIR / "scenario_metrics.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        st.subheader("📄 Scenari analizzati")
        st.dataframe(df, use_container_width=True)

        # ============================================================
        # ❌ ELIMINA SINGOLO SCENARIO (CON CONFERMA)
        # ============================================================
        st.divider()
        st.subheader("❌ Elimina singolo scenario")

        if "scenario_id" in df.columns:
            scenario_ids = df["scenario_id"].astype(str).tolist()

            selected_id = st.selectbox(
                "Seleziona Scenario ID da eliminare",
                scenario_ids,
                key="delete_scenario_select"
            )

            confirm_delete_one = st.checkbox(
                f"Confermo l'eliminazione dello scenario {selected_id}",
                key="confirm_delete_one"
            )

            if st.button(
                "🗑️ Elimina scenario selezionato",
                disabled=not confirm_delete_one
            ):
                df_new = df[df["scenario_id"].astype(str) != selected_id]

                if len(df_new) == len(df):
                    st.warning("Nessuna riga eliminata (ID non trovato).")
                else:
                    df_new.to_csv(csv_path, index=False)
                    st.success(f"Scenario {selected_id} eliminato correttamente.")
                    st.rerun()
        else:
            st.warning("Il CSV non contiene la colonna 'scenario_id'.")

        # ============================================================
        # 🧹 ELIMINA COMPLETAMENTE IL CSV (CON CONFERMA FORTE)
        # ============================================================
        st.divider()
        st.subheader("🧹 Elimina TUTTI i risultati")

        st.warning(
            "⚠️ Questa operazione eliminerà **TUTTI** gli scenari analizzati.\n"
            "Il file scenario_metrics.csv verrà cancellato definitivamente."
        )

        confirm_delete_all = st.checkbox(
            "Confermo l'eliminazione COMPLETA del CSV",
            key="confirm_delete_all"
        )

        if st.button(
            "🔥 Elimina TUTTO il CSV",
            disabled=not confirm_delete_all
        ):
            try:
                csv_path.unlink()
                st.success("CSV eliminato correttamente.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore durante l'eliminazione: {e}")

        # ============================================================
        # ⬇️ DOWNLOAD (SOLO SE ESISTE ANCORA)
        # ============================================================
        st.divider()
        if csv_path.exists():
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Scarica CSV",
                    data=f,
                    file_name="scenario_metrics.csv",
                    mime="text/csv"
                )

    else:
        st.info("Nessun risultato disponibile.")
