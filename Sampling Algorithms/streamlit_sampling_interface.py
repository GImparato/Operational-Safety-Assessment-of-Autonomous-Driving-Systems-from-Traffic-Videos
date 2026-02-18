# =========================================================
# STREAMLIT SAMPLING & RMSE INTERFACE
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import importlib
import inspect

from utils.operational_profile import generate_operational_profile

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Sampling & RMSE Interface", layout="wide")
st.title("Sampling Algorithms Evaluation")
st.caption("Final Streamlit – no-profile / profile, ODD, multi-algorithm")

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def normalize_outcome(df):
    if "Outcome" in df.columns:
        df["Outcome"] = df["Outcome"].astype(int)
        return df
    if "collision_occurred" in df.columns:
        df["Outcome"] = df["collision_occurred"].astype(int)
        return df
    if "collision" in df.columns:
        df["Outcome"] = df["collision"].astype(int)
        return df
    raise KeyError("No Outcome / collision column found")


def compute_true_accuracy(df):
    return 1.0 - df["Outcome"].mean()


def compute_rmse(estimates, true_value):
    est = np.asarray(estimates, dtype=float)
    return float(np.sqrt(np.mean((est - true_value) ** 2)))


def call_algorithm_safely(algo_fn, **candidate_kwargs):
    """
    Calls algo_fn passing ONLY the parameters
    present in its function signature.
    """
    sig = inspect.signature(algo_fn)
    accepted = sig.parameters.keys()
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    return algo_fn(**filtered_kwargs)


def build_odd_partition(df, odd_columns):
    df = df.copy()
    df["partition"] = df[odd_columns].astype(str).agg("_".join, axis=1)
    return df

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload scenario_metrics CSV", type=["csv"])
if uploaded_file is None:
    st.stop()

if "df_raw" not in st.session_state:
    df = pd.read_csv(uploaded_file)
    df = normalize_outcome(df)
    st.session_state["df_raw"] = df
else:
    df = st.session_state["df_raw"]

true_acc = compute_true_accuracy(df)
st.markdown(f"**True accuracy (full dataset):** `{true_acc:.5f}`")

# ---------------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------------
st.sidebar.subheader("Execution mode")

use_profile = st.sidebar.radio(
    "Sampling mode",
    ["Without operational profile", "With operational profile"]
) == "With operational profile"

# ---------------------------------------------------------
# OPERATIONAL PROFILE
# ---------------------------------------------------------
p_i = None
if use_profile:
    st.sidebar.subheader("Operational profile")

    profile_mode = st.sidebar.selectbox(
    "Profile type",
    ["uniform", "uniform_random", "dirichlet",
     "extreme_weather", "urban_complex"]
)


    profile_seed = st.sidebar.number_input("Profile seed", value=42)

if st.sidebar.button("Generate profile"):

    # --- NEW PROFILE LOGIC START ---

    if profile_mode in ["uniform", "uniform_random", "dirichlet"]:
        p_i = generate_operational_profile(df, profile_mode, profile_seed)

    elif profile_mode == "extreme_weather":

        illum_map = {"day": 0, "dusk_dawn": 0.5, "night": 1}
        vis_map = {"good": 0, "moderate": 0.5, "poor": 1}

        illumination = df["illumination"].map(illum_map).fillna(0)
        visibility = df["visibility"].map(vis_map).fillna(0)
        precipitation = df["precipitation_visible"].astype(int)

        traffic = df["traffic_density_value"]
        traffic_norm = (traffic - traffic.min()) / (traffic.max() - traffic.min())

        score = (
            0.3 * illumination +
            0.3 * visibility +
            0.2 * precipitation +
            0.2 * traffic_norm
        )

        alpha = 3
        p_i = np.exp(alpha * score)
        p_i = p_i / p_i.sum()

    elif profile_mode == "urban_complex":

        pedestrians = df["num_pedestrians"]
        ped_norm = (pedestrians - pedestrians.min()) / (pedestrians.max() - pedestrians.min())

        traffic = df["traffic_density_value"]
        traffic_norm = (traffic - traffic.min()) / (traffic.max() - traffic.min())

        odd_events = df["num_odd_events"]
        odd_norm = (odd_events - odd_events.min()) / (odd_events.max() - odd_events.min())

        urban_flag = df["odd_main_category"].isin(
            ["intersection", "roundabout"]
        ).astype(int)

        score = (
            0.3 * ped_norm +
            0.3 * traffic_norm +
            0.2 * odd_norm +
            0.2 * urban_flag
        )

        alpha = 3
        p_i = np.exp(alpha * score)
        p_i = p_i / p_i.sum()

    # --- NEW PROFILE LOGIC END ---

    st.session_state["p_i"] = p_i
    st.sidebar.success("Operational profile generated")


# ---------------------------------------------------------
# ALGORITHMS REGISTRY (NAME-ALIGNED)
# ---------------------------------------------------------
ALGORITHMS_NO_PROFILE = {
    "SRS":     ("no_profile.SRS", "srs_sampling"),
    "SUPS":    ("no_profile.SUPS", "sups_sampling"),
    "RHC":     ("no_profile.RHC", "rhc_sampling"),
    "SSRS":    ("no_profile.SSRS", "ssrs_sampling"),
    "2UPS":    ("no_profile.2UPS", "two_ups_sampling"),
    "DEEPEST": ("no_profile.DEEPEST", "deepest_sampling"),
}

ALGORITHMS_PROFILE = {
    "SRS_Prof":     ("profile.SRS_Prof", "srs_sampling_prof"),
    "SUPS_Prof":    ("profile.SUPS_Prof", "sups_sampling_prof"),
    "RHC_Prof":     ("profile.RHC_Prof", "rhc_sampling_prof"),
    "2UPS_Prof":    ("profile.2UPS_Prof", "two_ups_sampling_prof"),
    "DEEPEST_Prof": ("profile.DEEPEST_Prof", "deepest_sampling_prof"),
}

ALGORITHMS = ALGORITHMS_PROFILE if use_profile else ALGORITHMS_NO_PROFILE

# ---------------------------------------------------------
# SAMPLING CONFIG
# ---------------------------------------------------------
st.sidebar.subheader("Sampling")

multi_algo = st.sidebar.checkbox("Compare multiple algorithms")

if multi_algo:
    selected_algos = st.sidebar.multiselect(
        "Algorithms",
        list(ALGORITHMS.keys()),
        default=[list(ALGORITHMS.keys())[0]]
    )
else:
    selected_algos = [st.sidebar.selectbox("Algorithm", list(ALGORITHMS.keys()))]

budget = st.sidebar.slider(
    "Budget",
    min_value=1,
    max_value=len(df),
    value=min(200, len(df)),
    step=1
)

base_seed = st.sidebar.number_input("Base seed", value=123)

aux_candidates = [
    c for c in df.columns
    if ("ttc" in c.lower() or "prob" in c.lower() or "poc" in c.lower())
]

aux_var = st.sidebar.selectbox("Auxiliary variable", aux_candidates) if aux_candidates else None

# ---------------------------------------------------------
# ODD PARTITIONING (OPTIONAL)
# ---------------------------------------------------------
st.sidebar.subheader("ODD partitioning")

use_odd = st.sidebar.checkbox("Use ODD partitioning")

odd_columns = []
if use_odd:
    odd_candidates = [c for c in df.columns if c.lower().startswith("odd")]
    odd_columns = st.sidebar.multiselect("ODD columns", odd_candidates)

# ---------------------------------------------------------
# DEEPEST FEATURES (ONLY IF NEEDED)
# ---------------------------------------------------------
deepest_features = None

if any("DEEPEST" in a for a in selected_algos):
    st.sidebar.subheader("DEEPEST feature space")

    feature_candidates = [
        c for c in df.columns
        if c not in {"Outcome", "partition"}
        and np.issubdtype(df[c].dtype, np.number)
    ]

    deepest_features = st.sidebar.multiselect(
        "Feature columns (DEEPEST)",
        feature_candidates
    )

    if not deepest_features:
        st.warning("DEEPEST selected: please choose at least one feature.")
        st.stop()

# ---------------------------------------------------------
# RUN SAMPLING
# ---------------------------------------------------------
if st.button("Run sampling"):
    results = []

    if use_profile:
        if "p_i" not in st.session_state:
            st.error("Operational profile not generated.")
            st.stop()
        p_i = st.session_state["p_i"]


    for algo_name in selected_algos:
        module_name, fn_name = ALGORITHMS[algo_name]
        module = importlib.import_module(module_name)
        algo_fn = getattr(module, fn_name)

        df_run = df
        if use_odd and odd_columns:
            df_run = build_odd_partition(df_run, odd_columns)

        sampled_df = call_algorithm_safely(
            algo_fn,
            df=df_run,
            budget=budget,
            seed=seed,
            aux_var=aux_var,
            p_i=st.session_state.get("p_i") if use_profile else None,
            features=deepest_features
        )


        sampled_df = normalize_outcome(sampled_df)
        sampled_df["algorithm"] = algo_name
        results.append(sampled_df)

    out_df = pd.concat(results, ignore_index=True)
    st.session_state["sampled_df"] = out_df
    st.success("Sampling completed")

# ---------------------------------------------------------
# SHOW SAMPLE
# ---------------------------------------------------------
if "sampled_df" in st.session_state:
    st.subheader("Sampled dataset")
    st.dataframe(st.session_state["sampled_df"])

    buf = BytesIO()
    st.session_state["sampled_df"].to_csv(buf, index=False)
    st.download_button(
        "Download sampled CSV",
        buf.getvalue(),
        "sampled_results.csv",
        "text/csv"
    )

# ---------------------------------------------------------
# RMSE
# ---------------------------------------------------------
st.subheader("RMSE evaluation")

R = st.number_input("Repetitions (R)", min_value=1, max_value=100, value=30)

if st.button("Run RMSE"):
    rows = []

    for algo_name in selected_algos:
        module_name, fn_name = ALGORITHMS[algo_name]
        module = importlib.import_module(module_name)
        algo_fn = getattr(module, fn_name)

        estimates = []

        for r in range(R):
            seed = base_seed + r

            df_run = df
            if use_odd and odd_columns:
                df_run = build_odd_partition(df_run, odd_columns)

            sampled_df = call_algorithm_safely(
                algo_fn,
                df=df_run,
                budget=budget,
                seed=seed,
                aux_var=aux_var,
                p_i=st.session_state.get("p_i") if use_profile else None,
                features=deepest_features
            )


            sampled_df = normalize_outcome(sampled_df)
            acc = 1.0 - sampled_df["Outcome"].mean()
            estimates.append(acc)

        rows.append({
            "algorithm": algo_name,
            "budget": budget,
            "rmse": compute_rmse(estimates, true_acc),
            "mean_accuracy": np.mean(estimates),
            "runs": R
        })

    rmse_df = pd.DataFrame(rows)
    st.dataframe(rmse_df)

    buf = BytesIO()
    rmse_df.to_csv(buf, index=False)
    st.download_button(
        "Download RMSE CSV",
        buf.getvalue(),
        "rmse_results.csv",
        "text/csv"
    )
# ---------------------------------------------------------
# RQ2 – FAILURE DETECTION
# ---------------------------------------------------------

st.subheader("RQ2 – Failure detection capability")

if st.button("Run Failure Detection"):
    rows = []

    for algo_name in selected_algos:
        module_name, fn_name = ALGORITHMS[algo_name]
        module = importlib.import_module(module_name)
        algo_fn = getattr(module, fn_name)

        failures_list = []

        for r in range(R):
            seed = base_seed + r

            df_run = df
            if use_odd and odd_columns:
                df_run = build_odd_partition(df_run, odd_columns)

            sampled_df = call_algorithm_safely(
                algo_fn,
                df=df_run,
                budget=budget,
                seed=seed,
                aux_var=aux_var,
                p_i=st.session_state.get("p_i") if use_profile else None,
                features=deepest_features
            )

            sampled_df = normalize_outcome(sampled_df)

            failures = sampled_df["Outcome"].sum()
            failures_list.append(failures)

        rows.append({
            "algorithm": algo_name,
            "budget": budget,
            "mean_failures": float(np.mean(failures_list)),
            "std_failures": float(np.std(failures_list)),
            "runs": R
        })

    fail_df = pd.DataFrame(rows)
    st.dataframe(fail_df)

    buf = BytesIO()
    fail_df.to_csv(buf, index=False)
    st.download_button(
        "Download Failure Detection CSV",
        buf.getvalue(),
        "rq2_failure_results.csv",
        "text/csv"
    )
