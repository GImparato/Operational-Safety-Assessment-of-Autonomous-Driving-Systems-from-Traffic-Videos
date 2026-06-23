import streamlit as st
import pandas as pd
import numpy as np
import importlib
import time
from io import BytesIO

from utils.operational_profile import generate_operational_profile

# =========================================================
# PAGE
# =========================================================

st.set_page_config(layout="wide")
st.title("Full Experimental Results Generator")

# =========================================================
# DATASET
# =========================================================

DATASET_PATH = "scenario_metrics.csv"

df = pd.read_csv(DATASET_PATH)

# =========================================================
# NORMALIZE OUTCOME
# =========================================================

if "Outcome" not in df.columns:

    if "collision_occurred" in df.columns:
        df["Outcome"] = df["collision_occurred"].astype(int)

    elif "collision" in df.columns:
        df["Outcome"] = df["collision"].astype(int)

    else:
        raise ValueError("No Outcome column found")

# =========================================================
# CONFIG
# =========================================================

RUNS = 30

BUDGETS = [50, 100, 200, 300, 400]

ODD_COLUMNS = [
    "odd_main_category",
    "illumination",
    "visibility"
]

AUX_MAPPING = {
    "TTC": "ttc_min",
    "PC": "probability_of_collision"
}

# =========================================================
# METHODS
# =========================================================

METHODS = {

    "SRS": {
        "module": "profile.SRS_Prof",
        "sampling_fn": "srs_sampling_prof",
        "metrics_fn": "compute_metrics_srs_prof",
        "partitioning": ["none"]
    },

    "SUPS": {
        "module": "profile.SUPS_Prof",
        "sampling_fn": "sups_sampling_prof",
        "metrics_fn": "compute_metrics_sups_prof",
        "partitioning": ["none"]
    },

    "RHC": {
        "module": "profile.RHC_Prof",
        "sampling_fn": "rhc_sampling_prof",
        "metrics_fn": "compute_metrics_rhc_prof",
        "partitioning": ["none"]
    },

    "SSRS": {
    "module": "no_profile.SSRS",
    "sampling_fn": "ssrs_sampling",
    "metrics_fn": "compute_metrics_ssrs_prof",
    "partitioning": ["ODD", "HDBSCAN"]
    },

    "2UPS": {
        "module": "profile.2UPS_Prof",
        "sampling_fn": "two_ups_sampling_prof",
        "metrics_fn": "compute_metrics_2ups_prof",
        "partitioning": ["ODD", "HDBSCAN"]
    },

    "DEEPEST": {
        "module": "profile.DEEPEST_Prof",
        "sampling_fn": "deepest_sampling_prof",
        "metrics_fn": "compute_metrics_deepest_operational",
        "partitioning": ["none"]
    }
}

# =========================================================
# PROFILE GENERATION
# =========================================================

def generate_profile(df, profile_name, seed=42):

    N = len(df)

    if profile_name == "P0":
        return np.ones(N) / N

    elif profile_name == "P1":

        return generate_operational_profile(
            df,
            mode="uniform_random",
            seed=seed
        )

    elif profile_name == "P2":

        illum_map = {
            "day": 0,
            "dusk_dawn": 0.5,
            "night": 1
        }

        vis_map = {
            "good": 0,
            "moderate": 0.5,
            "poor": 1
        }

        illumination = df["illumination"].map(illum_map).fillna(0)

        visibility = df["visibility"].map(vis_map).fillna(0)

        precipitation = df["precipitation_visible"].astype(int)

        traffic = df["traffic_density_value"]

        traffic_norm = (
            (traffic - traffic.min()) /
            (traffic.max() - traffic.min() + 1e-12)
        )

        score = (
            0.3 * illumination +
            0.3 * visibility +
            0.2 * precipitation +
            0.2 * traffic_norm
        )

        p_i = np.exp(3 * score)
        p_i = p_i / p_i.sum()

        return p_i

    elif profile_name == "P3":

        pedestrians = df["num_pedestrians"]

        ped_norm = (
            (pedestrians - pedestrians.min()) /
            (pedestrians.max() - pedestrians.min() + 1e-12)
        )

        traffic = df["traffic_density_value"]

        traffic_norm = (
            (traffic - traffic.min()) /
            (traffic.max() - traffic.min() + 1e-12)
        )

        odd_events = df["num_odd_events"]

        odd_norm = (
            (odd_events - odd_events.min()) /
            (odd_events.max() - odd_events.min() + 1e-12)
        )

        urban_flag = df["odd_main_category"].isin(
            ["intersection", "roundabout"]
        ).astype(int)

        score = (
            0.3 * ped_norm +
            0.3 * traffic_norm +
            0.2 * odd_norm +
            0.2 * urban_flag
        )

        p_i = np.exp(3 * score)
        p_i = p_i / p_i.sum()

        return p_i

# =========================================================
# TRUE THETA
# =========================================================

def compute_theta_true(df, p_i):

    z = df["Outcome"].to_numpy(dtype=float)

    return float(np.sum(z * p_i))

# =========================================================
# EXECUTE
# =========================================================

if st.button("Generate Full Results CSV"):

    all_results = []

    total_configs = 0

    for m in METHODS.values():
        total_configs += (
            4 *
            len(AUX_MAPPING) *
            len(m["partitioning"]) *
            len(BUDGETS) *
            RUNS
        )

    progress = st.progress(0)

    counter = 0

    for method_name, config in METHODS.items():

        st.write(f"Running {method_name}")

        module = importlib.import_module(config["module"])

        sampling_fn = getattr(
            module,
            config["sampling_fn"]
        )

        # -------------------------------------------------
        # Metrics function import
        # -------------------------------------------------

        if method_name == "SSRS":

            metrics_module = importlib.import_module(
                "profile.SSRS_Prof"
            )

            metrics_fn = getattr(
                metrics_module,
                "compute_metrics_ssrs_prof"
            )

        else:

            metrics_fn = getattr(
                module,
                config["metrics_fn"]
            )

        for profile_name in ["P0", "P1", "P2", "P3"]:

            p_i = generate_profile(
                df,
                profile_name,
                seed=42
            )

            theta_true = compute_theta_true(df, p_i)

            for aux_label, aux_var in AUX_MAPPING.items():

                for partitioning in config["partitioning"]:

                    for budget in BUDGETS:

                        for run in range(1, RUNS + 1):

                            counter += 1

                            progress.progress(
                                counter / total_configs
                            )

                            seed = 123 + run

                            try:

                                df_run = df.copy()

                                kwargs = {
                                    "df": df_run,
                                    "budget": budget,
                                    "seed": seed
                                }

                                # -------------------------
                                # AUX
                                # -------------------------

                                if method_name != "SRS":
                                    kwargs["aux_var"] = aux_var

                                # -------------------------
                                # PROFILE
                                # -------------------------

                                df_run["p"] = p_i

                                if method_name == "SRS":
                                    kwargs["p_i"] = p_i

                                # -------------------------
                                # PARTITIONING
                                # -------------------------

                                if partitioning == "ODD":

                                    kwargs["partitioning"] = "odd"

                                    kwargs["odd_columns"] = ODD_COLUMNS

                                    # build partition also in full df
                                    df_run["partition"] = (
                                        df_run[ODD_COLUMNS]
                                        .astype(str)
                                        .agg("_".join, axis=1)
                                    )

                                elif partitioning == "HDBSCAN":

                                    kwargs["partitioning"] = "hdbscan"

                                    kwargs["min_cluster_size"] = 30

                                    # build partition also in full df
                                    from no_profile.SSRS import build_partitions_hdbscan

                                    df_run = build_partitions_hdbscan(
                                        df_run,
                                        aux_var,
                                        30
                                    )

                                # -------------------------
                                # SAMPLE
                                # -------------------------

                                start = time.time()

                                sampled_df = sampling_fn(**kwargs)

                                exec_time = (
                                    time.time() - start
                                )

                                # -------------------------
                                # THETA HAT
                                # -------------------------

                                if method_name == "SRS":

                                    accuracy_hat, failures_obs = (
                                        metrics_fn(sampled_df)
                                    )

                                    theta_hat = (
                                        1.0 - accuracy_hat
                                    )

                                elif method_name == "SUPS":

                                    (
                                        accuracy_hat,
                                        theta_hat,
                                        failures_obs
                                    ) = metrics_fn(
                                        sampled_df,
                                        p_i
                                    )

                                elif method_name == "RHC":

                                    (
                                        accuracy_hat,
                                        theta_hat,
                                        failures_obs
                                    ) = metrics_fn(
                                        sampled_df,
                                        p_i
                                    )

                                elif method_name == "SSRS":

                                    (
                                        accuracy_hat,
                                        theta_hat,
                                        failures_obs
                                    ) = metrics_fn(
                                        sampled_df,
                                        df_run,
                                        p_i
                                    )

                                elif method_name == "2UPS":

                                    accuracy_hat, failures_obs = (
                                        metrics_fn(
                                            sampled_df,
                                            p_i
                                        )
                                    )

                                    theta_hat = (
                                        1.0 - accuracy_hat
                                    )

                                elif method_name == "DEEPEST":

                                    (
                                        accuracy_hat,
                                        theta_hat,
                                        failures_obs
                                    ) = metrics_fn(
                                        sampled_df,
                                        profile_col="p"
                                    )

                                row = {

                                    "sampling_method": method_name,

                                    "profile": profile_name,

                                    "aux_var": aux_label,

                                    "partitioning": partitioning,

                                    "budget": budget,

                                    "run_id": run,

                                    "theta_hat": theta_hat,

                                    "theta_true": theta_true,

                                    "failures_obs": failures_obs,

                                    "execution_time": exec_time
                                }

                                all_results.append(row)

                            except Exception as e:

                                st.error(
                                    f"{method_name} | "
                                    f"{profile_name} | "
                                    f"{budget} | "
                                    f"run {run}"
                                )

                                st.exception(e)

    results_df = pd.DataFrame(all_results)

    st.success("Generation completed!")

    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Full Results CSV",
        csv,
        file_name="all_sampling_results.csv",
        mime="text/csv"
    )