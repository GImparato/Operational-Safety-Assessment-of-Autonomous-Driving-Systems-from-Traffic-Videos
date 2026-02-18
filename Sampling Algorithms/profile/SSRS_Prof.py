import numpy as np
import pandas as pd

# =========================================================
# SSRS WITH OPERATIONAL PROFILE – METRICS ONLY
# Option A: profile-weighted estimator (paper-consistent)
# =========================================================

def compute_metrics_ssrs_prof(
    sampled_df: pd.DataFrame,
    full_df: pd.DataFrame,
    p_i: np.ndarray,
):
    """
    SSRS with operational profile (Option A).

    Sampling:
      - IDENTICAL to standard SSRS (no profile involved)

    Estimator:
      - Operational profile is used ONLY in the estimator

    Let:
      - p_i be the operational profile over scenarios (sum p_i = 1)
      - h index the strata (partitions)
      - z_i = Outcome (0/1)

    Estimator:
      failure_rate_hat = sum_h ( P_h * mean_h(z) )

    where:
      P_h = sum_{i in stratum h} p_i
      mean_h(z) = mean Outcome of sampled points in stratum h

    Returns:
      accuracy_hat, failure_rate_hat, failures_obs
    """

    if len(sampled_df) == 0:
        return np.nan, np.nan, 0

    # ---------------------------------------------
    # Safety checks
    # ---------------------------------------------
    if "partition" not in sampled_df.columns:
        raise ValueError("sampled_df must contain 'partition' column")

    if len(p_i) != len(full_df):
        raise ValueError("p_i length must match full_df length")

    # ---------------------------------------------
    # Map scenario index -> p_i
    # ---------------------------------------------
    p_series = pd.Series(p_i, index=full_df.index)

    # ---------------------------------------------
    # P_h = sum of p_i per stratum
    # ---------------------------------------------
    if "partition" not in full_df.columns:
        raise ValueError("full_df must contain 'partition' column")

    P_h = p_series.groupby(full_df["partition"]).sum()

    # ---------------------------------------------
    # mean_h(z) from sampled data
    # ---------------------------------------------
    mean_h = sampled_df.groupby("partition")["Outcome"].mean()

    # align strata
    mean_h = mean_h.reindex(P_h.index)

    # ---------------------------------------------
    # Failure-rate estimator
    # ---------------------------------------------
    failure_rate_hat = float((P_h * mean_h).sum(skipna=True))
    accuracy_hat = 1.0 - failure_rate_hat

    # ---------------------------------------------
    # Observed failures in realized sample
    # ---------------------------------------------
    failures_obs = int((sampled_df["Outcome"] == 1).sum())

    return accuracy_hat, failure_rate_hat, failures_obs
