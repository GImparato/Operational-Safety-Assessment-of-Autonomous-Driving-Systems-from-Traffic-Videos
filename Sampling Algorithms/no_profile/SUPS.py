import numpy as np
import pandas as pd

# =========================================================
# SUPS SAMPLING
# No-profile version – Streamlit compatible
# =========================================================

def sups_sampling(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    SUPS: Unequal-probability sampling with replacement (PPS-WR)

    LOGIC IDENTICAL to the original implementation:
    - p_i ∝ aux_var[i]
    - sampling WITH replacement
    - Hansen–Hurwitz weights w_i = 1 / p_i
    """

    df = df.copy()

    # --- RNG handling (structural only) ---
    rng = np.random.default_rng(seed)

    # --- ORIGINAL LOGIC (UNCHANGED) ---
    x = df[aux_var].to_numpy(dtype=float)

    # Ensure non-negative auxiliary values
    if np.any(x < 0):
        x = x - x.min()

    total = x.sum()
    if total <= 0:
        p = np.ones(len(df)) / len(df)
    else:
        p = x / total

    idx = rng.choice(
        df.index.to_numpy(),
        size=min(budget, len(df)),
        replace=True,
        p=p
    )

    sampled_df = df.loc[idx].copy()

    # Hansen–Hurwitz weights
    p_map = pd.Series(p, index=df.index)
    sampled_df["weight"] = 1.0 / p_map.loc[idx].to_numpy()

    return sampled_df.reset_index(drop=True)


# =========================================================
# SUPS METRICS (PAPER-CONSISTENT)
# =========================================================

def compute_metrics_sups(
    sampled_df: pd.DataFrame,
    N: int,
):
    """
    Metrics for SUPS consistent with the paper formulation.

    LOGIC IDENTICAL to the original implementation.
    """

    w = sampled_df["weight"].to_numpy(dtype=float)
    y = sampled_df["Outcome"].to_numpy(dtype=float)
    n = len(sampled_df)

    if n == 0 or N <= 0:
        return np.nan, np.nan, 0

    failure_rate_hat = float((w * y).sum() / (n * N))
    accuracy_hat = 1.0 - failure_rate_hat

    failures_obs = int((sampled_df["Outcome"] == 1).sum())

    return accuracy_hat, failure_rate_hat, failures_obs
