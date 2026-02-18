import numpy as np
import pandas as pd

# =========================================================
# SUPS SAMPLING WITH OPERATIONAL PROFILE (SUPS_Prof)
# =========================================================

def sups_sampling_prof(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
):
    """
    SUPS: Unequal-probability sampling with replacement (PPS-WR)

    LOGIC IDENTICAL to the original implementation.

    - sampling WITH replacement
    - selection probability p_i ∝ aux_var[i]
    - Hansen–Hurwitz weights w_i = 1 / π_i
    """

    N = len(df)
    n = int(min(budget, N))

    if n <= 0 or N <= 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

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
        size=n,
        replace=True,
        p=p
    )

    sampled_df = df.loc[idx].copy()

    p_map = pd.Series(p, index=df.index)
    sampled_df["weight"] = 1.0 / p_map.loc[idx].to_numpy()

    return sampled_df.reset_index(drop=True)


# =========================================================
# SUPS METRICS WITH OPERATIONAL PROFILE (PAPER-CONSISTENT)
# =========================================================

def compute_metrics_sups_prof(
    sampled_df: pd.DataFrame,
    p_i: np.ndarray,
):
    """
    SUPS_Prof estimator (per professor's indication):

        θ̂ = (1/n) Σ (z_i * p_i) / π_i

    where:
        z_i = Outcome
        p_i = operational profile
        π_i = sampling probability (SUPS)

    LOGIC IDENTICAL to the original implementation.
    """

    n = len(sampled_df)
    if n == 0:
        return np.nan, np.nan, 0

    w = sampled_df["weight"].to_numpy(dtype=float)   # 1 / π_i
    y = sampled_df["Outcome"].to_numpy(dtype=float)
    idx = sampled_df.index.to_numpy()

    if len(p_i) <= idx.max():
        raise ValueError("p_i length must match the original dataset")

    p_prof = p_i[idx]

    failure_rate_hat = float((w * (y * p_prof)).sum() / n)
    accuracy_hat = 1.0 - failure_rate_hat

    failures_obs = int((sampled_df["Outcome"] == 1).sum())

    return accuracy_hat, failure_rate_hat, failures_obs
