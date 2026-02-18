import numpy as np
import pandas as pd

# =========================================================
# RHC SAMPLING (NO PROFILE IN SAMPLING)
# =========================================================

def rhc_sampling_prof(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    RHC-Sampling (WITHOUT replacement), paper-consistent.

    LOGIC IDENTICAL to the original implementation:
    - profile does NOT affect sampling
    - profile is used ONLY in the estimator

    Returns sampled_df with RHC weights.
    """

    df = df.copy()
    N = len(df)
    n = int(min(budget, N))

    if n <= 0 or N <= 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Build global c_i (PPS sizes)
    # -----------------------------------------------------
    x = df[aux_var].to_numpy(dtype=float)

    if np.any(x < 0):
        x = x - x.min()

    x = np.clip(x, eps, None)
    c = x / x.sum()

    # -----------------------------------------------------
    # Random grouping into n nearly equal groups
    # -----------------------------------------------------
    idx = df.index.to_numpy()
    perm = rng.permutation(idx)

    base = N // n
    rem = N % n
    group_sizes = [base + (1 if g < rem else 0) for g in range(n)]

    pos = pd.Series(np.arange(N), index=df.index)

    selected_rows = []
    start = 0

    for g, size in enumerate(group_sizes, start=1):
        group_idx = perm[start:start + size]
        start += size

        group_pos = pos.loc[group_idx].to_numpy()
        c_group = c[group_pos]
        delta = float(c_group.sum())

        p_within = c_group / delta
        chosen = rng.choice(group_idx, size=1, replace=False, p=p_within)[0]

        chosen_pos = int(pos.loc[chosen])
        c_chosen = float(c[chosen_pos])

        w = float(delta / c_chosen)

        row = df.loc[[chosen]].copy()
        row["group"] = g
        row["c"] = c_chosen
        row["delta"] = delta
        row["weight"] = w

        selected_rows.append(row)

    return pd.concat(selected_rows, ignore_index=True)


# =========================================================
# RHC METRICS WITH OPERATIONAL PROFILE (RHC_Prof)
# =========================================================

def compute_metrics_rhc_prof(
    sampled_df: pd.DataFrame,
    p_i: np.ndarray,
):
    """
    RHC_Prof estimator.

    failure_rate_hat = (1/n) * Σ (w_A * z_A * p_A)

    LOGIC IDENTICAL to the original implementation.
    """

    if len(sampled_df) == 0:
        return np.nan, np.nan, 0

    y = sampled_df["Outcome"].to_numpy(dtype=float)
    w = sampled_df["weight"].to_numpy(dtype=float)
    idx = sampled_df.index.to_numpy()

    if len(p_i) <= idx.max():
        raise ValueError("p_i length must match the original dataset")

    p_prof = p_i[idx]
    n = len(sampled_df)

    failure_rate_hat = float((w * (y * p_prof)).sum() / n)
    accuracy_hat = 1.0 - failure_rate_hat

    failures_obs = int((sampled_df["Outcome"] == 1).sum())

    return accuracy_hat, failure_rate_hat, failures_obs
