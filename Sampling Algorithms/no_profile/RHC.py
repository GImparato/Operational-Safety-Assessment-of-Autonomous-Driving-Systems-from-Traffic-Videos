import numpy as np
import pandas as pd

# =========================================================
# RHC SAMPLING (WITHOUT REPLACEMENT)
# No-profile version – Streamlit compatible
# =========================================================

def rhc_sampling(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    RHC-Sampling (without replacement) as described in the paper (RHC-S),
    for the classification case.

    LOGIC IDENTICAL to the original implementation.

    Steps:
      1) Randomly split N units into n = budget groups (nearly equal size).
      2) From each group, draw exactly 1 unit with PPS using c_i ∝ aux_var.
      3) Return the n selected units with RHC weights (Eq. 5).

    Returns
    -------
    sampled_df : pd.DataFrame (size = min(budget, N))
        Columns added:
          - group   : group id
          - c       : global normalized c_i
          - delta   : group sum Δ_A
          - weight  : Δ_A / c_selected
    """

    df = df.copy()
    N = len(df)
    n = int(min(budget, N))

    if n <= 0 or N <= 0:
        return df.iloc[0:0].copy()

    # --- RNG handling (structural only) ---
    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # Build global c_i (PPS "sizes")
    # -----------------------------------------------------
    x = df[aux_var].to_numpy(dtype=float)

    # Ensure non-negativity
    if np.any(x < 0):
        x = x - x.min()

    # Avoid zero probabilities
    x = np.clip(x, eps, None)

    c = x / x.sum()   # c_i = aux_i / sum_j aux_j

    # -----------------------------------------------------
    # Random grouping into n nearly equal groups
    # -----------------------------------------------------
    idx = df.index.to_numpy()
    perm = rng.permutation(idx)

    base = N // n
    rem = N % n
    group_sizes = [base + (1 if g < rem else 0) for g in range(n)]

    # Map index -> position
    pos = pd.Series(np.arange(N), index=df.index)

    selected_rows = []
    start = 0

    for g, size in enumerate(group_sizes, start=1):
        group_idx = perm[start:start + size]
        start += size

        group_pos = pos.loc[group_idx].to_numpy()
        c_group = c[group_pos]
        delta = float(c_group.sum())   # Δ_A

        # PPS within group
        p_within = c_group / delta
        chosen = rng.choice(group_idx, size=1, replace=False, p=p_within)[0]

        chosen_pos = int(pos.loc[chosen])
        c_chosen = float(c[chosen_pos])

        # RHC weight (Eq. 5): Δ_A / c_A
        w = float(delta / c_chosen)

        row = df.loc[[chosen]].copy()
        row["group"] = g
        row["c"] = c_chosen
        row["delta"] = delta
        row["weight"] = w

        selected_rows.append(row)

    return pd.concat(selected_rows, ignore_index=True)


# =========================================================
# RHC METRICS (PAPER-CONSISTENT)
# =========================================================

def compute_metrics_rhc(
    sampled_df: pd.DataFrame,
    N: int,
):
    """
    Classification metrics using the RHC estimator (Eq. 5).

    LOGIC IDENTICAL to the original implementation.
    """

    if len(sampled_df) == 0 or N <= 0:
        return np.nan, np.nan, 0

    y = sampled_df["Outcome"].to_numpy(dtype=float)   # I_A ∈ {0,1}
    w = sampled_df["weight"].to_numpy(dtype=float)    # Δ_A / c_A

    failure_rate_hat = float((w * y).sum() / N)
    accuracy_hat = 1.0 - failure_rate_hat

    failures_obs = int((sampled_df["Outcome"] == 1).sum())

    return accuracy_hat, failure_rate_hat, failures_obs
