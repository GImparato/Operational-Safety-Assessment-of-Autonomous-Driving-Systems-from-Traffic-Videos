import numpy as np
import pandas as pd

# =========================================================
# DEEPEST SAMPLING (WITH PROFILE COLUMN PRESENT IN df)
# =========================================================

def deepest_sampling_prof(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
    threshold: float = 0.7,
    r: float = 0.8,
) -> pd.DataFrame:
    """
    DEEPEST adaptive sampling for classification (no partitioning).

    Selection:
      - step 1: SRS (uniform, without replacement)
      - steps k>=2: with probability r -> WBS, else uniform

    WBS weight (your χ-based definition):
      w_{i,j} = χ(d_j) if χ(d_i) > threshold else 0

    Stored per sampled unit:
      - step
      - q : actual selection probability used at that step (mixture prob)
    """
    df_tmp = df.copy()
    N = len(df_tmp)
    n = int(min(budget, N))
    if n <= 0:
        return df_tmp.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    # ---- χ-based weight matrix ----
    chi = df_tmp[aux_var].to_numpy(dtype=float)
    mask = (chi > threshold).astype(float)
    W = mask[:, None] * chi[None, :]  # W[i,j] = χ_j if χ_i > threshold else 0

    all_idx = df_tmp.index.to_numpy()
    idx_to_pos = pd.Series(np.arange(N), index=df_tmp.index)

    remaining = set(all_idx.tolist())
    selected = []
    rows = []

    # ---- step 1: SRS ----
    first = rng.choice(all_idx, size=1, replace=False)[0]
    selected.append(first)
    remaining.remove(first)

    row = df_tmp.loc[[first]].copy()
    row["step"] = 1
    row["q"] = 1.0 / N
    rows.append(row)

    # ---- steps 2..n ----
    for step in range(2, n + 1):
        rem_list = np.array(list(remaining))
        m = len(rem_list)
        if m == 0:
            break

        sel_pos = idx_to_pos.loc[selected].to_numpy()
        rem_pos = idx_to_pos.loc[rem_list].to_numpy()

        # s_i = sum_{j in selected} w_{i,j}
        s = W[np.ix_(rem_pos, sel_pos)].sum(axis=1)

        # WBS distribution on remaining
        if s.sum() > 0:
            p_wbs = s / s.sum()
        else:
            p_wbs = np.ones(m) / m

        # uniform on remaining
        p_unif = np.ones(m) / m

        # mixture q_{k,i}
        q_vec = r * p_wbs + (1.0 - r) * p_unif

        chosen_idx = rng.choice(np.arange(m), p=q_vec)
        chosen = rem_list[chosen_idx]
        q_chosen = float(q_vec[chosen_idx])

        selected.append(chosen)
        remaining.remove(chosen)

        row = df_tmp.loc[[chosen]].copy()
        row["step"] = step
        row["q"] = q_chosen
        rows.append(row)

    return pd.concat(rows, ignore_index=True)

# =========================================================
# DEEPEST ESTIMATOR (WITH PROFILE )
# =========================================================


def compute_metrics_deepest_operational(
    sampled_df: pd.DataFrame,
    profile_col: str = "p",
):
    """
    Operational-profile DEEPEST estimator (classification):

    - Replace z_i with z_i * p_i everywhere (Eq. 10)
    - Remove the outer 1/N in Eq. 8

    Returns:
      accuracy_hat, failure_rate_hat_op, failures_obs
    """
    if len(sampled_df) == 0:
        return np.nan, np.nan, 0

    df_s = sampled_df.sort_values("step").reset_index(drop=True)

    z = df_s["Outcome"].to_numpy(dtype=float)       # 0/1
    q = df_s["q"].to_numpy(dtype=float)             # selection probs used at steps
    p = df_s[profile_col].to_numpy(dtype=float)     # operational profile mass
    n = len(df_s)

    failures_obs = int((df_s["Outcome"] == 1).sum())

    if np.any(q <= 0):
        raise ValueError("q must be > 0 for all sampled points.")
    if np.any(p < 0):
        raise ValueError("Operational profile p must be nonnegative.")

    z1p1 = float(z[0] * p[0])

    sum_prev = z1p1
    step_totals = []

    for t in range(1, n):
        zipi = float(z[t] * p[t])
        qi = float(q[t])
        tilde_theta = sum_prev + (zipi / qi)
        step_totals.append(tilde_theta)
        sum_prev += zipi

    failure_rate_hat_op = float((z1p1 + sum(step_totals)) / n)
    accuracy_hat = 1.0 - failure_rate_hat_op

    return accuracy_hat, failure_rate_hat_op, failures_obs