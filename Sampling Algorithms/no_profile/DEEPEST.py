import numpy as np
import pandas as pd

# =========================================================
# DEEPEST SAMPLING (NO PROFILE) — FIXED
# =========================================================

def deepest_sampling(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
    threshold: float = 0.7,
    r: float = 0.8,
) -> pd.DataFrame:
    """
    DEEPEST-like adaptive sampling for classification (no partitioning).

    FIX:
    - Weight matrix W is computed on auxiliary variable chi (aux_var),
      NOT on feature-space distances.

    Selection:
      - step 1: SRS (uniform, without replacement)
      - steps k>=2: with probability r -> WBS, else uniform

    WBS:
      W[i, j] = chi_j if chi_i > threshold else 0
      p_i ∝ sum_{j in selected} W[i, j]

    Stored per sampled unit:
      - step
      - q : actual selection probability used at that step
    """

    if aux_var is None:
        raise ValueError("DEEPEST requires 'aux_var' (chi) to be specified")

    df_tmp = df.copy()
    N = len(df_tmp)
    n = int(min(budget, N))

    if n <= 0:
        return df_tmp.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # AUXILIARY VARIABLE (chi)
    # -----------------------------------------------------
    chi = df_tmp[aux_var].to_numpy(dtype=float)

    # ---- weight matrix W (paper-consistent) ----
    mask = (chi > threshold).astype(float)     # shape (N,)
    W = mask[:, None] * chi[None, :]            # shape (N, N)

    # -----------------------------------------------------
    # INDEXING
    # -----------------------------------------------------
    all_idx = df_tmp.index.to_numpy()
    idx_to_pos = pd.Series(np.arange(N), index=df_tmp.index)

    remaining = set(all_idx.tolist())
    selected = []
    rows = []

    # -----------------------------------------------------
    # STEP 1 — SRS
    # -----------------------------------------------------
    first = rng.choice(all_idx, size=1, replace=False)[0]
    selected.append(first)
    remaining.remove(first)

    row = df_tmp.loc[[first]].copy()
    row["step"] = 1
    row["q"] = 1.0 / N
    rows.append(row)

    # -----------------------------------------------------
    # STEPS 2..n
    # -----------------------------------------------------
    for step in range(2, n + 1):
        rem_list = np.array(list(remaining))
        m = len(rem_list)
        if m == 0:
            break

        sel_pos = idx_to_pos.loc[selected].to_numpy()
        rem_pos = idx_to_pos.loc[rem_list].to_numpy()

        # ---- WBS scores ----
        s = W[np.ix_(rem_pos, sel_pos)].sum(axis=1)

        if s.sum() > 0:
            p_wbs = s / s.sum()
        else:
            p_wbs = np.ones(m) / m

        p_unif = np.ones(m) / m
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
