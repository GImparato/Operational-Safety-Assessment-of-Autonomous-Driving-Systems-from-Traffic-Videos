import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler

# =========================================================
# PARTITIONING
# =========================================================

def build_partitions_hdbscan(
    df: pd.DataFrame,
    aux_var: str,
    min_cluster_size: int,
):
    df = df.dropna(subset=[aux_var]).copy()
    X = StandardScaler().fit_transform(df[[aux_var]].values)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None
    ).fit_predict(X)

    df["partition"] = labels
    return df


def build_partitions_odd(
    df: pd.DataFrame,
    odd_columns: list[str],
):
    df = df.dropna(subset=odd_columns).copy()
    df["partition"] = df[odd_columns].astype(str).agg("_".join, axis=1)
    return df


# =========================================================
# SSRS – NEYMAN ALLOCATION (NO PROFILE)
# =========================================================

def ssrs_sampling(
    df: pd.DataFrame,
    budget: int,
    aux_var: str,
    seed: int = 42,
    partitioning: str = "hdbscan",
    min_cluster_size: int = 30,
    odd_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Stratified Sampling with Neyman allocation (SSRS).

    LOGIC IDENTICAL to the original implementation:
    - partitioning via HDBSCAN (aux_var) or ODD
    - Neyman allocation with std-based weights
    - fallback to SRS
    - top-up / trim safety steps
    """

    df = df.copy()

    # ---- Partitioning ----
    if partitioning == "hdbscan":
        df = build_partitions_hdbscan(df, aux_var, min_cluster_size)
    elif partitioning == "odd":
        if odd_columns is None:
            raise ValueError("odd_columns must be provided for ODD partitioning")
        df = build_partitions_odd(df, odd_columns)
    else:
        raise ValueError("Unknown partitioning mode")

    budget = int(min(budget, len(df)))
    if budget <= 0:
        return df.iloc[0:0].copy()

    group = df.groupby("partition", sort=False)
    Nh = group.size()
    stds = group["Outcome"].std(ddof=0).fillna(0)

    weights = Nh * stds

    # ---- fallback: pure SRS ----
    if weights.sum() == 0:
        return df.sample(n=budget, random_state=seed).reset_index(drop=True)

    # ---- fractional Neyman allocation ----
    target = weights / weights.sum() * budget
    alloc = np.floor(target).astype(int)
    remainder = target - alloc

    alloc = alloc.clip(upper=Nh)

    leftover = budget - int(alloc.sum())
    if leftover > 0:
        order = remainder.sort_values(ascending=False).index.tolist()
        for p in order:
            if leftover == 0:
                break
            if alloc.loc[p] < Nh.loc[p]:
                alloc.loc[p] += 1
                leftover -= 1

    rng = np.random.default_rng(seed)

    samples = []
    for p, n in alloc.items():
        if n <= 0:
            continue
        part_df = df[df["partition"] == p]
        rs = int(rng.integers(0, 2**32 - 1))
        samples.append(part_df.sample(n=n, random_state=rs))

    out = pd.concat(samples, ignore_index=True) if samples else df.iloc[0:0].copy()

    # ---- safety: top-up or trim ----
    if len(out) < budget:
        remaining = df.drop(out.index, errors="ignore")
        if len(remaining) > 0:
            rs = int(rng.integers(0, 2**32 - 1))
            topup = remaining.sample(
                n=min(budget - len(out), len(remaining)),
                random_state=rs
            )
            out = pd.concat([out, topup], ignore_index=True)

    if len(out) > budget:
        rs = int(rng.integers(0, 2**32 - 1))
        out = out.sample(n=budget, random_state=rs).reset_index(drop=True)

    return out.reset_index(drop=True)


# =========================================================
# SSRS METRICS (PAPER-CONSISTENT)
# =========================================================

def compute_metrics_ssrs(
    sampled_df: pd.DataFrame,
    full_df: pd.DataFrame,
):
    """
    SSRS estimator (paper-consistent).

    LOGIC IDENTICAL to the original implementation.
    """

    failures = int((sampled_df["Outcome"] == 1).sum())

    Nh = full_df.groupby("partition").size()
    N = Nh.sum()

    ph = sampled_df.groupby("partition")["Outcome"].mean()
    ph = ph.reindex(Nh.index)

    failure_rate_est = float((Nh / N * ph).sum(skipna=True))
    accuracy_est = 1.0 - failure_rate_est

    return accuracy_est, failures
