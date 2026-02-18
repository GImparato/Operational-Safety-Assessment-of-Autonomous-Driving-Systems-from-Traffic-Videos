import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler

# =========================================================
# PARTITIONING FUNCTIONS
# =========================================================

def build_partitions_hdbscan(
    df: pd.DataFrame,
    aux_var: str,
    min_cluster_size: int,
):
    df = df.dropna(subset=[aux_var]).copy()
    X = StandardScaler().fit_transform(df[[aux_var]].values)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size
    ).fit_predict(X)

    df["partition"] = labels
    return df


def build_partitions_odd(
    df: pd.DataFrame,
    odd_columns: list[str],
):
    df = df.dropna(subset=odd_columns).copy()
    df["partition"] = (
        df[odd_columns].astype(str).agg("_".join, axis=1)
    )
    return df


# =========================================================
# 2-UPS SAMPLING (NO PROFILE IN SAMPLING)
# =========================================================

def two_ups_sampling_prof(
    df: pd.DataFrame,
    budget: int,
    aux_var: str | None = None,
    seed: int = 42,
    partitioning: str = "hdbscan",
    min_cluster_size: int = 30,
    odd_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Two-Stage Unequal Probability Sampling (2-UPS).

    LOGIC IDENTICAL to the original implementation:
    - profile does NOT affect sampling
    - profile is used ONLY in the estimator
    """

    df = df.copy()

    if partitioning == "hdbscan":
        df = build_partitions_hdbscan(df, aux_var, min_cluster_size)
        scores = df.groupby("partition")[aux_var].mean()

    elif partitioning == "odd":
        if odd_columns is None:
            raise ValueError("odd_columns must be provided for ODD partitioning")
        df = build_partitions_odd(df, odd_columns)
        scores = df.groupby("partition")["Outcome"].mean()

    else:
        raise ValueError("Unknown partitioning mode")

    if scores.sum() == 0:
        probs = np.ones(len(scores)) / len(scores)
    else:
        probs = scores / scores.sum()

    partitions = scores.index.to_numpy()
    probs = probs.to_numpy()

    rng = np.random.default_rng(seed)

    chosen_partitions = rng.choice(
        partitions,
        size=budget,
        replace=True,
        p=probs
    )

    samples = []
    for p in chosen_partitions:
        part_df = df[df["partition"] == p]

        inner_seed = rng.integers(0, 2**32 - 1)
        s = part_df.sample(n=1, random_state=int(inner_seed))

        idx = np.where(partitions == p)[0][0]
        nk = len(part_df)
        s["weight"] = nk / probs[idx]

        samples.append(s)

    out = pd.concat(samples, ignore_index=True)

    if len(out) > budget:
        out = out.sample(n=budget, random_state=seed)

    return out.reset_index(drop=True)


# =========================================================
# 2-UPS METRICS WITH OPERATIONAL PROFILE (2UPS_Prof)
# =========================================================

def compute_metrics_2ups_prof(
    sampled_df: pd.DataFrame,
    p_i: np.ndarray,
):
    """
    Hansen–Hurwitz estimator with operational profile.

    failure_rate_hat = Σ (w_i * z_i * p_i) / Σ w_i

    LOGIC IDENTICAL to the original implementation.
    """

    w = sampled_df["weight"].to_numpy(dtype=float)
    y = sampled_df["Outcome"].to_numpy(dtype=float)
    idx = sampled_df.index.to_numpy()

    if len(p_i) <= idx.max():
        raise ValueError("p_i length must match the original dataset")

    p_prof = p_i[idx]

    failure_rate = float(np.sum(w * (y * p_prof)) / np.sum(w))
    accuracy = 1.0 - failure_rate

    failures = int(sampled_df["Outcome"].sum())

    return accuracy, failures
