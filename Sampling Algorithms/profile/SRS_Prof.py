import numpy as np
import pandas as pd

# =========================================================
# SRS WITH OPERATIONAL PROFILE (SRS_Prof)
# =========================================================

def srs_sampling_prof(
    df: pd.DataFrame,
    budget: int,
    p_i: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simple Random Sampling with operational profile (SRS_Prof).

    LOGIC IDENTICAL to the original implementation.

    - sampling WITH replacement
    - selection probability proportional to p_i

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must contain 'Outcome')
    budget : int
        Sampling budget
    p_i : np.ndarray
        Operational profile (sum p_i = 1, len = len(df))
    seed : int
        Random seed

    Returns
    -------
    sampled_df : pd.DataFrame
        Sampled dataset (size = min(budget, N))
    """

    N = len(df)
    n = int(min(budget, N))

    if n <= 0 or N <= 0:
        return df.iloc[0:0].copy()

    if len(p_i) != N:
        raise ValueError("Length of p_i must match number of rows in df")

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        df.index.to_numpy(),
        size=n,
        replace=True,
        p=p_i
    )

    return df.loc[indices].reset_index(drop=True)


# =========================================================
# SRS_Prof METRICS (EMPIRICAL ACCURACY)
# =========================================================

def compute_metrics_srs_prof(
    sampled_df: pd.DataFrame,
):
    """
    Metrics for SRS_Prof (empirical accuracy).

    LOGIC IDENTICAL to the original implementation.
    """

    total = len(sampled_df)
    failures = int(sampled_df["Outcome"].sum())

    accuracy = 1.0 - failures / total if total > 0 else np.nan

    return accuracy, failures
