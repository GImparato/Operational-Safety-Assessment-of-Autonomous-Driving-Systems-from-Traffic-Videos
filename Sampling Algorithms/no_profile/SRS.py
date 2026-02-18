import pandas as pd
import numpy as np

# =========================================================
# SIMPLE RANDOM SAMPLING (SRS)
# No-profile version – black-box compatible with Streamlit
# =========================================================

def srs_sampling(
    df: pd.DataFrame,
    budget: int,
    aux_var: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simple Random Sampling (without replacement).

    Identical logic to the original implementation:
    - uniform sampling
    - without replacement
    - pandas .sample with fixed random_state per run

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must contain 'Outcome').
    budget : int
        Sampling budget.
    aux_var : str or None
        Unused (kept for interface compatibility).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sampled dataset.
    """

    df = df.copy()

    n = min(budget, len(df))
    if n <= 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    # identical behavior to the original code:
    # generate a random_state from rng and pass it to pandas.sample
    rs = int(rng.integers(0, 2**32 - 1))

    sampled_df = df.sample(
        n=n,
        replace=False,
        random_state=rs
    )

    return sampled_df.reset_index(drop=True)
