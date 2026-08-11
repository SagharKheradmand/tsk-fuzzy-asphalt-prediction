import numpy as np
import pandas as pd
from typing import Tuple, List
import re
import unicodedata


class StandardScaler:
    """
    Standard normalization: (X - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Split dataset into train and test subsets.
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(X.shape[0])
        X = X[idx]
        Y = Y[idx]

    n = X.shape[0]
    split = int(n * train_ratio)
    return X[:split], X[split:], Y[:split], Y[split:]


def _norm_col(s: str) -> str:
    """
    Normalize column names to make matching tolerant:
    - lowercase
    - remove accents/unicode quirks
    - collapse whitespace
    - remove most punctuation
    """
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    # Treat one-letter symbols in parentheses as optional metadata, e.g. (η)/(n).
    s = re.sub(r"\(\s*[^)\s]{1}\s*\)", " ", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # keep % and numbers/letters, drop most other punctuation
    s = re.sub(r"[^a-z0-9% ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_columns(df: pd.DataFrame, requested: List[str]) -> List[str]:
    """
    Map requested column names to actual df columns using normalized matching.
    Raises a helpful error if any are missing.
    """
    actual_cols = list(df.columns)
    norm_to_actual = {}
    for c in actual_cols:
        key = _norm_col(c)
        # if duplicates after normalization, keep the first
        norm_to_actual.setdefault(key, c)

    resolved = []
    missing = []
    for r in requested:
        key = _norm_col(r)
        if key in norm_to_actual:
            resolved.append(norm_to_actual[key])
        else:
            missing.append(r)

    if missing:
        # show user what columns exist (normalized and raw)
        preview = "\n".join([f"- {c}" for c in actual_cols[:30]])
        raise KeyError(
            "Could not match these columns from config:\n"
            + "\n".join([f"- {m}" for m in missing])
            + "\n\nAvailable columns in your file (first 30):\n"
            + preview
            + "\n\nFix by updating config.py column names OR rely on this resolver "
            "and ensure your requested names are semantically the same."
        )
    return resolved


def extract_xy(
    df: pd.DataFrame, input_cols: List[str], output_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract input/output matrices with tolerant column matching.
    """
    in_cols = _resolve_columns(df, input_cols)
    out_cols = _resolve_columns(df, output_cols)

    X = df[in_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = df[out_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]

    if X.shape[0] < 10:
        raise ValueError("Too few valid rows after cleaning (NaN/inf removed).")

    return X, Y
