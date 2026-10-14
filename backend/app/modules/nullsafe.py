"""
Centralized null handling.

pandas represents missing cells as float('nan') (not None) -- both immediately
after CSV/Excel ingestion (na_values=[""]) and every time a dataframe round-trips
through parquet (save_layer/load_layer), even if the in-memory value was
explicitly set to None. Any code that does `str(v).strip() == ""` to detect a
missing value will NOT match `float('nan')`, since `str(float('nan'))` is the
literal string "nan".

To avoid re-deriving this fix in every metric/standardization function, all
values are normalized to real `None` at the single boundary where they leave a
DataFrame (`series_to_values`). Everything downstream can then safely assume
"missing" is always represented as `None`, never as NaN.
"""
from __future__ import annotations
import pandas as pd


def series_to_values(series: pd.Series) -> list:
    """Convert a pandas Series to a plain Python list, normalizing every
    NaN/NaT/pd.NA cell to None. Non-missing values are returned unchanged."""
    return [None if pd.isna(v) else v for v in series]
