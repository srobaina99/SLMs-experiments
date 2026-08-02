"""CSV-safe boolean coercion for pandas Series / scalars.

``Series.astype(bool)`` treats any non-empty string as True, so the CSV token
``\"False\"`` becomes True. Use these helpers for flags read from run CSVs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_FALSE_STRINGS = frozenset(
    {"", "0", "false", "f", "no", "n", "off", "none", "nan", "<na>"}
)
_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})


def coerce_bool(value: Any) -> bool:
    """Coerce one CSV / pandas value to bool (missing → False)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    # bool is a subclass of int; handled above.
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if number != number:  # NaN
            return False
        return number != 0.0
    text = str(value).strip().lower()
    if text in _FALSE_STRINGS:
        return False
    if text in _TRUE_STRINGS:
        return True
    return False


def coerce_bool_series(series: pd.Series) -> pd.Series:
    """Element-wise :func:`coerce_bool` → bool dtype Series."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(coerce_bool).astype(bool)
