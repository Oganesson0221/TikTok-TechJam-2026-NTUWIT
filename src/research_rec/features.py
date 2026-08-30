from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TEMPORAL_COLUMNS = ["hour", "day_of_week", "is_weekend"]


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive low-cardinality time features without looking beyond each row."""
    result = frame.copy()
    if "time_ms" in result:
        timestamp = pd.to_datetime(result["time_ms"], unit="ms", errors="coerce")
    elif "date" in result:
        raw_date = result["date"].astype("Int64").astype(str)
        timestamp = pd.to_datetime(raw_date, format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("temporal_features requires either a time_ms or date column")
    result["hour"] = timestamp.dt.hour.fillna(0).astype("int64")
    result["day_of_week"] = timestamp.dt.dayofweek.fillna(0).astype("int64")
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int64")
    return result


def merge_side_features(
    interactions: pd.DataFrame,
    user_features: pd.DataFrame | None,
    item_features: pd.DataFrame | None,
    user_column: str = "user_id",
    item_column: str = "video_id",
) -> pd.DataFrame:
    result = interactions
    if user_features is not None:
        if user_features[user_column].duplicated().any():
            raise ValueError(f"Duplicate {user_column} rows in user feature table")
        result = result.merge(user_features, on=user_column, how="left", validate="many_to_one")
    if item_features is not None:
        if item_features[item_column].duplicated().any():
            raise ValueError(f"Duplicate {item_column} rows in item feature table")
        result = result.merge(item_features, on=item_column, how="left", validate="many_to_one")
    return result


@dataclass
class CategoricalFeatureEncoder:
    """Training-only categorical vocabularies with 0 reserved for unknowns."""

    columns: list[str]
    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def _normalise(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__MISSING__")

    def fit(self, frame: pd.DataFrame) -> "CategoricalFeatureEncoder":
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing configured features: {sorted(missing)}")
        self.vocabularies = {}
        for column in self.columns:
            values = self._normalise(frame[column]).unique().tolist()
            self.vocabularies[column] = {value: index + 1 for index, value in enumerate(values)}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.vocabularies:
            raise RuntimeError("Feature encoder has not been fit")
        encoded = []
        for column in self.columns:
            values = self._normalise(frame[column])
            encoded.append(values.map(self.vocabularies[column]).fillna(0).to_numpy(dtype=np.int64))
        return np.column_stack(encoded)

    @property
    def field_dims(self) -> list[int]:
        return [len(self.vocabularies[column]) + 1 for column in self.columns]

    def state_dict(self) -> dict[str, object]:
        return {"columns": self.columns, "vocabularies": self.vocabularies}

