from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd

from debiaspy._exceptions import DebiasSchemaError

REPEATED_OBSERVATION_KEYS = ("mpd_source", "mpd_time")


def validation_join_keys(
    data: pd.DataFrame,
    reference: pd.DataFrame,
    base_keys: Sequence[str] = ("origin", "destination"),
) -> list[str]:
    """Return OD keys plus repeated-observation keys shared by both inputs."""
    return [
        *base_keys,
        *[
            key
            for key in REPEATED_OBSERVATION_KEYS
            if key in data.columns and key in reference.columns
        ],
    ]


def merge_validation_reference(
    data: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    how: Literal["left", "inner", "outer"],
    data_name: str,
    reference_name: str,
    sort: bool = False,
) -> pd.DataFrame:
    """Merge a validation table against an unambiguous reference table.

    Validation results are row-oriented around ``data``. A source- or
    time-specific reference must therefore expose the same observation key on
    both frames. Otherwise an OD-only join can multiply every data row by every
    reference observation for that OD pair.
    """
    keys = validation_join_keys(data, reference)
    duplicated = reference.duplicated(subset=keys, keep=False)
    if bool(duplicated.any()):
        missing_observation_keys = [
            key
            for key in REPEATED_OBSERVATION_KEYS
            if key in reference.columns and key not in data.columns
        ]
        detail = ""
        if missing_observation_keys:
            detail = (
                " Supply matching repeated-observation column(s) in "
                f"`{data_name}`: {', '.join(missing_observation_keys)}."
            )
        raise DebiasSchemaError(
            f"`{reference_name}` must contain at most one row per validation "
            f"join key ({', '.join(keys)}); the join would multiply rows."
            f"{detail}"
        )

    duplicated_data = data.duplicated(subset=keys, keep=False)
    if bool(duplicated_data.any()):
        missing_observation_keys = [
            key
            for key in REPEATED_OBSERVATION_KEYS
            if key in data.columns and key not in reference.columns
        ]
        detail = ""
        if missing_observation_keys:
            detail = (
                " Supply matching repeated-observation column(s) in "
                f"`{reference_name}`: {', '.join(missing_observation_keys)}."
            )
        raise DebiasSchemaError(
            f"`{data_name}` must contain at most one row per validation "
            f"join key ({', '.join(keys)}); duplicate adjusted rows would "
            f"be counted more than once.{detail}"
        )

    return data.merge(
        reference,
        how=how,
        on=keys,
        sort=sort,
        validate="one_to_one",
    )
