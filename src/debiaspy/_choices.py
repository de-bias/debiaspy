from __future__ import annotations

from collections.abc import Sequence


def match_choice(value: object, choices: Sequence[str], argument: str) -> str:
    """Resolve an R ``match.arg``-style exact or unambiguous prefix choice."""

    if not isinstance(value, str):
        raise ValueError(f"`{argument}` must be one of: {', '.join(choices)}")
    if value in choices:
        return value
    matches = [choice for choice in choices if choice.startswith(value)]
    if len(matches) != 1:
        raise ValueError(f"`{argument}` must be one of: {', '.join(choices)}")
    return matches[0]
