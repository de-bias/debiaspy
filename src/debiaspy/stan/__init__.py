"""Packaged Stan programs used by :mod:`debiaspy`."""

from __future__ import annotations

from pathlib import Path

CMDSTAN_COMPATIBLE_MAJOR_MINOR = (2, 32)


def latent_two_level_stan_path() -> Path:
    """Return the filesystem path to the packaged latent two-level Stan model."""

    path = Path(__file__).with_name("latent_two_level.stan")
    if not path.is_file():
        raise FileNotFoundError(
            "The packaged Stan model `debiaspy/stan/latent_two_level.stan` "
            "is missing; reinstall debiaspy from a complete wheel or source "
            "distribution."
        )
    return path


__all__ = [
    "CMDSTAN_COMPATIBLE_MAJOR_MINOR",
    "latent_two_level_stan_path",
]
