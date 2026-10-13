from __future__ import annotations

import argparse
import os
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

PINNED_CMDSTAN_VERSION = "2.32.2"


def main(argv: Sequence[str] | None = None) -> int:
    """Install and validate the CmdStan runtime used by latent models."""

    parser = argparse.ArgumentParser(
        description="Install the CmdStan runtime used by debiaspy latent models."
    )
    parser.add_argument("--version", default=PINNED_CMDSTAN_VERSION)
    parser.add_argument("--directory", type=Path, default=Path(".cmdstan"))
    parser.add_argument("--cores", type=int, default=min(os.cpu_count() or 1, 2))
    args = parser.parse_args(argv)

    import cmdstanpy

    install_dir = args.directory.resolve()
    cmdstan_dir = install_dir / f"cmdstan-{args.version}"
    if _is_complete(cmdstan_dir):
        cmdstanpy.set_cmdstan_path(str(cmdstan_dir))
        print(f"CmdStan {args.version} is ready at {cmdstan_dir}")
        return 0

    install_dir.mkdir(parents=True, exist_ok=True)
    installed = cmdstanpy.install_cmdstan(
        dir=str(install_dir),
        version=args.version,
        cores=max(args.cores, 1),
        progress=True,
        overwrite=False,
    )
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if not installed and apple_silicon:
        if not cmdstan_dir.is_dir():
            raise RuntimeError(
                "CmdStan download/extraction failed before the Apple Silicon "
                "compatibility build could run."
            )
        make_local = cmdstan_dir / "make" / "local"
        current = make_local.read_text() if make_local.exists() else ""
        compatibility_flag = (
            "CXXFLAGS += -Wno-missing-template-arg-list-after-template-kw\n"
        )
        if compatibility_flag not in current:
            make_local.write_text(current + compatibility_flag)
        subprocess.run(
            ["make", "build", f"-j{max(args.cores, 1)}", "arch=arm64"],
            cwd=cmdstan_dir,
            check=True,
        )

    if not _is_complete(cmdstan_dir):
        raise RuntimeError(f"CmdStan installation is incomplete at {cmdstan_dir}")
    cmdstanpy.set_cmdstan_path(str(cmdstan_dir))
    print(f"CmdStan {args.version} is ready at {cmdstan_dir}")
    return 0


def _is_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "bin" / "stanc").is_file()
        and (path / "bin" / "diagnose").is_file()
    )
