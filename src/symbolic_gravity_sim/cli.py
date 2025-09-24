# symbolic_gravity_sim/cli.py
from __future__ import annotations
import argparse
import sys
from typing import Optional

from . import __version__
from .symbolic_collapse_sim import run as run_sim


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic_gravity_sim",
        description="Symbolic Gravity Simulator CLI (11D-enabled)"
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    # --- run ---
    run_parser = sub.add_parser(
        "run",
        help="Run a single simulation and save outputs (PNG/JSON[/CSV])."
    )
    run_parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    run_parser.add_argument("--n_agents", type=int, default=4, help="Number of agents.")
    run_parser.add_argument("--steps", type=int, default=300, help="Simulation steps.")
    run_parser.add_argument(
        "--field",
        type=str,
        default="hillvalley",
        help="Base field type (e.g., hillvalley, gaussian, uniform, gradient).",
    )
    run_parser.add_argument(
        "--overlay_curvature",
        action="store_true",
        help="Overlay curvature on ψ_eff image."
    )
    run_parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.25,
        help="Alpha for curvature overlay."
    )
    run_parser.add_argument(
        "--dump_11d_means",
        action="store_true",
        help="Write per-run means of 11D fields to CSV."
    )
    run_parser.add_argument(
        "--save",
        type=str,
        default="runs/run_cli",
        help="Output prefix (e.g., runs/my_run -> PNG/JSON[/CSV])."
    )
    run_parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display matplotlib window."
    )

    # --- version ---
    sub.add_parser("version", help="Show package version.")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    run_sim(
        seed=args.seed,
        n_agents=args.n_agents,
        steps=args.steps,
        field=args.field,
        overlay_curvature=args.overlay_curvature,
        overlay_alpha=args.overlay_alpha,
        save=args.save,
        no_show=args.no_show,
        dump_11d_means=args.dump_11d_means,
    )
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "version":
        return _cmd_version(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
