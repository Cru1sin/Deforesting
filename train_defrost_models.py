"""Train, summarize or draw F_core, Chen B0 and G_core using the frozen Ridge106 protocol."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "summary", "figures"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--data", type=Path, default=Path("output/image_models/core_comparison_ridge106")
    )
    parser.add_argument(
        "--event-run",
        type=Path,
        default=Path("output/defrost_event_models/complete_defrost_events_zero"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/image_models/core_comparison_ridge106")
    )
    parser.add_argument(
        "--figure-output", type=Path, default=Path("output/cycles/core_comparison_ridge106")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("F_core", "B0", "G_core"),
        default=["F_core", "B0", "G_core"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--figure-seed", type=int, default=0)
    parser.add_argument("--heldout-experiment")
    parser.add_argument("--batch-size", type=int, default=8, help="whole cycles per batch")
    parser.add_argument("--maximum-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=6)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.n_jobs, args.batch_size, args.maximum_epochs, args.patience) < 1:
        raise ValueError("workers, batch size, epochs and patience must be positive")
    if args.action == "train":
        from image_models.core import run

        run(args)
    elif args.action == "summary":
        from image_models.core import PARENTS, summarize

        summarize(args.output, {name: PARENTS[name] for name in args.models})
    else:
        from plots.publication import render_stopping_loss_cycles

        render_stopping_loss_cycles(
            args.output, args.dataset, args.figure_output, args.n_jobs, seed=args.figure_seed
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
