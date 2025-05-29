"""
Command‑line front‑end:

    python -m evals.cli \
        --gt     ./dataset/ground_truth \
        --pred   ./results/my_model \
        [--direct]          # disable SDR‑based permutation
"""
from __future__ import annotations
import argparse
from pathlib import Path

from . import main


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two‑source separation evaluator")
    p.add_argument("--gt",   required=True, type=Path, help="Ground‑truth root")
    p.add_argument("--pred", required=True, type=Path, help="Prediction root")
    p.add_argument(
        "--direct", action="store_true",
        help="Skip SDR permutation and assume file order is correct",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    main(args.gt, args.pred, args.direct)
