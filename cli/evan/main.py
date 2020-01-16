import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import EVAN_CACHE_DIR
from .score import frechet_distance, inception_score, precision_recall


def do_compute_command(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        add_help=False,
        description="calculate evaluation score. please specify metric name",
        prog="evan compute",
    )
    p.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["inception-score", "frechet-distance", "precision-recall"],
        required=True,
        help="evaluation metric name",
    )
    args, rest_args = p.parse_known_args(argv)

    if args.metric == "inception-score":
        _p = argparse.ArgumentParser(
            description="calculate Inception Score",
            prog=f"evan compute -m {args.metric}",
        )
        _p.add_argument(
            "-g",
            "--gen-dir",
            type=Path,
            required=True,
            help="directory which contains generated videos by your model",
        )
        _p.add_argument(
            "-n",
            "--n_samples",
            type=int,
            default=-1,
            help="number of samples to use for evaluation. Use all samples by default",
        )
        params, _ = _p.parse_known_args(argv)
        inception_score(params.gen_dir.resolve(), params.n_samples)

    elif args.metric == "frechet-distance":
        _p = argparse.ArgumentParser(
            description="calculate Frechet Inception Distance",
            prog=f"evan compute -m {args.metric}",
        )
        _p.add_argument(
            "-g",
            "--gen-dir",
            type=Path,
            required=True,
            help="directory which contains generated videos by your model",
        )
        _p.add_argument(
            "-r",
            "--ref-dir",
            type=Path,
            required=True,
            help="directory path which contains reference videos to train your model",
        )
        _p.add_argument(
            "-n",
            "--n_samples",
            type=int,
            default=-1,
            help="number of samples to use for evaluation. Use all samples by default",
        )
        params, _ = _p.parse_known_args(argv)

        frechet_distance(
            params.gen_dir.resolve(), params.ref_dir.resolve(), params.n_samples
        )

    elif args.metric == "precision-recall":
        _p = argparse.ArgumentParser(
            description="calculate Precision and Recall for Distributions",
            prog=f"evan compute -m {args.metric}",
        )
        _p.add_argument(
            "-g",
            "--gen-dir",
            type=Path,
            required=True,
            help="directory which contains generated videos by your model.",
        )
        _p.add_argument(
            "-r",
            "--ref-dir",
            type=Path,
            required=True,
            help="directory path which contains reference videos to train your model.",
        )
        _p.add_argument(
            "-n",
            "--n_samples",
            type=int,
            default=-1,
            help="number of samples to use for evaluation. Use all samples by default.",
        )
        params, _ = _p.parse_known_args(argv)
        precision_recall(
            params.gen_dir.resolve(), params.ref_dir.resolve(), params.n_samples
        )
    else:
        raise NotImplementedError


def main():
    if not EVAN_CACHE_DIR.exists():
        EVAN_CACHE_DIR.mkdir(parents=True)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["compute"], help="command")
    args, rest_args = parser.parse_known_args(sys.argv[1:])

    if args.command == "compute":
        do_compute_command(rest_args)
    else:
        raise NotImplementedError

    # args = parser.parse_args()
    # if hasattr(args, "handler"):
    #     args.handler(args)
    # else:
    #     parser.print_help()
