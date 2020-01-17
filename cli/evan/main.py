import argparse
import sys
from pathlib import Path

from . import compute, plot
from .config import EVAN_CACHE_DIR


def add_compute_commands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(description="supported evaluation metrics.")

    # Inception Score
    parser_is = subparsers.add_parser(
        "inception-score", help="calculate Inception Score."
    )
    parser_is.add_argument(
        "-g",
        "--gen-dir",
        type=Path,
        required=True,
        help="directory which contains generated videos by your model",
    )
    parser_is.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=-1,
        help="number of samples to use for evaluation. Use all samples by default",
    )
    parser_is.set_defaults(handler=compute.inception_score)
    # inception_score(params.gen_dir.resolve(), params.n_samples)

    # Frechet Inception Distance
    parser_fid = subparsers.add_parser(
        "frechet-distance", help="calculate Frechet Inception Distance"
    )

    parser_fid.add_argument(
        "-g",
        "--gen-dir",
        type=Path,
        required=True,
        help="directory which contains generated videos by your model",
    )
    parser_fid.add_argument(
        "-r",
        "--ref-dir",
        type=Path,
        required=True,
        help="directory path which contains reference videos to train your model",
    )
    parser_fid.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=-1,
        help="number of samples to use for evaluation. Use all samples by default",
    )
    parser_fid.set_defaults(handler=compute.frechet_distance)

    # Precision Recall for Distributions
    parser_prd = subparsers.add_parser(
        "precision-recall", help="calculate Precision and Recall for Distributions"
    )
    parser_prd.add_argument(
        "-g",
        "--gen-dir",
        type=Path,
        required=True,
        help="directory which contains generated videos by your model.",
    )
    parser_prd.add_argument(
        "-r",
        "--ref-dir",
        type=Path,
        required=True,
        help="directory path which contains reference videos to train your model.",
    )
    parser_prd.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=-1,
        help="number of samples to use for evaluation. Use all samples by default.",
    )
    parser_prd.set_defaults(handler=compute.precision_recall)


def add_plot_commands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(description="supported evaluation metrics.")

    # Precision Recall for Distributions
    parser_prd = subparsers.add_parser("precision-recall")
    parser_prd.add_argument(
        "-i",
        "--input-jsons",
        type=Path,
        required=True,
        nargs="*",
        help="json files of evalauted result. you can pass multiple files.",
    )
    parser_prd.add_argument(
        "-o", "--output", type=Path, required=True, help="path for output image file.",
    )
    parser_prd.set_defaults(handler=plot.precision_recall)


def main():
    if not EVAN_CACHE_DIR.exists():
        EVAN_CACHE_DIR.mkdir(parents=True)

    parser = argparse.ArgumentParser(
        #         usage="""evan <command> <metric> ...
        #
        # example:
        #   - evan compute inception-score -g <dir> > result.json
        #   - evan plot inception-score -i result.json""",
        description="a tool for evaluation video GANs.",
    )
    subparsers = parser.add_subparsers(description="command names.")
    compute_parser = subparsers.add_parser("compute", help="compute evaluation score.")
    add_compute_commands(compute_parser)
    plot_parser = subparsers.add_parser("plot", help="visualize evaluation result.")
    add_plot_commands(plot_parser)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        # 未知のサブコマンドの場合はヘルプを表示
        parser.print_help()
