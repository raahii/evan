import argparse
import json
import sys
from pathlib import Path
from typing import List

from evan.metrics.precision_recall_distributions import plot as plot_prd


def precision_recall(metric: str, json_files: List[Path], save_path: Path):
    scores, labels = [], []
    for json_file in json_files:
        with open(json_file) as f:
            result = json.load(f)
            if result["type"] != metric:
                print(f"metric type of '{json_file}' is '{result['type']}'.", end=" ")
                print(f"please pass json file for {metric}.")
                sys.exit(1)

            scores.append(result["score"])
            labels.append(result["label"])

    scores = [[s["recall"], s["precision"]] for s in scores]
    plot_prd(scores, labels, save_path)


def do_plot_command(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        add_help=False,
        description="visualize evaluation result. please specify metric name.",
        prog="evan plot",
    )
    p.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["precision-recall"],
        required=True,
        help="evaluation metric name",
    )
    args, rest_args = p.parse_known_args(argv)

    if args.metric == "precision-recall":
        _p = argparse.ArgumentParser(
            description="visualize Precision and Recall for Distribution.s",
            prog=f"evan plot -m {args.metric}",
        )
        _p.add_argument(
            "-i",
            "--input-jsons",
            type=Path,
            required=True,
            nargs="*",
            help="json files of evalauted result. you can pass multiple files.",
        )
        _p.add_argument(
            "-o",
            "--output",
            type=Path,
            required=True,
            help="path for output image file.",
        )
        params = _p.parse_args(rest_args)

        precision_recall(args.metric, params.input_jsons, params.output)
