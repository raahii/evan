import argparse
import json
import sys
from pathlib import Path
from typing import List

from evan.metrics.precision_recall_distributions import plot as plot_prd


def precision_recall(args) -> None:
    scores, labels = [], []
    for json_file in args.input_jsons:
        with open(json_file) as f:
            result = json.load(f)
            if result["type"] != args.metric:
                print(f"metric type of '{json_file}' is '{result['type']}'.", end=" ")
                print(f"please pass json file for {args.metric}.")
                sys.exit(1)

            scores.append(result["score"])
            labels.append(result["label"])

    scores = [[s["recall"], s["precision"]] for s in scores]
    plot_prd(scores, labels, args.output)
