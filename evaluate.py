import argparse
import json
from pathlib import Path

import numpy as np

from .metrics.frechet_distance import compute_fid
from .metrics.inception_score import compute_is
from .metrics.precision_recall_distributions import compute_prd_from_embedding


def load_npy(directory, data_type, n_samples):
    path = directory / (data_type + ".npy")
    data = np.load(str(path))

    if n_samples is not None:
        np.random.shuffle(data)
        data = data[:n_samples]

    return data


def compute_metric(metric, result_dirs, n_samples=None):
    if metric == "is":
        data_type = "probs"
        directory = result_dirs[0]
        inception_preds = load_npy(directory, data_type, n_samples)
        score = compute_is(inception_preds)

        result = {
            "type": metric,
            "label": directory.name,
            "target": str(directory),
            "score": float(score),
        }

    elif metric == "fid":
        data_type = "features"
        dir1, dir2 = result_dirs[:2]
        features1 = load_npy(dir1, data_type, n_samples)
        features2 = load_npy(dir2, data_type, n_samples)
        score = compute_fid(features1, features2)

        result = {
            "type": metric,
            "label": dir2.name,
            "reference": str(dir1),
            "target": str(dir2),
            "score": score,
        }

    elif metric == "prd":
        data_type = "features"
        dir1, dir2 = result_dirs[:2]
        features1 = load_npy(dir1, data_type, n_samples)
        features2 = load_npy(dir2, data_type, n_samples)
        score = compute_prd_from_embedding(features2, features1)
        score = {"recall": score[0].tolist(), "precision": score[1].tolist()}

        result = {
            "type": metric,
            "label": dir2.name,
            "reference": str(dir1),
            "target": str(dir2),
            "score": score,
        }

    return result


def main():
    metrics = [
        "is",  # inception score
        "fid",  # frechet inception distance
        "prd",  # precision and recall for distributions
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=metrics)
    parser.add_argument("result_dirs", type=Path, nargs="*")
    parser.add_argument("--output_json", "-o", type=Path)
    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=None,
        help="Number of samples to use for evaluation.",
    )
    args = parser.parse_args()

    if len(args.result_dirs) not in [1, 2]:
        raise ValueError("One or two result directories must be specified.")

    result = compute_metric(args.metric)

    if args.output_json:
        args.output_json.parents[0].mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
            )
    else:
        print(result)


if __name__ == "__main__":
    main()
