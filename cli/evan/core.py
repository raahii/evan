import argparse
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from evan.metrics.frechet_distance import compute_fid
from evan.metrics.inception_score import compute_is
from evan.metrics.precision_recall_distributions import \
    compute_prd_from_embedding


def load_npy(path: Union[str, Path], n: Optional[int]) -> np.ndarray:
    if type(path) == Path:
        path = str(path)

    data = np.load(path)
    if n is not None:
        np.random.shuffle(data)
        data = data[:n]

    return data


def command_is(args):
    probs = load_npy(args.path / "probs.npy", args.n_samples)
    score = compute_is(probs)

    result = {
        "type": "inception-score",
        "label": args.path.name,
        "target": str(args.path),
        "score": float(score),
    }

    return result


def command_fid(args):
    features_ref = load_npy(args.path_ref / "features.npy", args.n_samples)
    features_gan = load_npy(args.path_gan / "features.npy", args.n_samples)
    score = compute_fid(features_ref, features_gan)

    result = {
        "type": "frechet-inceptin-distance",
        "label": args.gan.name,
        "reference": str(args.path_ref),
        "target": str(args.path_gan),
        "score": score,
    }

    return result


def command_prd(args):
    features_ref = load_npy(args.path_ref / "features.npy", args.n_samples)
    features_gan = load_npy(args.path_gan / "features.npy", args.n_samples)
    score = compute_prd_from_embedding(features2, features1)
    score = {"recall": score[0].tolist(), "precision": score[1].tolist()}

    result = {
        "type": "precision-recall-distributions",
        "label": args.gan.name,
        "reference": str(args.path_ref),
        "target": str(args.path_gan),
        "score": score,
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Inception Score
    parser_is = subparsers.add_parser(
        "is", help="Compute Inception Score. See `is -h` for detail."
    )
    parser_is.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Folder path which contains generated videos by your model.",
    )
    parser_is.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=None,
        help="Number of samples to use for evaluation. Use all samples by default.",
    )
    parser_is.set_defaults(handler=command_is)

    # Frechet Inception Distance
    parser_fid = subparsers.add_parser(
        "fid", help="Compute Geometry Score. See `fid -h` for detail."
    )
    parser_fid.add_argument(
        "-pr",
        "--path-ref",
        type=Path,
        required=True,
        help="Folder path which contains reference training videos by your model.",
    )
    parser_fid.add_argument(
        "-pg",
        "--path-gan",
        type=Path,
        required=True,
        help="Folder path which contains generated videos by your model.",
    )
    parser_fid.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=None,
        help="Number of samples to use for evaluation. Use all samples by default.",
    )
    parser_fid.set_defaults(handler=command_fid)

    # Precision Recall Distribution
    parser_prd = subparsers.add_parser(
        "prd",
        help="Compute Precision Recall for Distributions. See `prd -h` for detail.",
    )
    parser_prd.add_argument(
        "-pr",
        "--path-ref",
        type=Path,
        required=True,
        help="Folder path which contains reference training videos by your model.",
    )
    parser_prd.add_argument(
        "-pg",
        "--path-gan",
        type=Path,
        required=True,
        help="Folder path which contains generated videos by your model.",
    )
    parser_prd.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=None,
        help="Number of samples to use for evaluation. Use all samples by default.",
    )
    parser_prd.set_defaults(handler=command_prd)

    # run commands
    args = parser.parse_args()
    if hasattr(args, "handler"):
        result = args.handler(args)
    else:
        parser.print_help()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    )


if __name__ == "__main__":
    main()
