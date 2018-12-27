import argparse
from pathlib import Path
import inflection

import numpy as np

from metrics.frechet_distance import compute_fid
from metrics.inception_score import compute_is
from metrics.precision_recall_distributions import compute_prd_from_embedding

def load_npy(directory, mode):
    path = directory / (inflection.pluralize(mode) + ".npy")
    data = np.load(str(path))
    return data

def main():
    metrics = [
        "is",  # inception score
        "fid", # frechet inception distance
        "prd", # precision and recall for distributions
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=metrics)
    parser.add_argument("result_dir", type=Path, nargs="*")
    args = parser.parse_args()

    if len(args.result_dir) not in [1, 2]:
        raise ValueError("One or two result directories must be specified.")
    
    if args.metric == "is":
        directory = args.result_dir[0]
        mode = "score"
        inception_preds = load_npy(directory, mode)
        score = compute_is(inception_preds)

    elif args.metric  == "fid":
        dir1, dir2 = args.result_dir[:2]
        mode = "feature"
        features1 = load_npy(dir1, mode)
        features2 = load_npy(dir2, mode)
        score = compute_fid(features1, features2)

    elif args.metric  == "prd":
        dir1, dir2 = args.result_dir[:2]
        mode = "feature"
        features1 = load_npy(dir1, mode)
        features2 = load_npy(dir2, mode)
        score = compute_prd_from_embedding(features1, features2)

    print(f"{args.metric}: {score}")

if __name__=="__main__":
    main()
