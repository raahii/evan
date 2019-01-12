import argparse
from pathlib import Path

import numpy as np

from metrics.frechet_distance import compute_fid
from metrics.inception_score import compute_is
from metrics.precision_recall_distributions import compute_prd_from_embedding

def load_npy(directory, data_type, n_samples=None):
    path = directory / (data_type + ".npy")
    data = np.load(str(path))

    if n_samples is not None:
        np.random.shuffle(data)
        data = data[:n_samples]

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
    parser.add_argument("--n_samples", "-n", type=int, default=None,\
                        help="Number of samples to use for evaluation.")
    args = parser.parse_args()

    if len(args.result_dir) not in [1, 2]:
        raise ValueError("One or two result directories must be specified.")
    
    if args.metric == "is":
        directory = args.result_dir[0]
        data_type = "probs"
        inception_preds = load_npy(directory, data_type, args.n_samples)
        score = compute_is(inception_preds)

    elif args.metric  == "fid":
        dir1, dir2 = args.result_dir[:2]
        data_type = "features"
        features1 = load_npy(dir1, data_type, args.n_samples)
        features2 = load_npy(dir2, data_type, args.n_samples)
        score = compute_fid(features1, features2)

    elif args.metric  == "prd":
        dir1, dir2 = args.result_dir[:2]
        data_type = "features"
        features1 = load_npy(dir1, data_type, args.n_samples)
        features2 = load_npy(dir2, data_type, args.n_samples)
        score = compute_prd_from_embedding(features1, features2)

    print(score)

if __name__=="__main__":
    main()
