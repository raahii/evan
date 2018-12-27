import argparse
from pathlib import Path
import inflection

import numpy as np

def inception_score(pyx):
    # pyx: p(y|x)
    # py:  p(y)
    p_y = np.mean(pyx, axis=0)
    e = pyx*np.log(pyx/p_y) # KL divergence between p(y|x) and p(y)
    e = np.sum(e, axis=1)
    e = np.mean(e, axis=0)

    return np.exp(e)

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

    if len(args.result_dir) == 0:
        raise ValueError("One or two result directory must be specified.")
    
    if args.metric in ["is"]:
        directory = args.result_dir[0]
        mode = "score"
        inception_preds = load_npy(directory, mode)
        score = inception_score(inception_preds)
    elif args.metric in ["fid", "prd"]:
        mode = "feature"
        raise NotImplemented

    print(f"{args.metric}: {score}")

if __name__=="__main__":
    main()
