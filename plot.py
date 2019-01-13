import argparse
from pathlib import Path
import json

from metrics.precision_recall_distributions import plot as plot_prd

def main():
    metrics = [
        "is",  # inception score
        "fid", # frechet inception distance
        "prd", # precision and recall for distributions
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=metrics)
    parser.add_argument("json_files", type=Path, nargs="*")
    parser.add_argument("output_fig", type=Path)
    args = parser.parse_args()
    
    scores, labels = [], []
    for json_file in args.json_files:
        with open(json_file) as f:
            result = json.load(f)
            assert result["type"] == args.metric
            scores.append(result["score"])
            labels.append(result["label"])
    
    if args.metric == "is":
        raise NotImplemented
    elif args.metric  == "fid":
        raise NotImplemented
    elif args.metric  == "prd":
        scores = [[s["recall"], s["precision"]] for s in scores]
        plot_prd(scores, labels, args.output_fig)

if __name__=="__main__":
    main()
