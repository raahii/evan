import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from evan.dataset import VideoDataset
from evan.inception import forward_videos, prepare_inception_model
from evan.metrics.frechet_distance import compute_fid
from evan.metrics.inception_score import compute_is
from evan.metrics.precision_recall_distributions import \
    compute_prd_from_embedding

from .config import EVAN_CACHE_DIR


def load_npy(path: Union[str, Path], n: int) -> np.ndarray:
    if type(path) == Path:
        path = str(path)

    data = np.load(path)
    if n != -1:
        np.random.shuffle(data)
        data = data[:n]

    return data


def pretty_json(result):
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    )


def create_conv_features(videos_path: Path):
    batchsize = 10  # need to make editable

    # init model and load pretrained weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_inception_model(EVAN_CACHE_DIR, device)

    # load generated samples as pytorch dataset
    dataset = VideoDataset(videos_path)
    print(f"found {len(dataset)} samples.")
    dataloader = DataLoader(
        dataset, batch_size=batchsize, num_workers=0, pin_memory=True
    )

    # forward samples to the model and obtain results
    print(f"convert videos into conv features using inception model (on {device})...")
    features, probs = forward_videos(model, dataloader, device, True)

    del model

    np.save(str(videos_path / "features"), features)
    np.save(str(videos_path / "probs"), probs)


def inception_score(gen_dir: Path, n_samples: int):
    if not (gen_dir / "probs.npy").exists():
        create_conv_features(gen_dir)

    print(f"using {str(gen_dir / 'probs.npy')}...")
    probs = load_npy(gen_dir / "probs.npy", n_samples)
    score = compute_is(probs)

    result = {
        "metric": "inception-score",
        "label": gen_dir.name,
        "target": str(gen_dir),
        "score": float(score),
    }

    pretty_json(result)


def frechet_distance(gen_dir: Path, ref_dir: Path, n_samples: int):
    if not (gen_dir / "features.npy").exists():
        create_conv_features(gen_dir)

    if not (ref_dir / "features.npy").exists():
        create_conv_features(ref_dir)

    print(f"using {str(ref_dir / 'features.npy')}...")
    features_ref = load_npy(ref_dir / "features.npy", n_samples)
    print(f"using {str(gen_dir / 'features.npy')}...")
    features_gan = load_npy(gen_dir / "features.npy", n_samples)
    score = compute_fid(features_ref, features_gan)

    result = {
        "metric": "frechet-inceptin-distance",
        "label": gen_dir.name,
        "reference": str(ref_dir),
        "target": str(gen_dir),
        "score": score,
    }

    pretty_json(result)


def precision_recall(gen_dir: Path, ref_dir: Path, n_samples: int):
    if not (gen_dir / "features.npy").exists():
        create_conv_features(gen_dir)

    if not (ref_dir / "features.npy").exists():
        create_conv_features(ref_dir)

    print(f"using {str(ref_dir / 'features.npy')}...")
    features_ref = load_npy(ref_dir / "features.npy", n_samples)
    print(f"using {str(gen_dir / 'features.npy')}...")
    features_gan = load_npy(gen_dir / "features.npy", n_samples)
    score = compute_prd_from_embedding(features_ref, features_gan)
    score = {"recall": score[0].tolist(), "precision": score[1].tolist()}

    result = {
        "metric": "precision-recall-distributions",
        "label": gen_dir.name,
        "reference": str(ref_dir),
        "target": str(gen_dir),
        "score": score,
    }

    pretty_json(result)


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
        params = _p.parse_args(argv)

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
