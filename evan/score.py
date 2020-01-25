from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import VideoDataset
from .inception import create_conv_features
from .metrics.frechet_distance import compute_fid
from .metrics.inception_score import compute_is
from .metrics.precision_recall_distributions import compute_prd_from_embedding


def load_npy(path: Union[str, Path], n: int) -> np.ndarray:
    if type(path) == Path:
        path = str(path)

    data = np.load(path)
    if n != -1:
        np.random.shuffle(data)
        data = data[:n]

    return data


def compute_inception_score(
    _gen_dir: str, n_samples: int = -1, verbose: bool = False
) -> float:
    gen_dir: Path = Path(_gen_dir)

    if not (gen_dir / "probs.npy").exists():
        features, probs = create_conv_features(gen_dir, verbose=verbose)
        np.save(str(gen_dir / "features"), features)
        np.save(str(gen_dir / "probs"), probs)

    print(">> computing IS...")
    print(f"     generated samples: '{str(gen_dir / 'probs.npy')}'")
    probs = load_npy(gen_dir / "probs.npy", n_samples)
    score = compute_is(probs)

    return score


def compute_frechet_distance(
    _gen_dir: str, _ref_dir: str, n_samples: int = -1, verbose: bool = False
):
    gen_dir: Path = Path(_gen_dir)
    if (gen_dir / "features.npy").exists():
        features_gen = load_npy(gen_dir / "features.npy", n_samples)
    else:
        features_gen, probs_gen = create_conv_features(gen_dir, verbose=verbose)
        np.save(str(gen_dir / "features"), features_gen)
        np.save(str(gen_dir / "probs"), probs_gen)

    ref_dir: Path = Path(_ref_dir)
    if (ref_dir / "features.npy").exists():
        features_ref = load_npy(ref_dir / "features.npy", n_samples)
    else:
        features_ref, probs_ref = create_conv_features(ref_dir, verbose=verbose)
        np.save(str(ref_dir / "features"), features_ref)
        np.save(str(ref_dir / "probs"), probs_ref)

    print(">> computing FID...")
    print(f"     generated samples: {str(gen_dir / 'features.npy')}")
    print(f"     dataset samples: {str(ref_dir / 'features.npy')}")
    score = compute_fid(features_ref, features_gen)

    return float(score)


def compute_precision_recall(
    _gen_dir: str, _ref_dir: str, n_samples: int = -1, verbose: bool = False
):
    gen_dir: Path = Path(_gen_dir)
    if (gen_dir / "features.npy").exists():
        features_gen = load_npy(gen_dir / "features.npy", n_samples)
    else:
        features_gen, probs_gen = create_conv_features(gen_dir, verbose=verbose)
        np.save(str(gen_dir / "features"), features_gen)
        np.save(str(gen_dir / "probs"), probs_gen)

    ref_dir: Path = Path(_ref_dir)
    if (ref_dir / "features.npy").exists():
        features_ref = load_npy(ref_dir / "features.npy", n_samples)
    else:
        features_ref, probs_ref = create_conv_features(ref_dir, verbose=verbose)
        np.save(str(ref_dir / "features"), features_ref)
        np.save(str(ref_dir / "probs"), probs_ref)

    print(">> computing PRD...")
    print(f"     generated samples: {str(gen_dir / 'features.npy')}")
    print(f"     dataset samples: {str(ref_dir / 'features.npy')}")
    score = compute_prd_from_embedding(features_ref, features_gen)

    return {"recall": score[0].tolist(), "precision": score[1].tolist()}
