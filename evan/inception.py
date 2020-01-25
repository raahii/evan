import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import requests
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CACHE_DIR
from .dataset import VideoDataset
from .models import resnet


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def prepare_inception_model(weight_dir: Path, device: torch.device):
    filename = "resnet-101-kinetics-ucf101_split1.pth"
    weight_path = weight_dir / filename
    if not weight_path.exists():
        print("download trained model...")
        file_id = "1ACPeH9prQ7yBvb2d8QsW2kt4WNgb9JId"
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=weight_path)

    model = resnet.resnet101(
        num_classes=101, shortcut_type="B", sample_size=112, sample_duration=16
    )

    model_data = torch.load(str(weight_path), map_location="cpu")
    fixed_model_data = OrderedDict()
    for key, value in model_data["state_dict"].items():
        new_key = key.replace("module.", "")
        fixed_model_data[new_key] = value

    model.load_state_dict(fixed_model_data)
    model = model.to(device)
    model.eval()

    return model


def forward_videos(
    model, dataloader, device, verbose=False
) -> Tuple[np.ndarray, np.ndarray]:
    softmax = torch.nn.Softmax(dim=1)
    features, probs = [], []
    with torch.no_grad():
        for videos in tqdm(iter(dataloader), disable=not verbose):
            # foward samples
            videos = videos.to(device)
            _features, _probs = model(videos)

            # to cpu
            _features = _features.cpu().numpy()
            _probs = softmax(_probs).cpu().numpy()

            # add results
            features.append(_features)
            probs.append(_probs)

    return np.concatenate(features, axis=0), np.concatenate(probs, axis=0)


def create_conv_features(
    videos_path: Path, batchsize: int = 10, verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # init model and load pretrained weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_inception_model(CACHE_DIR, device)

    # load generated samples as pytorch dataset
    dataset = VideoDataset(videos_path)
    print(f">> found {len(dataset)} samples.")
    dataloader = DataLoader(
        dataset, batch_size=batchsize, num_workers=0, pin_memory=True
    )

    # forward samples to the model and obtain results
    print(
        f">> converting videos into conv features using inception model (on {device})..."
    )
    features, probs = forward_videos(model, dataloader, device, verbose)

    del model

    return features, probs
