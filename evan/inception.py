import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import VideoDataset
from .models import resnet


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def download_model(file_id: str, path: Path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768

    f = open(path, "wb")
    for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:
            f.write(chunk)
    f.close()


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


def forward_videos(model, dataloader, device, verbose=False):
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

    features = np.concatenate(features, axis=0)
    probs = np.concatenate(probs, axis=0)

    return features, probs


def convert(batchsize, result_dir, n_workers=0, verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model and load pretrained weights
    model = prepare_inception_model(device)

    # load generated samples as pytorch dataset
    dataset = VideoDataset(result_dir)
    # print(f"{len(dataset)} samples found!")
    dataloader = DataLoader(
        dataset, batch_size=batchsize, num_workers=n_workers, pin_memory=True
    )

    # forward samples to the model and obtain results
    features, probs = forward_videos(model, dataloader, device, verbose)

    del model

    return features, probs


def save(features, probs, save_path):
    # save the outputs as .npy
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path / "features"), features)
    np.save(str(save_path / "probs"), probs)
