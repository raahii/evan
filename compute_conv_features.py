import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import VideoDataet
from .models import resnet

root_path = Path(__file__).parent.resolve()
default_weight_path = root_path / "models/weights/resnet-101-kinetics-ucf101_split1.pth"


def prepare_inception_model(weight_path, use_cuda):
    model = resnet.resnet101(
        num_classes=101, shortcut_type="B", sample_size=112, sample_duration=16
    )

    if use_cuda:
        model.cuda()
        model_data = torch.load(weight_path)
    else:
        model_data = torch.load(weight_path, map_location="cpu")

    fixed_model_data = OrderedDict()
    for key, value in model_data["state_dict"].items():
        new_key = key.replace("module.", "")
        fixed_model_data[new_key] = value

    model.load_state_dict(fixed_model_data)
    model.eval()

    return model


def forward_videos(model, dataloader, use_cuda):
    softmax = torch.nn.Softmax(dim=1)
    features, probs = [], []
    with torch.no_grad():
        for videos in tqdm(
            iter(dataloader), "fowarding video samples to the inception model..."
        ):
            # foward samples
            videos = videos.cuda() if use_cuda else videos
            inputs = Variable(videos.float())
            _features, _probs = model(inputs)

            # to cpu
            _features = _features.data.cpu().numpy()
            _probs = softmax(_probs).data.cpu().numpy()

            # add results
            features.append(_features)
            probs.append(_probs)

    features = np.concatenate(features, axis=0)
    probs = np.concatenate(probs, axis=0)

    return features, probs


def convert(batchsize, result_dir, weight=default_weight_path, n_workers=8):
    use_cuda = torch.cuda.is_available()

    # init model and load pretrained weights
    model = prepare_inception_model(str(weight), use_cuda)

    # load generated samples as pytorch dataset
    dataset = VideoDataet(result_dir)
    print(f"{len(dataset)} samples found!")
    dataloader = DataLoader(
        dataset, batch_size=batchsize, num_workers=n_workers, pin_memory=False
    )

    # forward samples to the model and obtain results
    features, probs = forward_videos(model, dataloader, use_cuda)

    return features, probs


def save(features, probs, save_path):
    # save the outputs as .npy
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path / "features"), features)
    np.save(str(save_path / "probs"), probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", "-w", type=Path, default=default_weight_path)
    parser.add_argument("--batchsize", "-b", type=int, default="10")
    parser.add_argument("--n_workers", "-n", type=int, default=4)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("save_path", type=Path)
    args = parser.parse_args()

    features, probs = convert(
        args.batchsize, args.weight, args.result_dir, args.n_workers
    )
    save(features, probs, args.save_path)


if __name__ == "__main__":
    main()
