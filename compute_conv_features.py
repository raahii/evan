import sys, time
import argparse
from pathlib import Path
import inflection
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np 
from dataset import VideoDataet
from models import resnet

def prepare_inception_model(weight_path, use_cuda):
    model = resnet.resnet101(num_classes=101, shortcut_type='B',
                             sample_size=112, sample_duration=16)

    if use_cuda:
        model.cuda()
        model_data = torch.load(weight_path)
    else:
        model_data = torch.load(weight_path, map_location='cpu')
    
    fixed_model_data = OrderedDict()
    for key, value in model_data['state_dict'].items():
        new_key = key.replace('module.', '')
        fixed_model_data[new_key] = value

    model.load_state_dict(fixed_model_data)
    model.eval()

    return model

def forward_videos(model, dataloader, use_cuda):
    softmax = torch.nn.Softmax(dim=1)
    features, probs = [], []
    with torch.no_grad():
        for videos in tqdm(iter(dataloader), 'fowarding video samples to the inception model...'):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", '-w', default="models/weights/resnet-101-kinetics-ucf101_split1.pth")
    parser.add_argument("--batchsize", '-b', type=int, default='10')
    parser.add_argument("--n_workers", '-n', type=int, default=4)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("save_path", type=Path)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    
    # init model and load pretrained weights
    model = prepare_inception_model(args.weight, use_cuda)
    
    # load generated samples as pytorch dataset
    dataset = VideoDataet(args.result_dir)
    print(f"{len(dataset)} samples found!")
    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            num_workers=args.n_workers,
                            pin_memory=False)

    # forward samples to the model and obtain results
    features, probs = forward_videos(model, dataloader, use_cuda)

    # save the outputs as .npy
    args.save_path.mkdir(parents=True, exist_ok=True)
    np.save(str(args.save_path / "features"), features)
    np.save(str(args.save_path / "probs"), probs)

if __name__=="__main__":
    main()
