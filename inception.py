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

def prepare_inception_model(weight_path, use_cuda, mode='feature'):
    if mode not in ['score', 'feature']:
        raise ValueError

    last_fc = mode=='score'
    model = resnet.resnet34(num_classes=400, shortcut_type='A',
                            sample_size=112, sample_duration=16,
                            last_fc=last_fc)

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
    softmax = torch.nn.Softmax()
    outputs = []
    with torch.no_grad():
        for videos in tqdm(iter(dataloader), 'forwarding...'):
            videos = videos.cuda() if use_cuda else videos
            inputs = Variable(videos.float())
            output = softmax(model(inputs))
            outputs.append(output.data.cpu().numpy())

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", '-w', default="weights/resnet-34-kinetics.pth")
    parser.add_argument("--batchsize", '-b', type=int, default='32')
    parser.add_argument("--mode", '-m', choices=['score', 'feature'], default='feature')
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("save_path", type=Path)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    
    # init model and load pretrained weights
    model = prepare_inception_model(args.weight, use_cuda, args.mode)
    
    # load generated samples as pytorch dataset
    dataset = VideoDataet(args.result_dir)
    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            pin_memory=False)

    # forward samples to the model and obtain results
    outputs = forward_videos(model, dataloader, use_cuda)
    outputs = np.stack(outputs)
    dim_vector = outputs.shape[-1]
    outputs = outputs.reshape(-1, dim_vector)

    # save the outputs as .npy
    args.save_path.mkdir(parents=True, exist_ok=True)
    path = args.save_path / inflection.pluralize(args.mode)
    np.save(str(path), outputs)

if __name__=="__main__":
    main()
