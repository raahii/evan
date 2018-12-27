import sys, time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collections import OrderedDict

from dataset import VideoDataet
from models import resnet

def prepare_inception_model(weight_path, gpu, mode='feature'):
    if mode not in ['score', 'feature']:
        raise ValueError

    last_fc = mode=='score'
    model = resnet.resnet34(num_classes=400, shortcut_type='A',
                            sample_size=112, sample_duration=16,
                            last_fc=last_fc)

    if torch.cuda.is_available():
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", '-w', default="weights/resnet-34-kinetics.pth")
    parser.add_argument("--gpu", '-g', default="-1")
    parser.add_argument("--batchsize", '-b', type=int, default='32')
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    
    model = prepare_inception_model(args.weight, args.gpu)
    
    dataset = VideoDataet(args.result_dir)
    dataloader = DataLoader(dataset, batch_size=args.batchsize)

    print(len(dataset))
    s = time.time()
    for videos in iter(dataloader):
        print(videos.shape)
    print(time.time()-s, '[s]')

if __name__=="__main__":
    main()
