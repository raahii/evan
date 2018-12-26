import argparse

import torch

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
    
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", '-w', default="weights/resnet-34-kinetics-ucf101_split1.pth")
    parser.add_argument("--gpu", '-g', default="-1")
    args = parser.parse_args()
    
    model = prepare_inception_model(args.weight, args.gpu)
    print(model)

if __name__=="__main__":
    main()
