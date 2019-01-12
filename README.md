
## Main features

1. Convert video samples to convolutional features (embeddings of inception model) using inception model.

    I chose ResNet-101 trained with UCF-101 dataset as inception model. I borrowed the model and codes from [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch). You can use another models available on the repository.

2. Compute Various evaluation metrics for GANs are available.

   - [x] Inception Score [1]
   - [x] Frechet Inception Distace [2]
   - [x] Precision and Recall for Distributions [3]
## Requirements

- Python3
- Pytorch
- FFmpeg

## Getting Started

### 1. Install dependencies

I strongly recommend to use conda environment. For example, my environment is like following:

```
pyenv install miniconda-latest
pyenv local miniconda-latest
conda intall ffmpeg
pip install -r requirements.txt
```



### 2. Download pretrained weights of the inception model

Next, download pretrained weights from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M). Save `resnet-101-kinetics-ucf101_split1.pth` to under `models/weights/`.



### 3. Convert video samples to convolutional features

Inception Score needs probability of `y`. Moreover, all of other evaluation metrics can be computed more accurately by using convolutional features as `x` and it is standard. 

First of all, you must prepare a directory contains video samples in `.mp4` format. And then you can obtain convolutional features by:

```
python compute_conv_features.py --batchsize <batchsize> <video directory> <output directory>
```



This script outputs intermidiate convolutional features (`features.npy`) and probability for each classes  (`probs.npy`) under `<output directory>`.



### 4. Calculate evaluation score !

After you obtain convolutional features, you can finally evaluate your model.

For example, if you want to compute `Inception Score`, the metric needs only features of your model, so you can run:

```shell
python evaluation.py is <output directory>
```



If you want to compute `FID` or `PRD`, the metric needs the pair of dataset samples and generated samples, so you can run:

```shell
python evaluation.py fid <dataset directory1> <generated directory2>
```

```
python evaluation.py prd <dataset directory1> <generated directory2>
```



### Reference

- [1] "Improved Techniques for Training GANs", [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)

- [2] "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

- [3] "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)

