
## Main features

1. Calculate sample embeddings using inception model.

    I chose ResNet-34 as inception model. I borrowed the model and codes from [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).

2. Various evaluation metrics for GANs are available.

   - [x] Inception Score [1]
   - [x] Frechet Inception Distace [2]
   - [x] Precision and Recall for Distributions [3]
## Requirements

- Python3
- Pytorch
- FFmpeg

## Getting Started

I strongly recommend to use conda environment. For example, my environment is like following:

```
pyenv install miniconda-latest
pyenv local miniconda-latest
conda intall ffmpeg
pip install -r requirements.txt
```

Next, download pretrained weights from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M). Save `resnet-34-kinetics.pth` to under `models/weights/`


## Calculate embedding (convolutional feature) of the inception model

```shell
python inception.py <path containes generated samples> <path to save result> --mode <'score' or 'feature'>
```

## Calculate the evaluation score

```shell
python evaluation.py is <result directory>
```

```shell
python evaluation.py fid <result directory1> <result directory2>
```

```
python evaluation.py prd <result directory1> <result directory2>
```



### Reference

- [1] "Improved Techniques for Training GANs", [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)

- [2] "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

- [3] "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)

