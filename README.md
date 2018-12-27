
## Main features

1. Calculate video embeddings using inception model.

    I chose ResNet-34 as inception model. I borrowed the model and code from [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).

2. Various evaluation metrics for GANS are available.

  	- [ ] Inception Score[^is]
  	- [ ] Frechet Inception Distace[^fid]
  	- [ ] Precision and Recall for Distributions[^prd]

## Requirements

- Python3
- Pytorch
- ffmpeg

## Getting Started

I strongly recommend to use conda environment. For example, my environment is like following:

```
pyenv install miniconda-latest
pyenv local miniconda-latest
conda intall ffmpeg
pip install -r requirements.txt
```

Next, download pretrained weights from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M). 

Save `resnet-34-kinetics.pth` to under `weights/`


---

[^is]:  "Improved Techniques for Training GANs", [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
[^fid]: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
[^prd]: "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)

