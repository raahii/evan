GANs for video evaluation
--

### Main features

1. Calculate video embeddings using inception model.
  
  I choose ResNet-34 as inception model. I borrowed the model and code from [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).

2. Various evaluation metrics for GANS are available.
  - [] Inception Score[^is]
  - [] Frechet Inception Distace[^fid]
  - [] Precision and Recall for Distributions[^prd]

### Getting Started

I strongly recommend use conda environment. My environment is folloing:

```
pyenv install miniconda-latest
pyenv local miniconda-latest
conda intall ffmpeg
pip install -r requirements.txt
```

[^is]:  Improved Techniques for Training GANs, [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
[^fid]: GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
[^prd]: "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)

