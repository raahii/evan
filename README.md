These codes are useful for evaluating GANs for video generation. In order to make the codes can be used more generally, I designed the evaluation flow as follows.

<p align="center">
	<img src="https://user-images.githubusercontent.com/13511520/51083747-45652080-1762-11e9-880f-88139b9cc66d.png" width="80%">
</p>


## Main features

1. Convert video samples to convolutional features (embeddings) of the inception model.

    I chose ResNet-101 trained with UCF-101 dataset as inception model. I borrowed the model and codes from [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch). You can use another models available on the repository.

    

2. Perform evaluation! Various metrics for GANs are available.

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



### 3. Prepare your dataset or generated samples in a direcotory

  The evaluation codes in this repository is implemented to receive a path as input and read all `.mp4` files under the directory. Therefore, first of all you must save the dataset samples or generated samples by your model to a directory in `mp4` format.

### 4. Convert video samples to convolutional features

  Before evaluation, you need to convert the video samples using the Inception Model. In the first place, the Inception Score has to calculate the probabilities of each class output by the video classifier. In addtion, it has been pointed out that other metrics can be evaluated more accurately by treating the sample as an intermediate layer feature (convolutional feature) input to the Inception Model than by treating it as a pixel space feature[4]. So this is a standard procedure.



  To complete the above procedure, do the following:

```
python compute_conv_features.py --batchsize <batchsize> <video directory> <output directory>
```



`compute_conv_features.py ` reads all of `mp4` files in`<video directory>` , and convert them to convolutional features and probabilities of each class. The result is outputted to `features.npy`, `probs.npy` under `<output directory>`



### 5. Calculate evaluation score !

  Finally, you can peform evaluation using `evaluate.py`.  The program will read the necessary `npy` files accordingly and perform the evaluation by passing `<output directory>` in 3rd step as input.

  For example, the `Inception Score` can be calculated from a single set of video samples, and can be performed as follows:

```shell
python evaluation.py is <input directory>
```



  Other metrics, such as the `Frechet Inception Distance` and `Precision and Recall for Distributions`, are calculated using a pair of dataset samples and generated samples, and can be performed as follows:.

```shell
python evaluation.py fid -o result.json <dataset directory1> <generated directory2>
```

```
python evaluation.py prd -o result.json <dataset directory1> <generated directory2>
```



### 6. Visualize results

  You can also use visualization code if necessary. Especially for PRD, you need to plot the precision-recall curve to get the result. You can plot multiple evaluations together and save as image by doing the following．

```
python plot.py prd <result1.json> <result2.json> <prd_result.png>
```

<p align="center">
	<img src="https://user-images.githubusercontent.com/13511520/51083215-86a50280-1759-11e9-97c9-979ce04939cd.png" width="350px" height="350px">
</p>


### FAQ

  Not available yet.



### Reference

  - [1] "Improved Techniques for Training GANs", [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
  - [2] "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
  - [3] "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)
  - [4] "An empirical study on evaluation metrics of generative adversarial networks", https://arxiv.org/abs/1806.07755



### Credit

  - Icons made by [Smashicons](https://www.flaticon.com/authors/smashicons) from [www.flaticon.com](https://www.flaticon.com/) is licensed by [CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/).

