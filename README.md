"Evan" is a CLI tool (and python package) for evaluating video GANs.  You can calculates some famous evaluation metrics such as Incetion Score without preparing Incetion Model, codes for the metric.

<p align="center">
	<img src="https://user-images.githubusercontent.com/13511520/51083747-45652080-1762-11e9-880f-88139b9cc66d.png" width="80%">
</p>



## Main features

1. Convert video samples to convolutional features (embeddings) of the inception model.

    I chose ResNet-101 trained with UCF-101 dataset as inception model. I borrowed the model and codes from [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch). You can use another models available on the repository.

    

2. Perform evaluation. Various metrics for GANs are available.

   - [x] Inception Score [1]
   - [x] Frechet Inception Distace [2]
   - [x] Precision and Recall for Distributions [3]
     
## Requirements

- Python3
- FFmpeg
  

## Installation

```
pip install evan
```



## Getting Started

### 1. Prepare mp4 videos

  The evaluation codes in this repository is implemented to receive a path as input and read all `.mp4` files under the directory. Therefore, first of all you must save the dataset samples or generated samples by your model to a directory in `mp4` format.

### 2. Compute score

  Finally, you can peform evaluation using `evaluate.py`.  The program will read the necessary `npy` files accordingly and perform the evaluation by passing `<output directory>` in 3rd step as input.

  For example, the `Inception Score` can be calculated from a single set of video samples, and can be performed as follows:

```shell
evan -m inception-score <input directory>
```



  Other metrics, such as the `Frechet Inception Distance` and `Precision and Recall for Distributions`, are calculated using a pair of dataset samples and generated samples, and can be performed as follows:.

```shell
evan -m frechet-distance -r <dataset samples dir> -g <generated samples dir>
```

```shell
evan -m precision-recall -r <dataset samples dir> -g <generated samples dir>
```



### 3. Visualize results

  You can also use visualization code if necessary. Especially for PRD, you need to plot the precision-recall curve to get the result. You can plot multiple evaluations together and save as image by doing the following．

```shell
evan plot -m precision-recall -i <1st result json> <2nd result json> ... -o <output figure path>
```

<p align="center">
	<img src="https://user-images.githubusercontent.com/13511520/51083215-86a50280-1759-11e9-97c9-979ce04939cd.png" width="350px" height="350px">
</p>

### Reference

  - [1] "Improved Techniques for Training GANs", [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
  - [2] "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
  - [3] "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)
  - [4] "An empirical study on evaluation metrics of generative adversarial networks", https://arxiv.org/abs/1806.07755



## License:

MIT



## Authors:

[raahii](https://raahii.github.io/about)

### Credit

  - Icons made by [Smashicons](https://www.flaticon.com/authors/smashicons) from [www.flaticon.com](https://www.flaticon.com/) is licensed by [CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/).

