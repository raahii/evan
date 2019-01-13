GANを評価するのに便利なコードです．本レポジトリのコードがより汎用的に使えるように，以下のような手続きで評価を行うよう設計しました．



<p align="center">
	<img src="https://user-images.githubusercontent.com/13511520/51083747-45652080-1762-11e9-880f-88139b9cc66d.png" width="80%">
</p>


## Main features

1. 動画サンプルをInception Modelを使って畳み込み特徴量へと変換する．
   Inception ModelとしてUCF-101でプレトレーニングされたResNet-101を用いています．このモデルは[video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)から拝借したもので，用途に応じて他のモデルに変更することも可能です．

2. GANモデルの評価．下記の指標がサポートされています．
   - [x] Inception Score [1]
   - [x] Frechet Inception Distace [2]
   - [x] Precision and Recall for Distributions [3]



## Requirements

- Python3
- Pytorch
- FFmpeg



## Getting Started

### 1. Install dependencies

condaを使うことを推奨しています．例えば私の環境は次のとおりです．

```
pyenv install miniconda-latest
pyenv local miniconda-latest
conda intall ffmpeg
pip install -r requirements.txt
```



### 2. Download pretrained weights of the inception model

次に，[トレーニング済みの重み](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M)をダウンロードしてください．デフォルトのモデルの重みは`resnet-101-kinetics-ucf101_split1.pth` なので，これを `models/weights/` に保存してください．



### 3. Prepare your dataset or generated samples in a direcotory

本レポジトリの評価コードでは，ディレクトリのパスを渡すことでそれ以下の`.mp4`ファイルをすべて読むように実装されています．よって，まずは評価に必要なデータセットやGANによる生成サンプルを`mp4`形式で保存してください．



### 4. Convert video samples to convolutional features

評価を行う前に，Inception Modelを用いて動画サンプルを変換する必要があります．そもそもInception Scoreは動画分類器によって出力される各クラスの確率（`p(y)`）を計算する必要がありますし，それ以外の評価指標においても，サンプルをそのままのピクセル空間特徴として扱うよりも，Inception Modelに入力して得られる中間層特徴（畳み込み特徴量）として扱うほうが精確に評価できることが指摘されています[4]（そのためこれは標準的な手続きとなっています）．

この手続きを行うには，以下実行してください．

```
python compute_conv_features.py --batchsize <batchsize> <video directory> <output directory>
```

`compute_conv_features.py` は受け取った`video directory`以下の`mp4`ファイルを読み，すべて畳み込み特徴量と各クラスの所属確率へと変換します．結果は`<output directory>`以下にそれぞれ`features.npy`，`probs.npy`として保存されます．



### 5. Perform evaluation !

最後に，`evaluate.py`で評価を行うことができます．ステップ3で指定した`<output directory>`を渡すことで，プログラムは適宜必要な`.npy`を読んで評価を実行します．



例えば，`Inception Score`の場合は，単一の動画サンプル集合から算出できるので，以下のように実行できます．

```shell
python evaluation.py is <input directory>
```



その他の`Frechet Inception Distance`や`Precision and Recall for Distributions`などは，データセットサンプルの集合と生成されたサンプルの集合の2つを用いて算出するため，以下のように実行できます．

```shell
python evaluation.py fid -o result.json <dataset directory1> <generated directory2>
```

```
python evaluation.py prd -o result.json <dataset directory1> <generated directory2>
```



### 6. Visualize results

必要に応じて可視化用のコードも用意しています．特にPRDはprecision-recall曲線を書かないと結果がよくわかりません．次のように実行することで，複数の評価結果をまとめてプロットし，保存できます．

```
python plot.py prd <result1.json> <result2.json> <prd_result.png>
```

<p align="center">
	<img src="https://user-images.githubusercontent.com/13511520/51083215-86a50280-1759-11e9-97c9-979ce04939cd.png" width="350px" height="350px">
</p>

### FAQ

準備中

### Reference

- [1] "Improved Techniques for Training GANs", [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
- [2] "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
- [3] "Assessing Generative Models via Precision and Recall", [https://arxiv.org/abs/1806.00035](https://arxiv.org/abs/1806.00035)
- [4] "An empirical study on evaluation metrics of generative adversarial networks", https://arxiv.org/abs/1806.07755



### Credit

- Icons made by [Smashicons](https://www.flaticon.com/authors/smashicons) from [www.flaticon.com](https://www.flaticon.com/) is licensed by [CC 3.0 BY](http://creativecommons.org/licenses/by/3.0/).