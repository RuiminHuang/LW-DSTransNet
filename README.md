# LW-DSTransNet

[![Static Badge](https://img.shields.io/badge/building-pass-green?style=flat-square)](https://github.com/RuiminHuang/DSTransNet)
[![Static Badge](https://img.shields.io/badge/language-Python-blue?style=flat-square)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/framework-PyTorch-blue?style=flat-square)](https://pytorch.org/)
[![Static Badge](https://img.shields.io/badge/license-Apache2.0-blue?style=flat-square)](./LICENSE)
[![Static Badge](https://visitor-badge.laobi.icu/badge?page_id=RuiminHuang.LW-DSTransNet)](https://github.com/RuiminHuang/LW-DSTransNet)

The official implementation of the paper "Toward Extremely Efficient Infrared Small Target Detection Via Universal U-Net-based Knowledge Distillation" in PyTorch.

> This repository provides **clean and readable** code.

## Contents

- [:sparkles: 1. Introduction](#sparkles-1-introduction)
- [:building_construction: 2. The Network](#building_construction-2-the-network)
  - [:repeat: 2.1 Overall Pipeline](#repeat-21-overall-pipeline)
  - [:jigsaw: 2.2 Core Module](#jigsaw-22-core-module)
- [:rocket: 3. Installation](#rocket-3-installation)
- [:bar_chart: 4. Dataset Preparation](#bar_chart-4-dataset-preparation)
  - [:link: 4.1 Datasets Link](#link-41-datasets-link)
  - [:file_folder: 4.2 File Structure](#file_folder-42-file-structure)
- [:fire: 5. Train](#fire-5-train)
- [:dart: 6. Test](#dart-6-test)
- [:trophy: 7. Benchmark and Model Zoo](#trophy-7-benchmark-and-model-zoo)
  - [:chart_with_upwards_trend: 7.1 Quantitative Results](#chart_with_upwards_trend-71-quantitative-results)
  - [:framed_picture: 7.2 Qualitative Results](#framed_picture-72-qualitative-results)
  - [:package: 7.3 Model Zoo](#package-73-model-zoo)
- [:bookmark_tabs: 8. Citation](#bookmark_tabs-8-citation)
- [:star2: 9. Star History](#star2-9-star-history)
- [:email: 10. Contact](#email-10-contact)


---

## :sparkles: 1. Introduction

<div align="center">
  <img src="./figures/Figure1.png" width="500" alt="Figure1">
</div>

The heatmaps illustrate the areas of interest for various layers within the model. Layers subjected to specific analysis are indicated by red dots beneath the figure. (a) infrared image. (b) feature extraction stage, heatmap of focused features. (c) feature selection stage. Areas marked by red borders represent the suppression of false-alarm features, and areas marked by green borders signify the enhancement of small target features. (d) feature fusion-based reconstruction stage. Attention is paid to small target features during the reconstruction process. (e) ground truth label.

## :building_construction: 2. The Network

### :repeat: 2.1 Overall Pipeline
![Figure2](./figures/Figure2.png)
Overall architecture of the proposed DSTransNet. It incorporates three stages. The feature extraction stage is responsible for finer feature extraction. The feature selection stage aims to suppress false alarms similar to small targets and enhance real small target features. The fusion-based reconstruction stage emphasises small target features and reconstructs the final multi-scale small target mask.

### :jigsaw: 2.2 Core Module
![Figure4](./figures/Figure4.png)
Proposed RDSF module. It serves to suppress false alarms similar to small targets and enhance real small target features.


## :rocket: 3. Installation

* Step 1. Clone the repository

```shell
git clone git@github.com:RuiminHuang/DSTransNet.git
cd DSTransNet
```

* Step 2. Create environment and install dependencies

```shell
conda create --name DSTransNet python=3.12
conda activate DSTransNet
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tensorboard==2.19.0
pip install tqdm==4.65.0
```


## :bar_chart: 4. Dataset Preparation


### :link: 4.1 Datasets Link
The dataset comes from [this GitHub repository](https://github.com/GrokCV/SeRankDet). The datasets used in this project and the dataset split files can be downloaded from the following links:

* SIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1LgnBKcE8Cqlay5GnXfUaLA?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEgG0VEoH3nDbiWDA?e=gkUW2W)
* NUDT-SIRST Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/16BbL9H38cIcvaBh4tPNTCw?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEdBMrQDFM1Vi24DQ?e=vBNoN4)
* IRSTD1K Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1nRoZu1eI9BLnpmsxw0Kdwg?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEepi2ipymni0amNQ?e=XZILFh)
* SIRST-AUG Dataset
  * [Baidu Netdisk](https://pan.baidu.com/s/1_kAocokYSclQNf_ZLWPIhQ?pwd=grok)
  * [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEfdtbrZhLsbd0ITg?e=thyA6h)


### :file_folder: 4.2 File Structure

```shell
|- datasets
    |- NUAA
        |-trainval
            |-images
                |-Misc_1.png
                ......
            |-masks
                |-Misc_1.png
                ......
        |-test
            |-images
                |-Misc_50.png
                ......
            |-masks
                |-Misc_50.png
                ......
    |-NUDT
    |-IRSTD1k
    |-SIRSTAUG
```


Before running the code, make sure to update the dataset path in the config file:

![datasets_path](./figures/datasets_path.png)

## :fire: 5. Train

```shell
python train.py
```

Before training, specify the target datasets in the configuration file:

![train_config](./figures/train_config.png)

To view train process, run TensorBoard with:

```shell
tensorboard --port=8010 --samples_per_plugin=images=100000 --logdir=./
```

![train_log](./figures/train_log.png)


## :dart: 6. Test

```shell
python test.py
```

Before testing, specify the pre-trained weights and target datasets in the configuration file:

![test_config](./figures/test_config.png)


To view test results, run TensorBoard with:

```shell
tensorboard --port=8010 --samples_per_plugin=images=100000 --logdir=./
```

![train_log1](./figures/test_log1.png)
![train_log2](./figures/test_log2.png)



## :trophy: 7. Benchmark and Model Zoo

### :chart_with_upwards_trend: 7.1 Quantitative Results

| Methods                                 | Source      | Key Parameters Configurations                                                                                               |
| --------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| Filter-based Methods                    |             |                                                                                                                             |
| Max-Median \cite{deshpande1999max}      | SPIE'1999   | -$^{*}$                                                                                                                     |
| Top-Hat \cite{bai2010analysis}          | PR'2010     | ${R}*{0}$ = 9, ${R}*{i}$ = 4                                                                                                |
| FKRW \cite{qin2019infrared}             | TGRS'2019   | Windows size: 11, $K$ = 4, $p$ = 6, $\beta$ = 200                                                                           |
| HVS-based Methods                       |             |                                                                                                                             |
| MPCM \cite{wei2016multiscale}           | PR'2016     | $N = { 1, 3, ..., 9 }$, threshold $k$ = 3                                                                                   |
| RLCM \cite{han2018infrared}             | GRSL'2018   | ${k}*{1}$ = [2, 5, 9], ${k}*{2}$ = [4, 9, 16], scale: 3, threshold $k$ = 1                                                  |
| TLLCM \cite{han2019local}               | GRSL'2019   | Scales: [5, 7, 9], $\lambda = 0.5$, threshold $k$ = 3                                                                       |
| ILCM \cite{han2020infrared}             | GRSL'2020   | Cell size: $3 \times 3$, threshold $k$ = 3                                                                                  |
| GSWLCM \cite{qiu2022global}             | GRSL'2022   | Local Window Structure: [3, 5, 7, 9], $\delta$ = 0.01, $k$ = 20                                                             |
| LRM-based Methods                       |             |                                                                                                                             |
| IPI \cite{gao2013infrared}              | TIP'2013    | Patch size: 50, sliding step: 10, $\lambda = 1/\sqrt{\mathrm{min}(m,n)}$, $\epsilon = 10^{-7}$, $k = 8$                     |
| RIPT \cite{dai2017reweighted}           | JSTARS'2017 | Patch size: 50, sliding step: 10, $\lambda = 2/\sqrt{\mathrm{min}(m,n)}$, $\epsilon = 10^{-2}$, $\omega = 10^{-7}$, $k = 5$ |
| PSTNN \cite{zhang2019infrared_PSTNN}    | RS'2019     | Patch size: 40, sliding step: 40, $\lambda = 0.6/\sqrt{\mathrm{max}(n_1,n_2)\times n_3}$, $\epsilon = 10^{-7}$, $k = 2$     |
| Deep Learning Methods (Lightweight)     |             |                                                                                                                             |
| LW-IRSTNet \cite{kou2023lw}             | TGRS'2023   | Channels: [8, 32, 64], kernel size: [7, 7, 7, 7]                                                                            |
| IRPruneDeXt \cite{zhang2025irprunedext} | TNNLS'2025  | -$^{*}$                                                                                                                     |
| LW-DSTransNet-Nano                      | -           | Channels: [4, 8, 16, 32, 32], Patch size: [16, 8, 4, 2]                                                                     |
| Deep Learning Methods (Standard)        |             |                                                                                                                             |
| ACM \cite{dai2021asymmetric}            | WACV'2021   | Backbone: UNet, layer blocks: [3, 3, 3], channels: [8, 16, 32, 64], fuse mode: AsymBi                                       |
| ALCNet \cite{dai2021attentional}        | TGRS'2021   | Backbone: FPN, layer blocks: [4, 4, 4], channels: [8, 16, 32, 64], fuse mode: AsymBi                                        |
| DNANet \cite{li2022dense}               | TIP'2022    | Backbone: resnet18, layer blocks: [2, 2, 2, 2], filter: [16, 32, 64, 128, 256]                                              |
| ISTDU-Net \cite{Hou2022ISTDUNetIS}      | GRSL'2022   | Down Channels: [16, 32, 64, 128], up channels: [512, 256, 128, 64]                                                          |
| RDIAN \cite{sun2023receptive}           | TGRS'2023   | Channels: [32, 32, 32, 32], layer blocks: [1, 2, 2, 2], kernel size: [1, 3, 5, 7]                                           |
| UIUNet \cite{wu2022uiu}                 | TIP'2023    | Channels: [64, 128, 256, 512], fuse mode: AsymBi                                                                            |
| SCTransNet \cite{yuan2024sctransnet}    | TGRS'2024   | Channels: [32, 64, 128, 256], Patch size: [16, 8, 4, 2]                                                                     |
| MLP-Net \cite{wang2024mlp}              | TGRS'2025   | Patch size: 4, embed dim: 64, layer blocks: [4, 4, 4, 4]                                                                    |
| LW-DSTransNet-Tiny                      | -           | Channels: [8, 16, 32, 64, 64], Patch size: [16, 8, 4, 2]                                                                    |
| LW-DSTransNet-Small                     | -           | Channels: [16, 32, 64, 128, 128], Patch size: [16, 8, 4, 2]                                                                 |



### :framed_picture: 7.2 Qualitative Results
![Figure8](./figures/Figure8.png)
2D visualization of detection results across different methods on representative images from SIRST and IRSTD1K datasets. Blue, yellow, and red circles denote correct detections, missed detections, and false alarms, respectively.

### :package: 7.3 Model Zoo

TensorBoard logs, train logs, test logs, pre-trained weights, and test results are available on [Google Drive](https://drive.google.com/drive/folders/1Cktwh19m4gm0PVY63o_HWHXe6CXkqhOf?usp=sharing). Just download and unzip it to the [log path](./logs/).

## :bookmark_tabs: 8. Citation

If you find this repository useful for your research, please consider citing our paper using the following BibTeX entry.

```bibtex
@article{huang2026toward,
  title={Toward Extremely Efficient Infrared Small Target Detection Via Universal U-Net-based Knowledge Distillation},
  author={Huang, Ruimin and Huang, Jun and Ma, Yong and Fan, Fan and Zhu, Yiming},
  journal={xxxx},
  volume={xx},
  pages={x--xx},
  year={xxxx},
  publisher={xxxx}
}
```

## :star2: 9. Star History

If you find this repository useful for your research, please consider giving it a star.

[![Star History Chart](https://api.star-history.com/svg?repos=RuiminHuang/LW-DSTransNet&type=Date)](https://star-history.com/#RuiminHuang/LW-DSTransNet&Date)

## :email: 10. Contact

Please feel free to raise issues or email to [huang_ruimin@whu.edu.cn](huang_ruimin@whu.edu.cn) for any questions regarding our DSTransNet.
