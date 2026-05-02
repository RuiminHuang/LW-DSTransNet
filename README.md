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

| Method                                         | Params ↓ | FLOPs ↓ | SIRST mIoU ↑ | SIRST nIoU ↑ | SIRST ${P}_{d}$ ↑ | SIRST ${F}_{a}$ ↓ | NUDT-SIRST mIoU ↑ | NUDT-SIRST nIoU ↑ | NUDT-SIRST ${P}_{d}$ ↑ | NUDT-SIRST ${F}_{a}$ ↓ | IRSTD-1k mIoU ↑ | IRSTD-1k nIoU ↑ | IRSTD-1k ${P}_{d}$ ↑ | IRSTD-1k ${F}_{a}$ ↓ |
| ---------------------------------------------- | -------: | ------: | -----------: | -----------: | ----------------: | ----------------: | ----------------: | ----------------: | ---------------------: | ---------------------: | --------------: | --------------: | -------------------: | -------------------: |
| Filter-based Methods                           |          |         |              |              |                   |                   |                   |                   |                        |                        |                 |                 |                      |                      |
| Max-Median \cite{deshpande1999max}             |        - |       - |        4.172 |        12.31 |             69.20 |             55.33 |             4.197 |             3.674 |                  58.41 |                  36.89 |           6.998 |           3.051 |                65.21 |                59.73 |
| Top-Hat \cite{bai2010analysis}                 |        - |       - |        7.143 |        18.27 |             79.84 |              1012 |             20.72 |             28.98 |                  78.41 |                  166.7 |           10.06 |           7.438 |                75.11 |                 1432 |
| FKRW \cite{qin2019infrared}                    |        - |       - |        21.25 |        27.69 |             78.21 |             16.63 |              9.67 |             16.12 |                  69.47 |                  66.85 |            9.75 |           16.19 |                67.14 |                24.15 |
| HVS-based Methods                              |          |         |              |              |                   |                   |                   |                   |                        |                        |                 |                 |                      |                      |
| MPCM \cite{wei2016multiscale}                  |        - |       - |        24.65 |        26.56 |             65.35 |             45.02 |             26.34 |             37.44 |                  53.37 |                  11.79 |           19.93 |           22.36 |                56.48 |                31.06 |
| RLCM \cite{han2018infrared}                    |        - |       - |        25.45 |        27.78 |             86.68 |             62.31 |             18.37 |             18.11 |                  79.57 |                  68.96 |           16.92 |           21.03 |                70.36 |                67.60 |
| TLLCM \cite{han2019local}                      |        - |       - |        18.66 |        27.71 |             82.24 |             16.95 |             11.08 |             23.82 |                  77.97 |                  67.26 |           10.26 |           18.24 |                68.55 |                24.48 |
| ILCM \cite{han2020infrared}                    |        - |       - |        18.27 |        19.70 |             38.58 |             40.26 |             12.84 |             13.69 |                  72.62 |                  43.21 |           13.05 |           11.46 |                43.55 |                47.32 |
| GSWLCM \cite{qiu2022global}                    |        - |       - |        15.42 |        13.24 |             72.68 |             21.73 |             10.82 |             18.71 |                  67.32 |                  53.31 |           12.82 |           13.72 |                70.24 |                13.92 |
| LRM-based Methods                              |          |         |              |              |                   |                   |                   |                   |                        |                        |                 |                 |                      |                      |
| IPI \cite{gao2013infrared}                     |        - |       - |        25.67 |        33.57 |             85.55 |             11.47 |             28.63 |             38.18 |                  74.49 |                  41.23 |           27.92 |           30.12 |                81.37 |                16.18 |
| RIPT \cite{dai2017reweighted}                  |        - |       - |        11.05 |        19.91 |             79.08 |             22.61 |             29.17 |             36.12 |                  91.85 |                 344.30 |           14.11 |           17.43 |                77.55 |                28.31 |
| PSTNN \cite{zhang2019infrared_PSTNN}           |        - |       - |        22.40 |        29.59 |             77.95 |             29.11 |             27.72 |             39.80 |                  66.13 |                  44.17 |           24.57 |           28.71 |                71.99 |                35.26 |
| Deep Learning Methods (Lightweight)            |          |         |              |              |                   |                   |                   |                   |                        |                        |                 |                 |                      |                      |
| LW-IRSTNet \cite{kou2023lw}                    |    0.163 |   0.215 |        70.16 |        71.77 |      <u>95.41</u> |      <u>13.75</u> |             73.99 |             77.41 |                  96.03 |                  19.38 |    <u>66.44</u> |       **66.83** |            **93.27** |             **8.88** |
| IRPruneDeXt \cite{zhang2025irprunedext} $^{*}$ |    0.172 |   0.927 |    **76.17** |    **75.08** |         **99.08** |          **1.29** |      <u>86.30</u> |      <u>85.10</u> |           <u>97.80</u> |            <u>8.50</u> |           65.69 |           63.68 |         <u>93.17</u> |         <u>13.53</u> |
| LW-DSTransNet-Nano                             |    0.107 |   0.155 | <u>74.25</u> | <u>74.45</u> |         **99.08** |             36.46 |         **86.77** |         **87.32** |              **98.83** |               **3.20** |       **67.81** |    <u>65.76</u> |                91.92 |                18.83 |
| Deep Learning Methods (Standard)               |          |         |              |              |                   |                   |                   |                   |                        |                        |                 |                 |                      |                      |
| ACM \cite{dai2021asymmetric}                   |    0.398 |   0.352 |        68.93 |        69.18 |             91.63 |             15.23 |             61.12 |             64.40 |                  93.12 |                  55.22 |           59.23 |           57.03 |                93.27 |                65.28 |
| ALCNet \cite{dai2021attentional}               |    0.427 |   0.302 |        70.83 |        71.05 |             94.30 |             36.15 |             64.74 |             67.20 |                  94.18 |                  34.61 |           60.60 |           57.14 |                92.98 |                58.80 |
| DNANet \cite{li2022dense}                      |    4.697 |  14.261 |        52.29 |        64.50 |             84.40 |             11.36 |             85.85 |             86.42 |                  99.07 |                   5.49 |           68.87 |    <u>67.53</u> |            **94.95** |         <u>13.38</u> |
| ISTDU-Net \cite{Hou2022ISTDUNetIS}             |    2.752 |   7.944 |        55.00 |        62.00 |             91.74 |            112.80 |             89.55 |             90.48 |                  97.67 |                  13.44 |           66.36 |           63.86 |                93.60 |                53.10 |
| RDIAN \cite{sun2023receptive}                  |    0.217 |   3.718 |        68.72 |        75.39 |             93.54 |             43.29 |             76.28 |             79.14 |                  95.77 |                  34.56 |           56.45 |           59.72 |                88.55 |                26.63 |
| UIUNet \cite{wu2022uiu}                        |   50.540 |  54.426 |        62.35 |        69.45 |             91.74 |             23.55 |      <u>93.48</u> |      <u>93.89</u> |                  98.31 |                   7.79 |           66.15 |           66.66 |         <u>93.98</u> |                22.07 |
| SCTransNet \cite{yuan2024sctransnet}           |   11.191 |  10.119 |        60.74 |        67.11 |             90.83 |             27.55 |         **94.63** |         **94.55** |           <u>99.30</u> |                   2.29 |           67.43 |       **68.05** |                91.58 |             **8.46** |
| MLP-Net \cite{wang2024mlp}                     |    5.007 |   4.326 |        72.33 |        74.85 |         **97.25** |             25.69 |             91.43 |             91.56 |              **99.53** |            <u>1.93</u> |           64.99 |           65.72 |                91.92 |                22.53 |
| LW-DSTransNet-Tiny                             |    0.409 |   0.581 | <u>74.73</u> | <u>75.50</u> |      <u>96.33</u> |          **9.94** |             90.97 |             91.33 |           <u>99.30</u> |                   2.34 |    <u>69.17</u> |           66.73 |                90.57 |                27.31 |
| LW-DSTransNet-Small                            |    1.601 |   2.253 |    **75.47** |    **76.40** |      <u>96.33</u> |      <u>10.11</u> |             93.08 |             93.41 |              **99.53** |               **0.15** |       **70.69** |           66.85 |                92.59 |                21.22 |



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
