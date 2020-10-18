# STGAT
STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction

## Correction
Our statement about Average Displacement Error (ADE) in the paper is wrong, and it should be RMSE or L2 distance (as in [SocialAttention](https://arxiv.org/pdf/1710.04689.pdf) and [SocialGan](https://arxiv.org/pdf/1803.10892.pdf)).

## Requirements
* Python 3
* PyTorch (1.2)
* Matplotlib

## Datasets
All the data comes from the [SGAN](https://github.com/agrimgupta92/sgan) model without any further processing.

## How to Run
* First `cd STGAT`
* To train the model run `python train.py` (see the code to understand all the arguments that can be given to the command)
* To evalutae the model run `python evaluate_model.py`
* Using the default parameters in the code, you can get most of the numerical results presented in the paper. But a reasonable attention visualization may require trained for a longer time and tuned some parameters. For example, for the zara1 dataset and `pred_len` is 8 time-steps,, you can set `num_epochs` to `600` (line 36 in `train.py`), and the `learning rate` in step3 to `1e-4` (line 180 in `train.py`).
* The attachment folder contains the code that produces the attention figures presented in the paper
* Check out the issue of this repo to find out how to get better results on the ETH dataset.

## Acknowledgments
All data and part of the code comes from the [SGAN](https://github.com/agrimgupta92/sgan) model. If you find this code useful in your research then please also cite their paper.

If you have any questions, please contact huangyingfan@outlook.com, and if you find this repository useful for your research, please cite the following paper:
```
@InProceedings{Huang_2019_ICCV,
author = {Huang, Yingfan and Bi, Huikun and Li, Zhaoxin and Mao, Tianlu and Wang, Zhaoqi},
title = {STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
