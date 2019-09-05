# STGAT
STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction

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
* The attachment folder contains the code that produces the attention figures presented in the paper

## Acknowledgments
All data and part of the code comes from the [SGAN](https://github.com/agrimgupta92/sgan) model. If you find this code useful in your research then please also cite their paper.