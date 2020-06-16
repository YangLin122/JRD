# Joint Representation Diversification
PyTorch Implementation: "Distance Metric Learning with Joint Representation Diversification"

## Usage: Train on CUB-200-2011

1. Obtain dataset
The CUB dataset can be downloaded from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html .


2. Train the model
```
python train.py --gpu 0 --margin 0.1 --scale 20.0 --alpha 1.0 --embDim 512 --batchsize 100 --data_path [data folder]
```

## Requirements
* Python 3.6.9
* PyTorch 1.2.0
* numpy 1.16.2

## Note
The backbone architecture is modified from https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/bninception.py .
