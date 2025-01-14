#!/bin/bash

# Use pre-trained models to get predictions for complete datasets
cd Evaluation

python predict_all.py --dataset "Walmart"
python predict_all.py --dataset "WalmartR"
python predict_all.py --dataset "Favorita"
