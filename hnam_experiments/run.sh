#!/bin/bash

# Place the original kaggle data from Favorita and Walmart (kaggle competitions) inside Datasets

# Preprocess the raw data
cd Preprocessing
python pre_walmart.py
python pre_favorita.py

# Use pre-trained models to get predictions for complete datasets
cd ../Evaluation

python predict_all.py --dataset "Walmart"
python predict_all.py --dataset "WalmartR"
python predict_all.py --dataset "Favorita"
