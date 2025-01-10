#!/bin/bash

# Place the original kaggle data from Favorita and Walmart (kaggle competitions) inside Datasets

# Preprocess the raw data
cd Preprocessing
python pre_walmart.py
python pre_favorita.py

# Use pre-trained models to get predictions for complete datasets
cd ../Evaluation
datasets=("Walmart" "WalmartR" "Favorita")

for D in "${datasets[@]}"
do
    echo "Running prediction with dataset: $D"
    python predict_all.py --dataset "$D"
done
