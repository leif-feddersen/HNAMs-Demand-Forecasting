#!/bin/bash

# Place the original kaggle data from Favorita and Walmart (kaggle competitions) inside Datasets

# Preprocess the raw data
cd Preprocessing
python pre_walmart.py
python pre_favorita.py