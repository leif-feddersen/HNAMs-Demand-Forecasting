#!/bin/bash

# Place the original kaggle data from Favorita and Walmart (kaggle competitions) inside Datasets

# Preprocess the raw data
cd Preprocessing
uv run python pre_walmart.py
uv run python pre_favorita.py