#!/bin/bash

# Use pre-trained models to get predictions for complete datasets
cd Evaluation

uv run python predict_all.py --dataset "Walmart"
uv run python predict_all.py --dataset "WalmartR"
uv run python predict_all.py --dataset "Favorita"
