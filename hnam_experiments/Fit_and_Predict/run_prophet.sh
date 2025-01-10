#!/bin/bash

# Define the list of datasets
datasets=("Retail" "Walmart" "Favorita")

# Loop through each dataset and execute the script
for D in "${datasets[@]}"
do
    echo "Running prophet_general.py with dataset: $D"
    python prophet_general.py --dataset "$D"
done