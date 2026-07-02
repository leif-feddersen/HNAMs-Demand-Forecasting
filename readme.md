# HNAM Experiments
This repository contains the experimental code, data, and pre-trained model checkpoints used in the paper:
> **Hierarchical Neural Additive Models for Interpretable Demand Forecasts**  
> *Published in the [International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207025000317), 42(1), 216–234*
>
> Assembled on Feb 5th 2025 by Leif Feddersen at Kiel University 
> 

---

## Overview

1. **Purpose**  
   - Demonstrates the use of **Hierarchical Neural Additive Models (HNAMs)** for interpretable demand forecasting on two real-world datasets, obtainable from Kaggle with a Kaggle account:
     - [Favorita Grocery Sales Forecasting (Kaggle)](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/)
     - [Walmart M5 Forecasting (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/)
   - Includes scripts for data preprocessing, evaluation notebooks (accuracy/interpretability/runtime), and pre-trained HNAM checkpoints.

2. **Repository Structure**  
   - **`Datasets/`**
     - Place original CSVs from the Kaggle competitions inside `Datasets/Favorita` and `Datasets/Walmart` if you wish to reproduce preprocessing.
     - Processed files are stored in `Processed/` (provided in this repo).
   - **`Preprocessing/`**  
     - Python scripts for data cleaning, transformation, and serialization.  
   - **`Processed/`**
     - Intermediary `.pkl` files from the preprocessing stage.   
   - **`Fit_and_Predict/`**  
     - Scripts to train various models (HNAMs, TFT, ARIMA, Prophet, etc.) and generate test set predictions.
     - Script for hyperparamter tuning of HNAMs and TFT.
     - Model checkpoints and hyperparameter optimization results are stored here.
   - **`Evaluation/`**  
     - Jupyter notebooks (`evaluation-*.ipynb`) for analyzing accuracy, interpretability, runtime, and reversed hierarchy.
     - `predict_all.py` for generating forecasts (using pre-trained models) and storing predictions in `.pkl` files.
     - Figures, logs, and saved predictions also reside here.  
   - **`preprocess.sh`**  
     - A single script to run the entire data preprocessing pipeline.
   - **`predict.sh`**  
     - A single script to generate neural network forecasts for the complete datasets using pre-trained models.

---

## Getting Started

### 1. Clone This Repository

```bash
git clone https://github.com/leif-feddersen/HNAMs-Demand-Forecasting.git
cd HNAMs-Demand-Forecasting
```

### 2. Install the Environment with uv

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`).

Then, from the repository root:

```bash
uv sync
```

This installs Python 3.10 and all pinned dependencies (see `pyproject.toml` / `uv.lock`) into a local `.venv`, including [this custom PyTorch Forecasting fork](https://github.com/leif-feddersen/pytorch-forecasting/tree/hnam-mods) which adds HNAMs.

> **Where is the HNAM model code?** It is *not* in this repository and *not* on the fork's default (`master`) branch. It lives on the [`hnam-mods` branch](https://github.com/leif-feddersen/pytorch-forecasting/tree/hnam-mods) of the fork, mainly in [`pytorch_forecasting/models/hnam/__init__.py`](https://github.com/leif-feddersen/pytorch-forecasting/blob/hnam-mods/pytorch_forecasting/models/hnam/__init__.py). Note that GitHub's code search does not index non-default branches, so you must switch branches to see it.

Run any script with `uv run python <script.py>`, or activate the environment with `source .venv/bin/activate`.
To use the environment in the Jupyter notebooks, select the `.venv` interpreter as kernel (ipykernel is installed).


---

**If you wish to re-run the entire pipeline (preprocessing, fitting models locally, and generating in-sample predictions for full interpretability), continue with steps 3, 4, and 5.**  
**Otherwise, you can skip them and still run the evaluation notebooks with the provided preprocessed data and model checkpoints.**

---

### 3. Place Raw Data and Preprocess (Optional)

If you want to replicate preprocessing locally (rather than using the pre-processed pickles under `Processed/`):

1. Obtain the original CSVs from Kaggle for both **Favorita** and **Walmart**.
2. Place them in the following structure:

```
hnam_experiments/
└── Datasets/
    ├── Favorita/
    │   ├── holidays_events.csv
    │   ├── items.csv
    │   ├── oil.csv
    │   ├── sample_submission.csv
    │   ├── stores.csv
    │   ├── test.csv
    │   ├── train.csv
    │   └── transactions.csv
    └── Walmart/
        ├── calendar.csv
        ├── sales_train_evaluation.csv
        ├── sales_train_validation.csv
        ├── sample_submission.csv
        └── sell_prices.csv
```

Then execute:

```bash
bash preprocess.sh
```

This runs `python pre_walmart.py` and `python pre_favorita.py`, creating `.pkl` files under `Processed/`.
Note that preprocessing is memory intense and takes roughly 30GB for Favorita.

### 4. Run Hyperparameter Tuning (Optional)

To find optimal hyperparameters for HNAMs and TFT, run `uv run python nn_hparam.py --dataset DATASET --model MODEL`. Results from hyperparameter tuning are found within `hp_optim` as CSV files.

### 5. Fit Models and Generate Predictions (Optional)

To train and predict with various models, run the corresponding scripts in `Fit_and_Predict`. For example:

- `uv run python nn_general.py --dataset Walmart --model HNAM`
- `uv run python prophet_general.py --dataset Favorita`
- etc.

**Approximate runtimes**:  
- HNAM: up to ~5 hours on an Rocky Linux 8.6 system with H100 GPU and Intel Xeon Gold CPU.  
- TFT: up to ~30 hours.  
- Statistical models train much faster (see Table 2 in the paper).

### 6. Local Inference on Complete Datasets (Optional)

To generate in-sample predictions for the full interpretability evaluation, run:

```bash
bash predict.sh
```

This script calls `predict_all.py` on each dataset (`Walmart`, `WalmartR`, and `Favorita`) using validated model checkpoints. 
(**WalmartR** uses a reversed hierarchy for comparison.)

---

## Results and Evaluation

1. **Evaluation Notebooks**  
   - In `Evaluation/`, the notebooks (`evaluation-*.ipynb`) analyze:
     - **`evaluation-1_accuracy.ipynb`**: Accuracy results (Tables 3 & 4 in the paper).  
     - **`evaluation-2_interpretability.ipynb`**: Interpretability results (Figures 2–8).  
     - **`evaluation-3_runtime.ipynb`**: Runtime comparison (Table 2).  
     - **`evaluation-4_reversed.ipynb`**: Reversed hierarchy comparison (Table 6).

2. **Figures and Logs**  
   - Plots appear in `Evaluation/figures/`.
   - PyTorch Lightning logs are under `lightning_logs/`.
   - The figures in the Paper are generated by running `evaluation-2-interpretability.ipynb` with `DATASET` arguments `Favorita` and `Walmart`.
     - Figure 2: `walmart_global_dataset.tiff`
     - Figure 3: `favorita_global_dataset.tiff`
     - Figure 4: `walmart_global_time_series.tiff`
     - Figure 5: `favorita_local_Many_Effects.tiff`
     - Figure 6: `walmart_local_Best_vs_TFT.tiff`
     - Figure 7: `walmart_local_Worst_vs_TFT.tiff`
     - Figure 8a: `walmart_robustness_global.tiff`
     - Figure 8b: `favorita_robustness_global.tiff`

---

## Notes

The Retail dataset in the paper is a proprietary dataset which can not be shared.

## Citing this Work

Feddersen, L., & Cleophas, C. (2026). Hierarchical neural additive models for interpretable demand forecasts. *International Journal of Forecasting*, 42(1), 216–234. [https://doi.org/10.1016/j.ijforecast.2025.03.003](https://doi.org/10.1016/j.ijforecast.2025.03.003)

```bibtex
@article{feddersen2026hierarchical,
  title   = {Hierarchical neural additive models for interpretable demand forecasts},
  author  = {Feddersen, Leif and Cleophas, Catherine},
  journal = {International Journal of Forecasting},
  volume  = {42},
  number  = {1},
  pages   = {216--234},
  year    = {2026},
  doi     = {10.1016/j.ijforecast.2025.03.003}
}
```

---

## Contact & Support

Leif Feddersen feddersen@bwl.uni-kiel.de

---

Happy forecasting!
