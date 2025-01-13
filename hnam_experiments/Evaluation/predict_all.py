# %%
import os
import io
import numpy as np
import pandas as pd
import torch
import sys
from dataclasses import dataclass
import argparse
current_dir = os.getcwd()
parallel_folder = os.path.abspath(os.path.join(current_dir, '../Fit_and_Predict'))
sys.path.append(parallel_folder)

from config import get_model_ds_kwargs,get_dataset_kwargs
from config import get_ptfc_model, get_dataset_kwargs, get_model_ds_kwargs, get_model_kwargs, get_cutoffs, count_trainable_parameters, get_best_hparams

from pytorch_forecasting import TimeSeriesDataSet, RMSE, SMAPE, MAE, MAPE,HNAM
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
ACC = 'cpu'
NW = 0  # number of threads (cores), mac requires 0, use cpu count with other machines
ND = 1
STR = 'auto'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Favorita", help="Dataset name")
args = parser.parse_args()

DATASET = args.dataset
if DATASET.endswith('R'):
    DATASET = DATASET[:-1]
    reverse_hierarchy = True
else:
    reverse_hierarchy = False
MODEL = 'HNAM'
BS = 256


DS_KWARGS = get_dataset_kwargs(DATASET)
MODEL_DS_KWARGS = get_model_ds_kwargs(MODEL, DATASET)  


path = f'../Fit_and_Predict/models/{DATASET}/HNAM/lightning_logs/version_4/checkpoints/'
if reverse_hierarchy:
    path =  f'../Fit_and_Predict/models/{DATASET}R/HNAM/lightning_logs/version_4/checkpoints/'
path = path + os.listdir(path)[0]

# path stores the checkpoint for the currently best model
checkpoint = torch.load(path, map_location=torch.device('cpu'))
checkpoint['__special_save__']['loss'] = checkpoint['hyper_parameters']['loss'] = RMSE()
checkpoint['__special_save__']['logging_metrics'] =checkpoint['hyper_parameters']['logging_metrics'] = torch.nn.ModuleList([SMAPE(),RMSE(),MAE(),MAPE()])
buffer = io.BytesIO()
torch.save(checkpoint, buffer)
buffer.seek(0)
# this works previously, model was trained on gpu, now we load it on cpu
model = PTFC_MODEL = HNAM.load_from_checkpoint(buffer)
ENC_LENGTH = model.hparams['max_encoder_length']
model.rescale_on()
df = pd.read_pickle(f'../Processed/{DATASET}/{DATASET.lower()}_data.pkl')
ds = TimeSeriesDataSet(df,
                        target='sales',
                        group_ids=['time_series'],
                        time_idx='time_idx',
                        min_encoder_length=ENC_LENGTH,
                        max_encoder_length=ENC_LENGTH,
                        add_relative_time_idx=True,
                        allow_missing_timesteps=True,
                        scale_target=True,
                        **DS_KWARGS
                        )

dl = ds.to_dataloader(train=False,num_workers=0,batch_size=BS)
raw = model.predict(dl, mode="raw", return_x=True, return_index=True,trainer_kwargs=dict(accelerator="cpu"))

index = raw.index
preds = raw.output.prediction
dec_len = preds.shape[1]
n_quantiles = preds.shape[-1]
covs = list(raw.output.keys()[1:])
quantiles = None

preds_df = pd.DataFrame(index.values.repeat(dec_len * n_quantiles, axis=0),columns=index.columns)
preds_df = preds_df.assign(h=np.tile(np.repeat(np.arange(1,1+dec_len),n_quantiles),len(preds_df)//(dec_len*n_quantiles)))
preds_df = preds_df.assign(q=np.tile(np.arange(n_quantiles),len(preds_df)//n_quantiles))
preds_df['pred_idx'] = preds_df['time_idx'] + preds_df['h'] - 1
if quantiles is not None:
    preds_df['q'] = preds_df['q'].map({i:q for i,q in enumerate(quantiles)})
elif preds_df.q.nunique() == 1:
    preds_df = preds_df.drop(columns=['q'])
preds_df['pred'] = preds.flatten().cpu()
preds_df['true'] = torch.repeat_interleave(raw.x['decoder_target'].flatten().cpu(),n_quantiles)

if MODEL == "HNAM":
    for k in covs:
        if k in ['level_past']:
            continue
        if k in ['alpha','beta','phi']:
            preds_df[k] = raw.output[k][:,-DS_KWARGS['max_prediction_length']:].flatten().repeat_interleave(DS_KWARGS['max_prediction_length']).cpu()
        else:
            preds_df['effect_'+k] = raw.output[k][:,-DS_KWARGS['max_prediction_length']:].flatten().cpu()


if reverse_hierarchy:
    preds_df.to_pickle(f'{DATASET}/{MODEL.lower()}R_all.pkl')
else:
    preds_df.to_pickle(f'{DATASET}/{MODEL.lower()}_all.pkl')



