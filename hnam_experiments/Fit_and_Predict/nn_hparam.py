# %%
### IMPORTS

import argparse
import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from config import get_ptfc_model, get_dataset_kwargs, get_model_ds_kwargs, get_model_kwargs, get_cutoffs, get_hparams, count_trainable_parameters

from pytorch_forecasting import TimeSeriesDataSet, RMSE, NaNLabelEncoder, GroupNormalizer, EncoderNormalizer, TemporalFusionTransformer, HNAM
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

### SEEDING

np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    ACC = 'cuda'
    NW = int(os.environ.get('OMP_NUM_THREADS', 0))
    ND = int(os.environ.get('DEVICES', 1))
    torch.set_float32_matmul_precision('medium')
    if ND>1:
        STR = 'ddp' #'ddp_find_unused_parameters_true'
    else:
        STR = 'auto'
elif torch.backends.mps.is_available():
    ACC = 'mps'
    NW = 0  # MPS doesn't use CPU threads in the same way
    ND = 1
    STR = 'auto'
else:
    ACC = 'cpu'
    NW = 0  # number of threads (cores), mac requires 0, use cpu count with other machines
    ND = 1
    STR = 'auto'

print(f'Using accelerator {ACC}')
print(f'Using n devices ', ND)
print(f'Using {NW} threads')
print(f'Using strategy {STR}')

# %%
### MAJOR CONFIGURATION
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Favorita", help="Dataset name")
parser.add_argument("--model", type=str, default="HNAM", help="Model name")
args = parser.parse_args()

DATASET = args.dataset
MODEL = args.model

LIMIT_TS = False # limit number of time series for testing #R!
print('Optimizing HPARAMS for', DATASET, 'using', MODEL)
if LIMIT_TS:
    print('Limiting number of time series to', LIMIT_TS)

### VARIABLE HPARAMS
LR = None # set by hparams
BS = None # set by hparams
ENC_WEEKS = None # set by hparams
GRAD_CLIP = None # set by hparams

### FIXED HPARAMS
EPOCHS = 100 # maximum number of epochs #R!
FINE_TUNE = 30
PATIENCE = 20 # patience for early stopping
LOSS = RMSE(quantiles=[0.5])
RLRP = 10   # reduce on plateau patience
WD = 1e-2   # weight decay
OPTIMIZER='adamw'

print('Max Epochs:', EPOCHS)

PTFC_MODEL = get_ptfc_model(MODEL)
DS_KWARGS = get_dataset_kwargs(DATASET)
MODEL_DS_KWARGS = get_model_ds_kwargs(MODEL, DATASET)  
MODEL_KWARGS = get_model_kwargs(MODEL)
MODEL_KWARGS.update(MODEL_DS_KWARGS)

VAL_LENGTH = DS_KWARGS['max_prediction_length'] * 2

### HPARAMS
HPARAMS = get_hparams(MODEL, 30)


### CREATE LOGGING FOLDER
os.makedirs(f'hp_optim/models/{DATASET}/{MODEL}', exist_ok=True)
os.makedirs(f'../Evaluation/{DATASET}', exist_ok=True)

# %%
### LOAD DATA
df = pd.read_pickle(f'../Processed/{DATASET}/{DATASET.lower()}_data.pkl')

# %%
### GET CUTOFF DATES
test_dates, test_end_dates = get_cutoffs(DATASET,hp_optim=True)
train_ends = np.sort(df[df.date.isin(pd.to_datetime(test_dates))]['time_idx'].unique()-1)

### LIMIT NO OF TIME SERIES IF SUPPLIED
if LIMIT_TS:
    sel = list(df.groupby('time_series')['sales'].sum().sort_values(ascending=False).index)[:LIMIT_TS]
    df = df[df['time_series'].isin(sel)]

### FIT THE CATEGORICAL ENCODERS
for cat_col in DS_KWARGS['categorical_encoders']:
    DS_KWARGS['categorical_encoders'][cat_col] = DS_KWARGS['categorical_encoders'][cat_col].fit(df[cat_col])



# %%


for i,hparams in enumerate(HPARAMS):
    hparams['encoder_length'] = int(hparams['encoder_weeks'] * DS_KWARGS['max_prediction_length']/2)
    print(f'Fitting model {i} with hparams: {hparams}')
    print('\n'*3)
    LR = hparams['learning_rate']
    BS = hparams['batch_size']
    GRAD_CLIP = hparams['max_gradient_norm']
    ENC_LENGTH = hparams['encoder_length']
    del hparams['learning_rate'], hparams['batch_size'], hparams['max_gradient_norm'], hparams['encoder_weeks'], hparams['encoder_length']
    MODEL_KWARGS.update(hparams)


    first_run = True
    for train_end,test_end_date in zip(train_ends,test_end_dates): 
        np.random.seed(0)
        torch.manual_seed(0)


        df_train = df[df['time_idx'] <= train_end - VAL_LENGTH]
        df_val = df[(df['time_idx'] > train_end - VAL_LENGTH - ENC_LENGTH) & (df['time_idx'] <= train_end)]
        df_test = df[(df['time_idx'] > train_end - ENC_LENGTH) & (df['date'] < test_end_date)]  
        ds_train = TimeSeriesDataSet(df_train, 
                                target='sales',
                                group_ids=['time_series'],
                                time_idx='time_idx',
                                min_encoder_length=ENC_LENGTH,
                                max_encoder_length=ENC_LENGTH,
                                add_relative_time_idx=True,
                                allow_missing_timesteps=False,
                                scale_target=True,
                                **DS_KWARGS
                                )

        ds_val = TimeSeriesDataSet.from_dataset(ds_train, df_val, stop_randomization=True,scale_target=True)
        ds_test = TimeSeriesDataSet.from_dataset(ds_train, df_test, stop_randomization=True,scale_target=True)
        dl_train = ds_train.to_dataloader(train=True,num_workers = NW, batch_size=BS)
        dl_val = ds_val.to_dataloader(train=False,num_workers = NW, batch_size=BS)
        dl_test = ds_test.to_dataloader(train=False,num_workers = NW, batch_size=BS)

        if first_run:
            model = PTFC_MODEL.from_dataset(ds_train,
                                            loss=LOSS,
                                            learning_rate=LR,
                                            weight_decay=WD,
                                            reduce_on_plateau_patience=RLRP,
                                            optimizer=OPTIMIZER,
                                            **MODEL_KWARGS)
            n_params = count_trainable_parameters(model)
        else:
            model = PTFC_MODEL.load_from_checkpoint(best_model_path)
        

        model.rescale_off() # we scale the target thus do not rescale the model output

        checkpoint_callback = ModelCheckpoint(  # used for early stopping
            monitor='val_loss',
            filename='model-{epoch:02d}-{val_loss:.2f}',  # Save the model with the epoch number and the validation metric
            save_top_k=1,  # Save only the best model
            mode='min',  # The best model is the one with the lowest validation loss
            auto_insert_metric_name=False,  # Prevent prepending monitored metric name to the filename
        )
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=PATIENCE,
            verbose=True,
            mode="min", 
        )
        trainer = pl.Trainer(
            max_epochs= EPOCHS if first_run else FINE_TUNE, 
            accelerator=ACC,
            devices = ND,
            gradient_clip_val=GRAD_CLIP,
            strategy = STR,
            callbacks=[early_stop_callback, checkpoint_callback],  # Add the checkpoint_callback to the list of callbacks
            default_root_dir=f'hp_optim/models/{DATASET}/{MODEL}/',  # The path where the runs and models will be saved
        )


        if not first_run:
            # Validate the model on the validation data loader
            validation_results = trainer.validate(model, dataloaders=dl_val)
            initial_val_loss = validation_results[0]['val_loss']
            print(f'Initial Validation Loss: {initial_val_loss}')
            
            # Update the checkpoint callback's best score with the initial validation loss
            checkpoint_callback.best_model_score = torch.tensor(initial_val_loss)
            checkpoint_callback.best_model_path = os.path.join(
                checkpoint_callback.dirpath, 
                f"model-initial-val_loss={initial_val_loss:.2f}.ckpt"
            )
            
            trainer.save_checkpoint(checkpoint_callback.best_model_path)



        start_training = pd.Timestamp.now()
        trainer.fit(model, dl_train,val_dataloaders=dl_val)
        end_training = pd.Timestamp.now()
        seconds_training = (end_training - start_training).total_seconds()

        val_loss = trainer.checkpoint_callback.best_model_score.item()

        best_model_path = trainer.checkpoint_callback.best_model_path
        model = PTFC_MODEL.load_from_checkpoint(best_model_path)
        validation_results = trainer.validate(model, dataloaders=dl_test)
        test_loss = validation_results[0]['val_loss']

        first_run = False
        # LOG HPARAMS, SCORE, PATH
        # first create a csv file that indicates dataset and model, only if the file is not already there, one column per hparam
        if not os.path.exists(f'hp_optim/{DATASET}_{MODEL}_results.csv'):
            pd.DataFrame(columns=list(HPARAMS[0].keys())+ ['lr','bs','grad_clip','enc_length'] + ['n_params','train_end','val_loss','test_loss','seconds_training','path']).to_csv(f'hp_optim/{DATASET}_{MODEL}_results.csv', index=False)
        # append values to the file
        pd.DataFrame({**hparams,'lr':LR,'bs':BS,'grad_clip':GRAD_CLIP,'enc_length':ENC_LENGTH,'n_params':n_params,'train_end':train_end, 'val_loss': val_loss,'test_loss': test_loss,'seconds_training':seconds_training, 'path': best_model_path}, index=[0]).to_csv(f'hp_optim/{DATASET}_{MODEL}_results.csv', mode='a', header=False, index=False)


