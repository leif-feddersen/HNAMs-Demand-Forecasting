# %%
### IMPORTS
import argparse
import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from config import get_ptfc_model, get_dataset_kwargs, get_model_ds_kwargs, get_model_kwargs, get_cutoffs, count_trainable_parameters, get_best_hparams

from pytorch_forecasting import TimeSeriesDataSet, RMSE
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

### SEEDING

np.random.seed(0)
torch.manual_seed(0)

### ACCELERATOR AND WORKERS CONFIGURATION

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
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, default="Favorita", help="Dataset name")
# parser.add_argument("--model", type=str, default="HNAM", help="Model name")
# args = parser.parse_args()

DATASET = 'Walmart'
MODEL = 'HNAM'
LIMIT_TS = False # limit number of time series for testing

MODEL_KWARGS, best_ds_optim_kwargs = get_best_hparams(DATASET, MODEL)

### VARIABLE HPARAMS
WD = 1e-2   # weight decay
LR = best_ds_optim_kwargs['lr']
BS = best_ds_optim_kwargs['bs']
ENC_LENGTH = best_ds_optim_kwargs['enc_length']
GRAD_CLIP = best_ds_optim_kwargs['grad_clip']

### FIXED HPARAMS
EPOCHS = 300 # maximum number of epochs 
FINE_TUNE = 100 # fine tune epochs
PATIENCE = 30 # patience for early stopping
LOSS = RMSE(quantiles=[0.5])
RLRP = 10   # reduce on plateau patience
OPTIMIZER='adamw'

PTFC_MODEL = get_ptfc_model(MODEL)
DS_KWARGS = get_dataset_kwargs(DATASET)
MODEL_DS_KWARGS = get_model_ds_kwargs(MODEL, DATASET)  
MODEL_KWARGS.update(MODEL_DS_KWARGS)
MODEL_KWARGS['causal'] = ['weekday','snap','relprice','National','Cultural','Sporting'][::-1]

VAL_LENGTH = DS_KWARGS['max_prediction_length'] * 2

### GET CUTOFF DATES
test_dates, test_end_dates = get_cutoffs(DATASET)
### CREATE LOGGING FOLDER
os.makedirs(f'models/{DATASET}/{MODEL}', exist_ok=True)
os.makedirs(f'../Evaluation/{DATASET}', exist_ok=True)

# %%
### LOAD DATA
df = pd.read_pickle(f'../Processed/{DATASET}/{DATASET.lower()}_data.pkl')

# %%
### GET END OF TRAINING TIME INDICES FOR VALIDATION SET CREATION
train_ends = np.sort(df[df.date.isin(pd.to_datetime(test_dates))]['time_idx'].unique()-1)

### LIMIT NO OF TIME SERIES IF SUPPLIED
if LIMIT_TS:
    sel = list(df.groupby('time_series')['sales'].sum().sort_values(ascending=False).index)[:LIMIT_TS]
    df = df[df['time_series'].isin(sel)]

### FIT THE CATEGORICAL ENCODERS
for cat_col in DS_KWARGS['categorical_encoders']:
    DS_KWARGS['categorical_encoders'][cat_col] = DS_KWARGS['categorical_encoders'][cat_col].fit(df[cat_col])

# %%
all_preds = pd.DataFrame()
all_insample = pd.DataFrame() if MODEL == "HNAM" else None
best_model_path = None

first_run = True
for train_end,test_end_date in zip(train_ends,test_end_dates): 

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
                            allow_missing_timesteps=True,
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

    model.rescale_off()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='model-{epoch:02d}-{val_loss:.3f}',  # Save the model with the epoch number and the validation metric
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
        default_root_dir=f'models/{DATASET}/{MODEL}/',  # The path where the runs and models will be saved
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

    best_model_path = trainer.checkpoint_callback.best_model_path


    model = PTFC_MODEL.load_from_checkpoint(best_model_path)
    val_loss = trainer.checkpoint_callback.best_model_score.item()
    validation_results = trainer.validate(model, dataloaders=dl_test)
    test_loss = validation_results[0]['val_loss']

    model.rescale_on()
    ds_test.scale_target = False
    dl_test = ds_test.to_dataloader(train=False,num_workers = NW, batch_size=BS)

    start_inference = pd.Timestamp.now()
    raw = model.predict(dl_test, mode="raw", return_x=True, return_index=True,trainer_kwargs=dict(accelerator="cpu"))
    end_inference = pd.Timestamp.now()
    seconds_inference = (end_inference - start_inference).total_seconds()

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

    all_preds = pd.concat([all_preds,preds_df])

    if MODEL == "HNAM":
        insample_df = pd.DataFrame(index.values.repeat(ENC_LENGTH * n_quantiles, axis=0),columns=index.columns)
        insample_df = insample_df.assign(h=np.tile(np.repeat(np.arange(-ENC_LENGTH,0),n_quantiles),len(insample_df)//(ENC_LENGTH*n_quantiles)))
        insample_df = insample_df.assign(q=np.tile(np.arange(n_quantiles),len(insample_df)//n_quantiles))
        insample_df['pred_idx'] = insample_df['time_idx'] + insample_df['h']
        if quantiles is not None:
            insample_df['q'] = insample_df['q'].map({i:q for i,q in enumerate(quantiles)})
        elif insample_df.q.nunique() == 1:
            insample_df = insample_df.drop(columns=['q'])

        insample_df['true'] = torch.repeat_interleave(raw.x['encoder_target'].flatten().cpu(),n_quantiles)

        for k in covs:
            if k in ['level_pred']:
                continue
            elif k in ['alpha','beta','phi']:
                insample_df[k] = raw.output[k][:,:ENC_LENGTH].flatten().repeat_interleave(ENC_LENGTH).cpu()
            else:
                insample_df['effect_'+k] = raw.output[k][:,:ENC_LENGTH].flatten().cpu()

        all_insample = pd.concat([all_insample,insample_df])


    first_run = False


    if not os.path.exists(f'models/{DATASET}_{MODEL}_results.csv'):
        pd.DataFrame(columns=['n_params','train_end','val_loss','test_loss','seconds_training','seconds_inference','path']).to_csv(f'models/{DATASET}_{MODEL}_results.csv', index=False)
    # append values to the file
    pd.DataFrame({'n_params':n_params,'train_end':train_end, 'val_loss': val_loss,'test_loss': test_loss,'seconds_training':seconds_training,'seconds_inference':seconds_inference, 'path': best_model_path}, index=[0]).to_csv(f'models/{DATASET}_{MODEL}_results.csv', mode='a', header=False, index=False)

MODEL = MODEL+'_R'
all_preds['model'] = MODEL
all_preds.to_pickle(f'../Evaluation/{DATASET}/{MODEL.lower()}.pkl')
if MODEL == "HNAM":
    all_insample.to_pickle(f'../Evaluation/{DATASET}/{MODEL.lower()}_insample.pkl')

