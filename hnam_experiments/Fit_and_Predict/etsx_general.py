# %%
import argparse
import pandas as pd
from statsforecast.models import AutoETS
from joblib import Parallel, delayed
import multiprocessing
import os
from config import get_dataset_kwargs_nixtla, get_cutoffs, quantiles_to_conflevels
from sklearn.linear_model import LinearRegression
import traceback
import numpy as np

### MAJOR CONFIGURATION
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Favorita", help="Dataset name")
args = parser.parse_args()

DATASET = args.dataset
MODEL = 'ETS'
LIMIT_TS = False
HORIZON = 14 # days not timesteps. two weeks in all datasets.

DS_KWARGS = get_dataset_kwargs_nixtla(DATASET)
COVS_REAL = [c for c in DS_KWARGS['time_varying_known_reals'] if c not in ['doy_sine','doy_cosine','time_idx','relative_time_idx']]
COVS_CAT =  [c for c in DS_KWARGS['time_varying_known_categoricals'] if c not in ['weekday']]
PRED_LENGTH = DS_KWARGS['max_prediction_length']
SEASON_LENGTH = DS_KWARGS['season_length']
QUANTILES = [0.5]
HORIZON = 14 # days not timesteps. two weeks in all datasets.

### GET CUTOFF DATES
conflevels,replacer = quantiles_to_conflevels(QUANTILES)
test_dates, test_end_dates = get_cutoffs(DATASET)
test_dates = pd.to_datetime(test_dates)
test_end_dates = pd.to_datetime(test_end_dates)
### CREATE LOGGING FOLDER
os.makedirs(f'../Models/{DATASET}/{MODEL}', exist_ok=True)
os.makedirs(f'../Evaluation/{DATASET}', exist_ok=True)
df = pd.read_pickle(f'../Processed/{DATASET}/{DATASET.lower()}_data.pkl')
if LIMIT_TS:
    sel = list(df.groupby('time_series')['sales'].sum().sort_values(ascending=False).index)[:LIMIT_TS]
    df = df[df['time_series'].isin(sel)]
df = df[['time_series','date','time_idx','sales']+COVS_REAL+COVS_CAT]


dummies = pd.get_dummies(df[COVS_CAT],drop_first=True)
df = pd.concat([df,dummies],axis=1).drop(COVS_CAT,axis=1)

COVS_CAT = dummies.columns.to_list()
COVS = COVS_REAL + COVS_CAT

if MODEL == 'ETS':
    COVS = [c.replace(' ','_') for c in COVS]
    COVS = [c.replace("'",'') for c in COVS]
    df.columns = [c.replace(' ','_') for c in df.columns]
    df.columns = [c.replace("'",'') for c in df.columns]
# valid as in /valid dates where there is an observation, not validation set
valid_dates = pd.to_datetime(pd.Series(df.date.unique()).sort_values().reset_index(drop=True))
# for looping
all_time_series = df['time_series'].unique()

print(f'Forecasting {DATASET} with {MODEL} model')
print(f'Forecasting {len(all_time_series)} time series')
print(f'Using Covariates {COVS}')

def clean_cols(covs):
    columns_to_drop = []
    duplicates = covs.T.duplicated()
    duplicate_columns = covs.columns[duplicates]
    print("Duplicate columns:")
    print(duplicate_columns.tolist())
    columns_to_drop.extend(duplicate_columns.tolist())
    constant_columns = [col for col in covs.columns if covs[col].nunique() <= 1]
    print("Constant columns:")
    print(constant_columns)
    columns_to_drop.extend(constant_columns)
    covs_reduced = covs.drop(columns=columns_to_drop)
    corr_matrix = covs_reduced.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    perfect_corr_columns = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] >= 1.0)
    ]
    print("Columns to drop due to perfect correlation:")
    print(perfect_corr_columns)
    columns_to_drop.extend(perfect_corr_columns)

    return columns_to_drop

def forecast_time_series(time_series_i):
    try:
        all_preds_ts = pd.DataFrame()
        df_ts = df.query('time_series == @time_series_i')
        to_clean = clean_cols(df_ts.query('date < @test_dates[0]')[COVS])
        covs_i = [c for c in COVS if c not in to_clean]
        df_ts = df_ts.drop(to_clean,axis=1)

        for first_test,end_test in zip(test_dates,test_end_dates):

            past_data = df_ts.query('date < @first_test').drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)
            future_covs = df_ts.query('date >= @first_test & date <= @end_test')
            time_idx_i  = future_covs['time_idx'].min().astype(int)
            future_covs = future_covs.drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)
            max_pred = first_test + pd.Timedelta(days=HORIZON-1)

            model = AutoETS(season_length = SEASON_LENGTH)
            model.fit(past_data['sales'].values)  # fitted ets model

            first_run = True
            while max_pred < end_test:
                preds = model.forward(y = past_data['sales'].values,
                                X = None,
                                X_future = None,
                                h = PRED_LENGTH,
                                level = conflevels,
                                fitted = True if first_run else False)
                
                if first_run:
                    insample_df = pd.DataFrame()
                    insample_df['insample_ETS'] = preds['fitted']
                    insample_df['true'] = past_data['sales'].values
                    insample_df['insample_residual'] = insample_df['true'] - insample_df['insample_ETS']
                    

                    # sklearn ols model
                    reg = LinearRegression(fit_intercept=False)
                    reg.fit(past_data[covs_i],insample_df['insample_residual'])

                    insample_df['predicted_residual'] = reg.predict(past_data[covs_i])
                    insample_df['insample_ETSx'] = insample_df['insample_ETS'] + insample_df['predicted_residual']
                    insample_df['time_idx'] = time_idx_i
                    insample_df['time_series'] = time_series_i
                first_run = False

                preds = preds = {k:v for k,v in preds.items() if 'fitted' not in k}
                preds = pd.DataFrame(preds).rename(columns=replacer)[QUANTILES].melt(value_name='pred_ETS',var_name='q')
                preds['time_idx'] = time_idx_i
                preds['time_series'] = time_series_i
                preds['h'] = preds.index + 1
                preds['pred_idx'] = preds['time_idx'] + preds['h']
                preds['true'] = future_covs['sales'].values[:PRED_LENGTH]
                preds['predicted_residual'] = reg.predict(future_covs[covs_i][:PRED_LENGTH])
                preds['pred_ETSX'] = preds['pred_ETS'] + preds['predicted_residual']


                all_preds_ts = pd.concat([all_preds_ts,preds])

                first_test = first_test + pd.Timedelta(days=1)
                while first_test not in valid_dates.values:  
                    first_test = first_test + pd.Timedelta(days=1)
                
                max_pred = first_test + pd.Timedelta(days=HORIZON-1)
                
                past_data = df.query('date < @first_test & time_series == @time_series_i').drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)
                future_covs = df.query('date >= @first_test & date <= @end_test & time_series == @time_series_i')
                time_idx_i  = future_covs['time_idx'].min().astype(int)
                future_covs = future_covs.drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)

        print(f'Finished {time_series_i}')
        return all_preds_ts, insample_df
                    
    except Exception as e:
        print(f'Error with {time_series_i}')
        traceback.print_exc()
        return None

num_cores = multiprocessing.cpu_count()

start_training = pd.Timestamp.now()
result = Parallel(n_jobs=num_cores)(delayed(forecast_time_series)(ts) for ts in all_time_series)
end_training = pd.Timestamp.now()
training_seconds = (end_training - start_training).seconds
print('Training Time:',training_seconds)
# log total training time
if not os.path.exists(f'models/{DATASET}_{MODEL}_results.csv'):
    with open(f'models/{DATASET}_{MODEL}_results.csv','w') as f:
        f.write('dataset,model,training_time\n')
        f.write(f'{DATASET},{MODEL},{training_seconds}\n')


all_preds, all_insample = zip(*result)

all_preds = pd.concat(all_preds, axis=0)
all_preds = all_preds[['time_series','time_idx','pred_idx','h','true','pred_ETS','pred_ETSX']]
    # Assuming your original DataFrame is named 'df'
all_preds = pd.melt(
    all_preds,
    id_vars=['time_series', 'time_idx', 'pred_idx', 'h', 'true'],
    value_vars=['pred_ETS', 'pred_ETSX'],
    var_name='model',
    value_name='pred'
)

all_preds['model'] = all_preds['model'].str.replace('pred_','')
all_preds = all_preds[['time_series', 'time_idx', 'pred_idx', 'h', 'true', 'pred', 'model']]

all_preds.query('model=="ETS"').to_pickle(f'../Evaluation/{DATASET}/ets.pkl')
all_preds.query('model=="ETSX"').to_pickle(f'../Evaluation/{DATASET}/etsx.pkl')


