import argparse
import pandas as pd
import numpy as np
from prophet import Prophet
from joblib import Parallel, delayed
import multiprocessing
import os
from config import get_dataset_kwargs_nixtla, get_cutoffs, quantiles_to_conflevels

### MAJOR CONFIGURATION
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Favorita", help="Dataset name")
args = parser.parse_args()

DATASET = args.dataset
MODEL = 'Prophet'
LIMIT_TS = False
HORIZON = 14 # days not timesteps. two weeks in all datasets.

DS_KWARGS = get_dataset_kwargs_nixtla(DATASET)
COVS_REAL = [c for c in DS_KWARGS['time_varying_known_reals'] if c not in ['doy_sine','doy_cosine','time_idx','relative_time_idx']]
COVS_CAT =  [c for c in DS_KWARGS['time_varying_known_categoricals'] if c not in ['weekday']]
PRED_LENGTH = DS_KWARGS['max_prediction_length']
SEASON_LENGTH = DS_KWARGS['season_length']
QUANTILES = [0.5]

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
df = df.rename(columns={'date':'ds','sales':'y'})


dummies = pd.get_dummies(df[COVS_CAT],drop_first=True)
df = pd.concat([df,dummies],axis=1).drop(COVS_CAT,axis=1)

COVS_CAT = dummies.columns.to_list()
COVS = COVS_REAL + COVS_CAT

# valid as in /valid dates where there is an observation, not validation set
valid_dates = pd.to_datetime(pd.Series(df.ds.unique()).sort_values().reset_index(drop=True))
# date to time index
dtot = df[['ds','time_idx']].drop_duplicates().set_index('ds').sort_index()['time_idx'].to_dict()
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
    all_preds_ts = pd.DataFrame()
    df_ts = df.query('time_series == @time_series_i')
    to_clean = clean_cols(df_ts.query('ds < @test_dates[0]')[COVS])
    covs_i = [c for c in COVS if c not in to_clean]
    df_ts = df_ts.drop(to_clean,axis=1)

    for first_test,end_test in zip(test_dates,test_end_dates):

        past_data = df_ts.query('ds < @first_test').drop('time_series',axis=1)
        future_covs = df_ts.query('ds >= @first_test & ds <= @end_test')

        
        actuals = future_covs['y'].values[:PRED_LENGTH]
        future_covs = future_covs[covs_i+['ds']]
        max_pred = first_test + pd.Timedelta(days=HORIZON-1)


        model_add = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
        model_mul = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
        for cov in covs_i:
            model_add.add_regressor(cov)
            model_mul.add_regressor(cov)
        model_add.fit(past_data)
        model_mul.fit(past_data)
        pred_add = model_add.predict(past_data)
        pred_mul = model_mul.predict(past_data)

        # calculate rmse
        rmse_add = np.sqrt(((past_data['y'] - pred_add['yhat'])**2).mean())
        rmse_mul = np.sqrt(((past_data['y'] - pred_mul['yhat'])**2).mean())
        # select model
        if rmse_add.mean() < rmse_mul.mean():
            modeltype = 'add'
        else:
            modeltype = 'mul'
       
        while max_pred < end_test:

            if modeltype == 'add':
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')
            else:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')

            for cov in covs_i:
                model.add_regressor(cov)

            model.fit(past_data)
            forecast = model.make_future_dataframe(periods=30, include_history=False, freq='D')
            forecast = forecast[forecast.ds <= max_pred]
            forecast = forecast[forecast.ds.isin(valid_dates.values)]
            forecast = forecast.merge(future_covs, on='ds', how='left')
            forecast = model.predict(forecast)
            forecast = forecast[['ds', 'yhat']]
            forecast.columns = ['pred_date', 'pred']
            forecast['pred_idx'] = forecast['pred_date'].map(dtot)
            forecast['time_idx'] = dtot[first_test]
            forecast['h'] = forecast['pred_idx'] - forecast['time_idx'] + 1
            forecast['time_series'] = time_series_i
            forecast['q'] = 0.5 # could expand for probabilistic forecasts
            forecast['true'] = actuals
            all_preds_ts = pd.concat([all_preds_ts,forecast])

            #### PREPARE NEXT LOOP

            first_test = first_test + pd.Timedelta(days=1)
            while first_test not in valid_dates.values:
                first_test = first_test + pd.Timedelta(days=1)

            past_data = df.query('ds < @first_test & time_series == @time_series_i')
            future_covs = df.query('ds >= @first_test & ds <= @end_test & time_series == @time_series_i')
            actuals = future_covs['y'].values[:PRED_LENGTH]
            future_covs = future_covs[covs_i+['ds']]

            max_pred = first_test + pd.Timedelta(days=HORIZON-1)

    print(f'Finished {time_series_i}')
    return all_preds_ts

# Run the forecasts in parallel
num_cores = multiprocessing.cpu_count()
start_training = pd.Timestamp.now()
results = Parallel(n_jobs=num_cores)(delayed(forecast_time_series)(ts) for ts in all_time_series)
end_training = pd.Timestamp.now()
training_seconds = (end_training - start_training).seconds
print('Training Time:',training_seconds)

if not os.path.exists(f'models/{DATASET}_{MODEL}_results.csv'):
    with open(f'models/{DATASET}_{MODEL}_results.csv','w') as f:
        f.write('dataset,model,training_time\n')
        f.write(f'{DATASET},{MODEL},{training_seconds}\n')

# Concatenate all results
all_preds = pd.concat(results, axis=0)
all_preds['model'] = MODEL
all_preds = all_preds[['time_series','time_idx','pred_idx','h','true','pred','model']]

all_preds.to_pickle(f'../Evaluation/{DATASET}/{MODEL.lower()}.pkl')