# %%
# %%
import argparse
import pandas as pd
from statsforecast.models import AutoARIMA
from joblib import Parallel, delayed
import multiprocessing
import os
from config import get_dataset_kwargs_nixtla, get_cutoffs, quantiles_to_conflevels
import traceback
import numpy as np
# import standard scaler
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

### MAJOR CONFIGURATION
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Favorita", help="Dataset name")
args = parser.parse_args()

DATASET = args.dataset
MODEL = 'Lasso'
LIMIT_TS = False


DS_KWARGS = get_dataset_kwargs_nixtla(DATASET)
PRED_LENGTH = DS_KWARGS['max_prediction_length']
SEASON_LENGTH = DS_KWARGS['season_length']
LASSO_REGS = ['lag_1','lag_3',f'lag_{SEASON_LENGTH}',f'lag_{SEASON_LENGTH*2}'
              ,f'rolling_{SEASON_LENGTH}',f'rolling_{SEASON_LENGTH*2}']
COVS_REAL = [c for c in DS_KWARGS['time_varying_known_reals'] if c not in ['doy_sine','doy_cosine','time_idx','relative_time_idx']] 
COVS_CAT =  [c for c in DS_KWARGS['time_varying_known_categoricals'] if c not in ['weekday']]

QUANTILES = [0.5]
HORIZON = 14

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

for lags in [1,3,SEASON_LENGTH,SEASON_LENGTH*2]:
    df[f'lag_{lags}'] = df.groupby('time_series')['sales'].shift(lags)
for rolling_mean in [SEASON_LENGTH,SEASON_LENGTH*2]:
    df[f'rolling_{rolling_mean}'] = df.groupby('time_series')['sales'].transform(lambda x: x.rolling(rolling_mean).mean())
df = df[['time_series','date','time_idx','sales']+COVS_REAL+COVS_CAT+LASSO_REGS]
dummies = pd.get_dummies(df[COVS_CAT],drop_first=True)
df = pd.concat([df,dummies],axis=1).drop(COVS_CAT,axis=1).dropna()
COVS_CAT = dummies.columns.to_list()
COVS = COVS_REAL + COVS_CAT
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
print(f'Using Covariates {COVS+LASSO_REGS}')

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
        covs_i = [c for c in COVS+LASSO_REGS if c not in to_clean]
        df_ts = df_ts.drop(to_clean,axis=1)

        X = df_ts.query('date < @test_dates[0]').drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)[covs_i]
        # fit standard scaler to X
        scaler = StandardScaler()
        # scaler.fit(X)

        for first_test,end_test in zip(test_dates,test_end_dates):

            past_data = df_ts.query('date < @first_test').drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)
            future_covs = df_ts.query('date >= @first_test & date <= @end_test')
            time_idx_i  = future_covs['time_idx'].min().astype(int)
            future_covs = future_covs.drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)
            max_pred = first_test + pd.Timedelta(days=HORIZON-1)


            scaler.fit(past_data[covs_i])
            features_scaled = scaler.transform(past_data[covs_i])
            target = past_data['sales'].values

            model = LassoCV(cv=TimeSeriesSplit(n_splits=5), n_jobs=1)  # Use single core per model
            model.fit(features_scaled, target)

            while max_pred < end_test:

                sales_history = past_data['sales'].values
                for h in range(PRED_LENGTH):
                    features = future_covs.iloc[[h]][[c for c in covs_i if c not in LASSO_REGS]]
                    features['lag_1'] = sales_history[-1]
                    features['lag_3'] = sales_history[-3]
                    features[f'lag_{SEASON_LENGTH}'] = sales_history[-SEASON_LENGTH]
                    features[f'lag_{SEASON_LENGTH*2}']= sales_history[-SEASON_LENGTH*2]
                    features[f'rolling_{SEASON_LENGTH}'] = sales_history[-SEASON_LENGTH:].mean()
                    features[f'rolling_{SEASON_LENGTH*2}'] = sales_history[-SEASON_LENGTH*2:].mean()

                    features_scaled = scaler.transform(features[covs_i]).reshape(1,-1)
                    pred = model.predict(features_scaled)

                    # add pred to sales_history at last position
                    sales_history = np.append(sales_history,pred)

                    pred_df = pd.DataFrame({'time_idx':time_idx_i,'time_series':time_series_i,
                                            'h':h+1,'pred_idx':time_idx_i+h,'pred':pred,'true':future_covs['sales'].values[h]})
                    
                    all_preds_ts = pd.concat([all_preds_ts,pred_df])

                    
                first_test = first_test + pd.Timedelta(days=1)
                while first_test not in valid_dates.values:  
                    first_test = first_test + pd.Timedelta(days=1)
                
                max_pred = first_test + pd.Timedelta(days=HORIZON-1)

                past_data = df_ts.query('date < @first_test').drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)
                future_covs = df_ts.query('date >= @first_test & date <= @end_test')
                time_idx_i  = future_covs['time_idx'].min().astype(int)
                future_covs = future_covs.drop('time_series',axis=1).set_index('date').asfreq('D').fillna(0).astype(float)

        print(f'Finished {time_series_i}')
        return all_preds_ts
    except Exception as e:
        print(f'Error with {time_series_i}')
        traceback.print_exc()
        return None

num_cores = multiprocessing.cpu_count()
start_training = pd.Timestamp.now()
all_preds = Parallel(n_jobs=num_cores)(delayed(forecast_time_series)(ts) for ts in all_time_series)
end_training = pd.Timestamp.now()
training_seconds = (end_training - start_training).seconds
print('Training Time:', training_seconds)

# Log total training time
if not os.path.exists(f'models/{DATASET}_{MODEL}_results.csv'):
    with open(f'models/{DATASET}_{MODEL}_results.csv', 'w') as f:
        f.write('dataset,model,training_time\n')
        f.write(f'{DATASET},{MODEL},{training_seconds}\n')

all_preds = pd.concat(all_preds, axis=0)
all_preds = all_preds[['time_series', 'time_idx', 'pred_idx', 'h', 'true', 'pred']]
all_preds['model'] = MODEL

all_preds = all_preds[['time_series', 'time_idx', 'pred_idx', 'h', 'true', 'pred', 'model']]
all_preds.to_pickle(f'../Evaluation/{DATASET}/{MODEL.lower()}.pkl')


