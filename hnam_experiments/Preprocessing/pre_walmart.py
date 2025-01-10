# %%
### IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import copy
import sys


current_dir = os.getcwd()
parallel_folder = os.path.abspath(os.path.join(current_dir, '../Fit_and_Predict'))
sys.path.append(parallel_folder)

from config import get_cutoffs, get_dataset_kwargs_nixtla

# %%
DATASET = 'Walmart'

DECODER_LENGTH = get_dataset_kwargs_nixtla(DATASET)['max_prediction_length']
VALIDATION_LENGTH = DECODER_LENGTH * 2 
MAX_ENCODER_LENGTH = DECODER_LENGTH * 12

_, test_ends = get_cutoffs(DATASET)
test_ends = pd.to_datetime(test_ends)

hp_test,_ = get_cutoffs(DATASET,hp_optim=True)

min_date = pd.to_datetime(hp_test[0]) - pd.Timedelta(days=MAX_ENCODER_LENGTH+DECODER_LENGTH+VALIDATION_LENGTH)
max_date = test_ends[-1] - pd.Timedelta(days=1)
eval_dates = pd.date_range(min_date,max_date,freq='D')

if DATASET == 'Retail':
    eval_dates = eval_dates[eval_dates.weekday != 6]

# %%
data = pd.read_csv('../Datasets/Walmart/sales_train_evaluation.csv')
prices = pd.read_csv('../Datasets/Walmart/sell_prices.csv')
cal = pd.read_csv('../Datasets/Walmart/calendar.csv')
# _______________________________________________

replacer = {'item_id':'art',
            'store_id':'pos',
            'dept_id':'dept',
            'cat_id':'cat',
            'state_id':'state',}

data = data.rename(columns=replacer)
data = data.drop('id', axis=1)

prices = prices.rename(columns=replacer)
# _______________________________________________

# %%
# data = data[data['cat'].str.contains('FOODS')]
art_info = data.iloc[:,:5].drop_duplicates()
data = data.iloc[:,5:].T
data.columns = art_info.set_index(['art','pos']).index

prices = prices.merge(cal[['wm_yr_wk','date']], on='wm_yr_wk', how='left').drop('wm_yr_wk', axis=1)
prices['date'] = pd.to_datetime(prices['date'])
priceswide = prices.pivot_table(index='date',columns=['art','pos'],values='sell_price').fillna(method='ffill').fillna(method='bfill')

cal['date'] = pd.to_datetime(cal['date'])
data.index = data.index.map(cal.set_index('d')['date'])
data = data.loc[data.index < test_ends[-1]]

# %%
### GET SALES PIVOTED
sales_pivoted = data.copy().asfreq('D').replace(0,np.nan)
sales_pivoted.index.name = 'date'

# %%
### make any observations prior to a zero sequence also nan
zs = DECODER_LENGTH
zero_seq_long_rev = sales_pivoted.iloc[::-1].isnull().rolling(zs).sum().loc[sales_pivoted.index]
zero_seq_long = sales_pivoted.isnull().rolling(zs).sum()
zero_mask = (zero_seq_long + zero_seq_long_rev).ge(zs+1)            # zero seqs are set to true
zero_mask = zero_mask.replace(False, np.nan).bfill().fillna(False)  # good values get false, replaced with np.nan and backfilled so that
                                                                    # all values prior to a zero seq are also set to nan later (currently set to False)

sales_pivoted[zero_mask] = np.nan

mask_nan = sales_pivoted.copy()
mask_nan[mask_nan.ffill().notnull()] = 0
mask_nan = mask_nan.fillna(1).astype(bool)
full_mask = mask_nan | zero_mask

sales_pivoted[full_mask] = np.nan

# %%
# Find a date at which a product must be launched so that it's available for hp optimization
sel = []
for c in sales_pivoted.columns:
    if len(sales_pivoted[c].dropna()) == 0:
        continue
    if sales_pivoted[c].first_valid_index() <= eval_dates[0]:
        sel.append(c)

print("Initial Products ", len(sales_pivoted.columns))
print("Remaining Products: ",len(sel))
sales_pivoted = sales_pivoted[sel].copy()

# %%
for sales_thresh in range(5,100):
    sel = sales_pivoted.loc[eval_dates].median() > sales_thresh
    sel = sel[sel].index
    samples = (~np.isnan(sales_pivoted[sel].values.flatten())).sum()
    products = sales_pivoted[sel].shape[1]
    if  samples < 1_500_000:
        print(f"Median Sales threshold: {sales_thresh}")
        print(f"Number of samples: {samples}")
        print(f"Number of products: {products}")
        break

sales_pivoted = sales_pivoted[sel]

# %%
### FIRST SEE THAT FOR THE RELEVANT PERIODS WE HAVE ENOUGH DATA
for THRESH_MISSINGS in np.arange(0.0,0.3,0.001):
    sel = (sales_pivoted.loc[eval_dates].isnull().mean()<THRESH_MISSINGS)
    sel = sel[sel].index
    samples = (~np.isnan(sales_pivoted[sel].values.flatten())).sum()
    products = sales_pivoted[sel].shape[1]
    print(f"With Threshold: {THRESH_MISSINGS:.4f}, Samples: {samples}")
    if samples > 500_000:
        break

print(f"Number of samples: {samples}")
print(f"Number of products: {products}")

sales_pivoted = sales_pivoted[sel].copy()

# %%
fill_mask = sales_pivoted.isnull()
plt.imshow(fill_mask)

# %%
ts_info = pd.DataFrame(index=sales_pivoted.columns,columns=['first_valid_index','pct_missing_or_zero','length_of_time_series'])
for c in sales_pivoted.columns:
    fvi = sales_pivoted[c].first_valid_index()
    ts_info.loc[c,'first_valid_index'] = fvi
    ts_info.loc[c,'pct_missing_or_zero'] = sales_pivoted.loc[fvi:,c].replace(0,np.nan).isnull().mean()
    ts_info.loc[c,'length_of_time_series'] = sales_pivoted.loc[fvi:,c].notnull().sum()
ts_info['pct_missing_or_zero'].sort_values()

# %%
fill_values = sales_pivoted.rolling(DECODER_LENGTH,min_periods=1).mean()[sales_pivoted.isnull()]
sales_pivoted[fill_mask] = fill_values

# %%
for c in sales_pivoted.columns:
    fvi = sales_pivoted[c].first_valid_index()
    ts_info.loc[c,'missing_after'] = sales_pivoted.loc[fvi:,c].replace(0,np.nan).isnull().mean()

# %%
plt.imshow(sales_pivoted.isnull())

# %%
data = sales_pivoted.melt(ignore_index=False, value_name='sales').dropna()
data = data.reset_index()

### TIME IDX
mapping = pd.Series(data['date'].unique())
mapping.index = mapping.values
mapping = mapping.rank().astype(int)
data['time_idx'] = data['date'].map(mapping)

# %%
### PROCESSING HOLIDAY

holiday = cal[['date','event_type_1','event_name_1','event_type_2','event_name_2']]

hdays = pd.DataFrame()
htypes = list(holiday[['event_type_1','event_type_2']].melt()['value'].dropna().unique())
for htype in htypes:
    h1 = holiday.query('event_type_1 == @htype')[['date','event_name_1']].dropna().rename(columns={'event_name_1':htype})
    h2 = holiday.query('event_type_2 == @htype')[['date','event_name_2']].dropna().rename(columns={'event_name_2':htype})
    h = pd.concat([h1,h2],axis=0).set_index('date')
    hdays = pd.concat([hdays,h],axis=1)

# %%
hdays = hdays.reset_index()
extra_lags = pd.DataFrame()
for lags in [1,2,3,4,5]:
    thdays = hdays.assign(date = hdays['date'] - pd.Timedelta(days=lags)).copy()
    thdays[['Sporting','Cultural','National','Religious']] += '-'+str(lags)
    extra_lags= pd.concat([extra_lags,thdays])
extra_lags = extra_lags.sort_values('date')
hdays = pd.concat([hdays,extra_lags],ignore_index=True)


# %%
hdays = hdays.groupby('date').first().reset_index()

# %%
### PROCESSING SNAP

data = data.merge(cal[['date','snap_CA','snap_TX','snap_WI']], on='date', how='left')
data[['snap_CA','snap_TX','snap_WI']] = data[['snap_CA','snap_TX','snap_WI']].apply(lambda x: x.map({0: '._.', 1: x.name.split('_')[1]}))
snap = ((data['pos'].apply(lambda x:x.split('_')[0]) == data['snap_CA']) | (data['pos'].apply(lambda x:x.split('_')[0]) == data['snap_TX']) | (data['pos'].apply(lambda x:x.split('_')[0]) == data['snap_WI']))
data['snap'] = snap.replace(False,np.nan)
data = data.drop(['snap_CA','snap_TX','snap_WI'],axis=1)

# %%
### MERGING AND DTYPES

prices = priceswide[sales_pivoted.columns].melt(ignore_index=False,value_name='price')
data = data.merge(fill_values.notnull().melt(ignore_index=False,value_name='interpolated'), on=['date','art','pos'], how='left')
data = data.merge(prices, on=['date','art','pos'], how='left')
data = data.merge(art_info,on=['art','pos'],how='left')
data = data.merge(hdays,on='date',how='left')

data['weekday'] = data['date'].dt.weekday.astype(str)


cats = ['art','pos','dept','cat','state','snap','weekday'] + htypes
for c in cats:
    data[c] = data[c].replace(True,'x').replace(False,'-')
    data[c] = data[c].replace(1,'x').replace(0,'-')
    data[c] = data[c].replace(np.nan,'-')
    data[c] = data[c].astype('category')

# convert all numerical columns to float 32
for c in data.select_dtypes('number').columns:
    data[c] = data[c].astype('float32')
    
### TIME FEATURES
data['time_idx'] = data['time_idx'].astype(int)
data['time_series'] = data['art'].astype(str)+'_' + data['pos'].astype(str)
data = data.sort_values(['time_series','time_idx'])
data['relprice'] = data.groupby('time_series')['price'].rolling(30).mean().bfill().values
data['relprice'] = data['price'] / data['relprice'] - 1
data['relprice'] = data['relprice'].fillna(0)

data['dayofyear'] = data['date'].dt.dayofyear.astype(str)
data['weekofyear'] = data['date'].dt.isocalendar().week.astype(str)


data['snap'] = data['snap'].astype(str)
sales_pivoted = data.pivot_table(index='date',columns=['art','pos'],values='sales')
for c in data.columns:
    data.rename(columns={c:c.replace(' ','').replace('-','').replace("'",'')},inplace=True)
    
data['storesales'] = data.groupby(['date','pos'])['sales'].transform('sum')
data = data[data.storesales > 0]

data['day_of_year'] = data['date'].dt.dayofyear
data['doy_sine'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
data['doy_cosine'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
data['year'] = data['date'].dt.year

# %%
len(data)/1000,data.time_series.nunique()

# %%
scaling = data.groupby(['time_series'])['sales'].agg(['mean','std'])
ttod = data[['time_idx','date']].drop_duplicates().sort_values('time_idx').set_index('time_idx')['date']

data.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_data.pkl')
ts_info.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_ts_info.pkl')
scaling.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_scaling.pkl')
ttod.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_ttod.pkl')
