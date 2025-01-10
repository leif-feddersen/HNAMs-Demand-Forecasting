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
DATASET = 'Favorita'

DECODER_LENGTH = get_dataset_kwargs_nixtla(DATASET)['max_prediction_length']
VALIDATION_LENGTH = DECODER_LENGTH * 2 
MAX_ENCODER_LENGTH = DECODER_LENGTH * 12

_, test_ends = get_cutoffs(DATASET)
test_ends = pd.to_datetime(test_ends)

hp_test,_ = get_cutoffs(DATASET,hp_optim=True)

min_date = pd.to_datetime(hp_test[0]) - pd.Timedelta(days=MAX_ENCODER_LENGTH+DECODER_LENGTH+VALIDATION_LENGTH)
max_date = test_ends[-1] - pd.Timedelta(days=1)
eval_dates = pd.date_range(min_date,max_date,freq='D')

# %%
# _______________________________________________

data = pd.read_csv('../Datasets/Favorita/train.csv')
items = pd.read_csv('../Datasets/Favorita/items.csv')
holiday = pd.read_csv('../Datasets/Favorita/holidays_events.csv')
stores = pd.read_csv('../Datasets/Favorita/stores.csv')
oil = pd.read_csv('../Datasets/Favorita/oil.csv')
# _______________________________________________

replacer = {'item_nbr':'art',
            'store_nbr':'pos',
            'unit_sales':'sales',}

# %%
holiday = holiday[~holiday.transferred]
holiday['state'] = holiday[holiday['locale'] == 'Regional']['locale_name']
holiday['city'] = holiday[holiday['locale'] == 'Local']['locale_name']

# %%
### GETTING SALES PIVOTED
data = data.rename(columns=replacer)
items = items.rename(columns=replacer)
data = data.drop('id',axis=1)
data['onpromotion'] = data['onpromotion'].replace(False,np.nan)
data['date'] = pd.to_datetime(data['date'])
data = data[data['sales'] >= 0]
data = data[data.date < test_ends[-1]]
sales_pivoted = data.pivot_table(index='date', columns=['art','pos'], values='sales', aggfunc='sum').asfreq('D').replace(0,np.nan)
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

sales_pivoted = sales_pivoted[sel].copy()

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
melted = sales_pivoted.melt(ignore_index=False,value_name='sales').dropna()
data = melted.merge(data[['art','pos','date','onpromotion']],how='left',on=['art','pos','date'])
data.dtypes

# %%
# interpolated values
data = data.merge(fill_values.notnull().melt(ignore_index=False,value_name='interpolated'), on=['date','art','pos'], how='left')
### HOLIDAY AND STORES
holiday['date'] = pd.to_datetime(holiday['date'])
# holiday.rename(columns={'locale_name':'state'},inplace=True)

holiday = holiday.groupby('date').first().reset_index()
stores = stores.rename(columns=replacer)
data = data.merge(stores,how='left',on='pos')

# %%
# add pre holiday
holiday = holiday[~holiday['description'].str.contains(r'-\d+', regex=True)]
extra_lags = pd.DataFrame()
for lags in [1,2,3,4,5]:
    extra_lags= pd.concat([extra_lags,holiday[(holiday.description != "Navidad+1") & (~holiday.description.str.contains("Terremoto"))].assign(description = holiday['description'] + f'-{lags}').assign(date = holiday['date'] - pd.Timedelta(days=lags))])
holiday = pd.concat([holiday,extra_lags],ignore_index=True)

# %%
# add national holiday
data = data.merge(holiday.loc[holiday['locale'] == 'National',['date','description']].groupby('date')['description'].first().reset_index(),how='left',on=['date'])
data = data.rename(columns={'description':'national_holiday'})
# merge regional holiday based on date and state
data = data.merge(holiday.loc[holiday['locale'] == 'Regional',['date','state','description']].groupby(['date','state'])['description'].first().reset_index(),how='left',on=['date','state'])
data = data.rename(columns={'description':'regional_holiday'})
# merge local holiday based on date and city
data = data.merge(holiday.loc[holiday['locale'] == 'Local',['date','city','description']].groupby(['date','city'])['description'].first().reset_index(),how='left',on=['date','city'])
data = data.rename(columns={'description':'local_holiday'})

# %%

### OIL
oil['date'] = pd.to_datetime(oil['date'])
oil = oil.set_index('date').asfreq('D')
oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()
oil = oil.bfill().ffill()
data = data.merge(oil,how='left',on=['date'])


data['weekday'] = data['date'].dt.weekday.astype(str)
cats = ['art','pos','onpromotion','city','state','type','cluster','national_holiday','regional_holiday','local_holiday','weekday']
for c in cats:
    data[c] = data[c].replace(np.nan,'-').astype(str).astype('category')

# convert all numerical columns to float 32
for c in data.select_dtypes('number').columns:
    data[c] = data[c].astype('float32')

columns = data.select_dtypes('datetime').columns.append(
    data.select_dtypes('number').columns).append(
        data.select_dtypes('category').columns)


data = data[columns.to_list()+['interpolated']]
data['dayofyear'] = data['date'].dt.dayofyear.astype(str)
data['weekofyear'] = data['date'].dt.isocalendar().week.astype(str)
data['time_series'] = data['art'].astype(str)+'_' + data['pos'].astype(str)
data['storesales'] = data.groupby(['date','pos'])['sales'].transform('sum')

# %%
### TIME INDEX MAPPING
mapping = pd.Series(data['date'].unique())
mapping.index = mapping.values
mapping = mapping.rank().astype(int)

data['time_idx'] = data['date'].map(mapping).astype(int)
data['time_idx'] = data['time_idx'].astype(int)

# %%
### DATE FEATURES
data['day_of_year'] = data['date'].dt.dayofyear
data['doy_sine'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
data['doy_cosine'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
data['year'] = data['date'].dt.year

# %%
#(712.007, 665)
len(data)/1000,data.time_series.nunique()


# %%
scaling = data.groupby(['time_series'])['sales'].agg(['mean','std'])
ttod = data[['time_idx','date']].drop_duplicates().sort_values('time_idx').set_index('time_idx')['date']

data.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_data.pkl')
ts_info.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_ts_info.pkl')
scaling.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_scaling.pkl')
ttod.to_pickle(f'../Processed/{DATASET.capitalize()}/{DATASET.lower()}_ttod.pkl')


