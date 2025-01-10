from itertools import product
import numpy as np
import pandas as pd
from pytorch_forecasting import NaNLabelEncoder, GroupNormalizer, EncoderNormalizer, HNAM, TemporalFusionTransformer

def quantiles_to_conflevels(quantiles):
    conflevels = [int(abs(100 - (100-sl*100)*2)) for sl in quantiles]
    clevel = [str(int(abs(100 - (100-sl*100)*2))) for sl in quantiles]
    clevel_r = []
    for q,s in zip(quantiles,clevel):
        prefix = 'lo-' if q < 0.5 else 'hi-'
        clevel_r.append(prefix+s)
    replacer = dict(zip(clevel_r,quantiles))
    conflevels = list(set(conflevels))
    return conflevels,replacer


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_ptfc_model(model_name):
    models = {
        "tft": TemporalFusionTransformer,
        "hnam": HNAM
    }

    if model_name.lower() in models:
        return models[model_name.lower()]
    else:
        raise ValueError("Unknown model name")
    
def get_dataset_kwargs_nixtla(dataset_name):
    configs = {
        "favorita": {
            'static_categoricals': ['art', 'pos', 'cluster'],  #R leave out cluster
            'time_varying_unknown_reals': ['sales', 'dcoilwtico'],
            'time_varying_known_reals': ['time_idx', 'doy_sine', 'doy_cosine'],
            'time_varying_known_categoricals': ['weekday', 'onpromotion', 'national_holiday', 'regional_holiday', 'local_holiday'],
            'min_prediction_length': 14,
            'max_prediction_length': 14,
            'season_length':7
        },
        "walmart": {
            'static_categoricals': ['art', 'pos'],
            'time_varying_unknown_reals': ['sales'],
            'time_varying_known_reals': ['time_idx', 'relprice', 'doy_sine', 'doy_cosine'],
            'time_varying_known_categoricals': ['weekday', 'snap', 'Sporting', 'Cultural', 'National'],
            'min_prediction_length': 14,
            'max_prediction_length': 14,
            'season_length':7
        }
    }

    if dataset_name.lower() in configs:
        return configs[dataset_name.lower()]
    else:
        raise ValueError("Unknown dataset name")


def get_dataset_kwargs(dataset_name):
    configs = {
        "favorita": {
            'static_categoricals': ['art', 'pos', 'cluster'],  #R leave out cluster
            'time_varying_unknown_reals': ['sales', 'dcoilwtico'],
            'time_varying_known_reals': ['time_idx', 'doy_sine', 'doy_cosine'],
            'time_varying_known_categoricals': ['weekday', 'onpromotion', 'national_holiday', 'regional_holiday', 'local_holiday'],
            'target_normalizer': GroupNormalizer(groups=['time_series']),
            'categorical_encoders': {
                'national_holiday': NaNLabelEncoder(add_nan=False),
                'regional_holiday': NaNLabelEncoder(add_nan=False),
                'local_holiday': NaNLabelEncoder(add_nan=False)
            },
            'min_prediction_length': 14,
            'max_prediction_length': 14
        },
        "walmart": {
            'static_categoricals': ['art', 'pos'],
            'time_varying_unknown_reals': ['sales'],
            'time_varying_known_reals': ['time_idx', 'relprice', 'doy_sine', 'doy_cosine'],
            'time_varying_known_categoricals': ['weekday', 'snap', 'Sporting', 'Cultural', 'National'],
            'target_normalizer': GroupNormalizer(groups=['time_series']),
            'categorical_encoders': {
                'Sporting': NaNLabelEncoder(add_nan=False),
                'Cultural': NaNLabelEncoder(add_nan=False),
                'National': NaNLabelEncoder(add_nan=False)
            },
            'scalers': {
                'price': GroupNormalizer(groups=['time_series']),
                'relprice': EncoderNormalizer('identity')
            },
            'min_prediction_length': 14,
            'max_prediction_length': 14
        }
    }

    if dataset_name.lower() in configs:
        return configs[dataset_name.lower()]
    else:
        raise ValueError("Unknown dataset name")
    
    
def get_model_ds_kwargs(model_name, dataset_name):
    configs = {"TFT":{  "Favorita":{}, "Walmart":{}},
               
              "HNAM":{  "Favorita": dict(base=        ['time_idx','art','pos','doy_sine','doy_cosine'],
                                         causal=      ['weekday','onpromotion','national_holiday','regional_holiday','local_holiday'],
                                         one_hot_not_minus_one = ['weekday']),
                        "Walmart":  dict(base=           ['time_idx','art','pos','doy_sine','doy_cosine'],
                                         causal=         ['weekday','snap','relprice','National','Cultural','Sporting'],
                                         one_hot_not_minus_one = ['weekday']),
              }}
    
    if model_name in configs:
        if dataset_name in configs[model_name]:
            return configs[model_name][dataset_name]
        else:
            raise ValueError("Unknown dataset name")
    else:
        raise ValueError("Unknown model name")


def get_model_kwargs(model_name):
    configs = {"TFT":   dict(hidden_size=64),
               "HNAM":  dict(cov_emb=32,factor=4)}
    
    if model_name in configs:
        return configs[model_name]
    else:
        raise ValueError("Unknown model name")


def get_cutoffs(dataset_name, hp_optim=False):

    configs = {"Favorita": {'test_dates': ['2016-12-01', '2017-01-02', '2017-02-01', '2017-03-01', '2017-04-01'],
                            'test_end_dates': ['2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01']},
               "Walmart":  {'test_dates': ['2015-12-01', '2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01'],
                            'test_end_dates': ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01']}
    }

    configs_hp = {"Favorita": {'test_dates': ['2016-10-01', '2016-11-01'],
                            'test_end_dates': ['2016-11-01', '2016-12-01']},
               "Walmart":  {'test_dates': ['2015-10-01', '2015-11-01'],
                            'test_end_dates': ['2015-11-01','2015-12-01']}
    }


    dataset_name = dataset_name.capitalize()
    if dataset_name in configs:
        if not hp_optim:
            return ( configs[dataset_name]["test_dates"],
                    configs[dataset_name]["test_end_dates"])
        elif hp_optim:
            return (configs_hp[dataset_name]["test_dates"],
                    configs_hp[dataset_name]["test_end_dates"])
    else:
        raise ValueError("Unknown dataset name")
    

def get_hparams(model_name, n_trials):

    hparams_tft = dict(
        learning_rate = [1e-2,1e-3],
        batch_size = [128,256],
        max_gradient_norm = [0.1,1,100],
        encoder_weeks = [4,12],
        dropout = [0.1,0.3,0.5],
        attention_head_size = [1,4],
        hidden_size = [32,128,240],
    )

    hparams_hnam = dict(
        learning_rate = [1e-2,1e-3],
        batch_size = [128,256],
        max_gradient_norm = [0.1,1,100],
        encoder_weeks = [4,12],
        dropout = [0.1,0.3,0.5],
        cov_heads = [1,4],
        cov_emb = [8,32,64],
        factor = [1,2,4],
    )

    optimal_per_tft_paper = dict(
        learning_rate = 1e-3,
        batch_size = 128,
        max_gradient_norm = 100,
        encoder_weeks = 12,
        dropout = 0.1,
        attention_head_size = 4,
        hidden_size = 240,
    )

    assert model_name.lower() in ['tft', 'hnam'], 'Invalid model name'
    hparams = hparams_tft if model_name.lower() == 'tft' else hparams_hnam

    hparam_names = list(hparams.keys())
    hparam_values = list(hparams.values())

    all_hparams = list(product(*hparam_values))
    np.random.seed(0)
    np.random.shuffle(all_hparams)
    model_list = [dict(zip(hparam_names, hparam_values)) for hparam_values in all_hparams[:n_trials]]

    if model_name.lower() == 'tft':
        model_list = [optimal_per_tft_paper] + model_list

    return model_list

def get_best_hparams(DATASET,MODEL):
    hp_data = pd.read_csv(f'hp_optim/{DATASET.capitalize()}_{MODEL.upper()}_results.csv')

    model_kwargs = {'HNAM':['dropout','cov_heads','cov_emb','factor'],
                    'TFT':['dropout','attention_head_size','hidden_size']}[MODEL]

    ds_optim_kwargs = ['lr','bs','grad_clip','enc_length']

    best_model = hp_data.groupby(model_kwargs+ds_optim_kwargs)['test_loss'].mean().sort_values().index
    best_model = dict(zip(best_model.names, best_model.values[0]))

    best_model_kwargs = {k:best_model[k] for k in model_kwargs}
    best_ds_optim_kwargs = {k:best_model[k] for k in ds_optim_kwargs}

    return best_model_kwargs, best_ds_optim_kwargs

