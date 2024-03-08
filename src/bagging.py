import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train import (sparse_features, dense_features, varlen_features)
from preprocess import preprocess, generate_datasets


def get_prediction(model_ls, model_input):

    pred_ls = []
    for model in model_ls:
        pred_val = model.predict(model_input, batch_size=8192).reshape(-1)
        pred_ls.append(pred_val)
        torch.cuda.empty_cache()

    return pred_ls


def get_auc(pred_ls, target):

    auc_ls = []
    for pred in pred_ls:
        auc_ls.append(roc_auc_score(target, pred))
    # auc_dict['bagging'] = roc_auc_score(
    #     target, np.mean(list(pred_dict.values()), axis=0))

    return auc_ls


def get_preds_auc(models_dict, model_input, target):

    preds_dict = {}
    auc_dict = {}
    for k, v in models_dict.items():
        preds_dict[k] = get_prediction(v, model_input)
        auc_dict[k] = get_auc(preds_dict[k], target)

    return preds_dict, auc_dict


def get_top_models(models_dict, preds_dict, auc_dict, n_best: int = 2):

    top_models = {}
    for model_name, model_ls in models_dict.items():
        best_model_idx = np.argsort(auc_dict[model_name])[::-1][:n_best]
        for idx in best_model_idx:
            top_models[f'{model_name}_{idx}'] = (model_ls[idx],
                                                 preds_dict[model_name][idx],
                                                 auc_dict[model_name][idx])
    return top_models


if __name__ == "__main__":
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    songs_df = pd.read_csv('../data/songs.csv')
    members_df = pd.read_csv('../data/members.csv')

    tr_song_msno_df, val_song_msno_df, ts_song_msno_df, item2idx = preprocess(
        train_df, test_df, songs_df, members_df)

    (tr_model_input, val_model_input, ts_model_input, linear_feature_columns,
     dnn_feature_columns) = generate_datasets(tr_song_msno_df,
                                              val_song_msno_df,
                                              ts_song_msno_df, sparse_features,
                                              dense_features, varlen_features,
                                              item2idx)

    models_dict = defaultdict(list)
    checkpoint_fd = '../checkpoints/'
    for model in os.listdir(checkpoint_fd):
        for i, ckpts in enumerate(os.listdir(f"{checkpoint_fd}/{model}")):
            if not ckpts.endswith('ckpt'):
                continue
            models_dict[model].append(
                torch.load(f"{checkpoint_fd}/{model}/{ckpts}"))

    val_preds_dict, val_auc_dict = get_preds_auc(models_dict, val_model_input,
                                                 val_song_msno_df['target'])
    top_models = get_top_models(models_dict,
                                val_preds_dict,
                                val_auc_dict,
                                n_best=None)

    print("ROC-AUC")
    print(pd.Series({k: v[2] for k, v in top_models.items()}))
    print()
    print("Correlation")
    print(pd.DataFrame({k: v[1] for k, v in top_models.items()}).corr())

    ts_pred_ls = get_prediction([v[0] for v in top_models.values()],
                                ts_model_input)
    ts_song_msno_df['target'] = np.mean(ts_pred_ls, axis=0)
    output_df = ts_song_msno_df[['id', 'target']].drop_duplicates('id')
    output_df.to_csv('../data/output.csv', index=False, encoding='utf-8')
