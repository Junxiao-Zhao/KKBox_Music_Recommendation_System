import os
import json
import random
from typing import List

import torch
import torch.optim as optim
import pandas as pd
from deepctr_torch.models import *
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import preprocess, generate_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset and features
sparse_features = [
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'composer',
    'lyricist',
    'language',
    'is_featured',
    'city',
    'registered_via',
    'gender',
]
dense_features = [
    'song_length',
    'num_artist',
    'num_composer',
    'num_lyricist',
    'bd',
    'registration_init_time',
    'expiration_date',
    'duration',
]
varlen_features = [
    'genre_ids',
    'artist_name',
]


def random_features(features: List[str]):

    sample_size = random.randint(1, len(features))
    sampled_features = random.sample(features, sample_size)

    return sampled_features


def train_pipeline(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   songs_df: pd.DataFrame,
                   members_df: pd.DataFrame,
                   deepctr_model,
                   num_models: int = 5,
                   **kwargs):

    tr_song_msno_df, val_song_msno_df, ts_song_msno_df, item2idx = preprocess(
        train_df, test_df, songs_df, members_df)

    trained_models = {}
    used_features = set()
    ckpt_fp = f'../checkpoints/{deepctr_model.__name__}'
    os.makedirs(ckpt_fp, exist_ok=True)

    i = 0
    while i < num_models:
        sampled_sparse = random_features(sparse_features)
        sampled_dense = random_features(dense_features)
        sampled_varlen = random_features(varlen_features)

        sampled_all = tuple(
            sorted(sampled_sparse + sampled_dense + sampled_varlen))
        if sampled_all in used_features:
            continue
        used_features.add(sampled_all)

        (tr_model_input, val_model_input, ts_model_input,
         linear_feature_columns,
         dnn_feature_columns) = generate_datasets(tr_song_msno_df,
                                                  val_song_msno_df,
                                                  ts_song_msno_df,
                                                  sampled_sparse,
                                                  sampled_dense,
                                                  sampled_varlen,
                                                  item2idx,
                                                  embed_dim=64)

        # train model
        model = deepctr_model(linear_feature_columns,
                              dnn_feature_columns,
                              task='binary',
                              device=DEVICE,
                              **kwargs)
        es = EarlyStopping(monitor='val_auc',
                           min_delta=0,
                           verbose=1,
                           patience=1,
                           mode='max')
        mdckpt = ModelCheckpoint(filepath=f'{ckpt_fp}/model_{i}.ckpt',
                                 monitor='val_auc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

        model.compile(
            # optimizer="adam",
            optimizer=optim.RMSprop(model.parameters(), lr=1e-3),
            loss="binary_crossentropy",
            metrics=['auc'])
        model.fit(tr_model_input,
                  tr_song_msno_df['target'].values,
                  batch_size=8192,
                  epochs=10,
                  verbose=2,
                  validation_data=(val_model_input,
                                   val_song_msno_df['target'].values),
                  shuffle=True,
                  callbacks=[es, mdckpt])

        if mdckpt.best >= 0.68:
            trained_models[i] = mdckpt.best
            i += 1
            print()

    print(trained_models)
    with open(f'{ckpt_fp}/result.json', 'w', encoding='utf-8') as f:
        json.dump(trained_models, f)


if __name__ == "__main__":
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    songs_df = pd.read_csv('../data/songs.csv')
    members_df = pd.read_csv('../data/members.csv')

    train_pipeline(
        train_df,
        test_df,
        songs_df,
        members_df,
        WDL,
        l2_reg_embedding=1e-4,
        dnn_dropout=0.3,
        l2_reg_dnn=1e-4,
        dnn_use_bn=True,
    )

    
