import os
import json

import ast
import torch
import torch.optim as optim
import pandas as pd
from tensorflow.keras.utils import pad_sequences
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import (SparseFeat, DenseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # load data
    tr_song_msno_df = pd.read_csv("../../data/prep_tr_song_msno.csv")
    val_song_msno_df = pd.read_csv("../../data/prep_val_song_msno.csv")
    ts_song_msno_df = pd.read_csv("../../data/prep_ts_song_msno.csv")

    with open('../../data/item2idx.json', 'r', encoding='utf-8') as f:
        item2idx = json.load(f)

    # dataset and features
    sparse_features = [
        'msno',
        'song_id',
        'source_system_tab',
        'source_screen_name',
        'source_type',
        # 'composer',
        # 'lyricist',
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
    embed_dim = 64

    for feat in varlen_features:
        tr_song_msno_df[feat] = tr_song_msno_df[feat].apply(ast.literal_eval)
        val_song_msno_df[feat] = val_song_msno_df[feat].apply(ast.literal_eval)
        ts_song_msno_df[feat] = ts_song_msno_df[feat].apply(ast.literal_eval)

    full_df = pd.concat([tr_song_msno_df, val_song_msno_df, ts_song_msno_df],
                        ignore_index=True)

    fixlen_feature_columns = [
        SparseFeat(feat, len(full_df[feat].unique()), embedding_dim=embed_dim)
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]

    varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat(feat,
                                    vocabulary_size=len(item2idx[feat]),
                                    embedding_dim=embed_dim),
                         maxlen=max(full_df[feat].apply(len)),
                         combiner='mean') for feat in varlen_features
    ]

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns +
                                      dnn_feature_columns)

    tr_model_input = {name: tr_song_msno_df[name] for name in feature_names}
    val_model_input = {name: val_song_msno_df[name] for name in feature_names}
    ts_model_input = {name: ts_song_msno_df[name] for name in feature_names}

    for feat in varlen_features:
        tr_model_input[feat] = pad_sequences(tr_song_msno_df[feat],
                                             maxlen=max(
                                                 full_df[feat].apply(len)),
                                             padding='post')
        val_model_input[feat] = pad_sequences(val_song_msno_df[feat],
                                              maxlen=max(
                                                  full_df[feat].apply(len)),
                                              padding='post')
        ts_model_input[feat] = pad_sequences(ts_song_msno_df[feat],
                                             maxlen=max(
                                                 full_df[feat].apply(len)),
                                             padding='post')

    # train model
    model = DeepFM(linear_feature_columns,
                   dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5,
                   dnn_dropout=0.3,
                   l2_reg_dnn=1e-5,
                   dnn_use_bn=True,
                   device=DEVICE)
    es = EarlyStopping(monitor='val_auc',
                       min_delta=0,
                       verbose=1,
                       patience=2,
                       mode='max')
    os.makedirs('../../checkpoints/DeepFM/', exist_ok=True)
    mdckpt = ModelCheckpoint(filepath='../../checkpoints/DeepFM/model.ckpt',
                             monitor='val_auc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

    model.compile(
        # optimizer="adam",
        optimizer=optim.RMSprop(model.parameters(), lr=1e-3),
        loss="binary_crossentropy",
        metrics=['auc'])
    history = model.fit(tr_model_input,
                        tr_song_msno_df['target'].values,
                        batch_size=8192,
                        epochs=10,
                        verbose=2,
                        validation_data=(val_model_input,
                                         val_song_msno_df['target'].values),
                        shuffle=True,
                        callbacks=[es, mdckpt])
