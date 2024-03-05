import os
import sys
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

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocess import preprocess, generate_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # load data
    train_df = pd.read_csv('../../data/train.csv')
    test_df = pd.read_csv('../../data/test.csv')
    songs_df = pd.read_csv('../../data/songs.csv')
    members_df = pd.read_csv('../../data/members.csv')

    tr_song_msno_df, val_song_msno_df, ts_song_msno_df, item2idx = preprocess(
        train_df, test_df, songs_df, members_df)

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
        # 'num_artist',
        # 'num_composer',
        # 'num_lyricist',
        'bd',
        'registration_init_time',
        'expiration_date',
        # 'duration',
    ]
    varlen_features = [
        'genre_ids',
        'artist_name',
    ]

    (tr_model_input, val_model_input, ts_model_input, linear_feature_columns,
     dnn_feature_columns) = generate_datasets(tr_song_msno_df,
                                              val_song_msno_df,
                                              ts_song_msno_df,
                                              sparse_features,
                                              dense_features,
                                              varlen_features,
                                              item2idx,
                                              embed_dim=64)

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
