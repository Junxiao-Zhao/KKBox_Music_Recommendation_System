import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from deepctr_torch.models import DCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.preprocess import Preprocesser

if __name__ == "__main__":
    train_df = pd.read_csv("../../data/sampled_train.csv")
    val_df = pd.read_csv("../../data/pseudo_test.csv")
    songs_df = pd.read_csv("../../data/songs.csv")
    song_extra_info_df = pd.read_csv("../../data/song_extra_info.csv")
    members_df = pd.read_csv("../../data/members.csv")
    real_train_df = pd.read_csv("../../data/train.csv")
    real_test_df = pd.read_csv("../../data/test.csv")

    preprocesser = Preprocesser()
    preprocesser.fit_train_test(pd.concat([real_train_df, real_test_df]))
    preprocesser.transform_train_test(train_df)
    preprocesser.transform_train_test(val_df)
    preprocesser.preprocess_songs(songs_df)
    preprocesser.preprocess_members(members_df)

    # merge data and use label encoder
    tr_song_df = train_df.merge(songs_df, how='inner', on='song_id')
    tr_song_msno_df = tr_song_df.merge(members_df, how='inner', on='msno')
    val_song_df = val_df.merge(songs_df, how='inner', on='song_id')
    val_song_msno_df = val_song_df.merge(members_df, how='inner', on='msno')
    preprocesser.transform_msno_song(tr_song_msno_df)
    preprocesser.transform_msno_song(val_song_msno_df)

    tr_genre = preprocesser.padding_genre(tr_song_msno_df['genre_ids'])
    val_genre = preprocesser.padding_genre(val_song_msno_df['genre_ids'])

    # preprocess dataset and assign features
    sparse_features = [
        'msno', 'song_id', 'source_system_tab', 'source_screen_name',
        'source_type', 'artist_name', 'composer', 'lyricist', 'language',
        'city', 'gender', 'registered_via'
    ]
    dense_features = ['song_length', 'bd', 'duration']
    fixlen_feature_columns = [
        SparseFeat(feat, preprocesser.vocab_size[feat], embedding_dim=64)
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]

    varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat('genre_ids',
                                    vocabulary_size=len(
                                        preprocesser.genre2idx),
                                    embedding_dim=64),
                         maxlen=preprocesser.genre_maxlen,
                         combiner='mean')
    ]

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns +
                                      dnn_feature_columns)

    tr_model_input = {name: tr_song_msno_df[name] for name in feature_names}
    val_model_input = {name: val_song_msno_df[name] for name in feature_names}
    tr_model_input['genre_ids'] = tr_genre
    val_model_input['genre_ids'] = val_genre

    # train model
    model = DCN(linear_feature_columns,
                dnn_feature_columns,
                task='binary',
                l2_reg_embedding=1e-5,
                device=device)
    es = EarlyStopping(monitor='val_auc',
                       min_delta=0,
                       verbose=1,
                       patience=3,
                       mode='max')
    os.makedirs('../../checkpoints/DCN/', exist_ok=True)
    mdckpt = ModelCheckpoint(filepath='../../checkpoints/DCN/model.ckpt',
                             monitor='val_auc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

    model.compile("adam", "binary_crossentropy", metrics=['auc'])
    history = model.fit(tr_model_input,
                        tr_song_msno_df['target'].values,
                        batch_size=64,
                        epochs=10,
                        verbose=1,
                        validation_data=(val_model_input,
                                         val_song_msno_df['target'].values),
                        shuffle=True,
                        callbacks=[es, mdckpt])