from typing import Dict, List

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from deepctr_torch.inputs import (SparseFeat, DenseFeat, VarLenSparseFeat,
                                  get_feature_names)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_idx(item_ls: List[str], item2idx: Dict[str, int]):
    for item in item_ls:
        if item not in item2idx:
            item2idx[item] = len(item2idx)

    return list(map(lambda item: item2idx[item], item_ls))


def preprocess_songs(songs_df: pd.DataFrame):

    # standardize song length
    songlen_ss = StandardScaler()
    songs_df['song_length'].fillna(songs_df['song_length'].median(),
                                   inplace=True)
    songs_df['song_length'] = songlen_ss.fit_transform(
        songs_df['song_length'].to_numpy().reshape(-1, 1)).reshape(-1)

    # genre ids
    genre2idx = {'[PAD]': 0}
    songs_df['genre_ids'].fillna('Unknown', inplace=True)
    songs_df['genre_ids'] = songs_df['genre_ids'].astype(str).str.split('|')
    songs_df['genre_ids'] = songs_df['genre_ids'].apply(map_idx,
                                                        item2idx=genre2idx)

    def name_split(col_name: str):
        return songs_df[col_name].str.split(
            r"and|,|feat\.|featuring|&|\. |\||/|\\|;",
            regex=True).apply(lambda ls: list(
                map(lambda s: s.strip(), filter(lambda s: s.strip(), ls))))

    # names
    songs_df['artist_name'].fillna("", inplace=True)
    songs_df['composer'].fillna("", inplace=True)
    songs_df['lyricist'].fillna("", inplace=True)
    songs_df['is_featured'] = songs_df['artist_name'].apply(
        lambda x: 1 if 'feat' in str(x) else 0)
    songs_df['artist_name'] = name_split('artist_name')
    songs_df['composer'] = name_split('composer')
    songs_df['lyricist'] = name_split('lyricist')
    songs_df['num_artist'] = songs_df['artist_name'].apply(len)
    songs_df['num_composer'] = songs_df['composer'].apply(len)
    songs_df['num_lyricist'] = songs_df['lyricist'].apply(len)

    for col in ['num_artist', 'num_composer', 'num_lyricist']:
        songs_df[col] = StandardScaler().fit_transform(
            songs_df[col].to_numpy().reshape(-1, 1)).reshape(-1)

    artist2idx = {'[PAD]': 0}
    songs_df['artist_name'] = songs_df['artist_name'].apply(
        map_idx, item2idx=artist2idx)

    for col in ['composer', 'lyricist']:
        songs_df[col] = LabelEncoder().fit_transform(
            songs_df[col].apply(lambda x: x[0] if x else np.nan))

    # language
    songs_df['language'].fillna(-1, inplace=True)
    songs_df['language'] = LabelEncoder().fit_transform(songs_df['language'])

    item2idx = {'genre_ids': genre2idx, 'artist_name': artist2idx}

    return item2idx


def preprocess_members(members_df: pd.DataFrame):

    # age
    members_df['bd'] = members_df['bd'].apply(lambda x: np.nan
                                              if x <= 5 or x >= 75 else x)
    members_df['bd'].fillna(members_df['bd'].median(), inplace=True)
    members_df['bd'] = MinMaxScaler().fit_transform(
        members_df['bd'].to_numpy().reshape(-1, 1)).reshape(-1)

    # encode geograph info
    columns = ['city', 'gender', 'registered_via']
    for column in columns:
        column_encoder = LabelEncoder()
        members_df[column] = column_encoder.fit_transform(members_df[column])

    # preprocess dates
    members_df['registration_init_time'] = pd.to_datetime(
        members_df['registration_init_time'].astype(str))
    members_df['expiration_date'] = pd.to_datetime(
        members_df['expiration_date'].astype(str))

    members_df['registration_init_time'] = members_df[
        'registration_init_time'].apply(lambda x: np.nan
                                        if x < datetime(2000, 1, 1) else x)
    members_df['registration_init_time'].fillna(
        members_df['registration_init_time'].min(), inplace=True)

    members_df.loc[members_df['expiration_date'] <
                   members_df['registration_init_time'],
                   'expiration_date'] = np.nan
    members_df['expiration_date'].fillna(members_df['expiration_date'].max(),
                                         inplace=True)

    dur_mm = MinMaxScaler()
    dur_col = (members_df['expiration_date'] -
               members_df['registration_init_time']
               ).apply(lambda x: 0 if x < timedelta(0) else x.days).fillna(0)
    members_df['duration'] = dur_mm.fit_transform(dur_col.to_numpy().reshape(
        -1, 1)).reshape(-1)

    members_df['registration_init_time'] = members_df[
        'registration_init_time'].apply(lambda x: x.timestamp())
    members_df['registration_init_time'] = MinMaxScaler().fit_transform(
        members_df['registration_init_time'].to_numpy().reshape(-1,
                                                                1)).reshape(-1)
    members_df['expiration_date'] = members_df['expiration_date'].apply(
        lambda x: x.timestamp())
    members_df['expiration_date'] = MinMaxScaler().fit_transform(
        members_df['expiration_date'].to_numpy().reshape(-1, 1)).reshape(-1)


def generate_datasets(
    tr_song_msno_df: pd.DataFrame,
    val_song_msno_df: pd.DataFrame,
    ts_song_msno_df: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    varlen_features: List[str],
    item2idx: Dict[str, dict],
    embed_dim: int = 64,
):
    full_df = pd.concat([tr_song_msno_df, val_song_msno_df, ts_song_msno_df],
                        ignore_index=True)
    varlen = {feat: max(full_df[feat].apply(len)) for feat in varlen_features}

    fixlen_feature_columns = [
        SparseFeat(feat, len(full_df[feat].unique()), embedding_dim=embed_dim)
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]

    varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat(feat,
                                    vocabulary_size=len(item2idx[feat]),
                                    embedding_dim=embed_dim),
                         maxlen=varlen[feat],
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
                                             maxlen=varlen[feat],
                                             padding='post')
        val_model_input[feat] = pad_sequences(val_song_msno_df[feat],
                                              maxlen=varlen[feat],
                                              padding='post')
        ts_model_input[feat] = pad_sequences(ts_song_msno_df[feat],
                                             maxlen=varlen[feat],
                                             padding='post')

    return (tr_model_input, val_model_input, ts_model_input,
            linear_feature_columns, dnn_feature_columns)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame,
               songs_df: pd.DataFrame, members_df: pd.DataFrame):

    train_test_df = pd.concat([train_df, test_df], ignore_index=True)

    # filter songs and members
    songs_df = songs_df[songs_df['song_id'].isin(train_test_df['song_id'])]
    members_df = members_df[members_df['msno'].isin(train_test_df['msno'])]

    # add missing songs
    missing_song_ids = train_test_df[~train_test_df['song_id'].
                                     isin(songs_df['song_id'])][['song_id']]
    songs_df = pd.concat((songs_df, missing_song_ids), ignore_index=True)

    # encode song ids
    song_id_encoder = LabelEncoder()
    song_id_encoder.fit(train_test_df['song_id'])
    songs_df['song_id'] = song_id_encoder.transform(songs_df['song_id'])
    train_df['song_id'] = song_id_encoder.transform(train_df['song_id'])
    test_df['song_id'] = song_id_encoder.transform(test_df['song_id'])

    # encode msno
    msno_encoder = LabelEncoder()
    msno_encoder.fit(members_df['msno'])
    members_df['msno'] = msno_encoder.transform(members_df['msno'])
    train_df['msno'] = msno_encoder.transform(train_df['msno'])
    test_df['msno'] = msno_encoder.transform(test_df['msno'])

    # preprocess songs and members
    item2idx = preprocess_songs(songs_df)
    preprocess_members(members_df)

    # encode source*
    columns = ['source_system_tab', 'source_screen_name', 'source_type']
    for column in columns:
        column_encoder = LabelEncoder()
        column_encoder.fit(train_test_df[column])
        train_df[column] = column_encoder.transform(train_df[column])
        test_df[column] = column_encoder.transform(test_df[column])

    # train, validation, test
    tr_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)
    tr_song_df = tr_df.merge(songs_df, how='left',
                             on='song_id').drop_duplicates(['msno', 'song_id'])
    tr_song_msno_df = tr_song_df.merge(members_df, how='left', on='msno')
    val_song_df = val_df.merge(songs_df, how='left', on='song_id')
    val_song_msno_df = val_song_df.merge(members_df, how='left', on='msno')
    ts_song_df = test_df.merge(songs_df, how='left', on='song_id')
    ts_song_msno_df = ts_song_df.merge(members_df, how='left', on='msno')

    return tr_song_msno_df, val_song_msno_df, ts_song_msno_df, item2idx
