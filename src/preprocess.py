from typing import List

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def preprocess_train(train_df: pd.DataFrame):
    """Preprocess train

    :param train_df: the train dataframe
    """

    # fill Nan
    train_df['source_system_tab'].fillna('Unknown', inplace=True)
    train_df['source_screen_name'].fillna('Unknown', inplace=True)
    train_df['source_type'].fillna('Unknown', inplace=True)

    # encode
    train_df['source_system_tab'] = LabelEncoder().fit_transform(
        train_df['source_system_tab'])
    train_df['source_screen_name'] = LabelEncoder().fit_transform(
        train_df['source_screen_name'])
    train_df['source_type'] = LabelEncoder().fit_transform(
        train_df['source_type'])


def preprocess_songs(songs_df: pd.DataFrame):
    """Preprocess songs

    :param songs_df: the song dataframe
    """

    # fill Nan
    songs_df['language'] = songs_df['language'].fillna(0.0).astype(str)
    songs_df['genre_ids'].fillna('Unknown', inplace=True)
    songs_df['composer'].fillna(songs_df['artist_name'], inplace=True)
    songs_df['lyricist'].fillna(songs_df['lyricist'], inplace=True)

    # encode & preprocess
    genre2idx = {}

    def map_idx(genre_ls: List[str]):
        for genre in genre_ls:
            if genre not in genre2idx:
                genre2idx[genre] = len(genre2idx) + 1

        return list(map(lambda genre: genre2idx[genre], genre_ls))

    songs_df['genre_ids'] = songs_df['genre_ids'].astype(str).str.split('|')
    songs_df['genre_ids'] = songs_df['genre_ids'].apply(map_idx)

    songs_df['song_length'] = StandardScaler().fit_transform(
        songs_df['song_length'].to_numpy().reshape(-1, 1)).reshape(-1)

    songs_df['artist_name'] = LabelEncoder().fit_transform(
        songs_df['artist_name'])
    songs_df['composer'] = LabelEncoder().fit_transform(songs_df['composer'])
    songs_df['lyricist'] = LabelEncoder().fit_transform(songs_df['lyricist'])
    songs_df['language'] = LabelEncoder().fit_transform(songs_df['language'])


def preprocess_members(members_df: pd.DataFrame):
    """Preprocess members

    :param members_df: the member dataframe
    """

    # duration
    members_df['registration_init_time'] = pd.to_datetime(
        members_df['registration_init_time'].astype(str))
    members_df['expiration_date'] = pd.to_datetime(
        members_df['expiration_date'].astype(str))
    members_df['registration_init_time'] = members_df[
        'registration_init_time'].apply(lambda x: np.nan
                                        if x < datetime(2005, 10, 1) else x)
    members_df['expiration_date'] = members_df['expiration_date'].apply(
        lambda x: np.nan if x >= datetime(2017, 9, 27) else x)
    dur_col = (members_df['expiration_date'] -
               members_df['registration_init_time']
               ).apply(lambda x: 0 if x < timedelta(0) else x.days)
    members_df['duration'] = MinMaxScaler().fit_transform(
        dur_col.to_numpy().reshape(-1, 1)).reshape(-1)

    # age
    members_df['bd'] = members_df['bd'].apply(lambda x: np.nan
                                              if x <= 5 or x >= 75 else x)

    # encode
    members_df['city'] = LabelEncoder().fit_transform(members_df['city'])
    members_df['registered_via'] = LabelEncoder().fit_transform(
        members_df['registered_via'])


def padding_genre(split_genre: pd.Series):
    """Pad genres

    :param split_genre: a series of lists of genres
    :return: a tensor of size (batch, vocab)
    """

    return pad_sequence(list(map(torch.Tensor, split_genre.tolist())),
                        batch_first=True,
                        padding_value=0)
