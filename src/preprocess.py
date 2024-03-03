from typing import List

import pandas as pd
import numpy as np
from tensorflow.keras.utils import pad_sequences
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class Preprocesser:
    """Preprocesser for DeepCTR"""

    def __init__(self) -> None:
        """Constructor"""

        self.song_le = LabelEncoder()  # song_id
        self.msno_le = LabelEncoder()  # msno

        self.sst_le = LabelEncoder()  # source_system_tab
        self.ssn_le = LabelEncoder()  # source_screen_name
        self.st_le = LabelEncoder()  # source_type

        self.artist_le = LabelEncoder()  # artist_name
        self.composer_le = LabelEncoder()  # composer
        self.lyricist_le = LabelEncoder()  # lyricist
        self.language_le = LabelEncoder()  # language
        self.songlen_ss = StandardScaler()  # song_length
        self.genre2idx = {'[PAD]': 0}  # genre_ids

        self.dur_mm = MinMaxScaler()  # duration
        self.bd_mm = MinMaxScaler()  # age
        self.city_le = LabelEncoder()  # city
        self.via_le = LabelEncoder()  # registered_via
        self.gender_le = LabelEncoder()  # gender

        self.genre_maxlen = 8
        self.vocab_size = {}

    def fit_train_test(self, train_test_df: pd.DataFrame):
        """Fit train and test

        :param train_test_df: the train or test dataframe
        """

        # fill Nan
        train_test_df['source_system_tab'].fillna('Unknown', inplace=True)
        train_test_df['source_screen_name'].fillna('Unknown', inplace=True)
        train_test_df['source_type'].fillna('Unknown', inplace=True)

        # encode
        self.sst_le.fit(train_test_df['source_system_tab'])
        self.ssn_le.fit(train_test_df['source_screen_name'])
        self.st_le.fit(train_test_df['source_type'])

        # record vocab size
        self.vocab_size['source_system_tab'] = len(self.sst_le.classes_)
        self.vocab_size['source_screen_name'] = len(self.ssn_le.classes_)
        self.vocab_size['source_type'] = len(self.st_le.classes_)

    def transform_train_test(self, train_test_df: pd.DataFrame):
        """Transform train and test

        :param train_test_df: the train or test dataframe
        """

        # fill Nan
        train_test_df['source_system_tab'].fillna('Unknown', inplace=True)
        train_test_df['source_screen_name'].fillna('Unknown', inplace=True)
        train_test_df['source_type'].fillna('Unknown', inplace=True)

        # encode
        train_test_df['source_system_tab'] = self.sst_le.transform(
            train_test_df['source_system_tab'])
        train_test_df['source_screen_name'] = self.ssn_le.transform(
            train_test_df['source_screen_name'])
        train_test_df['source_type'] = self.st_le.transform(
            train_test_df['source_type'])

    def preprocess_songs(self, songs_df: pd.DataFrame):
        """Preprocess songs

        :param songs_df: the song dataframe
        """

        # fill Nan
        songs_df['language'] = songs_df['language'].fillna(0.0).astype(str)
        songs_df['genre_ids'].fillna('Unknown', inplace=True)
        songs_df['composer'].fillna(songs_df['artist_name'], inplace=True)
        songs_df['lyricist'].fillna(songs_df['lyricist'], inplace=True)

        # encode & preprocess
        def map_idx(genre_ls: List[str]):
            for genre in genre_ls:
                if genre not in self.genre2idx:
                    self.genre2idx[genre] = len(self.genre2idx)

            return list(map(lambda genre: self.genre2idx[genre], genre_ls))

        songs_df['genre_ids'] = songs_df['genre_ids'].astype(str).str.split(
            '|')
        songs_df['genre_ids'] = songs_df['genre_ids'].apply(map_idx)

        songs_df['song_length'] = self.songlen_ss.fit_transform(
            songs_df['song_length'].to_numpy().reshape(-1, 1)).reshape(-1)

        songs_df['artist_name'] = self.artist_le.fit_transform(
            songs_df['artist_name'])
        songs_df['composer'] = self.composer_le.fit_transform(
            songs_df['composer'])
        songs_df['lyricist'] = self.lyricist_le.fit_transform(
            songs_df['lyricist'])
        songs_df['language'] = self.language_le.fit_transform(
            songs_df['language'])

        self.song_le.fit(songs_df['song_id'])

        # record vocab size
        self.genre_maxlen = max(list(map(len, songs_df['genre_ids'])))
        self.vocab_size['artist_name'] = len(self.artist_le.classes_)
        self.vocab_size['composer'] = len(self.composer_le.classes_)
        self.vocab_size['lyricist'] = len(self.lyricist_le.classes_)
        self.vocab_size['language'] = len(self.language_le.classes_)
        self.vocab_size['song_id'] = len(self.song_le.classes_)

    def preprocess_members(self, members_df: pd.DataFrame):
        """Preprocess members

        :param members_df: the member dataframe
        """

        # duration
        members_df['registration_init_time'] = pd.to_datetime(
            members_df['registration_init_time'].astype(str))
        members_df['expiration_date'] = pd.to_datetime(
            members_df['expiration_date'].astype(str))
        members_df['registration_init_time'] = members_df[
            'registration_init_time'].apply(
                lambda x: np.nan if x < datetime(2005, 10, 1) else x)
        # members_df['expiration_date'] = members_df['expiration_date'].apply(
        #     lambda x: np.nan if x >= datetime(2017, 9, 27) else x)
        dur_col = (members_df['expiration_date'] -
                   members_df['registration_init_time']
                   ).apply(lambda x: 0 if x < timedelta(0) else x.days)
        dur_col.fillna(0, inplace=True)
        members_df['duration'] = self.dur_mm.fit_transform(
            dur_col.to_numpy().reshape(-1, 1)).reshape(-1)

        # age
        members_df['bd'] = members_df['bd'].apply(lambda x: np.nan
                                                  if x <= 5 or x >= 75 else x)
        members_df['bd'].fillna(members_df['bd'].median(), inplace=True)
        members_df['bd'] = self.bd_mm.fit_transform(
            members_df['bd'].to_numpy().reshape(-1, 1)).reshape(-1)

        # gender
        members_df['gender'].fillna('Unknown', inplace=True)
        members_df['gender'] = self.gender_le.fit_transform(
            members_df['gender'])

        # encode
        members_df['city'] = self.city_le.fit_transform(members_df['city'])
        members_df['registered_via'] = self.via_le.fit_transform(
            members_df['registered_via'])

        # msno
        self.msno_le.fit(members_df['msno'])

        # record vocab size
        self.vocab_size['gender'] = len(self.gender_le.classes_)
        self.vocab_size['city'] = len(self.city_le.classes_)
        self.vocab_size['registered_via'] = len(self.via_le.classes_)
        self.vocab_size['msno'] = len(self.msno_le.classes_)

    def padding_genre(self, split_genre: pd.Series):
        """Pad genres

        :param split_genre: a series of lists of genres
        :return: a tensor of size (batch, vocab)
        """

        return pad_sequences(split_genre,
                             maxlen=self.genre_maxlen,
                             padding='post')

    def transform_msno_song(self, union_df: pd.DataFrame):
        """Encode msno and song_ids

        :param union_df: the joined dataframes
        """

        union_df['msno'] = self.msno_le.transform(union_df['msno'])
        union_df['song_id'] = self.song_le.transform(union_df['song_id'])
