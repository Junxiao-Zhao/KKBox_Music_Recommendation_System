import os
import json
from typing import Dict, List, Iterable

import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class Tokenizer:

    def __init__(self,
                 vocab: Dict[str, Dict],
                 keyword_encoder: MultiLabelBinarizer = None) -> None:
        """Constructor

        :param vocab: a dict contains 'item2id' and 'id2item'
        :param keyword_encoder: a multi-hot encoder
        """

        self.vocab = vocab
        self.keyword_encoder = keyword_encoder

    @classmethod
    def construct(cls,
                  item_df: pd.DataFrame,
                  item_column: str,
                  keyword_ls: Iterable[Iterable[str]] = None):
        """Construct a Tokenizer

        :param item_df: a dataframe contains items
        :param item_column: a column contains items' ids
        :param keyword_ls: a set of keywords for each item

        :return: a Tokenizer
        """

        item2id = {
            item: idx + 2
            for idx, item in enumerate(item_df[item_column].unique())
        }
        item2id['[PAD]'] = 0
        item2id['[MASK]'] = 1
        id2item = {idx: item for item, idx in item2id.items()}

        if keyword_ls is not None:
            mlb = MultiLabelBinarizer()
            mlb.fit(keyword_ls)

        return cls({'item2id': item2id, 'id2item': id2item}, mlb)

    @classmethod
    def load(cls,
             vocab_fp: str = None,
             load_kw_enc: bool = False,
             keyword_enc_fp: str = None):
        """Load a Tokenizer

        :param vocab_fp: the vocab's file path; if None, use the default vocab
        :param load_kw_enc: whether to load the keyword multi-hot encoder;
        default False
        :param keyword_enc_fp: the multi-hot encoder's file path;
        if None, use the default multi-hot encoder

        :return: a Tokenizer
        """

        if not vocab_fp:
            vocab_fp = os.path.join(os.path.dirname(__file__), 'vocab.json')
            if not os.path.exists(vocab_fp):
                raise FileNotFoundError('No default vocab!')

        with open(vocab_fp, 'r', encoding='utf-8') as vocab_f:
            vocab = json.load(vocab_f)

        if load_kw_enc:
            if not keyword_enc_fp:
                keyword_enc_fp = os.path.join(os.path.dirname(__file__),
                                              'kw_enc.pkl')
                if not os.path.exists(vocab_fp):
                    raise FileNotFoundError('No default keyword encoder!')

            kw_enc = joblib.load(keyword_enc_fp)

        return cls(vocab, kw_enc)

    def save(self, vocab_fp: str = None, keyword_enc_fp: str = None):

        if not vocab_fp:
            vocab_fp = os.path.join(os.path.dirname(__file__), 'vocab.json')

        with open(vocab_fp, 'w', encoding='utf-8') as vocab_f:
            json.dump(self.vocab, vocab_f)

        if self.keyword_encoder is not None:
            if not keyword_enc_fp:
                keyword_enc_fp = os.path.join(os.path.dirname(__file__),
                                              'kw_enc.pkl')
            joblib.dump(self.keyword_encoder, keyword_enc_fp)
