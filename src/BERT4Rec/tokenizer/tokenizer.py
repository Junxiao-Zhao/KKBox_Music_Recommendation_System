import os
import json
import random
from typing import Dict, List, Iterable

import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class Tokenizer:
    """(Ke)BERT4Rec Tokenizer"""

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
        id2item = {str(idx): item for item, idx in item2id.items()}

        mlb = None
        if keyword_ls is not None:
            mlb = MultiLabelBinarizer()
            mlb.fit(keyword_ls)

        return cls({'item2id': item2id, 'id2item': id2item}, mlb)

    @staticmethod
    def padding(tokens: Iterable[str], side: str = 'left', max_len: int = 100):
        """Padding

        :param tokens: a sequence of tokens
        :param side: 'left' or 'right'; default 'left'
        :param max_len: the max sequence length
        :return: a sequence of tokens after padding
        """

        tokens = list(tokens)[-max_len:]
        if side == 'left':
            pad_tokens = ['[PAD]'] * (max_len - len(tokens)) + tokens
        elif side == 'right':
            pad_tokens = tokens + ['[PAD]'] * (max_len - len(tokens))
        else:
            raise ValueError('side should be either "left" or "right"')

        return pad_tokens

    @staticmethod
    def mask(tokens: Iterable[str], ratio: float = 0.2):
        """Mask randomly

        :param tokens: a sequence of tokens
        :param ration: the mask ratio
        :return: a sequence of masked tokens
        """

        tokens_list = list(tokens)
        total_tokens = len(tokens_list)
        num_to_mask = round(total_tokens * ratio)
        indices_to_mask = random.sample(range(total_tokens), num_to_mask)

        masked_tokens = [
            '[MASK]' if idx in indices_to_mask else token
            for idx, token in enumerate(tokens_list)
        ]

        return masked_tokens

    def convert_tokens_to_ids(self, tokens: Iterable[str]):
        """Convert tokens into ids

        :param tokens: a sequence of tokens
        :return: a tensor of ids
        """

        return torch.tensor(list(
            map(lambda item: self.vocab['item2id'].get(item, 1), tokens)),
                            dtype=torch.long)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        """Convert ids into tokens

        :param ids: a sequence of ids
        :return: a list of tokens
        """

        return list(map(lambda idx: self.vocab['id2item'][str(idx)], ids))

    def encode_keywords(self, keyword_ls: Iterable[Iterable[str]]):
        """Encode keywords using multi-hot encoding

        :param keyword_ls: a set of keywords for each item
        :return: a tensor of the multi-hot encodings
        """

        return torch.tensor(self.keyword_encoder.transform(keyword_ls),
                            dtype=torch.long)

    def decode_keywords(self, mulit_hot: np.ndarray):
        """Decode multi-hot encoding into keywords

        :param mulit_hot: a tensor of multi-hot encodings
        :return: a set of keywords for each item
        """

        return self.keyword_encoder.inverse_transform(mulit_hot)

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

        kw_enc = None
        if load_kw_enc:
            if not keyword_enc_fp:
                keyword_enc_fp = os.path.join(os.path.dirname(__file__),
                                              'kw_enc.pkl')
                if not os.path.exists(vocab_fp):
                    raise FileNotFoundError('No default keyword encoder!')

            kw_enc = joblib.load(keyword_enc_fp)

        return cls(vocab, kw_enc)

    def save(self, vocab_fp: str = None, keyword_enc_fp: str = None):
        """Save the Tokenizer

        :param vocab_fp: the vocab's file path; if None, use the default
        :param keyword_enc_fp: the multi-hot encoder's file path;
        if None, use the default
        """

        if not vocab_fp:
            vocab_fp = os.path.join(os.path.dirname(__file__), 'vocab.json')

        with open(vocab_fp, 'w', encoding='utf-8') as vocab_f:
            json.dump(self.vocab, vocab_f)

        if self.keyword_encoder is not None:
            if not keyword_enc_fp:
                keyword_enc_fp = os.path.join(os.path.dirname(__file__),
                                              'kw_enc.pkl')
            joblib.dump(self.keyword_encoder, keyword_enc_fp)
