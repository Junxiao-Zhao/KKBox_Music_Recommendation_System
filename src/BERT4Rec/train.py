import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pandas.core.groupby.generic import DataFrameGroupBy

from . import Tokenizer


class SequenceDataset(Dataset):
    """Sequence Dataset for KeBERT4Rec"""

    def __init__(self,
                 group_by: DataFrameGroupBy,
                 mode: str,
                 tokenizer: Tokenizer,
                 seq_len: int = 100) -> None:
        """Constructor

        :param group_by: a pandas groupby
        :param mode: 'train' or 'val'
        :param tokenizer: the Tokenizer
        :param seq_len: the sequence length
        """

        self.groups = list(group_by.groups.keys())
        self.group_by = group_by
        self.mode = mode
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):

        group = self.groups[index]
        group_df: pd.DataFrame = self.group_by.get_group(group)
        group_df = group_df.drop_duplicates(subset='song_id',
                                            ignore_index=True)

        # sample start and end
        end_idx = random.randint(
            10, max(group_df.shape[0],
                    10)) if self.mode == 'train' else group_df.shape[0]
        start_idx = max(0, end_idx - self.seq_len)
        group_df = group_df.iloc[start_idx:end_idx]

        # extract items and keywords
        target_items = group_df['song_id'].to_list()
        keywords_ls = group_df['genre_ids'].astype(str).str.split('|')
        target_keywords_tensor = self.tokenizer.encode_keywords(keywords_ls)

        # mask
        if self.mode == 'train':
            # only mask the last item, akin finetune
            if random.random() <= 0.05:
                source_items = target_items[:-1] + ['[MASK]']
            else:
                source_items = self.tokenizer.mask(target_items, ratio=0.2)
        else:
            source_items = target_items[:-1] + ['[MASK]']

        # padding
        pad_len = self.seq_len - len(target_items)
        pad_side = 'left' if random.random() <= 0.5 else 'right'
        target_items = self.tokenizer.padding(target_items,
                                              side=pad_side,
                                              max_len=self.seq_len)
        source_items = self.tokenizer.padding(source_items,
                                              side=pad_side,
                                              max_len=self.seq_len)
        target_keywords_tensor = F.pad(target_keywords_tensor,
                                       (0, 0, pad_len,
                                        0) if pad_side == 'left' else
                                       (0, 0, 0, pad_len))

        # convert to tensors
        target_item_tensor = self.tokenizer.convert_tokens_to_ids(target_items)
        source_item_tensor = self.tokenizer.convert_tokens_to_ids(source_items)
        mask = (target_item_tensor != source_item_tensor).to(torch.long)
        source_keywords_tensor = target_keywords_tensor.masked_fill(
            mask.unsqueeze(1) == 1, 1)  # mask keywords

        return (source_item_tensor, target_item_tensor, mask,
                source_keywords_tensor, target_keywords_tensor)
