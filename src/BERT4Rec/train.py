import os
import time
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from .tokenizer import Tokenizer
from .model import KeBERT4Rec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    """Sequence Dataset for KeBERT4Rec"""

    def __init__(self,
                 user_items: pd.DataFrame,
                 mode: str,
                 tokenizer: Tokenizer,
                 seq_len: int = 100) -> None:
        """Constructor

        :param user_items: a dataframe contain users and items
        :param mode: 'train' or 'val'
        :param tokenizer: the Tokenizer
        :param seq_len: the sequence length
        """

        self.mode = mode
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        user_seq = user_items.groupby('msno').apply(self.preprocess).to_list()
        self.user_seq = [(torch.tensor(items, dtype=torch.long, device=DEVICE),
                          torch.tensor(keywords,
                                       dtype=torch.long,
                                       device=DEVICE))
                         for items, keywords in user_seq]

    def preprocess(self, group_df: pd.DataFrame):
        """Preprocess each user-items dataframe after groupby

        :param group_df: a dataframe contain one user and the items
        :return: the encoded items and encoded keywords
        """

        group_df = group_df.drop_duplicates(subset='song_id',
                                            ignore_index=True)
        target_items = group_df['song_id'].to_list()
        keywords_ls = group_df['genre_ids'].astype(str).str.split('|')
        encoded_keywords = self.tokenizer.encode_keywords(keywords_ls)
        encoded_items = self.tokenizer.convert_tokens_to_ids(target_items)

        return encoded_items, encoded_keywords

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        target_items, target_keywords = self.user_seq[index]

        # sample start and end
        end_idx = random.randint(
            10, max(target_items.size(0),
                    10)) if self.mode == 'train' else target_items.size(0)
        start_idx = max(0, end_idx - self.seq_len)
        target_items = target_items[start_idx:end_idx]
        target_keywords = target_keywords[start_idx:end_idx]

        # mask
        if self.mode == 'train':
            # only mask the last item, akin finetune
            if random.random() <= 0.05:
                source_items = target_items.clone()
                source_items[-1] = self.tokenizer.MASK
                source_keywords = target_keywords.clone()
                source_keywords[-1] = self.tokenizer.MASK
            else:
                mask = torch.rand(target_items.shape, device=DEVICE) <= 0.2
                source_items = target_items.masked_fill(
                    mask, self.tokenizer.MASK)
                source_keywords = target_keywords.masked_fill(
                    mask.unsqueeze(1), self.tokenizer.MASK)
        else:
            source_items = target_items.clone()
            source_items[-1] = self.tokenizer.MASK
            source_keywords = target_keywords.clone()
            source_keywords[-1] = self.tokenizer.MASK

        # padding
        pad_len = self.seq_len - target_items.size(0)
        pad_side = 'left' if random.random() <= 0.5 else 'right'
        target_items = F.pad(target_items,
                             (pad_len, 0) if pad_side == 'left' else
                             (0, pad_len))
        target_keywords = F.pad(target_keywords,
                                (0, 0, pad_len, 0) if pad_side == 'left' else
                                (0, 0, 0, pad_len))
        source_items = F.pad(source_items,
                             (pad_len, 0) if pad_side == 'left' else
                             (0, pad_len))
        source_keywords = F.pad(source_keywords,
                                (0, 0, pad_len, 0) if pad_side == 'left' else
                                (0, 0, 0, pad_len))
        mask = (target_items != source_items).to(torch.long)

        return (source_items, target_items, mask, source_keywords,
                target_keywords)


def train(model: KeBERT4Rec, data_loader: DataLoader, num_epochs: int,
          checkpoint_fd: str, save_epochs: int):

    writer = SummaryWriter(r'../../tensorboard')

    os.makedirs(checkpoint_fd, exist_ok=True)

    for param in model.parameters():
        nn.init.trunc_normal_(param, mean=0, std=0.02, a=-0.02, b=0.02)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           betas=(0.9, 0.999),
                           weight_decay=0.01)
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda epoch: 1 - epoch / num_epochs)

    model.to(DEVICE)

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        total_acc = 0
        total_sim = 0
        ep_start = time.time()
        for i, batch in enumerate(data_loader):
            # start = time.time()
            (source_item, target_item, mask, source_keyword,
             target_keyword) = batch

            item_out, keyword_out = model(source_item, source_keyword)

            item_loss, item_acc = model.item_loss_acc(item_out, target_item,
                                                      mask)
            keyword_loss, keyword_sim = model.kw_loss_sim(
                keyword_out, target_keyword, mask)

            loss = item_loss + keyword_loss
            total_loss += loss.item()
            total_acc += item_acc.item()
            total_sim += keyword_sim.item()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            # print(f"Batch {i}, use {time.time()-start:.2f}")

        scheduler.step()

        print(
            "Epoch %d: avg loss %.2f, avg acc %.2f, avg sim %.2f, time %.2f" %
            (epoch, total_loss / len(data_loader),
             total_acc / len(data_loader), total_sim / len(data_loader),
             time.time() - ep_start))
        writer.add_scalar('Average Training Loss',
                          total_loss / len(data_loader), epoch)
        writer.add_scalar('Average Accuracy', total_acc / len(data_loader),
                          epoch)
        writer.add_scalar('Average Similarity', total_sim / len(data_loader),
                          epoch)

        if (epoch + 1) % save_epochs == 0:
            checkpoint_path = os.path.join(checkpoint_fd,
                                           f'checkpoint_epoch_{epoch}.pth')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss
                }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
