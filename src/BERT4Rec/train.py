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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
                mask = torch.rand(target_items.shape, device=DEVICE) <= 0.5
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

    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-4,
                            betas=(0.9, 0.999),
                            weight_decay=0.01)
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda epoch: 1 - epoch / num_epochs)
    model.to(DEVICE)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss, total_acc, total_sim = 0, 0, 0
        ep_start = time.time()

        for batch in data_loader:
            (source_item, target_item, mask, source_keyword,
             target_keyword) = batch
            item_out, keyword_out = model(source_item.to(DEVICE),
                                          source_keyword.to(DEVICE))
            item_loss, item_acc = model.item_loss_acc(item_out,
                                                      target_item.to(DEVICE),
                                                      mask.to(DEVICE))
            keyword_loss, keyword_sim = model.kw_loss_sim(
                keyword_out, target_keyword.to(DEVICE), mask.to(DEVICE))

            loss = item_loss + keyword_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            total_acc += item_acc.item()
            total_sim += keyword_sim.item()

        scheduler.step()
        epoch_duration = time.time() - ep_start
        avg_loss = total_loss / len(data_loader)
        avg_acc = total_acc / len(data_loader)
        avg_sim = total_sim / len(data_loader)

        print(
            "Epoch %d: avg loss %.2f, avg acc %.2f, avg sim %.2f, time %.2f" %
            (epoch, avg_loss, avg_acc, avg_sim, epoch_duration))
        writer.add_scalar('Average Training Loss', avg_loss, epoch)
        writer.add_scalar('Average Accuracy', avg_acc, epoch)
        writer.add_scalar('Average Similarity', avg_sim, epoch)

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


# train_df = pd.read_csv('../../data/sampled_train.csv')
# songs_df = pd.read_csv('../../data/songs.csv')
# tr_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)
# tr_song_df = tr_df.merge(songs_df, how='inner', on='song_id')

# tknr = Tokenizer.load(load_kw_enc=True)
# train_ds = SequenceDataset(tr_song_df, 'train', tknr, 50)
# train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
# model = KeBERT4Rec(len(tknr.vocab['id2item']),
#                    len(tknr.keyword_encoder.classes_)).to(DEVICE)

# train(model, train_dl, 2000, '../../checkpoints', 50)
