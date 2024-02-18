from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeBERT4Rec(nn.Module):
    """KeBERT4Rec"""

    def __init__(self,
                 item_size: int,
                 keyword_size: int,
                 dim: int = 64,
                 dropout: float = 0.4,
                 num_head: int = 4,
                 num_layers: int = 3) -> None:
        """Constructor

        :param item_size: total number of items
        :param keyword_size: total number of keywords
        :param dim: the dim of embeddings
        :param dropout: dropout rate
        :param num_layers: number of Encoder layers
        """

        super().__init__()

        # embeddings
        self.item_embeddings = nn.Embedding(item_size, dim)
        self.pos_embdddings = nn.Embedding(512, dim)
        self.kw_embdddings = nn.Linear(keyword_size, dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(dim,
                                                   nhead=num_head,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

        # output layer
        self.item_out = nn.Linear(dim, item_size)
        self.kw_out = nn.Linear(dim, keyword_size)

    def forward(
            self, source_items: torch.Tensor, source_keywords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        :param source_items: items
        :param source_keywords: keywords
        :return: predicted item logits and keyword logits
        """

        batch_size, seq_len = source_items.shape

        # encode input
        item_embed = self.item_embeddings(source_items)
        kw_embed = self.kw_embdddings(source_keywords.to(torch.float32))
        pos_encode = torch.arange(
            0, seq_len,
            device=source_items.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embed = self.pos_embdddings(pos_encode)

        # forward
        x = item_embed + kw_embed + pos_embed
        x = self.encoder(x)
        item_out = self.item_out(x)
        kw_out = self.kw_out(x)

        return item_out, kw_out

    @staticmethod
    def item_loss_acc(predict_items_logits: torch.Tensor,
                      target_items: torch.Tensor, mask: torch.Tensor):
        """Calculate item loss and accuracy

        :param predict_items_logits: the prediction logits
        :param target_items: the target items
        :param mask: the mask
        :return: the loss and the accuracy
        """
        predict_items_logits = predict_items_logits.view(
            -1, predict_items_logits.shape[-1])
        target_items = target_items.view(-1)
        mask = mask.flatten() == 1

        # loss
        loss = F.cross_entropy(predict_items_logits,
                               target_items,
                               reduction='none')
        loss *= mask
        loss = loss.sum() / (mask.sum() + 1e-8)

        # accuracy
        predict = predict_items_logits.argmax(dim=-1)
        y_true = target_items.masked_select(mask)
        y_predict = predict.masked_select(mask)
        acc = (y_true == y_predict).double().mean()

        return loss, acc

    @staticmethod
    def kw_loss_sim(predict_kw_logits: torch.Tensor, target_kw: torch.Tensor,
                    mask: torch.Tensor):
        """Calculate keywords loss and cosine similarity

        :param predict_kw_logits: the keywords logits
        :param target_kw: the target keywords
        :param mask: the mask
        :return: the loss and the cosine similarity
        """

        mask = mask == 1

        # loss
        loss = F.binary_cross_entropy_with_logits(predict_kw_logits,
                                                  target_kw.to(torch.float32),
                                                  reduction='none').sum(-1)
        loss *= mask
        loss = loss.sum() / (mask.sum() + 1e-8)

        # accuracy
        sim = torch.cosine_similarity(
            predict_kw_logits, target_kw,
            dim=-1).masked_select(mask == 1).sum() / (mask.sum() + 1e-8)

        return loss, sim
