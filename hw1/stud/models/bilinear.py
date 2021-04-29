import torch
from torch import nn

from typing import List, Dict

from stud.utils import text_length


class BilinearClassifier(nn.Module):
    def __init__(
        self,
        vectors_store: List[torch.Tensor],
        args,
    ) -> None:
        super().__init__()
        self.args = args
        ####### EMBEDDING LAYERS #######
        # sentences embedding
        if args.use_pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(
                vectors_store,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(
                len(vectors_store), args.sentence_embedding_size, padding_idx=0
            )

        # POS embedding
        if args.use_pos:
            self.embedding_pos = nn.Embedding(
                args.pos_vocab_size, args.pos_embedding_size, padding_idx=0
            )

        ####### CLASSIFICATION HEAD #######
        self.bilinear_layer = nn.Bilinear(
            args.sentence_embedding_size, args.sentence_embedding_size, args.bi_n_hidden
        )

        if args.use_pos:
            self.pos_bilinear_layer = nn.Bilinear(
                args.pos_embedding_size, args.pos_embedding_size, args.bi_n_hidden
            )
            self.last_bilinear_layer = nn.Bilinear(
                args.bi_n_hidden, args.bi_n_hidden, args.bi_n_hidden
            )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(args.bi_dropout)

        if args.use_pos:
            self.pos_bilinear = nn.Bilinear(
                args.bi_n_hidden, args.bi_n_hidden, args.bi_n_hidden
            )

        self.fc = nn.Linear(
            in_features=args.bi_n_hidden,
            out_features=args.bi_n_hidden,
        )

        self.last_layer = nn.Linear(in_features=args.bi_n_hidden, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Dict[str, torch.tensor]) -> torch.Tensor:
        sentence1 = batch["sentence1"]
        sentence2 = batch["sentence2"]

        # compute sentences length
        lengths1 = text_length(sentence1)
        lengths2 = text_length(sentence2)

        # text embeddings
        embeddings1 = self.embedding(sentence1)
        embeddings2 = self.embedding(sentence2)

        # compute the aggregation (i.e., mean) of the embeddings
        # only taking into consideration non-padding values
        embeddings1 = self._forward_embedding(embeddings1, lengths1)
        embeddings2 = self._forward_embedding(embeddings2, lengths2)

        # use pos embeddings
        if self.args.use_pos:
            pos1 = batch["pos1"]
            pos2 = batch["pos2"]
            pos_embedding1 = self.embedding_pos(pos1)
            pos_embedding2 = self.embedding_pos(pos2)
            pos_embedding1 = self._forward_embedding(pos_embedding1, lengths1)
            pos_embedding2 = self._forward_embedding(pos_embedding2, lengths2)
            pos_out = self.pos_bilinear_layer(pos_embedding1, pos_embedding2)
            pos_out = self.activation(pos_out)
            pos_out = self.dropout(pos_out)

        out = self.bilinear_layer(embeddings1, embeddings2)
        out = self.activation(out)
        out = self.dropout(out)

        if self.args.use_pos:
            out = self.pos_bilinear(out, pos_out)
        out = self.fc(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.last_layer(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)

    def _forward_embedding(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # create mask to compute mean excluding embeddings of pad index
        mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device)
            < lengths[..., None]
        )
        # compute mean
        embeddings = (embeddings * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        return embeddings