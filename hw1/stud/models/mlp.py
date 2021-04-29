import torch
from torch import nn

from typing import Dict

from stud.utils import text_length


class MlpClassifier(nn.Module):
    def __init__(
        self,
        vectors_store: torch.Tensor,
        args,
    ) -> None:
        super().__init__()
        self.args = args
        ####### EMBEDDING LAYERS #######
        # sentence embedding
        if args.use_pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(
                vectors_store,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(
                len(vectors_store), args.sentence_embedding_size, padding_idx=0
            )

        linear_features = 2 * args.sentence_embedding_size

        # POS embedding
        if args.use_pos:
            self.embedding_pos = nn.Embedding(
                args.pos_vocab_size, args.pos_embedding_size, padding_idx=0
            )
            linear_features += 2 * args.pos_embedding_size

        ####### CLASSIFICATION HEAD #######
        self.first_layer = nn.Linear(
            in_features=linear_features, out_features=args.mlp_n_hidden
        )
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(args.mlp_dropout)

        self.last_layer = nn.Linear(in_features=args.mlp_n_hidden, out_features=1)

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

        # concatenate aggregate embeddings of sentence1 and sentence2
        sentence_vector = torch.cat((embeddings1, embeddings2), dim=-1)

        # use pos embeddings
        if self.args.use_pos:
            pos1 = batch["pos1"]
            pos2 = batch["pos2"]
            pos_embedding1 = self.embedding_pos(pos1)
            pos_embedding2 = self.embedding_pos(pos2)
            pos_embedding1 = self._forward_embedding(pos_embedding1, lengths1)
            pos_embedding2 = self._forward_embedding(pos_embedding2, lengths2)
            sentence_vector = torch.cat(
                (sentence_vector, pos_embedding1, pos_embedding2), dim=-1
            )

        # first linear layer
        out = self.first_layer(sentence_vector)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.last_layer(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)

    def _forward_embedding(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device)
            < lengths[..., None]
        )
        # compute mean of embeddings (excluding padded elements)
        embeddings = (embeddings * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        return embeddings
