import torch
from torch import nn

from typing import List, Tuple, Optional, Dict, Callable

# MLP baseline model
class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_layers: int,
        hidden_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        linear_features = 2 * n_features
        self.first_layer = nn.Linear(
            in_features=linear_features, out_features=hidden_dim
        )

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            )

        self.activation = activation
        self.dropout = nn.Dropout(0.5)

        self.last_layer = nn.Linear(in_features=hidden_dim, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Dict[str, torch.tensor]) -> torch.Tensor:
        sentence_vector = batch["sentence_vector"]
        out = self.first_layer(
            sentence_vector
        )  # First linear layer, transforms the hidden dimensions from `n_features` (embedding dimension) to `hidden_dim`
        for layer in self.layers:  # Apply `k` (linear, activation) layer
            out = layer(out)
            out = self.activation(out)
            out = self.dropout(out)
        out = self.last_layer(
            out
        )  # Last linear layer to bring the `hiddem_dim` features to a binary space (`True`/`False`)

        out = self.sigmoid(out)
        return out.squeeze(-1)


# LSTM model
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vectors_store: torch.Tensor,
        n_hidden: int,
        num_layers: int,
        bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        use_lemma_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.vectors_store = vectors_store
        self.embedding_size = vectors_store.size(1)
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm_dropout = lstm_dropout
        self.use_lemma_embedding = use_lemma_embedding

        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(
            vectors_store,
            padding_idx=0,
        )

        # recurrent layer
        self.rnn = torch.nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.lstm_dropout,
        )

        linear_features = 2 * n_hidden

        if self.bidirectional:
            linear_features *= 2

        if self.use_lemma_embedding:
            linear_features += 2 * self.embedding_size

        # classification head
        self.lin1 = torch.nn.Linear(linear_features, linear_features)
        self.lin2 = torch.nn.Linear(linear_features, 1)

        self.dropout = nn.Dropout(0.3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Dict[str, torch.Tensor]):
        sentence1 = batch["sentence1"]
        sentence2 = batch["sentence2"]

        # compute sentences length
        lengths1 = self._text_length(sentence1)
        lengths2 = self._text_length(sentence2)

        # text embeddings
        embeddings1 = self.embedding(sentence1)
        embeddings2 = self.embedding(sentence2)

        # sentences1
        lstm_out1 = self._lstm_output(embeddings1, lengths1)

        # sentences2
        lstm_out2 = self._lstm_output(embeddings2, lengths2)

        # concatenate lstm outputs of both sentences
        lstm_out = torch.cat((lstm_out1, lstm_out2), dim=-1)

        if self.use_lemma_embedding:
            lemma1 = batch["lemma1"]
            lemma2 = batch["lemma2"]
            # lemma embeddings
            lemma_embedding1 = self.embedding(lemma1)
            lemma_embedding2 = self.embedding(lemma2)
            # concatenate target word embeddings embeddings
            lstm_out = torch.cat((lstm_out, lemma_embedding1, lemma_embedding2), dim=-1)

        # linear pass
        out = self.lin1(lstm_out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.lin2(out).squeeze(1)
        out = self.sigmoid(out)

        return out

    def _text_length(self, sentences: torch.Tensor) -> torch.Tensor:
        # search first zero
        lengths = (sentences == 0).int().argmax(axis=1).to("cpu")
        # length 0 only if sentence has max length
        # => replace 0 with max length
        lengths[lengths == 0] = sentences.shape[-1]
        return lengths

    def _lstm_output(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # pack input
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        # apply lstm
        packed_output, _ = self.rnn(packed_input)

        # pad packed batch
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # retrieve forward output of lstm
        out_forward = output[range(len(output)), lengths - 1, : self.n_hidden]

        # retrieve reverse output of lstm
        if self.bidirectional:
            out_reverse = output[:, 0, self.n_hidden :]
            out_reduced = torch.cat((out_forward, out_reverse), 1)
        else:
            out_reduced = out_forward

        return out_reduced