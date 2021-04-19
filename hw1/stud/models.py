import torch
from torch import nn

from typing import List, Tuple, Optional, Dict


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vectors_store: torch.Tensor,
        n_hidden: int,
        num_layers: int,
        bidirectional: bool = True,
        lstm_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.vectors_store = vectors_store
        self.input_size = vectors_store.size(1)
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm_dropout = lstm_dropout

        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(
            vectors_store,
            padding_idx=0,
        )

        # recurrent layer
        self.rnn = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.n_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.lstm_dropout,
        )

        linear_features = 2 * n_hidden

        if self.bidirectional:
            linear_features *= 2

        # classification head
        self.lin1 = torch.nn.Linear(linear_features, linear_features)
        self.lin2 = torch.nn.Linear(linear_features, 1)

        self.dropout = nn.Dropout(0.3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, sentence1, sentence2):
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