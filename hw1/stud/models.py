import torch
from torch import nn

from typing import List, Tuple, Optional, Dict, Callable


def text_length(sentences: torch.Tensor) -> torch.Tensor:
    # search first zero
    lengths = (sentences == 0).int().argmax(axis=1).to("cpu")
    # length 0 only if sentence has max length
    # => replace 0 with max length
    lengths[lengths == 0] = sentences.shape[-1]
    return lengths


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


class MlpClassifier(nn.Module):
    def __init__(
        self,
        vectors_store: List[torch.Tensor],
        n_features: int,
        num_layers: int,
        hidden_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            vectors_store,
            padding_idx=0,
        )

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
        sentence1 = batch["sentence1"]
        sentence2 = batch["sentence2"]

        # compute sentences length
        lengths1 = text_length(sentence1)
        lengths2 = text_length(sentence2)

        # text embeddings
        embeddings1 = self.embedding(sentence1)
        embeddings2 = self.embedding(sentence2)

        mask1 = (
            torch.arange(embeddings1.shape[1], device=embeddings2.device)
            < lengths1[..., None]
        )
        embeddings1 = (embeddings1 * mask1.unsqueeze(-1)).sum(1) / lengths1.unsqueeze(
            -1
        )

        mask2 = (
            torch.arange(embeddings2.shape[1], device=embeddings2.device)
            < lengths2[..., None]
        )
        embeddings2 = (embeddings2 * mask2.unsqueeze(-1)).sum(1) / lengths2.unsqueeze(
            -1
        )

        sentence_vector = torch.cat((embeddings1, embeddings2), dim=-1)

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
class LstmClassifier(nn.Module):
    def __init__(
        self,
        vectors_store: torch.Tensor,
        args,
    ) -> None:
        super().__init__()
        self.args = args

        # instantiate linear features as 0, add output dim for each lstm layer
        recurrent_output_size = 0

        ####### EMBEDDING LAYERS #######
        # sentence embedding
        self.embedding_words = torch.nn.Embedding.from_pretrained(
            vectors_store,
        )

        # POS embedding
        if args.use_pos:
            self.embedding_pos = nn.Embedding(
                args.pos_vocab_size, args.pos_embedding_size, padding_idx=0
            )

        ####### RECURRENT LAYERS #######
        # sentence recurrent layers
        self.rnn_sentence = torch.nn.LSTM(
            input_size=args.sentence_embedding_size,
            hidden_size=args.sentence_n_hidden,
            num_layers=args.sentence_num_layers,
            bidirectional=args.sentence_bidirectional,
            dropout=args.sentence_dropout,
            batch_first=True,
        )

        recurrent_output_size += 2 * (
            args.sentence_n_hidden
            if not args.sentence_bidirectional
            else args.sentence_n_hidden * 2
        )

        # pos recurrent layers
        if args.use_pos:
            self.rnn_pos = nn.LSTM(
                input_size=args.pos_embedding_size,
                hidden_size=args.pos_n_hidden,
                num_layers=args.pos_num_layers,
                bidirectional=args.pos_bidirectional,
                dropout=args.pos_dropout,
                batch_first=True,
            )

            recurrent_output_size += 2 * (
                args.pos_n_hidden
                if not args.pos_bidirectional
                else args.pos_n_hidden * 2
            )

        ####### CLASSIFICATION HEAD #######
        self.lin1 = torch.nn.Linear(recurrent_output_size, recurrent_output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lin2 = torch.nn.Linear(recurrent_output_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # extract elements from batch
        sentence1 = batch["sentence1"]
        sentence2 = batch["sentence2"]
        pos1 = batch["pos1"]
        pos2 = batch["pos2"]

        # compute sentences length
        lengths1 = text_length(sentence1)
        lengths2 = text_length(sentence2)

        # text embeddings and LSTM
        sentence_embeddings1 = self.embedding_words(sentence1)
        sentence_embeddings2 = self.embedding_words(sentence2)

        sentence_lstm_out1 = self._rnn_forward(
            self.rnn_sentence, sentence_embeddings1, lengths1
        )
        sentence_lstm_out2 = self._rnn_forward(
            self.rnn_sentence, sentence_embeddings2, lengths2
        )

        # concatenate sentence lstm outputs
        out = torch.cat((sentence_lstm_out1, sentence_lstm_out2), dim=-1)

        # pos embedding and LSTM
        if self.args.use_pos:
            pos_embedding1 = self.embedding_pos(pos1)
            pos_embedding2 = self.embedding_pos(pos2)

            sentence_lstm_out1 = self._rnn_forward(
                self.rnn_pos, pos_embedding1, lengths1
            )
            sentence_lstm_out2 = self._rnn_forward(
                self.rnn_pos, pos_embedding2, lengths2
            )

            # concatenate previous output and lstm outputs on pos embeddings
            out = torch.cat((out, sentence_lstm_out1, sentence_lstm_out2), dim=-1)

        # linear pass
        out = self.lin1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.lin2(out).squeeze(1)
        out = self.sigmoid(out)

        return out

    def _rnn_forward(self, rnn_layer, embeddings, lengths):
        # pack input
        n_hidden = rnn_layer.hidden_size
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        # apply rnn
        packed_output, _ = rnn_layer(packed_input)

        # pad packed batch
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # retrieve forward output of rnn
        out_forward = output[range(len(output)), lengths - 1, :n_hidden]

        # retrieve reverse output of rnn
        if rnn_layer.bidirectional:
            out_reverse = output[:, 0, n_hidden:]
            out_reduced = torch.cat((out_forward, out_reverse), 1)
        else:
            out_reduced = out_forward

        return out_reduced