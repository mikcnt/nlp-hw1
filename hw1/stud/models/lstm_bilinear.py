import torch
from torch import nn

from typing import Dict

from stud.utils import text_length


class LstmBilinearClassifier(nn.Module):
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
            self.embedding_words = nn.Embedding.from_pretrained(
                vectors_store,
                padding_idx=0,
            )
        else:
            self.embedding_words = nn.Embedding(
                len(vectors_store), args.sentence_embedding_size, padding_idx=0
            )

        # POS embedding
        if args.use_pos:
            self.embedding_pos = nn.Embedding(
                args.pos_vocab_size, args.pos_embedding_size, padding_idx=0
            )

        ####### RECURRENT LAYERS #######
        # sentence recurrent layers
        self.rnn_sentence = nn.LSTM(
            input_size=args.sentence_embedding_size,
            hidden_size=args.sentence_n_hidden,
            num_layers=args.sentence_num_layers,
            bidirectional=args.sentence_bidirectional,
            dropout=args.sentence_dropout,
            batch_first=True,
        )

        recurrent_output_size = (
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

            pos_recurrent_output_size = (
                args.pos_n_hidden
                if not args.pos_bidirectional
                else args.pos_n_hidden * 2
            )

            self.pos_bilinear_layer = nn.Bilinear(
                pos_recurrent_output_size, pos_recurrent_output_size, args.pos_n_hidden
            )

        ####### CLASSIFICATION HEAD #######
        self.bilinear_layer = nn.Bilinear(
            recurrent_output_size, recurrent_output_size, args.sentence_n_hidden
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.linear_dropout)
        # self.middle_layer = nn.Linear(args.sentence_n_hidden, args.sentence_n_hidden)
        self.final_layer = nn.Linear(args.sentence_n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # extract elements from batch
        sentence1 = batch["sentence1"]
        sentence2 = batch["sentence2"]

        # compute sentences length
        lengths1 = text_length(sentence1).to("cpu")
        lengths2 = text_length(sentence2).to("cpu")

        # text embeddings and LSTM
        sentence_embeddings1 = self.embedding_words(sentence1)
        sentence_embeddings2 = self.embedding_words(sentence2)

        sentence_lstm_out1 = self._rnn_forward(
            self.rnn_sentence, sentence_embeddings1, lengths1
        )
        sentence_lstm_out2 = self._rnn_forward(
            self.rnn_sentence, sentence_embeddings2, lengths2
        )

        # pos embedding and LSTM
        if self.args.use_pos:
            pos1 = batch["pos1"]
            pos2 = batch["pos2"]
            pos_embedding1 = self.embedding_pos(pos1)
            pos_embedding2 = self.embedding_pos(pos2)

            pos_lstm_out1 = self._rnn_forward(self.rnn_pos, pos_embedding1, lengths1)
            pos_lstm_out2 = self._rnn_forward(self.rnn_pos, pos_embedding2, lengths2)

        # bilinear and fully connected pass
        out = self.bilinear_layer(sentence_lstm_out1, sentence_lstm_out2)
        out = self.activation(out)
        out = self.dropout(out)

        # out = self.middle_layer(out)
        # out = self.activation(out)
        # out = self.dropout(out)

        out = self.final_layer(out).squeeze(1)
        out = self.sigmoid(out)

        return out

    def _rnn_forward(
        self, rnn_layer: nn.Module, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
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