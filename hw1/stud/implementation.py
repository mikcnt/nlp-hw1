import os
import pickle
import random
import string
from typing import Dict, List, Tuple

import nltk
import numpy as np
import torch
from model import Model
from torch import nn
from torch.utils.data import DataLoader

from stud.datasets.lstm_dataset import WiCDataset
from stud.models import BilinearClassifier
from stud.utils import index_dictionary, load_pickle

nltk.data.path.append("model/nltk_data")


def batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> List[torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_model(device: str) -> Model:
    return StudentModel(device)
    # return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ("True", 40000),
        ("False", 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [
            str(np.random.choice(self._options, 1, p=self._weights)[0])
            for x in sentence_pairs
        ]


class StudentModel:
    def __init__(self, device):
        super().__init__()

        class Args:
            # wandb
            save_wandb = False
            device = "cuda" if torch.cuda.is_available() else "cpu"
            save_labels = False

            # general parameters
            num_epochs = 15
            batch_size = 64
            lr = 0.0001
            weight_decay = 0.0001
            vocab_threshold = 0

            # dataset parameters
            remove_stopwords = True
            remove_digits = True
            remove_target_word = False
            target_window = None

            # model parameters
            model_type = "BILINEAR"
            use_pos = False

            # MLP Parameters
            if model_type == "MLP":
                mlp_n_features = 300
                mlp_num_layers = 2
                mlp_n_hidden = 512
                mlp_dropout = 0.3

            # LSTM Parameters
            if model_type == "LSTM":
                sentence_embedding_size = 300
                sentence_n_hidden = 512
                sentence_num_layers = 2
                sentence_bidirectional = True
                sentence_dropout = 0.3

            if model_type == "BILINEAR":
                bi_n_features = 300
                bi_n_hidden = 400
                bi_dropout = 0.3

            # POS parameters
            if use_pos:
                pos_embedding_size = 300
                pos_vocab_size = len(pos_all_tags)
                if model_type == "LSTM":
                    pos_n_hidden = 256
                    pos_num_layers = 2
                    pos_bidirectional = True
                    pos_dropout = 0.3

        # args for dataset and model
        self.args = Args()
        # vocabulary
        self.word_vectors = load_pickle("model/vocabulary.pkl")
        # vocabulary with indexes
        self.word_index, self.vectors_store = index_dictionary(self.word_vectors)

        # model definition
        self.model = BilinearClassifier(self.vectors_store, self.args).to(
            self.args.device
        )

        # load model weights
        weights = torch.load("model/002.pth", map_location=self.args.device)
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.eval()

        # dataset utils
        self.marker = "".join(random.choices(string.ascii_lowercase, k=20))

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        dataset = WiCDataset(sentence_pairs, self.word_index, self.marker, self.args)
        loader = DataLoader(dataset, batch_size=1)
        predicted = []
        for batch in loader:
            batch = batch_to_device(batch, self.args.device)
            pred = self.model(batch).round()
            predicted.append(pred)
        predicted = torch.stack(predicted, dim=0)
        # raise AssertionError("AOOOOOOOO {} AOOOOOOOOO".format(predicted))
        return [str(bool(x)) for x in predicted]
