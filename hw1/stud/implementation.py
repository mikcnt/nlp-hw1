import pickle
import random
import string
from typing import Dict, List

import nltk
import numpy as np
import torch
from torch import nn
from model import Model
from torch.utils.data import DataLoader

from stud.datasets.dataset import WiCDataset
from stud.models.bilinear import BilinearClassifier
from stud.utils import index_dictionary, load_pickle

nltk.data.path.append("model/nltk_data")


def batch_to_device(
    batch: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
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

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [
            str(np.random.choice(self._options, 1, p=self._weights)[0])
            for x in sentence_pairs
        ]


class StudentModel:
    def __init__(self, device):
        super().__init__()
        self.device = device

        class Args:
            # general parameters
            save_labels = False
            sentence_embedding_size = 300

            # dataset parameters
            remove_stopwords = True
            remove_digits = True
            remove_target_word = False
            target_window = None

            # model parameters
            model_type = "BILINEAR"
            use_pretrained_embeddings = True
            use_pos = False

            if model_type == "BILINEAR":
                bi_n_features = 300
                bi_n_hidden = 400
                bi_dropout = 0.3

        # args for dataset and model
        self.args = Args()
        # vocabulary
        self.word_vectors = load_pickle("model/vocabulary.pkl")
        # vocabulary with indexes
        self.word_index, self.vectors_store = index_dictionary(self.word_vectors)

        # model definition
        self.model = BilinearClassifier(self.vectors_store, self.args).to(self.device)

        # load model weights
        weights = torch.load("model/002.pth", map_location=self.device)
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.eval()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        dataset = WiCDataset(sentence_pairs, self.word_index, self.args)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        predicted = []
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            pred = self.model(batch).round()
            predicted.append(pred)
        predicted = torch.cat(predicted, dim=0)

        return [str(bool(x)) for x in predicted.reshape(-1)]
