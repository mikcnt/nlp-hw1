import numpy as np
from typing import List, Tuple, Dict

from model import Model

import pickle
import torch
from torch import nn
import os

def build_model(device: str) -> Model:
    # return StudentModel().to(device)
    return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Model()
        weights = torch.load(
            "model/022.pth", map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(weights["model_state_dict"])
        with open('model/vocabulary.pkl', 'rb') as f:
            self.vocabulary = pickle.load(f)

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        sentences = []
        for sentence in sentence_pairs:
            s1 = self.sentence2vector(sentence['sentence1'])
            s2 = self.sentence2vector(sentence['sentence2'])
            sentence_vector = torch.cat((s1, s2)).unsqueeze(0)
            sentences.append(sentence_vector)
        sentences = torch.stack(sentences)
        self.model.eval()
        predicted = self.model(sentences).round().int()
        return [str(bool(x.item())) for x in predicted]
    
    def sentence2vector(self, sentence: str):
        sentences_word_vector = [self.vocabulary[w] for w in sentence.split(' ') if w in self.vocabulary]
        
        if len(sentences_word_vector) == 0:
            return None

        sentences_word_vector = torch.stack(sentences_word_vector)
        return torch.mean(sentences_word_vector, dim=0)