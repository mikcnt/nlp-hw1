from typing import List, Tuple

import re

import nltk
from nltk.corpus import stopwords

import torch

set_stopwords = set(stopwords.words())


def custom_tokenizer(sentence: str, marker: str) -> List[str]:
    tokens = sentence.split()
    for i, tk in enumerate(tokens):
        if marker in tk:
            target_position = i
            tokens[i] = tk[20:]
    return tokens, target_position


def preprocess(sentence: str, target_word=None) -> str:
    # lowercase sentence
    sentence = sentence.lower()
    # remove punctuation
    sentence = re.sub(r"[^\w\s]", " ", sentence)
    # replace multiple adjacent spaces with one single space
    sentence = re.sub(" +", " ", sentence).strip()

    tokens = sentence.split()
    tokens_sw = [
        word for word in tokens if (not word in set_stopwords or word == target_word)
    ]

    return " ".join(tokens_sw)


def get_neighbourhood(
    tokens: List[str], target_position: int, width: int = 2
) -> Tuple[List[str], int]:
    neighbourhood = []
    new_position = width

    for pos in range(target_position - width, target_position + width + 1):
        if pos < 0:
            new_position -= 1
            continue
        if pos >= len(tokens):
            continue
        neighbourhood.append(tokens[pos])

    return neighbourhood, new_position


def tokens2indices(word_index, tokens: List[str]) -> torch.Tensor:
    return torch.tensor([word_index[word] for word in tokens], dtype=torch.long)