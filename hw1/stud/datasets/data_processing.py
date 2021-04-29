import re
from typing import List, Tuple, Optional

import nltk
import torch
from nltk.corpus import stopwords
from stud.datasets.pos import pos_indexes

nltk.data.path.append("model/nltk_data")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
set_stopwords = set(stopwords.words())


def custom_tokenizer(sentence: str, marker: str) -> List[str]:
    """Tokenize string, removing marker word from the sentence."""
    tokens = sentence.split()
    for i, tk in enumerate(tokens):
        if marker in tk:
            target_position = i
            tokens[i] = tk[len(marker) :]
    return tokens, target_position


def preprocess(
    sentence: str,
    target_word: str,
    remove_stopwords: bool,
    remove_digits: bool,
) -> str:
    """Preprocess text data: lowercase text, remove punctuation.
    Removes digits and stopwords if requested."""
    # lowercase sentence
    sentence = sentence.lower()
    # remove punctuation
    sentence = re.sub(r"[^\w\s]", " ", sentence)

    # remove digits
    if remove_digits:
        sentence = re.sub(r"\d", "", sentence).strip()
    # replace multiple adjacent spaces with one single space
    sentence = re.sub(" +", " ", sentence).strip()

    # remove stopwords
    if remove_stopwords:
        tokens = sentence.split()
        tokens = [
            word
            for word in tokens
            if (not word in set_stopwords or word == target_word)
        ]
        sentence = " ".join(tokens)

    return sentence


def get_neighbourhood(
    tokens: List[str],
    target_position: int,
    width: int,
) -> Tuple[List[str], int]:
    """Keep window of `target_position` around the target word.
    Edge cases are still functioning: e.g. if the word is in first
    or last position, `get_neighbourhood` stops there (or if there
    aren't enough words before/after the target word)."""
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
    """Convert list of tokens to list of indexes, using the index vocabulary."""
    return torch.tensor([word_index[word] for word in tokens], dtype=torch.long)


def compute_pos_tag_indexes(tokens: List[str]) -> torch.Tensor:
    """Use pretrained nltk pos-tagger to compute part-of-speech tags for
    a set of tokens."""
    tks_tags = nltk.pos_tag(tokens)
    indexes = torch.tensor([pos_indexes[tk_tag[1]] for tk_tag in tks_tags])
    return indexes
