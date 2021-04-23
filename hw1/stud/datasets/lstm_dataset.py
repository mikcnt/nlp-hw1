from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict
import jsonlines

import torch
from torch import nn

from .data_processing import (
    preprocess,
    custom_tokenizer,
    get_neighbourhood,
    tokens2indices,
    compute_pos_tag_indexes,
)


class IndicesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        word_index: Dict[str, int],
        marker: str,
        args,
    ) -> None:
        self.marker = marker
        self.word_index = word_index
        self.args = args

        self.create_dataset(dataset_path)

    def create_dataset(self, dataset_path: str) -> None:
        sentences1 = []
        sentences2 = []
        labels = []

        lemma_indexes1 = []
        lemma_indexes2 = []

        pos_indexes1 = []
        pos_indexes2 = []

        with jsonlines.open(dataset_path, "r") as f:
            for _, line in enumerate(f.iter()):
                # load sentences
                start1 = int(line["start1"])
                start2 = int(line["start2"])
                end1 = int(line["end1"])
                end2 = int(line["end2"])
                s1 = line["sentence1"]
                s2 = line["sentence2"]
                target_word1 = s1[start1:end1]
                target_word2 = s2[start2:end2]

                label = str(line["label"])

                # insert special characters to locate target word after preprocessing
                s1 = s1[:start1] + self.marker + s1[start1:]
                s2 = s2[:start2] + self.marker + s2[start2:]

                # preprocessing
                s1 = preprocess(
                    s1,
                    target_word1,
                    self.args.remove_stopwords,
                    self.args.remove_digits,
                )
                s2 = preprocess(
                    s2,
                    target_word2,
                    self.args.remove_stopwords,
                    self.args.remove_digits,
                )

                # tokenization
                t1, target_position1 = custom_tokenizer(s1, self.marker)
                t2, target_position2 = custom_tokenizer(s2, self.marker)

                # remove target word
                if self.args.remove_target_word:
                    t1 = [t1[i] for i in range(len(t1)) if i != target_position1]
                    t2 = [t2[i] for i in range(len(t2)) if i != target_position2]
                
                # POS indexes
                pos1 = compute_pos_tag_indexes(t1)
                pos2 = compute_pos_tag_indexes(t2)

                # get neighbourhood of words
                if self.args.target_window is not None:
                    t1, target_position1 = get_neighbourhood(
                        t1, target_position1, self.args.target_window
                    )
                    t2, target_position2 = get_neighbourhood(
                        t2, target_position2, self.args.target_window
                    )

                # sentences to indices
                indices1 = tokens2indices(self.word_index, t1)
                indices2 = tokens2indices(self.word_index, t2)

                # label is either 1
                label = torch.tensor(1.0) if label == "True" else torch.tensor(0.0)

                # keep track of sentences, labels, pos tags
                sentences1.append(indices1)
                sentences2.append(indices2)

                labels.append(label)

                pos_indexes1.append(pos1)
                pos_indexes2.append(pos2)

        # pad all sentences with max length
        # (both sentences1 and sentences2 with same padding length)
        sentences1, sentences2 = self.pad_and_split(sentences1, sentences2)

        # pad all pos tags with max length
        pos_indexes1, pos_indexes2 = self.pad_and_split(pos_indexes1, pos_indexes2)

        # data = dictionaries containing
        # indexes for sentence1, sentence2, lemma1, lemma2
        # and label
        self.data = {
            idx: {
                "sentence1": sentences1[idx],
                "sentence2": sentences2[idx],
                "pos1": pos_indexes1[idx],
                "pos2": pos_indexes2[idx],
                "label": labels[idx],
            }
            for idx in range(len(sentences1))
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

    def pad_and_split(self, elements1, elements2):
        padded_elements = nn.utils.rnn.pad_sequence(
            elements1 + elements2, batch_first=True, padding_value=0
        )
        elements1 = padded_elements[: len(elements1)]
        elements2 = padded_elements[len(elements1) :]
        return elements1, elements2