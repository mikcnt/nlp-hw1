from typing import List, Tuple, Optional, Dict
import jsonlines

import torch
from torch import nn

from .data_processing import (
    preprocess,
    custom_tokenizer,
    get_neighbourhood,
    tokens2indices,
)


class IndicesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        word_index: Dict[str, int],
        marker: str,
        padding_value: int = 0,
        neigh_width: Optional[int] = None,
    ):
        self.marker = marker
        self.word_index = word_index
        self.padding_value = padding_value
        self.neigh_width = neigh_width

        self.create_dataset(dataset_path)

    def create_dataset(self, dataset_path: str) -> None:
        sentences1 = []
        sentences2 = []
        labels = []

        with jsonlines.open(dataset_path, "r") as f:
            for _, line in enumerate(f.iter()):
                # load sentences
                start1 = int(line["start1"])
                start2 = int(line["start2"])
                s1 = line["sentence1"]
                s2 = line["sentence2"]

                # insert special characters to locate target word after preprocessing
                s1 = s1[:start1] + self.marker + s1[start1:]
                s2 = s2[:start2] + self.marker + s2[start2:]

                # preprocessing
                s1 = preprocess(s1)
                s2 = preprocess(s2)

                # tokenization
                t1, target_position1 = custom_tokenizer(s1, self.marker)
                t2, target_position2 = custom_tokenizer(s2, self.marker)

                # get neighbourhood of words
                if self.neigh_width:
                    t1, target_position1 = get_neighbourhood(
                        t1, target_position1, self.neigh_width
                    )
                    t2, target_position2 = get_neighbourhood(
                        t2, target_position2, self.neigh_width
                    )

                # sentences to indices
                indices1 = tokens2indices(self.word_index, t1)
                indices2 = tokens2indices(self.word_index, t2)

                # label is either 1
                label = (
                    torch.tensor(1.0) if line["label"] == "True" else torch.tensor(0.0)
                )

                # keep track of sentences and labels
                sentences1.append(indices1)
                sentences2.append(indices2)
                labels.append(label)

        # pad all sentences with max length
        # (both sentences1 and sentences2 with same padding length)
        padded_sentences = nn.utils.rnn.pad_sequence(
            sentences1 + sentences2, batch_first=True, padding_value=self.padding_value
        )

        # split back again sentences1 and sentences2
        sentences1 = padded_sentences[: len(sentences1)]
        sentences2 = padded_sentences[len(sentences1) :]

        # data = tuples of sentence1, sentence2 and corresponding label
        self.data = list(zip(sentences1, sentences2, labels))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]