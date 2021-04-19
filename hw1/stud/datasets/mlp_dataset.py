from typing import List, Tuple, Optional, Dict
import jsonlines

import torch
from torch import nn

from .data_processing import (
    preprocess,
    custom_tokenizer,
    get_neighbourhood,
)
from .manual_embedding import TokensEmbedder


class EmbeddedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, marker: str, embedder: TokensEmbedder, neigh_width: Optional[int] = None) -> None:
        self.data = {}
        self.marker = marker
        self.embedder = embedder
        self.neigh_width = neigh_width

        self.create_dataset(dataset_path)
        
    
    def create_dataset(self, dataset_path: str) -> None:
        with jsonlines.open(dataset_path, 'r') as f:
            for idx, line in enumerate(f.iter()):
                # load sentences
                start1 = int(line['start1'])
                start2 = int(line['start2'])
                s1 = line['sentence1']
                s2 = line['sentence2']
                # insert special characters to locate target word after preprocessing
                s1 = s1[:start1] + self.marker + s1[start1:]
                s2 = s2[:start2] + self.marker + s2[start2:]
                
                label = str(line["label"])
                
                # preprocessing
                s1 = preprocess(s1)
                s2 = preprocess(s2)
                
                # tokenization
                t1, target_position1 = custom_tokenizer(s1, self.marker)
                t2, target_position2 = custom_tokenizer(s2, self.marker)
                
                # if specified, get neighbourhood of target words
                # and recompute target position
                if self.neigh_width:
                    t1, target_position1 = get_neighbourhood(t1, target_position1, self.neigh_width)
                    t2, target_position2 = get_neighbourhood(t2, target_position2, self.neigh_width)
                
                # convert tokens to embeddings and aggregate
                v1 = self.embedder(t1, target_position1)
                v2 = self.embedder(t2, target_position2)
                
                # concatenate vectors
                sentence_vector = torch.cat((v1, v2))
                
                label = torch.tensor(1.) if label == "True" else torch.tensor(0.)
                self.data[idx] = {"sentence_vector": sentence_vector, "label": label}


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]