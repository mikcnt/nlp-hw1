from typing import List

import jsonlines
import torch
import re

from .embedder import TokensEmbedder

# Preprocess text data (lowercase, remove punctuation, remove multiple white spaces)
def preprocess(sentence: str) -> str:
    # lowercase sentence
    sentence = sentence.lower()
    # remove punctuation
    sentence = re.sub('[^\w\s]', ' ', sentence)
    # replace multiple adjacent spaces with one single space
    sentence = re.sub(' +', ' ', sentence).strip()
    return sentence

# Tokenize sentence and retrieve target word position
def custom_tokenizer(sentence: str, marker: str) -> List[str]:
    tokens = sentence.split()
    for i, tk in enumerate(tokens):
        if marker in tk:
            target_position = i
            tokens[i] = tk[20:]
    return tokens, target_position


# Main class for the WiC dataset
class WiCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, marker: str, embedder: TokensEmbedder) -> None:
        self.data = []
        self.marker = marker
        self.embedder = embedder

        self.create_dataset(dataset_path)
        
    
    def create_dataset(self, dataset_path: str) -> None:
        with jsonlines.open(dataset_path, 'r') as f:
            for line in f.iter():
                # load sentences
                start1 = int(line['start1'])
                start2 = int(line['start2'])
                s1 = line['sentence1']
                s2 = line['sentence2']
                # insert special characters to locate target word after preprocessing
                s1 = s1[:start1] + self.marker + s1[start1:]
                s2 = s2[:start2] + self.marker + s2[start2:]
                
                # preprocessing
                s1 = preprocess(s1)
                s2 = preprocess(s2)
                
                # tokenization
                t1, target_position1 = custom_tokenizer(s1, self.marker)
                t2, target_position2 = custom_tokenizer(s2, self.marker)
                
                # convert tokens to embeddings and aggregate
                v1 = self.embedder(t1, target_position1)
                v2 = self.embedder(t2, target_position2)
                
                # concatenate vectors
                sentence_vector = torch.cat((v1, v2))
                
                label = torch.tensor(1, dtype=torch.float32) if line['label'] == 'True' else torch.tensor(0, dtype=torch.float32)
                self.data.append((sentence_vector, label))


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]