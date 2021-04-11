from typing import Dict, List
import torch

class TokensEmbedder(object):
    def __init__(self, word_vectors: Dict[str, torch.Tensor]) -> None:
        self.word_vectors = word_vectors
    
    def compute_embeddings(self, tokens: List[str]) -> List[torch.Tensor]:
        word_embeddings = []
        for w in tokens:
            word_embeddings.append(self.word_vectors[w] if w in self.word_vectors else self.word_vectors['<unk>'])
        
        return word_embeddings
    
    def aggregate_embeddings(self, tokens: List[str], target_position: int) -> torch.Tensor:
        pass
    
    def __call__(self, tokens: List[str], target_position: int) -> torch.Tensor:
        return self.aggregate_embeddings(tokens, target_position)

class AverageEmbedder(TokensEmbedder):
    def __init__(self, word_vectors: Dict[str, torch.Tensor]) -> None:
        self.word_vectors = word_vectors
    
    def aggregate_embeddings(self, tokens: List[str], target_position: int) -> torch.Tensor:
        embeddings = torch.stack(self.compute_embeddings(tokens))
        return torch.mean(embeddings, dim=0)

class WeightedAverageEmbedder(TokensEmbedder):
    def __init__(self, word_vectors: Dict[str, torch.Tensor]) -> None:
        self.word_vectors = word_vectors
    
    def aggregate_embeddings(self, tokens: List[str], target_position: int) -> torch.Tensor:
        embeddings = torch.stack(self.compute_embeddings(tokens))
        # aliases for readibility
        n = len(embeddings)
        t = target_position

        # weights from 1 to 0
        weights = torch.linspace(5, 0.1, n).unsqueeze(1)

        # weighted vector
        new_vectors = embeddings

        # weighted average

        # right of the target word
        new_vectors[t:] = new_vectors[t:] * weights[:n - t]
        # left of the target word
        new_vectors[:t] = new_vectors[:t] * reversed(weights[1:t + 1])

        # denominator (sum of the weights)
        weights_sum = weights[:n - t].sum() + weights[1:t + 1].sum()

        return new_vectors.sum(dim=0) / weights_sum