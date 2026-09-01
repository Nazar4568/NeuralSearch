import torch.nn as nn
import torch

class InfoNCELoss(nn.Module):
    """
    Implements the InfoNCE (Multiple Negatives Ranking) Loss.

    This loss function treats retrieval as a classification problem over a batch.
    For each query, the positive document is the true class, and all other documents
    in the batch act as negative examples.
    """
    def __init__(self,temperature:float = 0.05 ):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive loss between query and document embeddings.

        Args:
            query_embeddings (torch.Tensor): Tensor of shape [batch_size, d_model].
            doc_embeddings (torch.Tensor): Tensor of shape [batch_size, d_model].

        Returns:
            torch.Tensor: A scalar tensor containing the computed loss.
        """
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        scores = (query_embeddings @ doc_embeddings.T)/self.temperature
        labels = torch.arange(query_embeddings.size(0), device=scores.device)
        return torch.nn.functional.cross_entropy(scores,labels)
