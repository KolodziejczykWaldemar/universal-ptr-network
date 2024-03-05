from typing import Tuple

import torch
from torch import nn


class SequenceEncoder(torch.nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_size: int) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self._embedding_dim,
            hidden_size=self._hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, raw_element_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = raw_element_embeddings.shape
        element_embeddings, encoded_sequence_hidden_states = self.lstm(raw_element_embeddings)
        return element_embeddings, encoded_sequence_hidden_states
