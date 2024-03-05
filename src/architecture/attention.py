from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class Attention(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.decoder_query_weights = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.encoder_query_weights = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.value_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self,
                encoded_elements: torch.Tensor,
                encoded_sequence: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, time_steps, hidden_dim = encoded_elements.shape

        encoded_sequence_hidden_states = encoded_sequence[0].view(batch_size, -1)

        decoder_query = self.decoder_query_weights(encoded_sequence_hidden_states)
        encoder_query = self.encoder_query_weights(encoded_elements)

        comparison = decoder_query.unsqueeze(1) + encoder_query
        comparison = torch.tanh(comparison)
        attention_scores = self.value_weights(comparison).squeeze(2)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=1)

        return attention_scores
