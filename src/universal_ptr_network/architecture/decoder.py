from typing import Optional, Tuple, List

import torch
from torch import nn

from universal_ptr_network.architecture.attention import Attention


class SequenceDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 only_uniques: bool = False,
                 max_seq_len: Optional[int] = None) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self._only_uniques = only_uniques
        self._max_seq_len = max_seq_len
        self.lstm = nn.LSTM(input_size=2 * self._hidden_size,
                            hidden_size=self._hidden_size,
                            bidirectional=True,
                            batch_first=True)
        self.attention = Attention(hidden_size=self._hidden_size)

    def forward(self,
                encoded_elements: torch.tensor,
                encoded_sequence: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor) -> Tuple[torch.tensor, List[int]]:
        return self._forward_max_steps(encoded_elements, encoded_sequence, attention_mask)

    def _forward_max_steps(self,
                           encoded_elements: torch.tensor,
                           encoded_sequence: Tuple[torch.Tensor, torch.Tensor],
                           attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = encoded_elements.shape
        overall_probabilities = []
        batch_identifier = torch.arange(batch_size, dtype=torch.long)

        # set initial token representing the start of the sequence
        decoder_input = torch.zeros_like(encoded_elements[:, 0, :]).unsqueeze(1)  # shape: (batch_size, 1, hidden_size)

        peak_indices = []
        for step in range(self._max_seq_len):
            _, encoded_sequence = self.lstm(decoder_input, encoded_sequence)

            # probabilities for some samples can be vectors of nan's if all elements are masked, i.e. attention_mask.sum(dim=1) == 0
            probabilities = self.attention(encoded_elements, encoded_sequence, attention_mask)

            _, peak_idx = probabilities.max(dim=1)

            peak_idx[attention_mask.sum(dim=1) == 0] = -1
            if self._only_uniques:
                attention_mask[batch_identifier, peak_idx] = 0

            decoder_input = encoded_elements[batch_identifier, peak_idx, :].unsqueeze(1)

            overall_probabilities.append(probabilities)
            peak_indices.append(peak_idx)

            if attention_mask.sum() == 0:
                break

        overall_probabilities = torch.stack(overall_probabilities).transpose(0, 1)
        peak_indices = torch.stack(peak_indices).t()
        return overall_probabilities, peak_indices
