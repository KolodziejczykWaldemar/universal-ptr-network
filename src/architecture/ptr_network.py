from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from src.architecture.feature_extractors.feature_extractor import FeatureExtractor
from src.architecture.feature_extractors.image_feature_extractor import ImageFeatureExtractor, Head


class PointerNetwork(torch.nn.Module):
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 embedding_dim: int,
                 hidden_size: int,
                 max_seq_len: Optional[int] = None,
                 only_uniques: bool = False) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._max_seq_len = max_seq_len
        self._only_uniques = only_uniques

        self._sequence_encoder = SequenceEncoder(embedding_dim=embedding_dim,
                                                 hidden_size=hidden_size)
        self._sequence_decoder = SequenceDecoder(hidden_size=hidden_size,
                                                 max_seq_len=max_seq_len,
                                                 only_uniques=only_uniques)

        self._attention = None

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._max_seq_len is not None:
            pass
            # TODO update mask with new column/row of zeros (unavailable tokens) if agile inference is required
            # TODO implement this functionality
        else:
            first_tokens = torch.zeros_like(x[:, 0, :]).unsqueeze(1)
            x = torch.cat([first_tokens, x], dim=1)
            # TODO update mask with new column/row of ones (available tokens)
            # TODO implement this functionality

        if attention_mask is None:
            attention_mask = torch.ones_like(x[:, :, 0])

        element_embeddings = self._feature_extractor(x)
        encoded_elements, encoded_sequence = self._sequence_encoder(element_embeddings)

        probabilities, peak_indices = self._sequence_decoder(encoded_elements, encoded_sequence, attention_mask)

        return probabilities, peak_indices


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
        for step in range(min(time_steps, self._max_seq_len)):
            _, encoded_sequence = self.lstm(decoder_input, encoded_sequence)

            if attention_mask.sum() == 0:
                break

            # probabilities for some samples can be vectors of nan's if all elements are masked, i.e. attention_mask.sum(dim=1) == 0
            probabilities = self.attention(encoded_elements, encoded_sequence, attention_mask)

            _, peak_idx = probabilities.max(dim=1)

            peak_idx[attention_mask.sum(dim=1) == 0] = -1
            if self._only_uniques:
                attention_mask[batch_identifier, peak_idx] = 0

            decoder_input = encoded_elements[batch_identifier, peak_idx, :].unsqueeze(1)

            overall_probabilities.append(probabilities)
            peak_indices.append(peak_idx)

        overall_probabilities = torch.stack(overall_probabilities).transpose(0, 1)
        peak_indices = torch.stack(peak_indices).t()
        return overall_probabilities, peak_indices


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


if __name__ == '__main__':
    x = torch.randn(2, 10, 1, 40, 45)
    feature_extractor = ImageFeatureExtractor(input_shape_chw=(1, 40, 45),
                                              hidden_size=20,
                                              hidden_layers=2,
                                              embedding_dim=64)

    x = torch.randn(2, 10, 1)

    # sample attention mask for trimming the sequence
    attention_mask = torch.ones(2, 10)  # shape: (batch_size, time_steps)
    attention_mask[0, 5:] = 0
    attention_mask[1, 7:] = 0

    mlp_feature_extractor = Head(input_size=1,
                                 hidden_size=20,
                                 hidden_layers=2,
                                 output_size=64)

    ptr_network = PointerNetwork(feature_extractor=mlp_feature_extractor,
                                 embedding_dim=64,
                                 hidden_size=50,
                                 max_seq_len=10,
                                 only_uniques=True)
    probabilities, peak_indices = ptr_network.forward(x, attention_mask=attention_mask)
    print(probabilities.shape)
    print(peak_indices.shape)
