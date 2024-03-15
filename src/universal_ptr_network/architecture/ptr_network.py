from typing import Optional, Tuple, Union

import torch

from universal_ptr_network.architecture.decoder import SequenceDecoder
from universal_ptr_network.architecture.encoder import SequenceEncoder
from universal_ptr_network.feature_extractors.feature_extractor import FeatureExtractor


class PointerNetwork(torch.nn.Module):
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 hidden_size: int,
                 max_seq_len: Optional[int] = None,
                 only_uniques: bool = False) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor
        self._hidden_size = hidden_size
        self._max_seq_len = max_seq_len
        self._only_uniques = only_uniques

        self._sequence_encoder = SequenceEncoder(embedding_dim=feature_extractor.embedding_dim,
                                                 hidden_size=hidden_size)
        self._sequence_decoder = SequenceDecoder(hidden_size=hidden_size,
                                                 max_seq_len=max_seq_len,
                                                 only_uniques=only_uniques)

    def forward(self,
                inputs: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_encoded_sequence: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # TODO implement masking aware loss function

        if self._max_seq_len is not None:
            pass
            # TODO update mask with new column/row of zeros (unavailable tokens) if agile inference is required
            # TODO implement this functionality
        else:
            first_tokens = torch.zeros_like(inputs[:, 0, :]).unsqueeze(1).to(inputs.device)
            inputs = torch.cat([first_tokens, inputs], dim=1)
            # TODO update mask with new column/row of ones (available tokens)
            # TODO implement this functionality

        if attention_mask is None:
            attention_mask = torch.ones_like(inputs[:, :, 0]).to(inputs.device)

        element_embeddings = self._feature_extractor(inputs)
        encoded_elements, encoded_sequence = self._sequence_encoder(element_embeddings)

        probabilities, peak_indices = self._sequence_decoder(encoded_elements, encoded_sequence, attention_mask)

        if return_encoded_sequence:
            _, bidir_lstm_cell_state = encoded_sequence
            # merge bidir_lstm_cell_state with shape (2, batch_size, hidden_size) into (batch_size, 2 * hidden_size)
            reshaped_encoded_sequence = torch.cat([bidir_lstm_cell_state[0], bidir_lstm_cell_state[1]], dim=1)
            return probabilities, peak_indices, reshaped_encoded_sequence

        return probabilities, peak_indices
