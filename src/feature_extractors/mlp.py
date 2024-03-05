import torch
import torch.nn.functional as F
from torch import nn

from src.feature_extractors.feature_extractor import FeatureExtractor


class MLPFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 hidden_layers: int,
                 output_size: int) -> None:
        """Embedding head for the feature extractor.

        input_size (int): The size of the input tensor.
        hidden_size (int): The size of the hidden layers.
        hidden_layers (int): The number of hidden layers excluding the last layer.
        output_size (int): The size of the output tensor.
        """
        super().__init__()
        self._output_size = output_size
        self.linear_first = nn.Linear(input_size, hidden_size)
        self.middle_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(max(0, hidden_layers - 1))]
        )
        self.linear_last = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear_first(x))
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        x = self.linear_last(x)
        return x

    @property
    def embedding_dim(self) -> int:
        return self._output_size
