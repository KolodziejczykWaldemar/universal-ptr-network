from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from universal_ptr_network.feature_extractors.feature_extractor import FeatureExtractor
from universal_ptr_network.feature_extractors.mlp import MLPFeatureExtractor


class ImageFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 input_shape_chw: Tuple[int, int, int],
                 hidden_size: int,
                 hidden_layers: int,
                 embedding_dim: int) -> None:
        """Feature extractor for image data with a convolutional neural network and MLP head.

        input_shape_chw (Tuple[int, int, int]): The shape of the input image in the format (C, H, W).
        hidden_size (int): The size of the hidden layers in the head.
        hidden_layers (int): The number of hidden layers in the head.
        embedding_dim (int): The size of the output embedding.
        """
        super().__init__()

        self._input_shape_chw = input_shape_chw
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers

        self._feature_extractor = ConvExtractor(input_shape_chw=input_shape_chw)
        flat_size = self._infer_flat_size()
        self._head = MLPFeatureExtractor(input_size=flat_size,
                                         hidden_size=self._hidden_size,
                                         hidden_layers=self._hidden_layers,
                                         output_size=self._embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, ,time_steps, channels, height, width)
        batch_size = x.size(0)
        x = x.view((-1, *self._input_shape_chw))
        # x.shape = (batch_size * time_steps, channels, height, width)
        x = self._feature_extractor(x)
        # x.shape = (batch_size * time_steps, conv_embedding_dim)
        x = self._head(x)
        # x.shape = (batch_size, time_steps, embedding_dim)
        x = x.view(batch_size, -1, self._embedding_dim)
        return x

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _infer_flat_size(self) -> int:
        with torch.no_grad():
            model_output = self._feature_extractor(torch.ones(1, *self._input_shape_chw))
        return int(np.prod(model_output.size()[1:]))


class ConvExtractor(nn.Module):
    def __init__(self,
                 input_shape_chw: Tuple[int, int, int]) -> None:
        """Convolutional feature extractor for image data.

        input_shape_chw (Tuple[int, int, int]): The input shape of the image in the form of (C, H, W)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape_chw[0], 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.shape[0], -1)
        return x

