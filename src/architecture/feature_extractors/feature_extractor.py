import torch


class FeatureExtractor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError
