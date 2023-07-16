from typing import TypeVar, Tuple

import torch
from torch import nn

Tensor = TypeVar('torch.tensor')


class BaseModel(nn.Module):
    
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def encode(self, input: Tensor) -> Tuple[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Tensor:
        raise NotImplementedError
