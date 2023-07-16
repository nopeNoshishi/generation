import sys
sys.path.append('./network')

from ff_vae import FeedForwardVae
from cnn_vae import CnnVae

__all__ = [
    FeedForwardVae,
    CnnVae
]
