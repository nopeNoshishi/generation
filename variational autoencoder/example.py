import logging
import torch

from network import FeedForwardVae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = FeedForwardVae(28*28, 2)
# model.load_state_dict(torch.load('model_weight.pth'))

x = torch.rand(1, 28*28)
y = model(x)

