from swd.entity_manager import EntityManager
from torch import nn
import torch.nn.functional as F


class TorchBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(1622, 300)
        self.linear2 = nn.Linear(300, EntityManager.cards_count() * 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        return x
