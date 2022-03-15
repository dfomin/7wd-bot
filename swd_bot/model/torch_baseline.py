from swd.entity_manager import EntityManager
from torch import nn
import torch.nn.functional as F


class TorchBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(600, 100)
        self.linear3 = nn.Linear(100, EntityManager.cards_count() * 2 + EntityManager.wonders_count())
        self.linear4 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        policy = F.softmax(self.linear3(x), dim=1)
        value = F.softmax(self.linear4(x), dim=1)
        return policy, value
