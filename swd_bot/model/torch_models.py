import torch
from swd.entity_manager import EntityManager
from torch import nn
import torch.nn.functional as F


class TorchBaseline(nn.Module):
    def __init__(self, game_features_count: int, cards_features_count: int):
        super().__init__()

        self.linear1 = nn.Linear(game_features_count, 50)
        self.linear2 = nn.Linear(50, EntityManager.cards_count() * 2 + EntityManager.wonders_count())
        self.linear3 = nn.Linear(50, 2)

    def forward(self, features, cards):
        x = features
        x = F.relu(self.linear1(x))
        policy = F.softmax(self.linear2(x), dim=1)
        value = F.softmax(self.linear3(x), dim=1)
        return policy, value


class TorchBaselineEmbeddings(nn.Module):
    def __init__(self, game_features_count: int, cards_features_count: int):
        super().__init__()

        self.linear1 = nn.Linear(game_features_count, 50)
        self.linear2 = nn.Linear(cards_features_count, 10)
        self.linear3 = nn.Linear(110, EntityManager.cards_count() * 2 + EntityManager.wonders_count())
        self.linear4 = nn.Linear(110, 2)

    def forward(self, features, cards):
        features = F.relu(self.linear1(features))
        cards = F.relu(self.linear2(cards)).flatten(1)
        x = torch.cat((features, cards), dim=1)
        policy = F.softmax(self.linear3(x), dim=1)
        value = F.softmax(self.linear4(x), dim=1)
        return policy, value
