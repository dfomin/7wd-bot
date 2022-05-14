import torch
from swd.entity_manager import EntityManager
from torch import nn
import torch.nn.functional as F


class TorchBaseline(nn.Module):
    def __init__(self,
                 game_features_count: int,
                 cards_features_count: int,
                 features_hidden: int):
        super().__init__()

        self.linear1 = nn.Linear(game_features_count, features_hidden)
        self.linear2 = nn.Linear(features_hidden, EntityManager.cards_count() * 2 + EntityManager.wonders_count())
        self.linear3 = nn.Linear(features_hidden, 2)

    def forward(self, features, cards):
        x = features
        x = F.relu(self.linear1(x))
        policy = self.linear2(x)
        value = self.linear3(x)
        return policy, value


class TorchBaselineEmbeddings(nn.Module):
    def __init__(self,
                 game_features_count: int,
                 cards_features_count: int,
                 game_features_hidden: int,
                 cards_features_hidden: int):
        super().__init__()

        self.linear1 = nn.Linear(game_features_count, game_features_hidden)
        self.linear2 = nn.Linear(cards_features_count, cards_features_hidden)
        hidden = game_features_hidden + cards_features_hidden * 6
        self.linear3 = nn.Linear(hidden, EntityManager.cards_count() * 2 + EntityManager.wonders_count())
        self.linear4 = nn.Linear(hidden, 2)

    def forward(self, features, cards):
        features = F.relu(self.linear1(features))
        cards = F.relu(self.linear2(cards)).flatten(1)
        x = torch.cat((features, cards), dim=1)
        policy = self.linear3(x)
        value = self.linear4(x)
        return policy, value
