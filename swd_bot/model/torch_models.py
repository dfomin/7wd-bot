from typing import List

import torch
from swd.entity_manager import EntityManager
from torch import nn
import torch.nn.functional as F


class TorchBaseline(nn.Module):
    def __init__(self,
                 game_features_count: int,
                 cards_features_count: int,
                 hidden_features_count: List[int]):
        super().__init__()

        prev_features = game_features_count
        layers = []
        for features_count in hidden_features_count:
            layers.append(nn.Linear(prev_features, features_count))
            layers.append(nn.LeakyReLU())
            prev_features = features_count

        self.backbone = nn.Sequential(*layers)
        self.head_policy = nn.Linear(prev_features, EntityManager.cards_count() * 2 + EntityManager.wonders_count())
        self.head_value = nn.Linear(prev_features, 2)

    def forward(self, features, cards):
        x = features
        x = self.backbone(x)
        policy = self.head_policy(x)
        value = self.head_value(x)
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
