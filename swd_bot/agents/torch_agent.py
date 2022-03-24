import random
from typing import Sequence

import numpy as np
import torch
from swd.action import Action, BuyCardAction, DiscardCardAction, BuildWonderAction
from swd.agents import Agent
from swd.entity_manager import EntityManager
from swd.states.game_state import GameState, GameStatus

from swd_bot.data_providers.feature_extractor import FlattenFeatureExtractor
from swd_bot.model.torch_models import TorchBaseline


class TorchAgent(Agent):
    def __init__(self):
        self.model = TorchBaseline(600, 0)
        self.model.load_state_dict(torch.load("../../models/model_flat_acc54.67.pth"))
        self.model.eval()

        self.feature_extractor = FlattenFeatureExtractor()

    def choose_action(self, state: GameState, possible_actions: Sequence[Action]) -> Action:
        if state.game_status != GameStatus.NORMAL_TURN:
            return random.choice(possible_actions)
        features, cards = self.feature_extractor.features(state)

        pred_actions, pred_winners = self.model(torch.FloatTensor(features)[None], torch.FloatTensor(cards)[None])
        action_predictions = pred_actions[0].detach().numpy()
        action_probs = []
        for action in possible_actions:
            if isinstance(action, BuyCardAction):
                action_probs.append(action_predictions[action.card_id])
            elif isinstance(action, DiscardCardAction):
                action_probs.append(action_predictions[action.card_id + EntityManager.cards_count()])
            elif isinstance(action, BuildWonderAction):
                action_probs.append(action_predictions[action.wonder_id + 2 * EntityManager.cards_count()])
            else:
                raise ValueError

        action_probs = np.exp(action_probs)
        action_probs /= action_probs.sum()
        print(action_probs)
        return random.choices(possible_actions, weights=action_probs)[0]
