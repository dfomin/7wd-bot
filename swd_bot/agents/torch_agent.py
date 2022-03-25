import random
from typing import Sequence, Tuple

import numpy as np
import torch
from swd.action import Action, BuyCardAction, DiscardCardAction, BuildWonderAction, PickWonderAction, \
    PickStartPlayerAction
from swd.agents import Agent
from swd.entity_manager import EntityManager
from swd.states.game_state import GameState, GameStatus

from swd_bot.agents.rule_based_agent import RuleBasedAgent
from swd_bot.data_providers.feature_extractor import FlattenFeatureExtractor
from swd_bot.model.torch_models import TorchBaseline


class TorchAgent(Agent):
    def __init__(self):
        self.model = TorchBaseline(600, 0)
        self.model.load_state_dict(torch.load("../models/model_flat_acc54.67.pth"))
        self.model.eval()

        self.feature_extractor = FlattenFeatureExtractor()

    def predict(self, state: GameState) -> Tuple[np.ndarray, np.ndarray]:
        features, cards = self.feature_extractor.features(state)
        pred_actions, pred_winners = self.model(torch.FloatTensor(features)[None], torch.FloatTensor(cards)[None])
        return pred_actions[0].detach().numpy(), pred_winners[0].detach().numpy()

    @staticmethod
    def normalize_actions(action_predictions: np.ndarray, possible_actions: Sequence[Action]) -> np.ndarray:
        actions_probs = np.zeros(len(possible_actions))
        for i, action in enumerate(possible_actions):
            if isinstance(action, BuyCardAction):
                actions_probs[i] = action_predictions[action.card_id]
            elif isinstance(action, DiscardCardAction):
                actions_probs[i] = action_predictions[action.card_id + EntityManager.cards_count()]
            elif isinstance(action, BuildWonderAction):
                actions_probs[i] = action_predictions[action.wonder_id + 2 * EntityManager.cards_count()]
            else:
                raise ValueError

        actions_probs = np.exp(actions_probs)
        actions_probs /= actions_probs.sum()

        return actions_probs

    def choose_action(self, state: GameState, possible_actions: Sequence[Action]) -> Action:
        if state.game_status == GameStatus.PICK_WONDER:
            RuleBasedAgent.pick_wonder(possible_actions)
        elif state.game_status == GameStatus.PICK_START_PLAYER:
            for action in possible_actions:
                if isinstance(action, PickStartPlayerAction) and action.player_index == state.current_player_index:
                    return action

        if state.game_status != GameStatus.NORMAL_TURN:
            return random.choice(possible_actions)

        actions_predictions, _ = self.predict(state)
        actions_probs = TorchAgent.normalize_actions(actions_predictions, possible_actions)

        # return possible_actions[action_probs.argmax()]

        # action_probs = np.power(action_probs, 2)
        # action_probs /= action_probs.sum()
        # for prob, action in zip(action_probs, possible_actions):
        #     print(f"{action}: {round(prob, 2)}")
        return random.choices(possible_actions, weights=actions_probs)[0]
