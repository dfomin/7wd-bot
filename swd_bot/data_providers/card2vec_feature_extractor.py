from typing import Tuple

import numpy as np
from swd.entity_manager import EntityManager
from swd.states.game_state import GameState

from swd_bot.data_providers.feature_extractor import FeatureExtractor
from swd_bot.state_features import StateFeatures


class Card2VecFeatureExtractor(FeatureExtractor):
    def __init__(self, path: str):
        self.card2vec = np.load(path)

    def features(self, state: GameState) -> Tuple[np.ndarray, np.ndarray]:
        len_bonuses = len(FeatureExtractor.card_features(card=EntityManager.card(0)))

        x = StateFeatures.extract_state_features_dict(state)
        output = [
            x["age"],
            x["current_player"]
        ]
        output.extend(x["tokens"])
        output.append(x["military_pawn"])
        output.extend(x["military_tokens"])
        output.append(x["game_status"])
        for i in range(2):
            output.append(x["players"][i]["coins"])
            output.extend(x["players"][i]["unbuilt_wonders"])
            output.extend(x["players"][i]["bonuses"])

        cards = []
        for card_id in x["cards_board"]:
            if card_id >= 0:
                c2v = self.card2vec[card_id]
                if card_id not in x["available_cards"]:
                    c2v = -1 * c2v
                cb = FeatureExtractor.card_features(card=EntityManager.card(card_id)) / 10  # эмбеддинг карты
                c2v = np.concatenate([c2v, cb])
            elif card_id == -1:
                c2v = np.ones(self.card2vec.shape[1] + len_bonuses)
            else:
                c2v = np.zeros(self.card2vec.shape[1] + len_bonuses)
            output.extend(list(c2v))

        return np.array(output), np.array(cards)
