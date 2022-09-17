from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from swd.bonuses import RESOURCES, INSTANT_BONUSES
from swd.cards import Card
from swd.entity_manager import EntityManager
from swd.game import Game

from swd_bot.state_features import StateFeatures


class FeatureExtractor(ABC):
    @abstractmethod
    def features(self, game: Game) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplemented

    @staticmethod
    def card_features(card: Card) -> np.ndarray:
        result = np.zeros(1 + len(card.price.resources) + 1 + len(card.bonuses) + len(INSTANT_BONUSES))
        result[0] = card.price.coins
        result[1: 1 + len(RESOURCES)] = card.price.resources
        # result[1 + len(RESOURCES)] = card.price.chain_symbol
        result[2 + len(RESOURCES): 2 + len(RESOURCES) + len(card.bonuses)] = card.bonuses
        result[2 + len(RESOURCES) + len(card.bonuses):] = card.instant_bonuses
        return result


class FlattenFeatureExtractor(FeatureExtractor):
    def features(self, game: Game) -> Tuple[np.ndarray, np.ndarray]:
        x = StateFeatures.extract_state_features_dict(game)
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
        for i in range(6):
            ohe = [0] * EntityManager.cards_count()
            if i < len(x["available_cards"]):
                ohe[x["available_cards"][i]] = 1
            output.extend(ohe)
        return np.array(output), np.array([])


class EmbeddingsFeatureExtractor(FeatureExtractor):
    def features(self, game: Game) -> Tuple[np.ndarray, np.ndarray]:
        x = StateFeatures.extract_state_features_dict(game)
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
        for i in range(6):
            if i < len(x["available_cards"]):
                card_id = x["available_cards"][i]
                ohe = np.zeros(EntityManager.cards_count())
                ohe[card_id] = 1
                card = EntityManager.card(card_id)
                cards.append(np.concatenate([FeatureExtractor.card_features(card), ohe]))
            else:
                card = EntityManager.card(0)
                features = FeatureExtractor.card_features(card)
                cards.append(np.zeros(len(features) + EntityManager.cards_count()))
        return np.array(output), np.array(cards)


class FlattenEmbeddingsFeatureExtractor(FeatureExtractor):
    def features(self, game: Game) -> Tuple[np.ndarray, np.ndarray]:
        x = StateFeatures.extract_state_features_dict(game)
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
        for i in range(6):
            if i < len(x["available_cards"]):
                card_id = x["available_cards"][i]
                ohe = [0] * EntityManager.cards_count()
                ohe[card_id] = 1
                card = EntityManager.card(card_id)
                output.extend(list(FeatureExtractor.card_features(card)))
                output.extend(ohe)
            else:
                card = EntityManager.card(0)
                features = FeatureExtractor.card_features(card)
                output.extend([0] * (len(features) + EntityManager.cards_count()))
        return np.array(output), np.array([])


class ManualFeatureExtractor(FeatureExtractor):
    def features(self, game: Game) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(StateFeatures.extract_manual_state_features(game)), np.array([])
