from typing import List, Any, Dict

import numpy as np
from swd.bonuses import BONUSES, SCIENTIFIC_SYMBOLS_RANGE, INSTANT_BONUSES
from swd.cards_board import AGES, CardsBoard

from swd.entity_manager import EntityManager
from swd.game import Game, GameState
from swd.player import Player


class StateFeatures:
    @staticmethod
    def extract_state_features(state: GameState) -> List[int]:
        features = [
            state.age,
            state.current_player_index
        ]

        features.extend([int(x in state.progress_tokens) for x in EntityManager.progress_token_names()])
        # features.extend([int(x in state.discard_pile) for x in range(EntityManager.cards_count())])

        # features.append(int(state.is_double_turn))

        for player_state in state.players_state:
            features.append(player_state.coins)
            unbuilt_wonders = [x[0] for x in player_state.wonders if x[1] is None]
            features.extend([int(x in unbuilt_wonders) for x in range(EntityManager.wonders_count())])
            features.extend(list(player_state.bonuses))

        features.append(state.military_track_state.conflict_pawn)
        features.extend(list(state.military_track_state.military_tokens))

        features.append(state.game_status.value)

        # features.extend(list(state.cards_board_state.card_places.flat))
        indices = np.flip(AGES[state.age] > 0, axis=0)
        features.extend(list(np.flip(state.cards_board_state.card_places, axis=0)[indices]))

        return features

    @staticmethod
    def extract_state_features_dict(state: GameState) -> Dict[str, Any]:
        features = {
            "age": state.age,
            "current_player": state.current_player_index,
            "tokens": [int(x in state.progress_tokens) for x in EntityManager.progress_token_names()],
            "discard_pile": state.discard_pile,
            "military_pawn": state.military_track_state.conflict_pawn,
            "military_tokens": list(state.military_track_state.military_tokens),
            "game_status": state.game_status.value,
            "players": []
        }

        for player_state in state.players_state:
            unbuilt_wonders = [x[0] for x in player_state.wonders if x[1] is None]
            player = {
                "coins": player_state.coins,
                "unbuilt_wonders": [int(x in unbuilt_wonders) for x in range(EntityManager.wonders_count())],
                "bonuses": list(player_state.bonuses)
            }
            features["players"].append(player)

        indices = np.flip(AGES[state.age] > 0, axis=0)
        available_cards = CardsBoard.available_cards(state.cards_board_state)
        features["cards_board"] = list(np.flip(state.cards_board_state.card_places, axis=0)[indices])
        features["available_cards"] = list(map(lambda x: x[0], available_cards))

        return features

    @staticmethod
    def extract_manual_state_features(state: GameState) -> List[int]:
        features = []

        features.extend([int(x in state.progress_tokens) for x in EntityManager.progress_token_names()])

        for i, player_state in enumerate(state.players_state):
            features.append(player_state.coins)
            features.extend(list(Game.points(state, i)))
            unbuilt_wonders = [x[0] for x in player_state.wonders if x[1] is None]
            features.append(len(unbuilt_wonders))
            if player_state.bonuses[BONUSES.index("theology")] > 0:
                features.append(len(unbuilt_wonders))
            else:
                features.append(len([x for x in unbuilt_wonders
                                     if INSTANT_BONUSES.index("double_turn") in EntityManager.wonder(x).instant_bonuses]))
            assets = Player.assets(player_state, Player.resources(state.players_state[1 - i]), None)
            features.extend(list(assets.resources))
            features.extend(list(assets.resources_cost))
            features.append(np.count_nonzero(player_state.bonuses[SCIENTIFIC_SYMBOLS_RANGE]))

        features.append(state.military_track_state.conflict_pawn)

        available_cards = [x[0] for x in CardsBoard.available_cards(state.cards_board_state)]
        features.extend([int(card_id in available_cards) for card_id in range(EntityManager.cards_count())])

        return features
