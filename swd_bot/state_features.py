from typing import List, Any, Dict

import numpy as np
from swd.bonuses import BONUSES, SCIENTIFIC_SYMBOLS_RANGE, INSTANT_BONUSES

from swd.entity_manager import EntityManager
from swd.game import Game


class StateFeatures:
    @staticmethod
    def extract_state_features(game: Game) -> List[int]:
        features = [
            game.age,
            game.current_player_index
        ]

        features.extend([int(x in game.progress_tokens) for x in range(EntityManager.progress_tokens_count())])
        # features.extend([int(x in state.discard_pile) for x in range(EntityManager.cards_count())])

        # features.append(int(state.is_double_turn))

        for player in game.players:
            features.append(player.coins)
            unbuilt_wonders = [x.id for x in player.wonders if not x.is_built]
            features.extend([int(x in unbuilt_wonders) for x in range(EntityManager.wonders_count())])
            features.extend([player.bonuses[x] for x in range(len(BONUSES))])

        features.append(game.military_track.conflict_pawn)
        features.extend(game.military_track.military_tokens)

        features.append(game.game_status.value)

        # features.extend(list(state.cards_board_state.card_places.flat))
        features.extend([x.card.id * (not x.is_taken) for row in game.cards_board.card_places for x in row])

        return features

    @staticmethod
    def extract_state_features_dict(game: Game) -> Dict[str, Any]:
        features = {
            "age": game.age,
            "current_player": game.current_player_index,
            "tokens": [int(x in game.progress_tokens) for x in range(EntityManager.progress_tokens_count())],
            "discard_pile": game.discard_pile,
            "military_pawn": game.military_track.conflict_pawn,
            "military_tokens": list(game.military_track.military_tokens),
            "game_status": game.game_status.value,
            "players": []
        }

        for player in game.players:
            unbuilt_wonders = [x.id for x in player.wonders if not x.is_built]
            player = {
                "coins": player.coins,
                "unbuilt_wonders": [int(x in unbuilt_wonders) for x in range(EntityManager.wonders_count())],
                "bonuses": list(player.bonuses)
            }
            features["players"].append(player)

        features["cards_board"] = [x.card.id * (not x.is_taken) for row in game.cards_board.card_places for x in row]
        features["available_cards"] = [x.card.id for x in game.cards_board.available_cards()]

        return features

    @staticmethod
    def extract_manual_state_features(game: Game) -> List[int]:
        features = []

        features.extend([int(x in game.progress_tokens) for x in range(EntityManager.progress_tokens_count())])

        for i, player in enumerate(game.players):
            features.append(player.coins)
            features.extend(list(Game.points(game)[i]))
            unbuilt_wonders = [x.id for x in player.wonders if not x.is_built]
            features.append(len(unbuilt_wonders))
            if player.bonuses[BONUSES.index("theology")] > 0:
                features.append(len(unbuilt_wonders))
            else:
                double_turn_index = INSTANT_BONUSES.index("double_turn")
                features.append(len([x for x in unbuilt_wonders
                                     if EntityManager.wonder(x).instant_bonuses[double_turn_index] > 0]))
            assets = player.assets(game.players[1 - i].bonuses, None)
            features.extend(list(assets.resources))
            features.extend(list(assets.resources_cost))
            features.append(np.count_nonzero(player.bonuses[SCIENTIFIC_SYMBOLS_RANGE]))

        features.append(game.military_track.conflict_pawn)

        available_cards = [x.card.id for x in game.cards_board.available_cards()]
        features.extend([int(card_id in available_cards) for card_id in range(EntityManager.cards_count())])

        return features
