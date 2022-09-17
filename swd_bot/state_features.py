from typing import List, Any, Dict

from swd.bonuses import BONUSES, SCIENTIFIC_SYMBOLS_RANGE

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
            features.extend([player.bonuses.get(x, 0) for x in range(len(BONUSES))])

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
            unbuilt_wonders = [x for x in player.wonders if not x.is_built]
            features.append(len(unbuilt_wonders))
            if player.has_theology:
                features.append(len(unbuilt_wonders))
            else:
                features.append(len([x for x in unbuilt_wonders if x.double_turn]))
            assets = player.assets(game.players[1 - i].bonuses, None)
            features.extend(list(assets.resources))
            features.extend(list(assets.resources_cost))
            features.append(sum(x in player.bonuses for x in SCIENTIFIC_SYMBOLS_RANGE))

        features.append(game.military_track.conflict_pawn)

        available_cards = [x.card.id for x in game.cards_board.available_cards()]
        features.extend([int(card_id in available_cards) for card_id in range(EntityManager.cards_count())])

        return features
