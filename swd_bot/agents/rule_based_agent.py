import random
from typing import Sequence, Optional

from swd.action import Action, PickWonderAction, PickProgressTokenAction, PickStartPlayerAction, \
    PickDiscardedCardAction, DestroyCardAction
from swd.agents import Agent
from swd.game import Game, GameStatus


class RuleBasedAgent(Agent):
    def choose_action(self, game: Game, possible_actions: Sequence[Action]) -> Action:
        if game.game_status == GameStatus.PICK_WONDER:
            return self.pick_wonder(possible_actions)
        elif game.game_status == GameStatus.PICK_PROGRESS_TOKEN:
            return self.pick_progress_token(game, possible_actions, True)
        elif game.game_status == GameStatus.PICK_REST_PROGRESS_TOKEN:
            return self.pick_progress_token(game, possible_actions, False)
        elif game.game_status == GameStatus.PICK_START_PLAYER:
            for action in possible_actions:
                if isinstance(action, PickStartPlayerAction) and action.player_index == state.current_player_index:
                    return action
        elif game.game_status in [GameStatus.DESTROY_BROWN, GameStatus.DESTROY_GRAY, GameStatus.SELECT_DISCARDED]:
            best_action: Optional[PickDiscardedCardAction] = None
            for action in possible_actions:
                if isinstance(action, (DestroyCardAction, PickDiscardedCardAction)):
                    if best_action is None or best_action.card.id < action.card.id:
                        best_action = action
            if best_action is not None:
                return best_action

        return random.choice(possible_actions)

    def pick_wonder(self, possible_actions: Sequence[Action]) -> Action:
        available_wonder_ids = {}
        for action in possible_actions:
            if isinstance(action, PickWonderAction):
                available_wonder_ids[action.wonder.id] = action
        sorted_wonders = [7, 11, 5, 0, 9, 3, 4, 6, 10, 1, 2, 8]
        for wonder_id in sorted_wonders:
            if wonder_id in available_wonder_ids:
                return available_wonder_ids[wonder_id]

        return random.choice(possible_actions)

    def pick_progress_token(self, game: Game, possible_actions: Sequence[Action], board: bool) -> Action:
        weights = {
            "agriculture": 6.5,
            "architecture": 1,
            "economy": 3.5,
            "law": 0,
            "masonry": 2,
            "mathematics": (len(game.current_player.progress_tokens) + 1) * 3,
            "philosophy": 7,
            "strategy": 4,
            "theology": 2.5,
            "urbanism": 3.5,
        }

        max_scientific_symbols = max(player.scientific_symbols_count for player in game.players)
        if board and max_scientific_symbols >= 3:
            weights["law"] = 100

        if abs(game.military_track.conflict_pawn) >= 3:
            weights["strategy"] = 50

        for token, _ in sorted(weights.items(), key=lambda x: -x[1]):
            for action in possible_actions:
                if isinstance(action, PickProgressTokenAction):
                    if token == action.progress_token:
                        return action

        return random.choice(possible_actions)
