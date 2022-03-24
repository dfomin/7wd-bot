import random
from typing import Sequence

from swd.action import Action, PickWonderAction
from swd.agents import Agent
from swd.states.game_state import GameState, GameStatus


class RuleBasedAgent(Agent):
    def choose_action(self, state: GameState, possible_actions: Sequence[Action]) -> Action:
        if state.game_status == GameStatus.PICK_WONDER:
            return RuleBasedAgent.pick_wonder(possible_actions)

    @staticmethod
    def pick_wonder(possible_actions: Sequence[Action]) -> Action:
        available_wonder_ids = {}
        for action in possible_actions:
            if isinstance(action, PickWonderAction):
                available_wonder_ids[action.wonder_id] = action
        sorted_wonders = [7, 11, 5, 0, 9, 3, 4, 6, 10, 1, 2, 8]
        for wonder_id in sorted_wonders:
            if wonder_id in available_wonder_ids:
                return available_wonder_ids[wonder_id]

        return random.choice(possible_actions)
