import random
from typing import Sequence

from swd.action import Action
from swd.agents import Agent
from swd.states.game_state import GameState

from swd_bot.mcts.mcts import MCTS


class MCTSAgent(Agent):
    mcts: MCTS

    def __init__(self, state: GameState):
        super().__init__()

        self.mcts = MCTS(state)

    def choose_action(self, state: GameState, possible_actions: Sequence[Action]) -> Action:
        actions_map = {str(action): action for action in possible_actions}
        actions_score = []
        for action, child in self.mcts.root.children.items():
            if self.mcts.root.current_player_index == child.current_player_index:
                rate = child.rate()
            else:
                rate = 1 - child.rate()
            actions_score.append((action, rate))
        for action in sorted(actions_score, key=lambda x: -x[1]):
            action_name = str(action[0])
            if action_name in actions_map:
                return actions_map[action_name]
        return random.choice(possible_actions)

    def on_action_applied(self, old_state: GameState, action: Action, new_state: GameState):
        self.mcts.shrink_tree(action, new_state)
