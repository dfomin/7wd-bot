import random
from typing import Sequence

import numpy as np
from swd.action import Action
from swd.agents import Agent
from swd.game import Game
from swd.states.game_state import GameState, GameStatus

from swd_bot.agents.rule_based_agent import RuleBasedAgent
from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.mcts.mcts import MCTS


class MCTSAgent(Agent):
    mcts: MCTS

    def __init__(self, state: GameState):
        super().__init__()

        self.torch_agent = TorchAgent()

        def evaluation_function(s: GameState):
            _, winners_predictions = self.torch_agent.predict(s)
            winners_predictions = np.exp(winners_predictions)
            winners_predictions /= winners_predictions.sum()
            return winners_predictions[s.current_player_index]

        self.mcts = MCTS(state, self.torch_agent, self.torch_agent, evaluation_function)

    def choose_action(self, state: GameState, possible_actions: Sequence[Action]) -> Action:
        if state.game_status == GameStatus.PICK_WONDER:
            return RuleBasedAgent.pick_wonder(possible_actions)

        print(Game.print(state))
        self.mcts.run(max_time=5, playout_limit=20)
        self.mcts.print_optimal_path()
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

    def on_action_applied(self, action: Action, new_state: GameState):
        self.mcts.shrink_tree(action, new_state)
