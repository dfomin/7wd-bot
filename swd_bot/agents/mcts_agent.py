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

        self.mcts.run(max_time=5, playout_limit=20, simulations=5000)
        self.mcts.print_optimal_path()

        actions_predictions, _ = self.torch_agent.predict(state)
        if state.game_status == GameStatus.NORMAL_TURN:
            actions_probs = TorchAgent.normalize_actions(actions_predictions, possible_actions)
            actions_probs /= 10
        else:
            actions_probs = np.zeros(len(possible_actions))

        for i, action in enumerate(possible_actions):
            if str(action) in self.mcts.root.children:
                child = self.mcts.root.children[str(action)]
                if self.mcts.root.current_player_index == child.current_player_index:
                    rate = child.rate()
                else:
                    rate = 1 - child.rate()
                actions_probs[i] += rate
        return possible_actions[actions_probs.argmax()]

    def on_action_applied(self, action: Action, new_state: GameState):
        self.mcts.shrink_tree(action, new_state)
