from typing import Sequence

import numpy as np
from swd.action import Action
from swd.agents import Agent
from swd.game import Game, GameStatus

from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.mcts.mcts import MCTS


class MCTSAgent(Agent):
    mcts: MCTS

    def __init__(self, game: Game):
        super().__init__()

        self.torch_agent = TorchAgent()

        def evaluation_function(game: Game):
            _, winners_predictions = self.torch_agent.predict(game)
            winners_predictions = np.exp(winners_predictions)
            winners_predictions /= winners_predictions.sum()
            return winners_predictions[game.current_player_index]

        self.mcts = MCTS(game, self.torch_agent, self.torch_agent, evaluation_function)

    def choose_action(self, game: Game, possible_actions: Sequence[Action]) -> Action:
        if game.game_status == GameStatus.PICK_WONDER:
            return self.torch_agent.choose_action(game, possible_actions)

        self.mcts.run(max_time=5, playout_limit=100, simulations=10000, playouts=100)
        self.mcts.print_optimal_path(1)

        actions_predictions, _ = self.torch_agent.predict(game)
        if game.game_status == GameStatus.NORMAL_TURN:
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

    def on_action_applied(self, action: Action, new_game: Game):
        self.mcts.shrink_tree(action, new_game)
