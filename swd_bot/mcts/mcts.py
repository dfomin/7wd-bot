import math
import time
from typing import Tuple

import numpy as np
from swd.action import Action
from swd.agents import RandomAgent
from swd.cards_board import CLOSED_CARD, CLOSED_PURPLE_CARD
from swd.game import Game
from swd.states.game_state import GameState
from tqdm import tqdm

from swd_bot.mcts.game_tree_node import GameTreeNode


class MCTS:
    root: GameTreeNode

    def __init__(self, state: GameState):
        self.prepare_mcts_root(state)

    def prepare_mcts_root(self, state: GameState):
        if state.cards_board_state.preset is not None:
            card_ids = []
            purple_card_ids = []
            for i in np.transpose(np.where(state.cards_board_state.card_places == -1)):
                card_id = state.cards_board_state.preset[2][tuple(i)]
                if card_id >= 66:
                    purple_card_ids.append(card_id)
                else:
                    card_ids.append(card_id)
            card_ids.extend([x for x in range(46, 66) if x not in state.cards_board_state.preset.flat])
            purple_card_ids.extend([x for x in range(66, 73) if x not in state.cards_board_state.preset.flat])
            cards_state = state.cards_board_state
            cards_state.card_ids = card_ids
            cards_state.purple_card_ids = purple_card_ids
            pos_to_replace = (cards_state.card_places == CLOSED_CARD) & (cards_state.preset[2] >= 66)
            cards_state.card_places[pos_to_replace] = CLOSED_PURPLE_CARD
            cards_state.preset = None
        self.root = GameTreeNode(state, Game.get_available_actions(state), state.current_player_index)

    def run(self,
            exploration_coefficient: float = math.sqrt(2),
            playouts: int = 1,
            simulations: int = 1_000_000,
            max_time: int = math.inf):
        start = time.time()
        for _ in tqdm(range(simulations)):
            if time.time() - start > max_time:
                break
            node = self.select(self.root, exploration_coefficient)
            wins, total_games = self.expand_and_play(node, playouts)
            self.propagate(node, wins, total_games)

    def select(self, node: GameTreeNode, exploration_coefficient: float):
        unchecked_action = None
        for action in node.actions:
            if str(action) not in node.children:
                unchecked_action = action
                break
        if unchecked_action is not None:
            action = unchecked_action
            new_node = self.create_next_node(node.game_state, action)
            node.children[str(action)] = new_node
            new_node.parent = node
            return new_node
        elif len(node.children) > 0:
            ucb = np.zeros(len(node.children))
            for i, child in enumerate(node.children.values()):
                if node.current_player_index == child.current_player_index:
                    rate = child.rate()
                else:
                    rate = 1 - child.rate()
                bonus = 0 if Game.is_finished(child.game_state) else 0.01
                # ucb[i] = rate + exploration_coefficient * math.sqrt(math.log(node.total_games) / child.total_games) + bonus
                ucb[i] = rate + exploration_coefficient * math.sqrt(node.total_games) / (child.total_games + 1) + bonus
            best_action = None
            for index in (-ucb).argsort():
                action_name = list(node.children.items())[index][0]
                for action in node.actions:
                    if str(action) == action_name:
                        best_action = action
                        break
                if best_action is not None:
                    break
            temp_node = self.create_next_node(node.game_state, best_action)
            next_node = node.children[str(best_action)]
            next_node.game_state = temp_node.game_state
            next_node.actions = temp_node.actions
            return self.select(next_node, exploration_coefficient)
        else:
            return node

    def expand_and_play(self, node: GameTreeNode, playouts: int = 1) -> Tuple[float, int]:
        wins = 0
        agent = RandomAgent()
        state = node.game_state.clone()
        for _ in range(playouts):
            while not Game.is_finished(state):
                actions = Game.get_available_actions(state)
                selected_action = agent.choose_action(state, actions)
                Game.apply_action(state, selected_action)
            if state.winner == node.current_player_index:
                wins += 1
        return wins / playouts, 1

    def propagate(self, node: GameTreeNode, wins: float, total_games: int):
        player_index = node.current_player_index
        while node is not None:
            node.total_games += total_games
            if node.current_player_index == player_index:
                node.wins += wins
            else:
                node.wins += 1 - wins
            node = node.parent

    def create_next_node(self, state: GameState, action: Action) -> GameTreeNode:
        state = state.clone()
        Game.apply_action(state, action)
        actions = Game.get_available_actions(state)
        return GameTreeNode(state, actions, state.current_player_index)

    def shrink_tree(self, made_action: Action, new_state: GameState):
        if str(made_action) in self.root.children:
            new_root = self.root.children[str(made_action)]
            new_root.parent = None
            available_actions = Game.get_available_actions(new_root.game_state)
            actions_to_delete = [x for x in new_root.children.keys() if x not in map(str, available_actions)]
            for action in actions_to_delete:
                new_root.children[action].parent = None
                del new_root.children[action]
            self.root = new_root
        else:
            self.prepare_mcts_root(new_state)
