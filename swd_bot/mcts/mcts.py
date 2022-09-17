import math
import time
from typing import Tuple, Callable

import numpy as np
from swd.action import Action
from swd.agents import Agent
from swd.game import Game
from tqdm import tqdm

from swd_bot.mcts.game_tree_node import GameTreeNode


class MCTS:
    root: GameTreeNode

    def __init__(self,
                 game: Game,
                 simulation_agent: Agent,
                 policy_agent: Agent,
                 evaluation_function: Callable[[Game], float]):
        self.prepare_mcts_root(game)
        self.simulation_agent = simulation_agent
        self.policy_agent = policy_agent
        self.evaluation_function = evaluation_function

    def prepare_mcts_root(self, game: Game):
        # if game.cards_board.preset is not None:
        #     card_ids = []
        #     purple_card_ids = []
        #     for row in game.cards_board.card_places:
        #         for board_card in row:
        #             if board_card.card is None:
        #                 card_id = game.cards_board.preset[2][board_card.row][board_card.column]
        #                 if board_card.is_purple_back:
        #                     purple_card_ids.append(card_id)
        #                 else:
        #                     card_ids.append(card_id)
        #     open_cards = [card.id for row in game.cards_board.preset[2] for card in row]
        #     card_ids.extend([x for x in range(46, 66) if x not in open_cards])
        #     purple_card_ids.extend([x for x in range(66, 73) if x not in open_cards])
        #     game.cards_board.card_ids = card_ids
        #     game.cards_board.purple_cards = purple_card_ids
        #     pos_to_replace = (cards_state.card_places == CLOSED_CARD) & (cards_state.preset[2] >= 66)
        #     cards_state.card_places[pos_to_replace] = CLOSED_PURPLE_CARD
        #     cards_state.preset = None
        self.root = GameTreeNode(game, game.get_available_actions(), game.current_player_index)

    def run(self,
            exploration_coefficient: float = math.sqrt(2),
            playouts: int = 1,
            playout_limit: int = 1_000,
            simulations: int = 1_000_000,
            max_time: int = math.inf):
        start = time.time()
        for _ in tqdm(range(simulations)):
            if time.time() - start > max_time:
                break
            node = self.select(self.root, exploration_coefficient)
            wins, total_games = self.expand_and_play(node, playouts, playout_limit)
            self.propagate(node, wins, total_games)

    def select(self, node: GameTreeNode, exploration_coefficient: float):
        unchecked_action = None
        for action in node.actions:
            if str(action) not in node.children:
                unchecked_action = action
                break
        if unchecked_action is not None:
            action = unchecked_action
            new_node = self.create_next_node(node.game, action)
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
                # bonus = 0 if Game.is_finished(child.game_state) else 0.01
                ucb[i] = rate + exploration_coefficient * math.sqrt(math.log(node.total_games) / child.total_games)
                # ucb[i] = rate + exploration_coefficient * math.sqrt(node.total_games) / (child.total_games + 1)
            best_action = None
            for index in (-ucb).argsort():
                action_name = list(node.children.items())[index][0]
                for action in node.actions:
                    if str(action) == action_name:
                        best_action = action
                        break
                if best_action is not None:
                    break
            temp_node = self.create_next_node(node.game, best_action)
            next_node = node.children[str(best_action)]
            next_node.game_state = temp_node.game
            next_node.actions = temp_node.actions
            return self.select(next_node, exploration_coefficient)
        else:
            return node

    def expand_and_play(self, node: GameTreeNode, playouts: int = 1, playout_limit: int = 1_000) -> Tuple[float, int]:
        wins = 0
        agent = self.simulation_agent
        game = node.game.clone()
        for _ in range(playouts):
            moves_count = 0
            while not game.is_finished and moves_count < playout_limit:
                actions = game.get_available_actions()
                selected_action = agent.choose_action(game, actions)
                game.apply_action(selected_action)
                moves_count += 1

            if game.is_finished:
                if game.winner == node.current_player_index:
                    wins += 1
            else:
                value = self.evaluation_function(game)
                if game.current_player_index != node.game.current_player_index:
                    value = 1 - value
                wins += value
        return wins / playouts, 1

    def propagate(self, node: GameTreeNode, wins: float, total_games: int):
        player_index = node.current_player_index
        while node is not None:
            node.total_games += total_games
            if node.current_player_index == player_index:
                node.wins += wins
            else:
                node.wins += total_games - wins
            node = node.parent

    def create_next_node(self, game: Game, action: Action) -> GameTreeNode:
        game = game.clone()
        game.apply_action(action)
        actions = game.get_available_actions()
        return GameTreeNode(game, actions, game.current_player_index)

    def shrink_tree(self, made_action: Action, new_game: Game):
        if str(made_action) in self.root.children:
            new_root = self.root.children[str(made_action)]
            new_root.game = new_game
            available_actions = Game.get_available_actions(new_root.game)
            new_root.actions = available_actions
            new_root.parent = None
            actions_to_delete = [x for x in new_root.children.keys() if x not in map(str, available_actions)]
            for action in actions_to_delete:
                new_root.children[action].parent = None
                del new_root.children[action]
            self.root = new_root
        else:
            self.prepare_mcts_root(new_game)

    def print_optimal_path(self, depth: int = 1):
        node = self.root
        count = 0
        while node is not None and count < depth:
            print(f"Player {node.current_player_index}")
            best_action = ""
            max_score = -1
            children = []
            for action, child in node.children.items():
                if node.current_player_index == child.current_player_index:
                    rate = child.rate()
                else:
                    rate = 1 - child.rate()
                children.append((action, rate, child))
                if rate > max_score:
                    best_action = action
                    max_score = rate
            for action, rate, child in sorted(children, key=lambda x: -x[1]):
                print(action, rate, child.wins, child.total_games)
            if best_action == "":
                print(f"Winner: {node.game.winner}")
                print(f"{Game.points(node.game)[0], node.game.players[0].coins} "
                      f"{Game.points(node.game)[1], node.game.players[1].coins}")
                break
            node = node.children[best_action]
            count += 1
