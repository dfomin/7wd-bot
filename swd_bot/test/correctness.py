from pathlib import Path
from typing import List, Union, Type

from swd.agents import Agent, RecordedAgent
from swd.game import Game
from swd.states.game_state import GameState

from swd_bot.test.game_processor import process_games
from swd_bot.thirdparty.loader import GameLogLoader


def test_game_correctness(state: GameState, agents: List[Agent], verbose: bool = False):
    state.price_cache = {0: {}, 1: {}}

    if verbose:
        print(state.meta_info["player_names"])

    while not Game.is_finished(state):
        actions = Game.get_available_actions(state)
        if Game.is_finished(state):
            break
        agent = agents[state.current_player_index]
        selected_action = agent.choose_action(state, actions)
        Game.apply_action(state, selected_action)

    for i, agent in enumerate(agents):
        if isinstance(agent, RecordedAgent):
            assert len(agent.actions) == 0
        else:
            assert False
        if "players" in state.meta_info:
            if verbose:
                print(state.players_state[i].coins, state.meta_info["players"][i]["coins"])
            assert state.players_state[i].coins == state.meta_info["players"][i]["coins"]

    if "result" in state.meta_info:
        if state.meta_info["result"]["victory"] != "tie" and state.winner != state.meta_info["result"]["winnerIndex"]:
            print(f"Winner: {state.winner}")
            print(Game.points(state, 0), Game.points(state, 1))
            print(state.meta_info["result"])
        if state.meta_info["result"]["victory"] == "tie":
            assert state.winner == -1
        else:
            assert state.winner == state.meta_info["result"]["winnerIndex"]


def test_games_correctness(path: Union[str, Path], loader: Type[GameLogLoader]):
    process_games(path, loader, test_game_correctness)
