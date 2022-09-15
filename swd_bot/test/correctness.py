from pathlib import Path
from typing import List, Union, Type

from swd.agents import Agent, RecordedAgent
from swd.game import Game

from swd_bot.test.game_processor import process_games
from swd_bot.thirdparty.loader import GameLogLoader


def test_game_correctness(game: Game, agents: List[Agent], verbose: bool = False):
    game.price_cache = {0: {}, 1: {}}

    if verbose:
        print(f'Season: {game.meta_info["season"]}, {" - ".join(game.meta_info["player_names"])}')

    while not game.is_finished:
        actions = game.get_available_actions()
        if game.is_finished:
            break
        agent = agents[game.current_player_index]
        selected_action = agent.choose_action(game, actions)
        game.apply_action(selected_action)

    for i, agent in enumerate(agents):
        if isinstance(agent, RecordedAgent):
            assert len(agent.actions) == 0
        else:
            assert False
        if "players" in game.meta_info:
            if verbose:
                print(game.players[i].coins, game.meta_info["players"][i]["coins"])
            assert game.players[i].coins == game.meta_info["players"][i]["coins"]

    if "result" in game.meta_info:
        if game.meta_info["result"]["victory"] != "tie" and game.winner != game.meta_info["result"]["winnerIndex"]:
            print(f"Winner: {game.winner}")
            print(game.points())
            print(game.meta_info["result"])
        if game.meta_info["result"]["victory"] == "tie":
            assert game.winner == -1
        else:
            assert game.winner == game.meta_info["result"]["winnerIndex"]


def test_games_correctness(path: Union[str, Path], loader: Type[GameLogLoader]):
    process_games(path, loader, test_game_correctness)
