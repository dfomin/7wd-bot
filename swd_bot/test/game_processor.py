from pathlib import Path
from typing import Type, Union, Callable, List

from swd.agents import Agent
from swd.game import Game
from tqdm import tqdm

from swd_bot.thirdparty.loader import GameLogLoader


def process_games(path: Union[str, Path],
                  loader: Type[GameLogLoader],
                  process_function: Callable[[Game, List[Agent]], None]):
    for game_log in tqdm(list(Path(path).rglob("*.json"))):
        # print(game_log)
        state, agents = loader.load(game_log)
        if state is None or agents is None:
            continue
        process_function(state, agents)
