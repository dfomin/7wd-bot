from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Optional, List, Union

from swd.agents import RecordedAgent
from swd.states.game_state import GameState


class GameLogLoader:
    @staticmethod
    @abstractmethod
    def load(path: Union[str, Path]) -> Tuple[Optional[GameState], Optional[List[RecordedAgent]]]:
        raise NotImplemented
