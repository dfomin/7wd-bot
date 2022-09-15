from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Optional, List, Union

from swd.agents import RecordedAgent
from swd.game import Game


class GameLogLoader:
    @staticmethod
    @abstractmethod
    def load(path: Union[str, Path]) -> Tuple[Optional[Game], Optional[List[RecordedAgent]]]:
        raise NotImplemented
