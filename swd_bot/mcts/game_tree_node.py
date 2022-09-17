from dataclasses import dataclass, field
from typing import Optional, Dict, List

from swd.action import Action, PickDiscardedCardAction
from swd.game import Game


@dataclass
class GameTreeNode:
    game: Game
    actions: List[str]
    current_player_index: int
    wins: float = 0
    total_games: int = 0
    parent: Optional['GameTreeNode'] = None
    children: Dict[str, 'GameTreeNode'] = field(default_factory=dict)

    def rate(self) -> float:
        rate = self.wins / self.total_games

        for action in self.actions:
            if str(action) in self.children:
                child = self.children[str(action)]
                if self.game.current_player_index == child.game.current_player_index:
                    child_rate = child.rate()
                else:
                    child_rate = 1 - child.rate()
                if child_rate > rate:
                    rate = child_rate
        return rate
