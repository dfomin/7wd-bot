from typing import List, Optional

import numpy as np
from swd.action import PickWonderAction
from swd.agents import RecordedAgent
from swd.bonuses import BONUSES, ImmediateBonus
from swd.entity_manager import EntityManager
from swd.game import Game
from swd.military_track import MilitaryTrack
from swd.player import Player
from swd.states.game_state import GameState


class GameFeatures:
    initial_state: GameState
    wonders_state: Optional[GameState]
    age_states: List[GameState]
    first_picked_wonders: List[int]
    double_turns: List[int]
    winner: Optional[int]
    victory: str
    division: Optional[int]
    path: Optional[str]
    players: Optional[str]

    def __init__(self, initial_state: GameState, agents: List[RecordedAgent]):
        self.initial_state = initial_state
        self.wonders_state = None
        self.age_states = []
        self.first_picked_wonders = []

        state = self.initial_state.clone()
        while not Game.is_finished(state):
            if len(state.wonders) == 0 and self.wonders_state is None:
                self.wonders_state = state.clone()
            if state.age > len(self.age_states):
                self.age_states.append(state.clone())

            actions = Game.get_available_actions(state)
            selected_action = agents[state.current_player_index].choose_action(state, actions)
            if state.wonders in [4, 8] and isinstance(selected_action, PickWonderAction):
                self.first_picked_wonders.append(selected_action.wonder_id)

            Game.apply_action(state, selected_action)
        self.age_states.append(state.clone())

        self.double_turns = [0, 0]
        for i, player_state in enumerate(self.wonders_state.players_state):
            for wonder in player_state.wonders:
                immediate_bonuses = EntityManager.wonder(wonder[0]).immediate_bonus
                self.double_turns[i] += ImmediateBonus.DOUBLE_TURN in immediate_bonuses

        self.winner = state.winner

        if self.winner is None:
            self.victory = "tie"
        elif MilitaryTrack.military_supremacist(state.military_track_state) is not None:
            self.victory = "military"
        elif max([np.count_nonzero(Player.scientific_symbols(p)) for p in state.players_state]) >= 6:
            self.victory = "science"
        else:
            self.victory = "score"

        self.division = self.initial_state.meta_info.get("division")
        self.path = self.initial_state.meta_info.get("path")
        self.players = self.initial_state.meta_info.get("player_names")
