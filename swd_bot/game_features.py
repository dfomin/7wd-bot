from typing import List

from swd.agents import RecordedAgent
from swd.game import Game
from swd.states.game_state import GameState


class GameFeatures:
    initial_state: GameState
    age_states: List[GameState]

    def __init__(self, initial_state: GameState, agents: List[RecordedAgent]):
        self.initial_state = initial_state
        self.age_states = []

        state = self.initial_state.clone()
        while not Game.is_finished(state):
            if state.age > len(self.age_states):
                self.age_states.append(state.clone())

            actions = Game.get_available_actions(state)
            selected_action = agents[state.current_player_index].choose_action(state, actions)
            Game.apply_action(state, selected_action)
        self.age_states.append(state.clone())
