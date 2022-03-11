import pickle
from pathlib import Path
from typing import List, Callable

import numpy as np
import pandas as pd
from swd.action import BuyCardAction, DiscardCardAction
from swd.agents import Agent
from swd.cards_board import AGES
from swd.entity_manager import EntityManager
from swd.game import Game
from swd.states.game_state import GameState
from tqdm import tqdm

from swd_bot.game_features import GameFeatures
from swd_bot.sevenee import SeveneeLoader
from swd_bot.state_features import StateFeatures


def generate_words():
    words = []
    for path in tqdm(list(Path("../../7wd/sevenee").rglob("*.json"))):
        state, agents = SeveneeLoader.load(path)
        if state is None:
            continue
        game_features = GameFeatures(state, agents)
        for age, age_state in enumerate(game_features.age_states):
            for player in range(2):
                d = {
                    "path": str(path),
                    "age": age,
                    "player": player,
                    "words": "_".join(map(str, age_state.players_state[player].cards))
                }
                words.append(d)
    df = pd.DataFrame(words)
    df.to_csv("words.csv", index=False)


def process_sevenee_games(process_function: Callable[[GameState, List[Agent]], None]):
    for file in Path(f"../../7wd/sevenee/").rglob("*.json"):
        state, agents = SeveneeLoader.load(file)
        if state is None:
            continue
        print(file)
        process_function(state, agents)


def collect_states_actions():
    saved_states = []
    saved_actions = []

    def save_state(state: GameState, agents: List[Agent]):
        while not Game.is_finished(state):
            actions = Game.get_available_actions(state)
            if Game.is_finished(state):
                break
            agent = agents[state.current_player_index]
            selected_action = agent.choose_action(state, actions)
            if isinstance(selected_action, BuyCardAction) or isinstance(selected_action, DiscardCardAction):
                saved_states.append(state.clone())
                saved_actions.append(selected_action)

            Game.apply_action(state, selected_action)

    process_sevenee_games(save_state)

    with open("../notebooks/states.pkl", "wb") as f:
        pickle.dump(saved_states, f)

    with open("../notebooks/actions.pkl", "wb") as f:
        pickle.dump(saved_actions, f)


def main():
    # with open("../notebooks/states.pkl", "rb") as f:
    #     states = pickle.load(f)
    # state: GameState = states[0]
    # print(StateFeatures.extract_state_features_dict(state))
    # print(EntityManager.card(0).bonuses)
    collect_states_actions()


if __name__ == "__main__":
    main()
