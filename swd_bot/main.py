import pickle
from pathlib import Path
from typing import List, Callable

import pandas as pd
import torch
from swd.action import BuyCardAction, DiscardCardAction, BuildWonderAction
from swd.agents import Agent, RecordedAgent
from swd.game import Game
from swd.states.game_state import GameState
from tqdm import tqdm

from swd_bot.data_providers.torch_data_provider import TorchDataset
from swd_bot.game_features import GameFeatures
from swd_bot.model.torch_models import TorchBaseline
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
    saved_states = [[], [], []]
    saved_actions = [[], [], []]

    def save_state(state: GameState, agents: List[Agent]):
        while not Game.is_finished(state):
            actions = Game.get_available_actions(state)
            if Game.is_finished(state):
                break
            agent = agents[state.current_player_index]
            selected_action = agent.choose_action(state, actions)
            if isinstance(selected_action, BuyCardAction) or isinstance(selected_action, DiscardCardAction) \
                    or isinstance(selected_action, BuildWonderAction):
                index = 0
                if state.meta_info["season"] % 5 == 0:
                    index = 2
                elif state.meta_info["season"] % 5 == 4:
                    index = 1
                saved_states[index].append(state.clone())
                saved_actions[index].append(selected_action)

            Game.apply_action(state, selected_action)

    process_sevenee_games(save_state)

    suffixes = ["_train", "_valid", "_test"]
    for i in range(3):
        with open(f"../datasets/buy_discard_build/states{suffixes[i]}.pkl", "wb") as f:
            pickle.dump(saved_states[i], f)

        with open(f"../datasets/buy_discard_build/actions{suffixes[i]}.pkl", "wb") as f:
            pickle.dump(saved_actions[i], f)


def collect_games_features():
    games_features = []

    def save_features(state: GameState, agents: List[Agent]):
        recorded_agents: List[RecordedAgent] = []
        for agent in agents:
            if isinstance(agent, RecordedAgent):
                recorded_agents.append(agent)
        features = GameFeatures(state, recorded_agents)
        games_features.append({
            "double_turns_0": features.double_turns[0],
            "double_turns_1": features.double_turns[1],
            "first_picked_wonder": features.first_picked_wonders,
            "winner": features.winner,
            "victory": features.victory,
            "division": features.division,
            "path": features.path,
            "players": features.players
        })

    process_sevenee_games(save_features)
    df = pd.DataFrame(games_features)
    df.to_csv("../notebooks/features.csv", index=False)


def test_model():
    model = TorchBaseline()
    model.load_state_dict(torch.load("../models/model_acc51.pth"))
    model.eval()

    # file = Path(f"../../7wd/sevenee/46/1/1/aRT22RJpJAGP8iPNs.json")
    file = Path(f"../../7wd/sevenee/46/1/1/SuJYYgWE7fMFS8Dfi.json")
    state, agents = SeveneeLoader.load(file)
    print(state.meta_info["player_names"])
    correct = 0
    all = 0
    while not Game.is_finished(state):
        actions = Game.get_available_actions(state)
        if Game.is_finished(state):
            break
        agent = agents[state.current_player_index]
        selected_action = agent.choose_action(state, actions)

        if len(state.wonders) == 0:
            features = TorchDataset.flatten_features(StateFeatures.extract_state_features_dict(state))
            action, winner = model(torch.tensor(features, dtype=torch.float)[None])
            if isinstance(selected_action, BuyCardAction) or isinstance(selected_action, DiscardCardAction)\
                    or isinstance(selected_action, BuildWonderAction):
                if action[0].argmax().item() == selected_action.card_id:
                    correct += 1
                all += 1
            print(round(winner[0][0].item(), 2), round(winner[0][1].item(), 2))

        Game.apply_action(state, selected_action)
    print(correct / all)


def main():
    # with open("../notebooks/states.pkl", "rb") as f:
    #     states = pickle.load(f)
    # state: GameState = states[0]
    # print(StateFeatures.extract_state_features_dict(state))
    # print(EntityManager.card(0).bonuses)
    # collect_states_actions()
    # collect_games_features()
    test_model()


if __name__ == "__main__":
    main()
