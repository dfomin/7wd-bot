import pickle
import time
from pathlib import Path
from typing import List, Callable, Optional

import pandas as pd
import torch
from swd.action import BuyCardAction, DiscardCardAction, BuildWonderAction
from swd.agents import Agent, RecordedAgent
from swd.game import Game
from tqdm import tqdm

from swd_bot.agents.mcts_agent import MCTSAgent
from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.data_providers.feature_extractor import FlattenEmbeddingsFeatureExtractor
# from swd_bot.game_features import GameFeatures
from swd_bot.model.torch_models import TorchBaseline
from swd_bot.thirdparty.sevenee import SeveneeLoader


torch_agent = TorchAgent()


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


def process_sevenee_games(process_function: Callable[[Game, List[Agent]], None]):
    # for file in Path(f"../../7wd/sevenee/").rglob("*.json"):
    #     state, agents = SeveneeLoader.load(file)
    #     if state is None:
    #         continue
    #     print(file)
    #     process_function(state, agents)
    state, agents = SeveneeLoader.load(Path(f"../../7wd/sevenee/48/0/0/FBtsCb8PDryQFrvaH.json"))
    process_function(state, agents)


def playout(original_game: Game) -> float:
    wins = 0
    total_games = 1
    for _ in tqdm(range(total_games)):
        game = original_game.clone()
        agent = TorchAgent()
        while not game.is_finished:
            actions = game.get_available_actions()
            selected_action = agent.choose_action(game, actions)
            game.apply_action(selected_action)
            agent.on_action_applied(selected_action, game)
        wins += game.winner == 0
    return wins / total_games


def estimate(game: Game, agent: Agent) -> float:
    actions = game.get_available_actions()
    agent.choose_action(game, actions)
    best_rate = None
    for action, child in agent.mcts.root.children.items():
        if action not in map(str, actions):
            continue
        if agent.mcts.root.current_player_index == child.current_player_index:
            rate = child.rate()
        else:
            rate = 1 - child.rate()
        if best_rate is None or best_rate < rate:
            best_rate = rate
    return best_rate if game.current_player_index == 1 else 1 - best_rate


def collect_states_actions():
    saved_states = [[], [], []]
    saved_actions = [[], [], []]
    saved_win_rates = [[], [], []]

    def save_state(game: Game, agents: List[Agent]):
        mcts_agent = None
        while not game.is_finished:
            actions = game.get_available_actions()
            agent = agents[game.current_player_index]
            selected_action = agent.choose_action(game, actions)
            if isinstance(selected_action, (BuyCardAction, DiscardCardAction, BuildWonderAction)):
                if mcts_agent is None:
                    mcts_agent = MCTSAgent(game.clone())

                index = 0
                if state.meta_info["season"] % 5 == 0:
                    index = 2
                elif state.meta_info["season"] % 5 == 4:
                    index = 1
                saved_states[index].append(state.clone())
                saved_actions[index].append(selected_action)

                win_rate = estimate(state, mcts_agent)
                saved_win_rates[index].append(win_rate)

            Game.apply_action(state, selected_action)
            if mcts_agent is not None:
                mcts_agent.on_action_applied(selected_action, state.clone())

    process_sevenee_games(save_state)

    for w in saved_win_rates[0]:
        print(w)

    suffixes = ["_train", "_valid", "_test"]
    for i in range(3):
        with open(f"../datasets/buy_discard_build/win_rates{suffixes[i]}.pkl", "wb") as f:
            pickle.dump(saved_win_rates[i], f)
    # for i in range(3):
    #     with open(f"../datasets/buy_discard_build/states{suffixes[i]}.pkl", "wb") as f:
    #         pickle.dump(saved_states[i], f)
    #
    #     with open(f"../datasets/buy_discard_build/actions{suffixes[i]}.pkl", "wb") as f:
    #         pickle.dump(saved_actions[i], f)


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
    model = TorchBaseline(505, 0, [50])
    model.load_state_dict(torch.load("../models/model_flat_acc53.42.pth"))
    model.eval()
    print(model)
    torch.onnx.export(model, (torch.randn(1, 505), None), "../models/model.onnx")
    return

    file = Path(f"../../7wd/sevenee/46/1/1/aRT22RJpJAGP8iPNs.json")
    # file = Path(f"../../7wd/sevenee/46/1/1/SuJYYgWE7fMFS8Dfi.json")
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
            extractor = FlattenEmbeddingsFeatureExtractor()
            features, cards = extractor.features(state)
            features = torch.tensor(features, dtype=torch.float)
            cards = torch.tensor(cards, dtype=torch.float)
            action, winner = model(features[None], cards[None])
            if isinstance(selected_action, BuyCardAction) or isinstance(selected_action, DiscardCardAction)\
                    or isinstance(selected_action, BuildWonderAction):
                if action[0].argmax().item() == selected_action.card_id:
                    correct += 1
                all += 1
            print(round(winner[0][0].item(), 2), round(winner[0][1].item(), 2))

        Game.apply_action(state, selected_action)
    print(correct / all)


def extract_state(game: Game, agents: List[Agent], stop_condition: Callable[[Game], bool]) -> Optional[Game]:
    while not game.is_finished:
        if stop_condition(game):
            return game
        actions = game.get_available_actions()
        selected_action = agents[game.current_player_index].choose_action(game, actions)
        game.apply_action(selected_action)


def main():
    # with open("../notebooks/states.pkl", "rb") as f:
    #     states = pickle.load(f)
    # state: GameState = states[0]
    # print(StateFeatures.extract_state_features_dict(state))
    # print(EntityManager.card(0).bonuses)
    # collect_states_actions()
    # collect_games_features()
    # test_model()
    # state, agents = SwdioLoader.load(Path("../../7wd/7wdio/log.json"))
    # test_game(state, agents, False)

    # state, agents = SeveneeLoader.load(Path(f"../../7wd/sevenee/48/1/1/MG4pdRWGH5aECmPKs.json"))
    # print(state.meta_info["player_names"])
    #
    # def f(s: GameState):
    #     return "DiscardCardAction(card_id=47, pos=(3, 1))" in map(str, Game.get_available_actions(s)) and s.age == 2
    # state = extract_state(state, agents, f)
    # agent = MCTSAgent(state)
    # estimate(state, agent)

    from swd_bot.editor.editor import play_against_ai
    play_against_ai()

    # start = time.time()
    # state = Game.create()
    # agent = MCTSAgent(state)
    # while not Game.is_finished(state):
    #     actions = Game.get_available_actions(state)
    #     selected_action = agent.choose_action(state, actions)
    #     Game.apply_action(state, selected_action)
    #     agent.on_action_applied(selected_action, state)
    # print(state.winner, time.time() - start)

    # test_games_correctness("../../7wd/sevenee/", SeveneeLoader)
    pass


if __name__ == "__main__":
    main()
