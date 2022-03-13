import json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from swd.action import Action, BuyCardAction, DiscardCardAction, DestroyCardAction, PickWonderAction, BuildWonderAction, \
    PickStartPlayerAction, PickProgressTokenAction, PickDiscardedCardAction
from swd.agents import RecordedAgent
from swd.cards_board import AGES, NO_CARD, CardsBoardState
from swd.game import GameState, GameStatus
from swd.military_track import MilitaryTrackState
from swd.player import PlayerState


class SeveneeLoader:
    @staticmethod
    def load(path: Path) -> Tuple[Optional[GameState], Optional[List[RecordedAgent]]]:
        game_log = json.loads(path.read_text())

        if game_log["result"] == "TERMINATED":
            return None, None

        victory = game_log["data"]["result"]["victory"]
        if victory == "system":
            return None, None

        tokens = []
        rest_tokens = []
        wonders = []
        sevenee_cards_preset = []
        cards_preset = np.zeros((3, AGES[0].shape[0], AGES[0].shape[1]), dtype=int) + NO_CARD

        actions: List[List[Action]] = [[], []]
        for i, action_item in enumerate(game_log["actionItems"]):
            agent = action_item["agent"]["t"]
            action = action_item["action"]
            action_type = action["type"]
            if action_type == "drawProgressTokens":
                tokens = action["progressTokens"]
                rest_tokens = action["reservedProgressTokens"]
            elif action_type == "drawWonders":
                wonders = action["wonders"]
            elif action_type == "drawCards":
                age_cards = []
                for row in action["cards"]:
                    for card in row:
                        if card is not None:
                            age_cards.append(card)
                sevenee_cards_preset.append(age_cards)
                for age in range(len(sevenee_cards_preset)):
                    cards_preset[age][AGES[age] > 0] = sevenee_cards_preset[age]
            if agent != "p":
                continue
            player_index = action["playerIndex"]
            if action_type == "buyCard":
                buy_action = BuyCardAction(action["card"], SeveneeLoader.card_pos(action["card"], cards_preset))
                actions[player_index].append(buy_action)
            elif action_type == "discardCard":
                discard_action = DiscardCardAction(action["card"], SeveneeLoader.card_pos(action["card"], cards_preset))
                actions[player_index].append(discard_action)
            elif action_type == "killCard":
                actions[player_index].append(DestroyCardAction(action["card"]))
            elif action_type == "pickWonder":
                actions[player_index].append(PickWonderAction(action["wonder"]))
            elif action_type == "buildWonder":
                pos = SeveneeLoader.card_pos(action["card"], cards_preset)
                build_action = BuildWonderAction(action["wonder"], action["card"], pos)
                actions[player_index].append(build_action)
            elif action_type == "pickStartPlayer":
                actions[player_index].append(PickStartPlayerAction(action["pickedPlayerIndex"]))
            elif action_type == "pickProgressToken":
                actions[player_index].append(PickProgressTokenAction(action["progressToken"]))
            elif action_type == "pickDiscardedCard":
                actions[player_index].append(PickDiscardedCardAction(action["card"]))
            else:
                print(action_type)
                assert False

        agents = []

        player_names = []
        for i in range(2):
            name = game_log["players"][i]["name"]
            player_names.append(name)
            agent = RecordedAgent(actions[i])
            agents.append(agent)

        game_state = GameState(0,
                               0,
                               tokens,
                               rest_tokens,
                               [],
                               False,
                               wonders,
                               [PlayerState(0), PlayerState(1)],
                               MilitaryTrackState(),
                               GameStatus.PICK_WONDER,
                               None,
                               CardsBoardState(0, np.array([]), np.array([]), np.array([]), cards_preset),
                               game_log["data"])
        game_state.meta_info["player_names"] = player_names
        game_state.meta_info["division"] = int(str(path).split("/")[-3])
        game_state.meta_info["season"] = int(str(path).split("/")[-4])
        game_state.meta_info["path"] = str(path)
        return game_state, agents

    @staticmethod
    def card_pos(card_id: int, cards_preset: np.ndarray) -> Tuple[int, int]:
        index = np.transpose(np.where(cards_preset == card_id))[0]
        return tuple(index[1:])
