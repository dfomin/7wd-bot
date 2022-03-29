import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
from swd.action import PickWonderAction, PickProgressTokenAction, BuyCardAction, BuildWonderAction, DiscardCardAction, \
    PickStartPlayerAction, DestroyCardAction, PickDiscardedCardAction, Action
from swd.agents import RecordedAgent
from swd.cards_board import AGES, NO_CARD
from swd.entity_manager import EntityManager
from swd.states.cards_board_state import CardsBoardState
from swd.states.game_state import GameState, GameStatus
from swd.states.military_state_track import MilitaryTrackState
from swd.states.player_state import PlayerState

from swd_bot.thirdparty.loader import GameLogLoader
from swd_bot.thirdparty.sevenee import SeveneeLoader

CARDS_MAP = {
    100: 0,
    101: 1,
    102: 2,
    103: 3,
    104: 4,
    105: 5,
    106: 6,
    107: 7,
    108: 8,
    109: 12,
    110: 13,
    111: 19,
    112: 20,
    113: 21,
    114: 9,
    115: 10,
    116: 11,
    117: 14,
    118: 15,
    119: 16,
    120: 17,
    121: 18,
    122: 22,
    200: 23,
    201: 24,
    202: 25,
    203: 26,
    204: 27,
    205: 28,
    206: 42,
    207: 43,
    208: 44,
    209: 37,
    210: 29,
    211: 30,
    212: 31,
    213: 32,
    214: 33,
    215: 34,
    216: 35,
    217: 36,
    218: 38,
    219: 39,
    220: 40,
    221: 41,
    222: 45,
    300: 46,
    301: 47,
    302: 51,
    303: 52,
    304: 61,
    305: 62,
    306: 63,
    307: 55,
    308: 56,
    309: 57,
    310: 48,
    311: 49,
    312: 50,
    313: 53,
    314: 54,
    315: 58,
    316: 59,
    317: 60,
    318: 64,
    319: 65,
    400: 66,
    401: 67,
    402: 68,
    403: 69,
    404: 70,
    405: 71,
    406: 72,
}


ACTIONS_MAP: Dict[int, type] = {
    2: PickWonderAction,
    3: PickProgressTokenAction,
    4: BuyCardAction,
    5: BuildWonderAction,
    6: DiscardCardAction,
    7: PickStartPlayerAction,
    8: DestroyCardAction,
    9: PickProgressTokenAction,
    11: PickDiscardedCardAction
}


class SwdioLoader(GameLogLoader):
    @staticmethod
    def load(path: Union[str, Path]) -> Tuple[Optional[GameState], Optional[List[RecordedAgent]]]:
        game_log = json.loads(path.read_text())
        return SwdioLoader.process(game_log)

    @staticmethod
    def process(game_log: List[Dict[str, Any]]):
        name_to_index = {
            game_log[0]["move"]["p1"]: 0,
            game_log[0]["move"]["p2"]: 1,
        }

        tokens = [EntityManager.progress_token_names()[x - 1] for x in game_log[0]["move"]["tokens"]]
        rest_tokens = [EntityManager.progress_token_names()[x] for x in range(10) if x not in tokens]
        wonders = [x - 1 for x in game_log[0]["move"]["wonders"]]

        cards_preset = np.zeros((3, AGES[0].shape[0], AGES[0].shape[1]), dtype=int) + NO_CARD
        for age in range(3):
            epoch_cards = list(map(lambda x: CARDS_MAP[x], game_log[0]["move"]["cards"].get(f"{age + 1}")))
            if epoch_cards is None:
                break
            cards_preset[age][AGES[age] > 0] = epoch_cards

        actions: List[List[Action]] = [[], []]
        for i, action_item in enumerate(game_log):
            move = action_item["move"]
            action_id = move["id"]
            if action_id in ACTIONS_MAP:
                player_index = name_to_index[action_item["meta"]["actor"]]
                params = {}
                if "wonder" in move:
                    params["wonder_id"] = move["wonder"] - 1
                if "card" in move:
                    card_id = CARDS_MAP[move["card"]]
                    params["card_id"] = card_id
                    params["pos"] = SeveneeLoader.card_pos(card_id, cards_preset)
                if "token" in move:
                    params["progress_token"] = EntityManager.progress_token_names()[move["token"] - 1]
                if "player" in move:
                    params["player_index"] = name_to_index[move["player"]]

                try:
                    actions[player_index].append(ACTIONS_MAP[action_id](**params))
                except TypeError:
                    params.pop("pos", None)
                    actions[player_index].append(ACTIONS_MAP[action_id](**params))

        agents = []
        player_names = []
        for i in range(2):
            name = list(name_to_index.keys())[i]
            player_names.append(name)
            agent = RecordedAgent(actions[i])
            agents.append(agent)

        state = GameState(0,
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
                          {})
        state.meta_info["player_names"] = player_names

        return state, agents

    @staticmethod
    def encode_action(action: Action) -> Dict[str, Any]:
        reversed_actions_map = {v: k for k, v in ACTIONS_MAP.items()}
        reversed_cards_map = {v: k for k, v in CARDS_MAP.items()}
        result = {"id": reversed_actions_map[type(action)]}

        if hasattr(action, "card_id"):
            result["card"] = reversed_cards_map[action.card_id]
        if hasattr(action, "wonder_id"):
            result["wonder"] = action.wonder_id + 1
        if hasattr(action, "progress_token"):
            result["token"] = EntityManager.progress_token_names().index(action.progress_token) + 1
        if hasattr(action, "player_index"):
            result["player"] = action.player_index

        return result
