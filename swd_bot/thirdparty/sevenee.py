import json
from pathlib import Path
from typing import Tuple, Optional, List, Union, Dict, Any

from swd.action import Action, BuyCardAction, DiscardCardAction, DestroyCardAction, PickWonderAction, \
    BuildWonderAction, PickStartPlayerAction, PickProgressTokenAction, PickDiscardedCardAction
from swd.agents import RecordedAgent
from swd.board_card import BoardCard
from swd.entity_manager import EntityManager
from swd.game import Game
from swd.player import Player

from swd_bot.thirdparty.loader import GameLogLoader


class SeveneeLoader(GameLogLoader):
    @staticmethod
    def load(path: Union[str, Path]) -> Tuple[Optional[Game], Optional[List[RecordedAgent]]]:
        game_log = json.loads(path.read_text())

        if game_log["result"] == "TERMINATED":
            return None, None

        victory = game_log["data"]["result"]["victory"]
        if victory == "system":
            return None, None

        tokens = []
        rest_tokens = []
        wonders = []
        cards_preset = []

        token_names_map = {}

        actions: List[List[Dict[str, Any]]] = [[], []]
        for i, action_item in enumerate(game_log["actionItems"]):
            agent = action_item["agent"]["t"]
            action = action_item["action"]
            action_type = action["type"]
            if action_type == "drawProgressTokens":
                for token_id in range(EntityManager.progress_tokens_count()):
                    token = EntityManager.progress_token(token_id)
                    token_names_map[token.name] = token.id
                tokens = [EntityManager.progress_token(token_names_map[name])
                          for name in action["progressTokens"]]
                rest_tokens = [EntityManager.progress_token(token_names_map[name])
                               for name in action["reservedProgressTokens"]]
            elif action_type == "drawWonders":
                wonders = [EntityManager.wonder(wonder_id) for wonder_id in action["wonders"]]
                for wonder in wonders:
                    wonder.card = None
            elif action_type == "drawCards":
                age_cards = []
                for row_id, row in enumerate(action["cards"]):
                    cards_row = []
                    for column, card in enumerate(row):
                        if card is not None:
                            cards_row.append(EntityManager.card(card))
                    age_cards.append(cards_row)
                cards_preset.append(age_cards)
            if agent != "p":
                continue
            player_index = action["playerIndex"]
            if action_type == "buyCard":
                actions[player_index].append({
                    "type": BuyCardAction,
                    "card_id": action["card"]
                })
            elif action_type == "discardCard":
                actions[player_index].append({
                    "type": DiscardCardAction,
                    "card_id": action["card"]
                })
            elif action_type == "killCard":
                actions[player_index].append({
                    "type": DestroyCardAction,
                    "card_id": action["card"]
                })
            elif action_type == "pickWonder":
                actions[player_index].append({
                    "type": PickWonderAction,
                    "wonder_id": action["wonder"]
                })
            elif action_type == "buildWonder":
                actions[player_index].append({
                    "type": BuildWonderAction,
                    "wonder_id": action["wonder"],
                    "card_id": action["card"]
                })
            elif action_type == "pickStartPlayer":
                actions[player_index].append({
                    "type": PickStartPlayerAction,
                    "player_index": action["pickedPlayerIndex"]
                })
            elif action_type == "pickProgressToken":
                token_id = token_names_map[action["progressToken"]]
                actions[player_index].append({
                    "type": PickProgressTokenAction,
                    "token_id": token_id
                })
            elif action_type == "pickDiscardedCard":
                actions[player_index].append({
                    "type": PickDiscardedCardAction,
                    "card_id": action["card"]
                })
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

        game = Game()
        game.progress_tokens = tokens
        game.rest_progress_tokens = rest_tokens
        game.wonders = wonders
        game.players = [Player(0), Player(1)]
        game.cards_board.preset = cards_preset

        game.meta_info = game_log["data"]
        game.meta_info["player_names"] = player_names
        game.meta_info["division"] = int(str(path).split("/")[-3])
        game.meta_info["season"] = int(str(path).split("/")[-4])
        game.meta_info["path"] = str(path)

        return game, agents

    @staticmethod
    def card_pos(card_id: int, cards_preset: List[List[List[int]]]) -> Tuple[int, int]:
        for age_preset in cards_preset:
            for row_id, row in enumerate(age_preset):
                if card_id in row:
                    return row_id, row.index(card_id)
        raise ValueError
