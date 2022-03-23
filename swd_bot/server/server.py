from typing import Optional, Dict, List, Any

from fastapi import FastAPI
from pydantic import BaseModel
from swd.game import Game

from swd_bot.swdio import SwdioLoader

app = FastAPI()


# class MetaItem(BaseModel):
#     actor: str
#
#
# class MoveItem(BaseModel):
#     id: int
#     p1: Optional[str]
#     p2: Optional[str]
#     cards: Optional[Dict[str, List[int]]]
#     tokens: Optional[List[int]]
#     wonders: Optional[List[int]]
#     wonder: Optional[int]
#     card: Optional[int]
#     token: Optional[int]
#     player: Optional[str]
#
#
# class Item(BaseModel):
#     meta: MetaItem
#     move: MoveItem


@app.post("/")
def process_game_log(log: List[Dict[str, Any]]):
    state, agents = SwdioLoader.process(log)
    while not Game.is_finished(state):
        actions = Game.get_available_actions(state)
        if Game.is_finished(state):
            break
        agent = agents[state.current_player_index]
        selected_action = agent.choose_action(state, actions)
        Game.apply_action(state, selected_action)
    return {"winner": state.winner}
