from typing import Dict, List, Any

from fastapi import FastAPI
from swd.game import Game

from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.thirdparty.swdio import SwdioLoader, ACTIONS_MAP

app = FastAPI()


@app.post("/7wd-bot/")
def process_game_log(log: List[Dict[str, Any]]):
    state, agents = SwdioLoader.process(log)
    while not Game.is_finished(state):
        actions = Game.get_available_actions(state)
        if Game.is_finished(state):
            break
        agent = agents[state.current_player_index]
        if len(agent.actions) == 0:
            break
        selected_action = agent.choose_action(state, actions)
        Game.apply_action(state, selected_action)

    if not Game.is_finished(state):
        agent = TorchAgent()
        actions = Game.get_available_actions(state)
        selected_action = agent.choose_action(state, actions)
        print(selected_action)
        for action_id, action_type in ACTIONS_MAP.items():
            if action_type == type(selected_action):
                return SwdioLoader.encode_action(selected_action)

    return {"winner": state.winner}
