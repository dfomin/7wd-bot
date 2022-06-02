import logging
from typing import Dict, List, Any

from fastapi import FastAPI
from swd.game import Game
from swd.states.game_state import GameStatus

from swd_bot.agents.mcts_agent import MCTSAgent
from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.thirdparty.swdio import SwdioLoader, ACTIONS_MAP


app = FastAPI()


@app.post("/7wd-bot/state/")
def process_game_state(state_description: Dict[str, Any]):
    state = SwdioLoader.parse_state(state_description)

    if not Game.is_finished(state):
        agent = MCTSAgent(state)
        actions = Game.get_available_actions(state)
        selected_action = agent.choose_action(state, actions)
        logging.info(selected_action)
        for action_id, action_type in ACTIONS_MAP.items():
            if action_type == type(selected_action):
                encoded_action = SwdioLoader.encode_action(selected_action)
                if state.game_status == GameStatus.PICK_PROGRESS_TOKEN:
                    encoded_action["id"] = 3
                elif state.game_status == GameStatus.PICK_REST_PROGRESS_TOKEN:
                    encoded_action["id"] = 9
                elif state.game_status == GameStatus.PICK_START_PLAYER:
                    if state.current_player_index == encoded_action["player"]:
                        encoded_action["player"] = state_description["state"]["me"]["name"]
                    else:
                        encoded_action["player"] = state_description["state"]["enemy"]["name"]
                return encoded_action

    return {"winner": state.winner}


@app.post("/7wd-bot/log/")
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
        logging.info(selected_action)
        for action_id, action_type in ACTIONS_MAP.items():
            if action_type == type(selected_action):
                return SwdioLoader.encode_action(selected_action)

    return {"winner": state.winner}


@app.get("/7wd-bot/ping/")
def process_ping():
    return "pong"
