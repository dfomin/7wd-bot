import pyglet
from swd.agents import ConsoleAgent, RandomAgent
from swd.game import Game

from swd_bot.agents.mcts_agent import MCTSAgent
from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.editor.game_view import GameWindow


def play_against_ai():
    state = Game.create()
    state.price_cache = {}
    GameWindow(state, [MCTSAgent(state), RandomAgent()])
    # GameWindow(state, [TorchAgent(), RandomAgent()])
    pyglet.app.run()
