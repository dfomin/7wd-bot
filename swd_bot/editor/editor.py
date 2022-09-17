import pyglet
from swd.agents import ConsoleAgent, RandomAgent
from swd.game import Game

from swd_bot.agents.mcts_agent import MCTSAgent
from swd_bot.agents.torch_agent import TorchAgent
from swd_bot.editor.game_view import GameWindow


def play_against_ai():
    game = Game()
    game.price_cache = {}
    GameWindow(game, [MCTSAgent(game), RandomAgent()])
    # GameWindow(state, [TorchAgent(), RandomAgent()])
    pyglet.app.run()
