import random

import gym
from swd.game import Game


class SWDEnv(gym.Env):
    def __init__(self):
        self.state = None
        self.ai_player_id = random.randint(0, 1)

    def step(self, action):
        self.state = Game.apply_action(self.state, action)
        reward = 0
        is_done = Game.is_finished(self.state)
        if is_done:
            reward = int(self.state.winner != self.ai_player_id)
        return self.state, reward, is_done, {}

    def reset(self):
        return Game.create()

    def render(self, mode="human"):
        pass
