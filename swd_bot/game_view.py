from enum import Enum, auto
from typing import List

import pyglet
from pyglet.sprite import Sprite
from pyglet.window import Window, key
from swd.agents import ConsoleAgent, Agent
from swd.bonuses import CARD_COLOR, BONUSES
from swd.cards_board import NO_CARD
from swd.entity_manager import EntityManager
from swd.game import Game
from swd.states.game_state import GameState
from swd.states.player_state import PlayerState

from swd_bot.agents.torch_agent import TorchAgent

MILITARY_TRACK_REGION = (123, 10, 104, 34)
PAWN_STEP = 23
PAWN_Y = 78
PROGRESS_TOKEN_Y = 84

WONDERS_SPRITE_LIST = [None, None, None, None, None, 0, 1, None, None, None, 3, 7, 5, 10, 9, 8, 2, 4, 11, 6]
CARDS_SPRITE_LIST = [
    None, None, None, None, None, None, None, None, None, None, None, None,
    72, None, None, None, None, None, None, None, None, None, None, None,
    55, 58, 59, 56, 60, 57, 66, 67, 68, 69, 70, 71,
    49, 48, 47, 51, 53, 52, 54, 65, 61, 62, 64, 63,
    33, 45, 42, 43, 44, 39, 38, 37, 40, 41, 50, 46,
    25, 24, 26, 27, 29, 30, 28, 31, 32, 35, 36, 34,
    14, 12, 15, 13, 22, 19, 20, 21, 16, 17, 18, 23,
    0, 5, 2, 1, 4, 3, 6, 7, 9, 10, 11, 8
]
TOKENS_SPRITE_LIST = [None, None, None, None, 8, 9, None, None, 4, 5, 6, 7, 0, 1, 2, 3]


class Mode(Enum):
    GAME_BOARD = auto()
    PLAYER1 = auto()
    PLAYER2 = auto()
    DISCARD_PILE = auto()


class GameWindow(Window):
    def __init__(self, state: GameState, agents: List[Agent]):
        super(GameWindow, self).__init__(1280, 720)

        self.state = state
        self.agents = agents
        self.mode = Mode.GAME_BOARD

        self.buildings = pyglet.image.ImageGrid(pyglet.resource.image("resources/buildings_v3.webp"), 8, 12)
        self.building_sprites = []

        self.wonders = pyglet.image.ImageGrid(pyglet.resource.image("resources/wonders_v3.webp"), 4, 5)
        self.wonder_sprites = []

        self.progress_tokens = pyglet.image.ImageGrid(pyglet.resource.image("resources/progress_tokens_v3.webp"), 4, 4)
        self.progress_tokens_sprites = []

        self.military_track = Sprite(pyglet.resource.image("resources/board.webp"))
        self.military_track.scale = 0.5
        self.military_track.rotation = -90
        self.military_track.x = self.width // 2 + self.military_track.height // 2
        self.military_track.y = 0

        self.conflict_pawn = Sprite(pyglet.resource.image("resources/sprites.webp").get_region(*MILITARY_TRACK_REGION))
        self.conflict_pawn.scale = 0.5
        self.conflict_pawn.rotation = 90
        self.conflict_pawn.y = PAWN_Y

        self.state_updated()

    def state_updated(self):
        self.building_sprites = []
        self.wonder_sprites = []
        self.progress_tokens_sprites = []

        x_shift = None
        x_space = None
        for row_index, row in enumerate(self.state.cards_board_state.card_places):
            for i, card_id in enumerate(row):
                if card_id >= 0:
                    sprite = Sprite(self.buildings[CARDS_SPRITE_LIST.index(card_id)])
                elif card_id == NO_CARD:
                    continue
                else:
                    sprite = Sprite(self.buildings[0])
                sprite.scale = 0.5
                if x_shift is None:
                    x_shift = self.width // 2 - sprite.width - sprite.width // 5
                    x_space = sprite.width // 5
                sprite.x = sprite.width * i - ((sprite.width + x_space) // 2) * row_index + x_space * i + x_shift
                sprite.y = self.height - sprite.height * (row_index + 1) + sprite.height // 2 * row_index
                self.building_sprites.append(sprite)

        for i, player_state in enumerate(self.state.players_state):
            for j, wonder in enumerate(player_state.wonders):
                sprite = Sprite(self.wonders[WONDERS_SPRITE_LIST.index(wonder[0])])
                sprite.scale = 0.5
                sprite.x = i * (self.width - sprite.width)
                sprite.y = self.height - (j + 1) * sprite.height - j * sprite.height // 10
                if wonder[1] is not None:
                    sprite.opacity = 32
                self.wonder_sprites.append(sprite)

        token_names = EntityManager.progress_token_names()
        for i, token in enumerate(self.state.progress_tokens):
            index = token_names.index(token)
            sprite = Sprite(self.progress_tokens[TOKENS_SPRITE_LIST.index(index)])
            sprite.scale = 0.25
            sprite.x = self.width // 2 - sprite.width // 2 + (i - 2) * sprite.width
            sprite.y = PROGRESS_TOKEN_Y
            self.progress_tokens_sprites.append(sprite)

        pawn_shift = self.state.military_track_state.conflict_pawn * PAWN_STEP
        self.conflict_pawn.x = self.width // 2 - self.conflict_pawn.height // 2 + pawn_shift

        for i, player_state in enumerate(self.state.players_state):
            print(f"Player {i}: {player_state.coins} {Game.points(self.state, i)[0]} "
                  f"{player_state.bonuses[0:3]}({player_state.bonuses[5]}) "
                  f"{player_state.bonuses[3:5]}({player_state.bonuses[6]})")

    def on_draw(self):
        self.clear()
        if self.mode == Mode.GAME_BOARD:
            self.draw_game_board()
        elif self.mode == Mode.PLAYER1:
            self.draw_player(0)
        elif self.mode == Mode.PLAYER2:
            self.draw_player(1)
        elif self.mode == Mode.DISCARD_PILE:
            self.draw_discard_pile()

    def draw_game_board(self):
        for sprite in self.building_sprites:
            sprite.draw()

        for sprite in self.wonder_sprites:
            sprite.draw()

        self.military_track.draw()
        self.conflict_pawn.draw()

        for sprite in self.progress_tokens_sprites:
            sprite.draw()

    def draw_player(self, player_index: int):
        player_state: PlayerState = self.state.players_state[player_index]
        self.draw_cards_and_tokens(player_state.cards, player_state.progress_tokens)

    def draw_discard_pile(self):
        self.draw_cards_and_tokens(self.state.discard_pile, self.state.rest_progress_tokens)

    def draw_cards_and_tokens(self, cards: List[int], tokens: List[str]):
        card_ids = []
        for color in CARD_COLOR:
            color_index = BONUSES.index(color)
            card_ids.extend([card_id for card_id in sorted(cards) if EntityManager.card(card_id).bonuses[color_index] > 0])
        for i, card_id in enumerate(card_ids):
            sprite = Sprite(self.buildings[CARDS_SPRITE_LIST.index(card_id)])
            sprite.scale = 0.5
            sprite.x = (i % 15) * sprite.width
            sprite.y = self.height - sprite.height * (i // 15 + 1)
            sprite.draw()

        token_names = EntityManager.progress_token_names()
        for i, token in enumerate(tokens):
            index = token_names.index(token)
            sprite = Sprite(self.progress_tokens[TOKENS_SPRITE_LIST.index(index)])
            sprite.scale = 0.5
            sprite.x = i * sprite.width
            sprite.y = 0
            sprite.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key._1:
            self.mode = Mode.PLAYER1
        elif symbol == key._2:
            self.mode = Mode.PLAYER2
        elif symbol == key._3:
            self.mode = Mode.DISCARD_PILE
        elif symbol == key._0:
            self.mode = Mode.GAME_BOARD
        elif symbol == key.ESCAPE:
            self.close()
        elif symbol == key.SPACE:
            self.move()

    def move(self):
        actions = Game.get_available_actions(self.state)
        selected_action = self.agents[self.state.current_player_index].choose_action(self.state, actions)
        Game.apply_action(self.state, selected_action)
        for agent in self.agents:
            agent.on_action_applied(selected_action, self.state)
        if Game.is_finished(self.state):
            print(f"Winner: {self.state.winner}")
        self.state_updated()
