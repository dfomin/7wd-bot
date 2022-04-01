from enum import Enum, auto
from typing import List, Optional, Tuple

import pyglet
from pyglet.sprite import Sprite
from pyglet.window import Window, key, mouse
from swd.action import PickWonderAction, BuyCardAction, DiscardCardAction, BuildWonderAction, Action, \
    PickStartPlayerAction, DestroyCardAction
from swd.agents import Agent
from swd.bonuses import CARD_COLOR, BONUSES
from swd.cards_board import NO_CARD
from swd.entity_manager import EntityManager
from swd.game import Game
from swd.states.game_state import GameState, GameStatus
from swd.states.player_state import PlayerState

from swd_bot.editor.sprites.card_sprite import CardSprite
from swd_bot.editor.sprites.draft_wonder_sprite import DraftWonderSprite
from swd_bot.editor.sprites.progress_token_sprite import ProgressTokenSprite
from swd_bot.editor.sprites.wonder_sprite import WonderSprite

MILITARY_TRACK_REGION = (123, 10, 104, 34)
PAWN_STEP = 23
PAWN_Y = 78
PROGRESS_TOKEN_Y = 84


class Mode(Enum):
    GAME_BOARD = auto()
    PLAYER1 = auto()
    PLAYER2 = auto()
    DISCARD_PILE = auto()
    EDITOR = auto()


class GameWindow(Window):
    def __init__(self, state: GameState, agents: List[Agent]):
        super(GameWindow, self).__init__(1280, 720)

        self.state = state
        self.agents = agents
        self.mode = Mode.GAME_BOARD

        self.selected_wonder = None
        self.editor_pos = None

        self.card_sprites = []
        self.pick_wonder_sprites = []
        self.wonder_sprites = []
        self.progress_tokens_sprites = []
        self.card_list_sprites = []
        self.progress_tokens_list_sprites = []

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
        self.pick_wonder_sprites = []
        self.card_sprites = []
        self.wonder_sprites = []
        self.progress_tokens_sprites = []

        if len(self.state.wonders) > 4:
            pick_wonders = self.state.wonders[:-4]
        else:
            pick_wonders = self.state.wonders

        for row, wonder in enumerate(pick_wonders):
            sprite = DraftWonderSprite(wonder)
            sprite.x = self.width // 2 - sprite.width // 2
            sprite.y = self.height - (row + 1) * sprite.height - row * sprite.height // 10
            self.pick_wonder_sprites.append(sprite)

        x_shift = None
        x_space = None
        for row_index, row in enumerate(self.state.cards_board_state.card_places):
            for i, card_id in enumerate(row):
                if card_id == NO_CARD:
                    continue
                else:
                    sprite = CardSprite(card_id, (row_index, i))
                if x_shift is None:
                    x_shift = self.width // 2 - sprite.width - sprite.width // 5
                    x_space = sprite.width // 5
                sprite.x = sprite.width * i - ((sprite.width + x_space) // 2) * row_index + x_space * i + x_shift
                sprite.y = self.height - sprite.height * (row_index + 1) + sprite.height // 2 * row_index
                self.card_sprites.append(sprite)

        for i, player_state in enumerate(self.state.players_state):
            for j, wonder in enumerate(player_state.wonders):
                sprite = WonderSprite(wonder[0])
                sprite.x = i * (self.width - sprite.width)
                sprite.y = self.height - (j + 1) * sprite.height - j * sprite.height // 10
                if wonder[1] is not None:
                    sprite.opacity = 32
                self.wonder_sprites.append(sprite)

        for i, token in enumerate(self.state.progress_tokens):
            sprite = ProgressTokenSprite(token)
            sprite.scale = 0.25
            sprite.x = self.width // 2 - sprite.width // 2 + (i - 2) * sprite.width
            sprite.y = PROGRESS_TOKEN_Y
            self.progress_tokens_sprites.append(sprite)

        pawn_shift = self.state.military_track_state.conflict_pawn * PAWN_STEP
        self.conflict_pawn.x = self.width // 2 - self.conflict_pawn.height // 2 + pawn_shift

        if Game.is_finished(self.state):
            print(f"Winner: {self.state.winner}")

        # for i, player_state in enumerate(self.state.players_state):
        #     opponent_state = self.state.players_state[1 - i]
        #     print(f"Player {i}: {player_state.coins} {Game.points(self.state, i)[0]} "
        #           f"{player_state.bonuses[0:3]}({player_state.bonuses[5]}) "
        #           f"{player_state.bonuses[3:5]}({player_state.bonuses[6]}) "
        #           f"{Player.assets(player_state, Player.resources(opponent_state), None).resources_cost} "
        #           f"{Player.scientific_symbols(player_state)}")

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
        elif self.mode == Mode.EDITOR:
            self.draw_editor()

    def on_mouse_release(self, x, y, button, modifiers):
        if self.mode == Mode.EDITOR:
            if self.editor_pos is None:
                for sprite in reversed(self.card_sprites):
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        if self.editor_pos is None:
                            self.editor_pos = sprite.pos
                            break
            else:
                for sprite in self.card_list_sprites:
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        old_id = self.state.cards_board_state.card_places[self.editor_pos]
                        if old_id != sprite.card_id:
                            self.state.cards_board_state.card_places[self.editor_pos] = sprite.card_id
                            for array in [self.state.cards_board_state.card_ids,
                                          self.state.cards_board_state.purple_card_ids]:
                                array[array == sprite.card_id] = old_id
                        self.editor_pos = None
            self.state_updated()
            return

        available_actions = Game.get_available_actions(self.state)

        if self.state.game_status == GameStatus.PICK_START_PLAYER:
            action = PickStartPlayerAction(0 if x <= self.width // 2 else 1)
            self.apply_action(action)
            return
        elif self.state.game_status in [GameStatus.DESTROY_BROWN, GameStatus.DESTROY_GRAY, GameStatus.SELECT_DISCARDED]:
            for sprite in self.card_list_sprites:
                if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                    for action in available_actions:
                        if action.card_id == sprite.card_id:
                            self.apply_action(action)
                            return
        elif self.state.game_status in [GameStatus.PICK_REST_PROGRESS_TOKEN]:
            for sprite in self.progress_tokens_list_sprites:
                if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                    for action in available_actions:
                        if action.progress_token == sprite.progress_token:
                            self.apply_action(action)
                            return
        elif self.state.game_status in [GameStatus.PICK_PROGRESS_TOKEN]:
            for sprite in self.progress_tokens_sprites:
                if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                    for action in available_actions:
                        if action.progress_token == sprite.progress_token:
                            self.apply_action(action)
                            return

        for sprite in self.pick_wonder_sprites:
            if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                action = PickWonderAction(sprite.wonder_id)
                if str(action) in map(str, available_actions):
                    self.apply_action(action)
                    return

        for sprite in self.card_sprites:
            if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                if self.selected_wonder is not None:
                    action = BuildWonderAction(self.selected_wonder.wonder_id, sprite.card_id, sprite.pos)
                elif button & mouse.LEFT:
                    action = BuyCardAction(sprite.card_id, sprite.pos)
                else:
                    action = DiscardCardAction(sprite.card_id, sprite.pos)
                if str(action) in map(str, available_actions):
                    self.apply_action(action)
                    return

        wonders = self.state.players_state[self.state.current_player_index].wonders
        unbuilt_wonders = [x[0] for x in wonders if x[1] is None]
        for sprite in self.wonder_sprites:
            if sprite.wonder_id not in unbuilt_wonders:
                continue
            if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                self.select_wonder(sprite)
                return

        self.deselect_wonder()

    def select_wonder(self, sprite: WonderSprite):
        self.deselect_wonder()
        self.selected_wonder = sprite
        sprite.opacity = 128

    def deselect_wonder(self):
        if self.selected_wonder is not None:
            self.selected_wonder.opacity = 255
        self.selected_wonder = None

    def draw_game_board(self):
        if self.state.game_status in [GameStatus.PICK_WONDER, GameStatus.NORMAL_TURN, GameStatus.PICK_PROGRESS_TOKEN,
                                      GameStatus.PICK_START_PLAYER, GameStatus.FINISHED]:
            for sprite in self.pick_wonder_sprites:
                sprite.draw()

            for sprite in self.card_sprites:
                sprite.draw()

            for sprite in self.wonder_sprites:
                sprite.draw()

            self.military_track.draw()
            self.conflict_pawn.draw()

            for sprite in self.progress_tokens_sprites:
                sprite.draw()
        elif self.state.game_status == GameStatus.PICK_REST_PROGRESS_TOKEN:
            # self.draw_cards_and_tokens([], [x.progress_token for x in Game.get_available_actions(self.state)])
            self.draw_cards_and_tokens([], self.state.rest_progress_tokens)
        elif self.state.game_status in [GameStatus.DESTROY_BROWN, GameStatus.DESTROY_GRAY, GameStatus.SELECT_DISCARDED]:
            self.draw_cards_and_tokens([x.card_id for x in Game.get_available_actions(self.state)], [])

    def draw_editor(self):
        if self.editor_pos is None:
            for sprite in self.card_sprites:
                sprite.draw()
        else:
            card_ids = []
            if self.state.age == 0:
                card_ids = list(range(23))
            elif self.state.age == 1:
                card_ids = list(range(23, 46))
            elif self.state.age == 2:
                card_ids = list(range(46, 73))
            self.draw_cards_and_tokens(card_ids, [])

    def draw_player(self, player_index: int):
        player_state: PlayerState = self.state.players_state[player_index]
        self.draw_cards_and_tokens(player_state.cards, player_state.progress_tokens)

    def draw_discard_pile(self):
        self.draw_cards_and_tokens(self.state.discard_pile, self.state.rest_progress_tokens)

    def draw_cards_and_tokens(self, cards: List[int], tokens: List[str]):
        card_ids = []
        self.card_list_sprites = []
        self.progress_tokens_list_sprites = []
        for color in CARD_COLOR:
            color_index = BONUSES.index(color)
            card_ids.extend([card_id for card_id in sorted(cards) if EntityManager.card(card_id).bonuses[color_index] > 0])
        for i, card_id in enumerate(card_ids):
            sprite = CardSprite(card_id)
            sprite.x = (i % 15) * sprite.width
            sprite.y = self.height - sprite.height * (i // 15 + 1)
            self.card_list_sprites.append(sprite)
            sprite.draw()

        for i, token in enumerate(tokens):
            sprite = ProgressTokenSprite(token)
            sprite.x = i * sprite.width
            sprite.y = 0
            self.progress_tokens_list_sprites.append(sprite)
            sprite.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key._1:
            self.mode = Mode.PLAYER1
        elif symbol == key._2:
            self.mode = Mode.PLAYER2
        elif symbol == key._3:
            self.mode = Mode.DISCARD_PILE
        elif symbol == key._4:
            self.mode = Mode.EDITOR
        elif symbol == key._0:
            self.mode = Mode.GAME_BOARD
        elif symbol == key.ESCAPE:
            self.close()
        elif symbol == key.SPACE:
            self.move()

    def apply_action(self, action: Action):
        Game.apply_action(self.state, action)
        for agent in self.agents:
            agent.on_action_applied(action, self.state)
        self.deselect_wonder()
        self.state_updated()

    def move(self):
        if Game.is_finished(self.state):
            return

        actions = Game.get_available_actions(self.state)
        selected_action = self.agents[self.state.current_player_index].choose_action(self.state, actions)
        Game.apply_action(self.state, selected_action)
        for agent in self.agents:
            agent.on_action_applied(selected_action, self.state)
        if Game.is_finished(self.state):
            print(f"Winner: {self.state.winner}")
        self.state_updated()
