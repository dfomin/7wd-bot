from enum import Enum, auto
from typing import List

import numpy as np
import pyglet
from pyglet.sprite import Sprite
from pyglet.text import Label
from pyglet.window import Window, key, mouse
from swd.action import PickWonderAction, BuyCardAction, DiscardCardAction, BuildWonderAction, Action, \
    PickStartPlayerAction, PickProgressTokenAction
from swd.agents import Agent
from swd.bonuses import CARD_COLOR, BONUSES, BONUSES_INDEX
from swd.entity_manager import EntityManager
from swd.game import Game, GameStatus
from swd.player import Player

from swd_bot.agents.mcts_agent import MCTSAgent
from swd_bot.editor.sprite_loader import SpriteLoader
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
    def __init__(self, game: Game, agents: List[Agent]):
        super(GameWindow, self).__init__(1280, 720)

        self.game = game
        self.prev_game = None
        self.last_action = None
        # self.agents = agents
        self.mcts = MCTSAgent(self.game)
        self.mode = Mode.GAME_BOARD

        self.selected_wonder = None
        self.editor_pos = None
        self.editor_wonder = None

        self.card_sprites = []
        self.pick_wonder_sprites = []
        self.wonder_sprites = []
        self.progress_tokens_sprites = []
        self.card_list_sprites = []
        self.progress_tokens_list_sprites = []
        self.wonder_list_sprites = []

        self.player_marker = Sprite(SpriteLoader.progress_tokens()[3])
        self.player_marker.scale = 0.5
        self.player_marker.x = 0
        self.player_marker.y = 20

        self.military_track = Sprite(pyglet.resource.image("resources/board.webp"))
        self.military_track.scale = 0.5
        self.military_track.rotation = -90
        self.military_track.x = self.width // 2 + self.military_track.height // 2
        self.military_track.y = 0

        self.conflict_pawn = Sprite(pyglet.resource.image("resources/sprites.webp").get_region(*MILITARY_TRACK_REGION))
        self.conflict_pawn.scale = 0.5
        self.conflict_pawn.rotation = 90
        self.conflict_pawn.y = PAWN_Y

        self.player_labels = [
            Label("", x=0, y=0, anchor_x="left", anchor_y="bottom"),
            Label("", x=self.width, y=0, anchor_x="right", anchor_y="bottom")
        ]

        self.best_move_label = Label("", x=self.width // 2, y=140, anchor_x="center", anchor_y="bottom")

        self.state_updated()

    def state_updated(self):
        self.pick_wonder_sprites = []
        self.card_sprites = []
        self.wonder_sprites = []
        self.progress_tokens_sprites = []

        if len(self.game.wonders) > 4:
            pick_wonders = self.game.wonders[:-4]
        else:
            pick_wonders = self.game.wonders

        self.player_marker.x = 0 if self.game.current_player_index == 0 else self.width - self.player_marker.width

        for row, wonder in enumerate(pick_wonders):
            sprite = DraftWonderSprite(wonder.id)
            sprite.x = self.width // 2 - sprite.width // 2
            sprite.y = self.height - (row + 1) * sprite.height - row * sprite.height // 10
            self.pick_wonder_sprites.append(sprite)

        x_shift = None
        x_space = None
        for row_index, row in enumerate(self.game.cards_board.card_places):
            for i, board_card in enumerate(row):
                sprite = CardSprite(board_card.card.id if board_card.card is not None else -1, (row_index, i))
                if x_shift is None:
                    x_shift = self.width // 2 - sprite.width - sprite.width // 5
                    x_space = sprite.width // 5
                sprite.x = sprite.width * i - ((sprite.width + x_space) // 2) * row_index + x_space * i + x_shift
                sprite.y = self.height - sprite.height * (row_index + 1) + sprite.height // 2 * row_index
                self.card_sprites.append(sprite)

        for i, player in enumerate(self.game.players):
            for j, wonder in enumerate(player.wonders):
                sprite = WonderSprite(wonder.id)
                sprite.x = i * (self.width - sprite.width)
                sprite.y = self.height - (j + 1) * sprite.height - j * sprite.height // 10
                if wonder.card is not None:
                    sprite.opacity = 32
                self.wonder_sprites.append(sprite)

        for i, token in enumerate(self.game.progress_tokens):
            sprite = ProgressTokenSprite(token.name)
            sprite.scale = 0.25
            sprite.x = self.width // 2 - sprite.width // 2 + (i - 2) * sprite.width
            sprite.y = PROGRESS_TOKEN_Y
            self.progress_tokens_sprites.append(sprite)

        pawn_shift = self.game.military_track.conflict_pawn * PAWN_STEP
        self.conflict_pawn.x = self.width // 2 - self.conflict_pawn.height // 2 + pawn_shift

        if self.game.is_finished:
            print(f"Winner: {self.game.winner}")

        # for i, player in enumerate(self.game.players):
            # opponent_state = self.state.players_state[1 - i]
            # self.player_labels[i].text = f"{player_state.coins} {Game.points(self.state, i)[0]} " + \
            #                              f"{player_state.bonuses[0:3]}({player_state.bonuses[5]}) " + \
            #                              f"{player_state.bonuses[3:5]}({player_state.bonuses[6]}) " + \
            #                              f"{Player.assets(player_state, Player.resources(opponent_state), None).resources_cost} " + \
            #                              f"{Player.scientific_symbols(player_state)}"

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
        self.best_move_label.text = ""

        if self.mode == Mode.EDITOR:
            if self.editor_pos is None and self.editor_wonder is None:
                for sprite in reversed(self.card_sprites):
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        if button & mouse.LEFT:
                            self.editor_pos = sprite.pos
                        elif self.game.age == 2:
                            board_card = self.game.cards_board.card_places[sprite.pos[0]][sprite.pos[1]]
                            if board_card.card is None:
                                board_card.is_purple_back = not board_card.is_purple_back
                        break
                for sprite in self.wonder_sprites:
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        self.editor_wonder = sprite.wonder_id
                        break
            elif self.editor_pos is not None:
                for sprite in self.card_list_sprites:
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        old_id = self.game.cards_board.card_places[self.editor_pos[0]][self.editor_pos[1]].card.id
                        if old_id != sprite.card_id:
                            board_state = self.game.cards_board

                            board_state.card_ids = board_state.card_ids[board_state.card_ids != sprite.card_id]
                            board_state.purple_card_ids = board_state.purple_card_ids[board_state.purple_card_ids != sprite.card_id]

                            if old_id >= 66:
                                if old_id not in board_state.purple_card_ids:
                                    board_state.purple_card_ids = np.append(board_state.purple_card_ids, old_id)
                            elif old_id >= 0:
                                if old_id not in board_state.card_ids:
                                    board_state.card_ids = np.append(board_state.card_ids, old_id)
                            self.game.cards_board.card_places[self.editor_pos[0]][self.editor_pos[1]] = sprite.card_id
                        self.editor_pos = None
            elif self.editor_wonder is not None:
                for sprite in self.wonder_list_sprites:
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        for player_state in self.game.players:
                            for i in range(len(player_state.wonders)):
                                if player_state.wonders[i][0] == self.editor_wonder:
                                    player_state.wonders[i] = sprite.wonder_id, None
                                    self.editor_wonder = None
                                    break
                for sprite in self.progress_tokens_list_sprites:
                    if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                        if sprite.progress_token in self.game.progress_tokens:
                            self.game.progress_tokens.remove(sprite.progress_token)
                            self.game.rest_progress_tokens.append(sprite.progress_token)
                        elif sprite.progress_token in self.game.rest_progress_tokens:
                            self.game.rest_progress_tokens.remove(sprite.progress_token)
                            self.game.progress_tokens.append(sprite.progress_token)

            self.state_updated()
            return

        if self.last_action is not None:
            return

        available_actions = self.game.get_available_actions()

        if self.game.game_status == GameStatus.PICK_START_PLAYER:
            action = PickStartPlayerAction(0 if x <= self.width // 2 else 1)
            self.apply_action(action)
            return
        elif self.game.game_status in [GameStatus.DESTROY_BROWN, GameStatus.DESTROY_GRAY, GameStatus.SELECT_DISCARDED]:
            for sprite in self.card_list_sprites:
                if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                    for action in available_actions:
                        if action.card_id == sprite.card_id:
                            self.apply_action(action)
                            return
        elif self.game.game_status in [GameStatus.PICK_REST_PROGRESS_TOKEN]:
            for sprite in self.progress_tokens_list_sprites:
                if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                    action = PickProgressTokenAction(sprite.progress_token)
                    self.apply_action(action)
                    return
        elif self.game.game_status in [GameStatus.PICK_PROGRESS_TOKEN]:
            for sprite in self.progress_tokens_sprites:
                if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                    for action in available_actions:
                        if action.progress_token == sprite.progress_token:
                            self.apply_action(action)
                            return

        for sprite in self.pick_wonder_sprites:
            if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                for action in available_actions:
                    if isinstance(action, PickWonderAction) and action.wonder.id == sprite.wonder_id:
                        self.apply_action(action)
                        return

        for sprite in self.card_sprites:
            if sprite.x <= x <= sprite.x + sprite.width and sprite.y <= y <= sprite.y + sprite.height:
                if self.selected_wonder is not None:
                    for action in available_actions:
                        if isinstance(action, BuildWonderAction) \
                                and action.wonder.id == self.selected_wonder.wonder_id \
                                and action.card.id == sprite.card_id:
                            self.apply_action(action)
                            return
                elif button & mouse.LEFT:
                    for action in available_actions:
                        if isinstance(action, BuyCardAction) and action.card.id == sprite.card_id:
                            self.apply_action(action)
                            return
                else:
                    for action in available_actions:
                        if isinstance(action, DiscardCardAction) and action.card.id == sprite.card_id:
                            self.apply_action(action)
                            return

        wonders = self.game.players[self.game.current_player_index].wonders
        unbuilt_wonders = [x.id for x in wonders if not x.is_built]
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
        self.best_move_label.draw()
        if self.game.game_status in [GameStatus.PICK_WONDER,
                                     GameStatus.NORMAL_TURN,
                                     GameStatus.PICK_PROGRESS_TOKEN,
                                     GameStatus.PICK_START_PLAYER,
                                     GameStatus.FINISHED]:
            for sprite in self.pick_wonder_sprites:
                sprite.draw()

            for sprite in self.card_sprites:
                sprite.draw()

            for sprite in self.wonder_sprites:
                sprite.draw()

            self.military_track.draw()
            self.conflict_pawn.draw()
            self.player_marker.draw()

            for sprite in self.progress_tokens_sprites:
                sprite.draw()

            for label in self.player_labels:
                label.draw()
        elif self.game.game_status == GameStatus.PICK_REST_PROGRESS_TOKEN:
            # self.draw_cards_and_tokens([], [x.progress_token for x in Game.get_available_actions(self.state)])
            self.draw_cards_and_tokens([], [x.name for x in self.game.rest_progress_tokens])
        elif self.game.game_status in [GameStatus.DESTROY_BROWN, GameStatus.DESTROY_GRAY, GameStatus.SELECT_DISCARDED]:
            self.draw_cards_and_tokens([x.card_id for x in Game.get_available_actions(self.game)], [])

    def draw_editor(self):
        if self.editor_pos is not None:
            board_cards = [board_card.card.id
                           for row in self.game.cards_board.card_places
                           for board_card in row
                           if not board_card.is_taken]
            card_ids = [card.id for card in self.game.cards_board.cards]
            purple_card_ids = [card.id for card in self.game.cards_board.purple_cards]
            self.draw_cards_and_tokens(board_cards + card_ids + purple_card_ids, [])
        elif self.editor_wonder is not None:
            self.draw_wonders()
        else:
            for sprite in self.card_sprites:
                sprite.draw()

            for sprite in self.wonder_sprites:
                sprite.draw()

    def draw_player(self, player_index: int):
        player: Player = self.game.players[player_index]
        self.draw_cards_and_tokens([x.id for x in player.cards], [x.name for x in player.progress_tokens])

    def draw_discard_pile(self):
        self.draw_cards_and_tokens([x.id for x in self.game.discard_pile], [x.name for x in self.game.rest_progress_tokens])

    def draw_wonders(self):
        self.wonder_list_sprites = []
        self.progress_tokens_list_sprites = []

        for i in range(EntityManager.wonders_count()):
            sprite = WonderSprite(i)
            sprite.x = (i % 4) * sprite.width
            sprite.y = self.height - sprite.height * (i // 4 + 1)
            self.wonder_list_sprites.append(sprite)
            sprite.draw()

        for i in range(EntityManager.progress_tokens_count()):
            name = EntityManager.progress_token(i).name
            sprite = ProgressTokenSprite(name)
            sprite.x = i * sprite.width
            sprite.y = 0
            if name in [x.name in self.game.rest_progress_tokens]:
                sprite.opacity = 128
            self.progress_tokens_list_sprites.append(sprite)
            sprite.draw()

    def draw_cards_and_tokens(self, cards: List[int], tokens: List[str]):
        card_ids = []
        self.card_list_sprites = []
        self.progress_tokens_list_sprites = []
        for color in CARD_COLOR:
            color_index = BONUSES_INDEX(color)
            card_ids.extend([card_id
                             for card_id in sorted(cards)
                             if color_index in EntityManager.card(card_id).bonuses])
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
        elif symbol == key._5:
            self.game = self.prev_game.clone()
            self.mcts = MCTSAgent(self.game)
            self.last_action = None
            self.state_updated()
        elif symbol == key._0:
            self.mode = Mode.GAME_BOARD
        elif symbol == key.ESCAPE:
            self.close()
        elif symbol == key.SPACE:
            if self.mode == Mode.GAME_BOARD:
                self.move()

    def apply_action(self, action: Action):
        self.prev_game = self.game.clone()
        self.game.apply_action(action)
        self.last_action = action
        # for agent in self.agents:
        #     agent.on_action_applied(action, self.state)
        self.deselect_wonder()
        self.state_updated()

    def move(self):
        if self.game.is_finished:
            return

        self.mcts.on_action_applied(self.last_action, self.game.clone())
        self.last_action = None

        actions = self.game.get_available_actions()
        selected_action = self.mcts.choose_action(self.game, actions)
        # selected_action = self.agents[self.state.current_player_index].choose_action(self.state, actions)
        self.best_move_label.text = str(selected_action)
        # Game.apply_action(self.state, selected_action)
        # for agent in self.agents:
        #     agent.on_action_applied(selected_action, self.state)
        # if Game.is_finished(self.state):
        #     print(f"Winner: {self.state.winner}")
        # self.state_updated()
