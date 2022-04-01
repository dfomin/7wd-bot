from typing import Optional

from pyglet.image import ImageGrid
from pyglet.resource import image
from swd.entity_manager import EntityManager

WONDERS_SPRITE_LIST = [None, None, None, None, None, 0, 1, None, None, None, 3, 7, 5, 10, 9, 8, 2, 4, 11, 6]
CARDS_SPRITE_LIST = [
    -1, -2, None, None, None, None, None, None, None, None, None, None,
    72, None, None, None, None, None, None, None, None, None, None, None,
    55, 58, 59, 56, 60, 57, 66, 67, 68, 69, 70, 71,
    49, 48, 47, 51, 53, 52, 54, 65, 61, 62, 64, 63,
    33, 45, 42, 43, 44, 39, 38, 37, 40, 41, 50, 46,
    25, 24, 26, 27, 29, 30, 28, 31, 32, 35, 36, 34,
    14, 12, 15, 13, 22, 19, 20, 21, 16, 17, 18, 23,
    0, 5, 2, 1, 4, 3, 6, 7, 9, 10, 11, 8
]
TOKENS_SPRITE_LIST = [None, None, None, None, 8, 9, None, None, 4, 5, 6, 7, 0, 1, 2, 3]


class SpriteLoader:
    _wonders: Optional[ImageGrid] = None
    _cards: Optional[ImageGrid] = None
    _progress_tokens: Optional[ImageGrid] = None

    @classmethod
    def wonders(cls):
        if cls._wonders is None:
            cls._wonders = ImageGrid(image("resources/wonders_v3.webp"), 4, 5)
        return cls._wonders

    @classmethod
    def cards(cls):
        if cls._cards is None:
            cls._cards = ImageGrid(image("resources/buildings_v3.webp"), 8, 12)
        return cls._cards

    @classmethod
    def progress_tokens(cls):
        if cls._progress_tokens is None:
            cls._progress_tokens = ImageGrid(image("resources/progress_tokens_v3.webp"), 4, 4)
        return cls._progress_tokens

    @classmethod
    def wonder(cls, wonder_id: int):
        index = WONDERS_SPRITE_LIST.index(wonder_id)
        return cls.wonders()[index]

    @classmethod
    def card(cls, card_id: int):
        index = CARDS_SPRITE_LIST.index(card_id)
        return cls.cards()[index]

    @classmethod
    def progress_token(cls, token: str):
        token_names = EntityManager.progress_token_names()
        token_id = token_names.index(token)
        index = TOKENS_SPRITE_LIST.index(token_id)
        return cls.progress_tokens()[index]
