from typing import Tuple, Optional

from pyglet.sprite import Sprite

from swd_bot.editor.sprite_loader import SpriteLoader


class CardSprite(Sprite):
    def __init__(self, card_id: int, pos: Optional[Tuple[int, int]] = None):
        super().__init__(SpriteLoader.card(card_id))

        self.scale = 0.5

        self.card_id = card_id
        self.pos = pos
