from pyglet.sprite import Sprite

from swd_bot.editor.sprite_loader import SpriteLoader


class WonderSprite(Sprite):
    def __init__(self, wonder_id: int):
        super().__init__(SpriteLoader.wonder(wonder_id))

        self.scale = 0.5

        self.wonder_id = wonder_id
