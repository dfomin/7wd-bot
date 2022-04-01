from pyglet.sprite import Sprite

from swd_bot.editor.sprite_loader import SpriteLoader


class ProgressTokenSprite(Sprite):
    def __init__(self, progress_token: str):
        super().__init__(SpriteLoader.progress_token(progress_token))

        self.scale = 0.5

        self.progress_token = progress_token
