from pynput.keyboard import Controller
from torch import Tensor

from buttons import ButtonsManager
import cv2


class VirtualKeyboard:
    def __init__(self):
        self.buttons_manager = ButtonsManager()
        self.keyboard = Controller()
        self.last_pt1: cv2.typing.Point = (20, 20)
        self.last_pt2: cv2.typing.Point = (100, 100)

    def _get_last_pts(self) -> tuple[cv2.typing.Point, cv2.typing.Point]:
        return self.last_pt1, self.last_pt2

    def _set_new_last_pts(self):
        pt1, pt2 = self._get_last_pts()

        pt1 = (pt1[0] + 110, pt1[1])
        pt2 = (pt2[0] + 110, pt2[1])

        self.last_pt1 = pt1
        self.last_pt2 = pt2

    def press_key(self, key):
        self.keyboard.press(key)

    def release_key(self, key):
        self.keyboard.release(key)

    def add_key(self, key: str):
        key = key.upper()
        key_press = lambda: self.keyboard.press(key)
        key_release = lambda: self.keyboard.release(key)

        pt1, pt2 = self._get_last_pts()

        self.buttons_manager.add_button(
            pt1=pt1,
            pt2=pt2,
            color=(0, 255, 0),
            thickness=2,
            text=key,
            func=key_press,
            end_func=key_release,
        )

        self._set_new_last_pts()

    def render_buttons(self, img: cv2.typing.MatLike):
        self.buttons_manager.render_buttons(img)

    def render_keypoints(self, img: cv2.typing.MatLike, hand: Tensor, keypoints: list[int]):
        for keypoint in keypoints:
            x, y = map(int, hand[keypoint].tolist())
            self.buttons_manager.render_circle(img, x, y)
            self.buttons_manager.process_function(x, y)
