from Xlib import XK
from pynput.keyboard import Controller, KeyCode
from torch import Tensor

from buttons import ButtonsManager
import cv2


class VirtualKeyboardController:
    def __init__(self, buttons_manager: ButtonsManager):
        self.buttons_manager = ButtonsManager()
        self.keyboard = Controller()
        self.last_pt1: cv2.typing.Point = (20, 20)
        self.last_pt2: cv2.typing.Point = (80, 80)

    def _get_last_pts(self) -> tuple[cv2.typing.Point, cv2.typing.Point]:
        return self.last_pt1, self.last_pt2

    def _set_new_last_pts(self):
        pt1, pt2 = self._get_last_pts()

        pt1 = (pt1[0] + 90, pt1[1])
        pt2 = (pt2[0] + 90, pt2[1])

        self.last_pt1 = pt1
        self.last_pt2 = pt2

    def press_key(self, key):
        self.keyboard.press(key)

    def release_key(self, key):
        self.keyboard.release(key)

    def add_key(self, key: str):
        key_name = key.lower()

        if key_name == "shift":
            key_name = "Shift_L"
        elif key_name == " ":
            key_name = "space"

        key_code = KeyCode.from_vk(XK.string_to_keysym(key_name))
        key_press = lambda: self.keyboard.press(key_code)
        key_release = lambda: self.keyboard.release(key_code)

        pt1, pt2 = self._get_last_pts()

        self.buttons_manager.add_button(
            pt1=pt1,
            pt2=pt2,
            color=(0, 255, 0),
            thickness=2,
            text="SPACE" if key_name == "space" else key.upper(),
            func=key_press,
            end_func=key_release,
        )

        self._set_new_last_pts()

    def render_buttons(self, img: cv2.typing.MatLike):
        self.buttons_manager.render_buttons(img)

    def render_keypoints(self, img: cv2.typing.MatLike, hand: Tensor, keypoints: list[int]):
        points = []
        for keypoint in keypoints:
            x, y = map(int, hand[keypoint].tolist())
            if x == 0 and y == 0:
                continue
            self.buttons_manager.render_circle(img, x, y)
            points.append((x, y))
        return points
