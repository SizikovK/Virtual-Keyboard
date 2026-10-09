from dataclasses import dataclass
from typing import Callable, Any, TypedDict, List
import cv2


@dataclass
class Button:
    pt1: cv2.typing.Point
    pt2: cv2.typing.Point
    b_text: str
    color: cv2.typing.Scalar | None
    thickness: int
    is_clicked: bool
    function: Callable[..., Any]
    end_function: Callable[[], Any] | None


class ButtonsManager:
    def __init__(self):
        self.buttons: List[Button] = []

    def add_button(
        self,
        pt1: cv2.typing.Point,
        pt2: cv2.typing.Point,
        thickness: int,
        text: str,
        func: Callable[[], Any],
        color: cv2.typing.Scalar | None = None,
        end_func: Callable[[], Any] | None = None,
    ):
        self.buttons.append(
            Button(
                pt1=pt1,
                pt2=pt2,
                b_text=text,
                color=color,
                thickness=thickness,
                is_clicked=False,
                function=func,
                end_function=end_func,
            )
        )

    @staticmethod
    def _calculate_color(button: Button) -> cv2.typing.Scalar:
        color = button.color
        if color is None:
            color = (0, 255, 0)

        if button.is_clicked:
            color = (0, 0, 255)

        return color

    @staticmethod
    def render_circle(img: cv2.typing.MatLike, x: int, y: int):
        cv2.circle(
            img,
            (x, y),
            8,
            (0, 255, 0),
            1,
        )

    def render_buttons(self, img: cv2.typing.MatLike,):
        for button in self.buttons:
            color = self._calculate_color(button)

            cv2.rectangle(
                img,
                button.pt1,
                button.pt2,
                color,
                button.thickness
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6

            (text_w, text_h), baseline = cv2.getTextSize(
                button.b_text,
                font,
                font_scale,
                button.thickness
            )

            center_x = (button.pt1[0] + button.pt2[0]) // 2
            center_y = (button.pt1[1] + button.pt2[1]) // 2

            text_x = center_x - text_w // 2
            text_y = center_y + (text_h - baseline) // 2

            cv2.putText(
                img,
                button.b_text,
                (text_x, text_y),
                font,
                font_scale,
                color,
                button.thickness
            )

    @staticmethod
    def _is_into_rectangle(
        pt1: cv2.typing.Point,
        pt2: cv2.typing.Point,
        x, y,
    ):
        x1, y1 = pt1
        x2, y2 = pt2
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        return xmin <= x <= xmax and ymin <= y <= ymax

    def process_function(self, points):
        for button in self.buttons:
            is_inside = any(
                self._is_into_rectangle(button.pt1, button.pt2, x, y)
                for x, y in points
            )
            if is_inside:
                if not button.is_clicked:
                    button.function()
                    button.is_clicked = True
            elif button.is_clicked:
                if button.end_function is not None:
                    button.end_function()
                button.is_clicked = False
