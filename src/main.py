import cv2
from ultralytics import YOLO
import argparse
import logging
from ultralytics.utils import LOGGER

from buttons import ButtonsManager
from vkeyboard import VirtualKeyboardController


class StreamWaitFilter(logging.Filter):
    def filter(self, record):
        return "Waiting for stream " not in record.getMessage()


def mirror_frames(predictor):
    paths, frames, info = predictor.batch
    predictor.batch = (
        paths,
        [cv2.flip(frame, 1) for frame in frames],
        info,
    )

def main():
    LOGGER.addFilter(StreamWaitFilter())
    parser = argparse.ArgumentParser(description="Виртуальная клавиатура")
    parser.add_argument('--weights', type=str, help="Файл с весами модели")
    args = parser.parse_args()
    weights = args.weights

    try:
        if weights is None:
            raise ValueError("Аргумент --weights не должен быть пустым!")
    except ValueError as e:
        print("Error:" + e.args[0])
        return

    model = YOLO(weights)

    button_manager = ButtonsManager()
    vkeyboard = VirtualKeyboardController(button_manager)

    vkeyboard.add_key("a")
    vkeyboard.add_key("w")
    vkeyboard.add_key("d")
    vkeyboard.add_key("s")
    vkeyboard.add_key("space")
    vkeyboard.add_key("shift")

    model.add_callback("on_predict_batch_start", mirror_frames)

    results = model.predict(
        source=0,
        imgsz=640,
        stream=True,
        conf=0.15,
        verbose=False,
    )

    try:
        for result in results:
            annotated_frame = result.plot()
            points = []
            if result.keypoints is not None:
                for hand in result.keypoints.xy:
                    points.extend(vkeyboard.render_keypoints(annotated_frame, hand, [4, 8, 12, 16, 20]))
            vkeyboard.buttons_manager.process_function(points)
            vkeyboard.render_buttons(annotated_frame)

            cv2.imshow("Hand tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vkeyboard.buttons_manager.process_function([])
        results.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit")
