import cv2
from numba.scripts.generate_lower_listing import description
from ultralytics import YOLO
import argparse
from vkeyboard import VirtualKeyboard


def mirror_frames(predictor):
    paths, frames, info = predictor.batch
    predictor.batch = (
        paths,
        [cv2.flip(frame, 1) for frame in frames],
        info,
    )

def main():
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

    vkeyboard = VirtualKeyboard()
    vkeyboard.add_key("a")
    vkeyboard.add_key("w")
    vkeyboard.add_key("d")
    vkeyboard.add_key("s")

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
            vkeyboard.render_buttons(annotated_frame)

            if result.keypoints is not None:
                for hand in result.keypoints.xy:
                    vkeyboard.render_keypoints(annotated_frame, hand, [13,])

            cv2.imshow("Hand tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        results.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit")
