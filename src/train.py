from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description="Скрипт для обучения модели")

    parser.add_argument('--epochs', type=int, help='Количество эпох обучения', default=150)
    parser.add_argument('--batch', type=int, help='Количество изображений в пакете', default=8)
    parser.add_argument('--device', type=str, help='0 - CUDA, cpu - CPU', default="cpu")

    args = parser.parse_args()
    model = YOLO("yolo26n-pose.pt")

    model.train(
        data="configs/hand-keypoints.yaml",
        device=args.device,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        fraction=1.0,
        patience=30,
        save_period=-1,
        amp=True,
        workers=2,
        cos_lr=True,
        project="hand-tracking",
    )


if __name__ == "__main__":
    main()
