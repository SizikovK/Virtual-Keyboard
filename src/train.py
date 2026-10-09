from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Скрипт для обучения модели")
model = YOLO("yolo26n-pose.pt")

parser.add_argument('--epochs', type=int, help='Количество эпох обучения', default=10)
parser.add_argument('--batch', type=int, help='Число тренировочных объектов обрабатываемых одновременно', default=8)

args = parser.parse_args()
epochs = args.epochs
batch = args.batch

if epochs is not None or batch is not None:
    model.train(
        data="configs/hand-keypoints.yaml",
        device="cpu",
        epochs=args.epochs,
        imgsz=416,
        batch=args.batch,
        fraction=0.1,
        patience=5,
        save_period=-1,
        amp=True,
        workers=2,
        project="hand-tracking",
    )
else:
    print("Проверьте корректность всех аргументов")