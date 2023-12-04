from ultralytics import YOLO
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.yaml", type=str,
                        help="specify yaml or saved model. e.g)last.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--save_period", type=int, default=50)
    args = parser.parse_args()

    model = YOLO(args.model)

    # Display model information (optional)
    model.info()

    results = model.train(data='coco.yaml', epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          save_period=args.save_period)
