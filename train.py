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

    parser.add_argument("--data", default="coco.yaml", type=str,
                        help="")
    parser.add_argument("--transfer", default="yolov8n.pt", type=str,
                        help="")

    args = parser.parse_args()

    model = YOLO(args.model)

    if args.transfer is not None and len(args.transfer) > 0:
        print(f'transfer weight from {args.transfer}')
        model = model.load(args.transfer)

    # Display model information (optional)
    model.info()
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          save_period=args.save_period)
