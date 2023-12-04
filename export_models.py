from ultralytics import YOLO
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/yolov8n.pt", type=str)
    parser.add_argument("--format", default="tflite", type=str)
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)

    if args.format == 'tflite':
        model.export(format='tflite', int8=True)
    elif args.format  == 'onnx':
        model.export(format='onnx')
    else:
        raise Exception(f'unknown format: {args.format}')
