from ultralytics import YOLO
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/yolov8n.pt", type=str)
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)

    # Export tf model first as it generates .onnx file then export it to tflite
    model.export(format='tflite', int8=True)

    # However, that onnx file does not have image size exported correctly
    # so just export it to onnx format again to get correct onnx file we want
    model.export(format='onnx')
