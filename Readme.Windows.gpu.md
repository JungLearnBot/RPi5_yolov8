# Windows 11

## Prerequisite
CUDA GPU is needed for GPU training.
Tested with RTX 2080 super(8GB), windows 11.

## Create environment
````
conda create -n yolov8 python=3.9
conda activate yolov8
pip install ultralytics==8.0.221
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````

## Export yolov8n to tflite and onnx format 
python export_models.py

## Run 

### Run yolov8n.pt

- --debug option show debug window with annotation, dood for debugging but slows down the fps
- --print_fps option prints fps every 1 sec.
```
python main.py --debug
python main.py --print_fps
```

### Run exported models
```
python main.py --model=./models/yolov8n.onnx --debug
python main.py --model=./models/yolov8n_saved_model/yolov8n_integer_quant.tflite --debug
```


## Train

Training yolov8n - low resolution(320) with coco dataset.

```
python train.py
# or
python train.py --model=yolov8n.yaml --imgsz=320 --batch 128
```