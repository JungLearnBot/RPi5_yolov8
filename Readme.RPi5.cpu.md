# Raspberry Pi 5 OS (64bit)

## Prerequisite
CMake is needed for tflite format later
````
sudo apt-get install cmake 
````

## Create environment
````
conda create -n yolov8_cpu python=3.9
conda activate yolov8_cpu
pip install ultralytics==8.0.221
pip install tensorflow==2.13.1
pip install onnx==1.15.0 onnxruntime==1.16.3 onnxsim==0.4.33
pip install -U --force-reinstall flatbuffers==23.5.26
````

Installing tensorflow and onnx are required if you want to convert yolov8 model to tflite.
I also had to upgrade flatbuffers for tflite export

## Export yolov8n to tflite and onnx format
```
python export_models.py
python export_models.py --format onnx
```
Note, It seems like there is a bug when I export tflite and onnx at the same time.
So for now export them separately.

## Run 

Set utf8 format for python if you are getting strange error with latin1 encoding
```
export PYTHONUTF8=1 
```

### Run yolov8n.pt

- --debug option show debug window with annotation, good for debugging but slows down the fps
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