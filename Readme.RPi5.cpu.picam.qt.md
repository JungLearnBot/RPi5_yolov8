# Raspberry Pi 5 OS (64bit)
Pi Camera Module 3 does not play well with OpenCV VideoCapture. There is some work around in some other OS but I haven't found any that works in Raspberry Pi 5 OS (64)

The only solution seems to use **picamera2** package but it does not install virtual environment:
```
https://github.com/raspberrypi/picamera2/issues/446
https://github.com/raspberrypi/picamera2/issues/503
```
due to libcamera can only be installed by sudo apt install

So as a workaround, I fixed the conda python version to 3.11, created environment 
and copied necessary libraries so I can have a separate conda environment.

There is another problem I encountered with **picamera2** is that it doesn't work with cv2.imshow. It gets stuck and frozen. I couldn't find obvious solution. So I decided to use Qt for visualisation.
But this time Qt have problem with opencv in Pi OS, so as a workround **opencv-python-headless** is installed.


## Prerequisite
You need Pi Camera Module 3. (only tested with Pi Camera Module 3)

CMake is needed for tflite format later
````
sudo apt-get install cmake 
````

## Create environment
````
conda create -n yolov8_picam python=3.11
conda activate yolov8_picam
pip install ultralytics==8.0.221
pip install tensorflow==2.13.1
pip install onnx==1.15.0 onnxruntime==1.16.3 onnxsim==0.4.33
pip install -U --force-reinstall flatbuffers==23.5.26
````

Installing tensorflow and onnx are required if you want to convert yolov8 model to tflite.
I also had to upgrade flatbuffers for tflite export

As libcamera does not get installed thru pip install we do a hack, install on global python.
And copy the libraries to conda environment. This only works because we set the python version to 3.11.


```
sudo apt install -y python3-libcamera python3-kms++
sudo apt install -y python3-pyqt5 python3-prctl libatlas-base-dev ffmpeg python3-pip
```

```
pip install picamera2
sudo cp -r /usr/lib/python3/dist-packages/libcamera ~/miniconda3/envs/yolov8_picam/lib/python3.11/site-packages/
sudo cp -r /usr/lib/python3/dist-packages/pykms ~/miniconda3/envs/yolov8_picam/lib/python3.11/site-packages/

cd ~/miniconda3/envs/yolov8_picam/lib
mv -vf libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./libstdc++.so.6
```

Now install QT5

```
conda install pyqt
pip uninstall opencv-python
pip install opencv-python-headless==4.6.0.66
```


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

--debug option show debug window with annotation, good for debugging but slows down the fps

```
python main_picam.py --debug
```

### Run exported models
```
python main_picam --model=./models/yolov8n.onnx --debug
python main_picam --model=./models/yolov8n_saved_model/yolov8n_integer_quant.tflite
```