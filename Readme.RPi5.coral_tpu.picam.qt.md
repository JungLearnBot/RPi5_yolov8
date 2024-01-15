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

Now install pycoral from: https://github.com/oberluz/pycoral/releases/
This is not the official build from google. However, there is no official release for python 3.11 but I really want to use python 3.11 due to picamera2.


```
wget https://github.com/oberluz/pycoral/releases/download/2.13.0/pycoral-2.13.0-cp311-cp311-linux_aarch64.whl
pip install pycoral-2.13.0-cp311-cp311-linux_aarch64.whl --no-deps
pip install tflite-runtime==2.14.0
```

Please note this is work-around-hack and only for testing. If you try to install pycoral-2.13.0-cp311-cp311-linux_aarch64.whl, it will not work as the compiled pycoral depends on tflite-runtime==2.13.0 but 2.13 does not exist for python 3.11. I tried to compile pycoral myself targeting specific version of tensorflow but it was way too much work.


I also tried to downgrade python to python 3.10 and install pycoral-2.13.0-cp310-cp310-linux_aarch64.whl
and recompile libcamera which is required by picamera2. I managed to compile libcamera for python 3.10 with python bindings but it did not load for python 3.10 for some reason and it only showed very vauge error message. I also found other people had the same issue. So I gave up on this approach.

At the end, I decided to force-ignore dependency of tflite-runtime==2.13.0 and then install tflite-runtime==2.14.0. I was going to give up on picamera2+pycoral combination if this did not work but it worked.
Good enough for me now ;)

### Export model to edge tpu

```
yolo export model=yolov8n.pt format=edgetpu imgsz=192,192
```

Note, the resoultion is forced to 192x192 since it was the only resolution that worked. Any higher resolution failed to run on the M.2 accelerator(https://coral.ai/products/m2-accelerator-bm).
It seems like other people got to the same point (https://github.com/ultralytics/ultralytics/issues/4089)

## Run 
```
export PYTHONUTF8=1 
python main_picam_coral_tpu.py
```
