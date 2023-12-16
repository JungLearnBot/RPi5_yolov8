import os

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from yolo_manager import YoloDetectorWrapper
from utils import SimpleFPS, draw_fps, draw_annotation
import argparse
import time
from pathlib import Path
from picamera2 import Picamera2
import subprocess


class RecordingSaver:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def __init__(self, file_path):
        self.video_out = cv2.VideoWriter(file_path, RecordingSaver.fourcc, 30.0, (640, 480))

    def write(self, frame):
        self.video_out.write(frame)

    def __del__(self):
        self.video_out.release()


class RecordingManager:
    def __init__(self):
        self.on_duration = -1
        self.start_time = -1
        self.last_flag = False

        self.saver = None
        self.video_index = 0
        self.video_out_dir = Path('./clips/')
        os.makedirs(self.video_out_dir, exist_ok=True)

    def record(self, on_duration):
        self.on_duration = on_duration
        self.start_time = time.time()

    def update(self, frame):
        is_on, was_on = self.get_flags()

        if is_on:
            if self.saver is None:
                out_filename = self.video_out_dir / f'video_{self.video_index:04d}.mp4'
                self.video_index += 1
                self.saver = RecordingSaver(str(out_filename))

            self.saver.write(frame)
        else:
            # finish saver
            if was_on:
                del self.saver
                self.saver = None

    def get_flags(self):
        last_flag = self.last_flag
        is_on_flag = time.time() - self.start_time <= self.on_duration
        self.last_flag = is_on_flag
        return is_on_flag, last_flag

    def __del__(self):
        if self.saver is not None:
            del self.saver

class VideoBroadcaster:
    def __init__(self, rtmp_url):

        # gather video info to ffmpeg
        fps = 30
        width = 640
        height = 480

        # command and params for ffmpeg
        command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                '-loglevel', 'error',
                # "-fflags", "nobuffer",
                # "-flags", "low_delay",
                rtmp_url]

        # using subprocess and pipe to fetch frame data
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    def update(self, frame):
        self.p.stdin.write(frame.tobytes())


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, rotate_camera, rtmp_url):
        super().__init__()
        self.detect_frame = True
        self.should_run = True
        self.rotate_camera = rotate_camera

        self.recording_manager = RecordingManager()
        self.rtmp_url = rtmp_url

    def run(self):
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size":(640,480),"format":"RGB888"}, raw={"size": (640, 480)})
        # camera_config = picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
        picam2.configure(camera_config)
        picam2.start()

        video_broadcaster = None
        if self.rtmp_url is not None:
            print(f'streaming to {self.rtmp_url}')
            video_broadcaster = VideoBroadcaster(rtmp_url=self.rtmp_url)

        while self.should_run:
            frame = picam2.capture_array()
            
            if frame is not None:

                if self.rotate_camera:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                if video_broadcaster is not None:
                    video_broadcaster.update(frame)
                self.recording_manager.update(frame)

                if self.detect_frame:
                    self.change_pixmap_signal.emit(frame)
                    self.detect_frame = False
            else:
                time.sleep(0.0001)

        del self.recording_manager
        self.recording_manager = None
        print('VideoThread finished!')

    def stop(self):
        self.should_run = False

    def record_video(self):
        if self.recording_manager is not None:
            self.recording_manager.record(5)  # once triggered, record minimum 5 sec


class FrameCounter:
    def __init__(self, detection_target_indices, num_frames):
        self.num_frames = num_frames
        self.detection_target_indices = detection_target_indices
        self.counter = 0

    def check_detection_results(self, detection_results):

        target_found = False
        for detection_result in detection_results:
            
            if len(detection_result.boxes) > 0:
                cls_id = int(detection_result.boxes.cls[0])
                if cls_id in self.detection_target_indices:
                    target_found = True
                    break

        if target_found:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter >= self.num_frames


class App(QWidget):
    def __init__(self, camera_test_only, rotate_camera, rtmp_url):
        super().__init__()

        self.camera_test_only = camera_test_only

        if camera_test_only:
            self.yolo_detector = None
        else:
            self.yolo_detector = YoloDetectorWrapper(args.model)

        target_indices = {14, 15, 16}  # bird, cat, dog
        # if we find targets in least 2 frames in a row, we start recording
        self.detection_counter = FrameCounter(target_indices, 2)

        self.setWindowTitle("Qt UI")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.fps_util = SimpleFPS()

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread(rotate_camera, rtmp_url)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""

        if self.yolo_detector is None:
            display_img = cv_img
        else:
            results = self.yolo_detector.predict(cv_img)
            display_img = draw_annotation(cv_img, self.yolo_detector.get_label_names(), results)

            if self.detection_counter.check_detection_results(results):
                self.thread.record_video()

        fps, _ = self.fps_util.get_fps()
        draw_fps(display_img, fps)

        qt_img = self.convert_cv_qt(display_img)
        self.image_label.setPixmap(qt_img)
        self.thread.detect_frame = True

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        print('closeEvent')
        self.thread.stop()
        self.thread.wait()
        event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/yolov8n.pt")
    parser.add_argument("--rtmp_url", type=str, default=None)
    
    parser.add_argument('--camera_test', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rotate_camera', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    app = QApplication(sys.argv)
    a = App(camera_test_only=args.camera_test, rotate_camera=args.rotate_camera,
            rtmp_url=args.rtmp_url)
    a.show()
    sys.exit(app.exec_())
