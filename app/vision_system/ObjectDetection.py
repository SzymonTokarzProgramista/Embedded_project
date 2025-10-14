from ImageCapture import ImageCapture
from Preprocessing import Preprocessing
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, model_path="models/yolov5s.pt"):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = YOLO(self.model_path)
        return model

    def detect_objects(self, image):
        results = self.model(image)
        return results